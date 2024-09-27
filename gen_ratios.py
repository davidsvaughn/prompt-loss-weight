"""
Compute Generation Ratios (Rg) for HuggingFace Instruction Datasets
-------------------------------------------------------------------

The generation ratio (Rg) of an instruction dataset is defined as the mean ratio of 
the completion length to the prompt length. Ideally, Rg should be computed on tokenized text, 
since token counts are more relevant to LLM processing.  But using character counts can also 
yield a good estimate of Rg, when tokenization is too time-consuming.  This script allows for
both options, through the 'tokenize' argument in the ScriptArguments class (default is True).

"""
#--------------------------------------------------------------------------------------------------
import random, sys
import numpy as np
import bisect
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field
from typing import List

#--------------------------------------------------------------------------------------------------

# define command-line arguments class
@dataclass
class ScriptArguments:
    model_id:       str             = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "The HuggingFace model id"})
    max_samples:    int             = field(default=100000, metadata={"help": "Max number of random samples to use for Rg estimation"})
    tokenize:       bool            = field(default=True, metadata={"help": "Tokenize before computing Rg"})
    plot:           bool            = field(default=True, metadata={"help": "Plot Rg distribution for each dataset"})

# parse script arguments
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args()

print("\n-------- Script Arguments ------------")
for key, value in vars(script_args).items():
    print(f"{key}: {value}")

#--------------------------------------------------------------------------------------------------

# define dataset arguments class
@dataclass
class DatasetArguments:
    dataset_id:         List[str]       = field(default_factory=list)   # HuggingFace dataset ID
    name:               str             = field(default="")             # dataset name
    prompt_template:    str             = field(default="")             # custom prompt template
    completion_template:str             = field(default="")             # custom completion template


# HuggingFace instruction datasets
datasets = [
    DatasetArguments(
        name = "Alpaca",
        dataset_id = ["yahma/alpaca-cleaned"],
        prompt_template = f"{{instruction}}\n{{input}}",
        completion_template = f"{{output}}",
    ),
    DatasetArguments(
        name = "OpenHermes",
        dataset_id = ["teknium/openhermes"],
        prompt_template = f"{{instruction}}\n{{input}}",
        completion_template = f"{{output}}",
    ),
    DatasetArguments(
        name = "Python-18k",
        dataset_id = ["iamtarun/python_code_instructions_18k_alpaca"],
        prompt_template = f"{{instruction}}\nInput:{{input}}",
        completion_template = f"{{output}}",
    ),
    DatasetArguments(
        name = "Databricks-Dolly-15k",
        dataset_id = ["databricks/databricks-dolly-15k"],
        prompt_template = "Follow the instruction based on the context." \
                          f"\nInstruction:{{instruction}}\nContext:{{context}}",
        completion_template = f"{{response}}",
    ),
    DatasetArguments(
        name = "OpenOrca",
        dataset_id = ["polinaeterna/OpenOrca"],
        prompt_template = f"{{system_prompt}}\n{{question}}",
        completion_template = f"{{response}}",
    ),
    DatasetArguments(
        name = "SAMSum",
        dataset_id = ["knkarthick/samsum"],
        prompt_template = f"Summarize the following dialogue:\n{{dialogue}}",
        completion_template = f"{{summary}}",
    ),
    DatasetArguments(
        name = "XSum",
        dataset_id = ["EdinburghNLP/xsum"],
        prompt_template = f"Summarize the following document:\n{{document}}",
        completion_template = f"{{summary}}",
    ),
        DatasetArguments(
        name = "RACE",
        dataset_id = ["ehovy/race", "all"], # use 'all' to load all splits
        prompt_template = "Choose the correct option based on the context." \
                        f"\nContext:{{article}}\nQuestion:{{question}}\nOptions:{{options}}",
        completion_template = f"{{answer}}",
    ),
]

#--------------------------------------------------------------------------------------------------

def compute_generation_ratios(tokenizer, dataset_args, script_args):
    print(f"\nLoading {dataset_args.name} dataset...")

    # Load dataset from HuggingFace hub
    dataset = load_dataset(*dataset_args.dataset_id)
    
    # print splits and number of samples
    dataset_keys = list(dataset.keys())
    for k in dataset_keys:
        print(f"Number of {k} samples: {len(dataset[k])}")

        # if number of samples is more than max_samples, randomly select max_samples
        if len(dataset[k]) > script_args.max_samples:
            dataset[k] = dataset[k].shuffle(seed=42).select(range(script_args.max_samples))
            print(f"Randomly selected {script_args.max_samples} samples from {k} split")

    # tokenize and encode batch of samples
    def tokenize_batch(batch):
        tokenized_text = tokenizer(batch["text"], add_special_tokens=False, return_offsets_mapping=True,)
        data = {k: tokenized_text[k] for k in tokenized_text.keys()}

        # use offset_mappings to find the index of the last token of the prompt
        gen_ratio = []
        for offset_mapping, idx in zip(data["offset_mapping"], batch["idx"]):
            num_prompt_tokens = bisect.bisect_right(offset_mapping, (idx,))-1
            gen_ratio += [(len(offset_mapping) - num_prompt_tokens) / num_prompt_tokens] # compute Rg with token counts

        data["gen_ratio"] = gen_ratio
        del data["offset_mapping"]
        return data
    
    # apply instruction template and chat template to each sample
    def format_sample(sample):
        # get the instruction and correct output
        user_text = dataset_args.prompt_template.format(**sample)
        asst_text = dataset_args.completion_template.format(**sample)

        # use the tokenizer's chat template to format the prompt/completion chat dialogue
        messages = [{"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text }]

        sample["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_txt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

        sample["idx"] = num_prompt_chars = len(prompt_txt)
        sample["gen_ratio"] = (len(sample["text"]) - num_prompt_chars) / num_prompt_chars # compute Rg with character counts
        return sample

    # format each sample
    dataset = DatasetDict({ k : dataset[k].map(format_sample) for k in dataset_keys })

    if script_args.tokenize: # tokenize dataset (batched)
        dataset = DatasetDict({ k : dataset[k].map(tokenize_batch, batched=True) for k in dataset_keys })

    # collect generation ratios over all splits
    gen_ratios = np.sort(np.concatenate([dataset[k]["gen_ratio"] for k in dataset_keys]))

    # remove top and bottom q quintiles
    q = 0.0025
    gen_ratios = gen_ratios[int(q*len(gen_ratios)):int((1-q)*len(gen_ratios))]
    return gen_ratios

#--------------------------------------------------------------------------------------------------

# load tokenizer (for chat template)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

# loop over all datasets and compute generation ratios
avg_gen_ratios, all_gen_ratios = [], []
for dataset_args in datasets:
    gen_ratios = compute_generation_ratios(tokenizer, dataset_args, script_args)
    Rg = np.mean(gen_ratios)
    avg_gen_ratios.append((dataset_args.name, Rg))
    all_gen_ratios.append((dataset_args.name, gen_ratios))
    print(f"{dataset_args.name} Rg\t= {Rg:.4g}")

# print final result table
headers = ["Dataset", "Generation Ratio (Rg)"]
table_data = [(name, f"{value:.3f}") for name, value in avg_gen_ratios]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

#--------------------------------------------------------------------------------------------------

# plot Rg distributions for all datasets
if script_args.plot:

    plt.rcParams.update({'font.size': 14})
    n_datasets = len(all_gen_ratios)
    fig = plt.figure(figsize=(10*n_datasets + 2, 10))

    # Create grid specification
    gs = gridspec.GridSpec(2, n_datasets)
    gs.update(left=0.05, right=0.95, hspace=0.2, wspace=0.1)

    for i, (name, gen_ratios) in enumerate(all_gen_ratios):
        # Create subplot for normal KDE plot
        ax1 = plt.subplot(gs[0, i])
        kde = gaussian_kde(gen_ratios)
        x_range = np.linspace(gen_ratios.min(), gen_ratios.max(), 1000)
        ax1.plot(x_range, kde(x_range))
        ax1.fill_between(x_range, kde(x_range), alpha=0.5)
        ax1.set_title(f"{name}: Rg", fontsize=18)
        ax1.set_yticklabels([])
        ax1.tick_params(axis='x', which='major', labelsize=14)
        ax1.set_ylabel('')  # Remove y-axis label

        # Create subplot for log-scaled KDE plot
        ax2 = plt.subplot(gs[1, i])
        log_x_range = np.logspace(np.log10(gen_ratios.min()), np.log10(gen_ratios.max()), 1000)
        log_kde = gaussian_kde(np.log10(gen_ratios))
        ax2.plot(log_x_range, log_kde(np.log10(log_x_range)))
        ax2.fill_between(log_x_range, log_kde(np.log10(log_x_range)), alpha=0.5)
        ax2.set_xscale('log')
        ax2.set_title("Log(Rg)", fontsize=18)
        ax2.set_yticklabels([])
        ax2.tick_params(axis='x', which='major', labelsize=14)
        ax2.set_ylabel('')  # Remove y-axis label

    plt.tight_layout()
    plt.show()
