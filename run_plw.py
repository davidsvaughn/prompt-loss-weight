"""
Instruction Tuning with Prompt-Loss-Weight
------------------------------------------

This module demonstrates an end-to-end pipeline for fine-tuning a HuggingFace Causal Language Model (CLM) 
on an instruction dataset using a tunable Prompt-Loss Weight (PLW) parameter. By default this script uses the 
RACE multiple choice dataset, but can be easily adapted to other instruction datasets, as long as the correct 
prompt and completion templates are provided. Likewise, the default model is the Llama-2-7b-chat-hf model, 
but can be changed by specifying a different model_id.

Main sections:
1. GPU Setup and Utilities
2. Argument Parsing
3. Logging setup and random seed initialization
4. Sample Packing Implementation
5. Dataset Loading and Preprocessing
6. Model Initialization
7. Custom Metrics
8. Custom Trainer & Custom Loss

Usage: to use defaults, run script without any command line arguments

    single GPU training:
        python run_plw.py [args]

    multi-GPU training:
        torchrun --nproc_per_node [num_gpus] run_plw.py [args]

"""
#--------------------------------------------------------------------------------------------------
# 1. GPU Setup and Utilities - housekeeping related to single/multi-GPU training

import os,sys
import torch

if 'LOCAL_RANK' not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # non-DDP training (single GPU)

# detect if this is the rank=0 (or only) GPU process
def is_main():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    # also return True if not using DDP, or only one GPU is available
    return True

# decorator to run 'func' only on main (rank=0) GPU process
# - all other GPU processes will just return None
# - for DDP / multi-GPU training
def main(func):
    def wrapper(*args, **kwargs):
        if is_main():
            result = func(*args, **kwargs)
            # Synchronize all the processes
            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Wait for rank 0 to finish
            return result
        else:
            # If not rank 0, wait for rank 0 to finish
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return None
    return wrapper

# print to stdout only on main GPU process
@main
def printmain(s):
    print(s)

# show available GPUs
@main
def print_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"Number of CUDA devices: {num_gpus}")
        for idx, name in enumerate(gpu_names):
            print(f"GPU {idx}: {name}")
    else:
        print("CUDA is not available.")
print_gpus()

#--------------------------------------------------------------------------------------------------
# 2. Argument Parsing
# - cmd line arguments take precedence over these defaults
# - can be run with no cmd line arguments, or with any of the following
# - all arguments are logged to wandb

from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, asdict, field
from typing import List, Optional, cast

@dataclass
class ScriptArguments:
    model_id:       str             = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "The HuggingFace model id"})
    dataset_id:     List[str]       = field(default_factory=lambda: ["ehovy/race", "all"], metadata={"help": "The HuggingFace dataset id"})
    prompt_template:str             = field(default="Choose the correct option based on the context.\nContext:{article}\nQuestion:{question}\nOptions:{options}", 
                                            metadata={"help": "The prompt template to use"})
    completion_template: str        = field(default="{answer}", metadata={"help": "The completion template to use"})
    data_dir:       Optional[str]   = field(default='~/data', metadata={"help": "The directory to store the data"})
    rand_seed:      Optional[int]   = field(default=1234, metadata={"help": "The random seed to use"})
    max_seq_length: Optional[int]   = field(default=2048, metadata={"help": "The maximum sequence length"})
    subsample_train:Optional[float] = field(default=2000, metadata={"help": "The number of training samples to use"})
    subsample_eval: Optional[float] = field(default=500, metadata={"help": "The number of evaluation samples to use"})
    max_samples:    Optional[int]   = field(default=500000, metadata={"help": "The maximum number of samples to load"})
    lora_alpha:     Optional[int]   = field(default=64, metadata={"help": "The LoRA alpha parameter"})
    lora_r:         Optional[int]   = field(default=64, metadata={"help": "The LoRA r parameter"})
    lora_dropout:   Optional[float] = field(default=0.1, metadata={"help": "The LoRA dropout rate"})
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"help": "The attention implementation to use"})
    use_4bit:       Optional[bool]  = field(default=False, metadata={"help": "Whether to use 4-bit quantization"})
    use_double_quant:Optional[bool] = field(default=False, metadata={"help": "Whether to use double quantization"})
    shuffle:        Optional[bool]  = field(default=False, metadata={"help": "Whether to shuffle the training data"})
    prompt_loss_weight: Optional[float] = field(default=1.0, metadata={"help": "The prompt loss weight"})

# instantiate default training arguments
training_args = TrainingArguments(
    num_train_epochs                = 10,       # number of training epochs
    per_device_train_batch_size     = 8,        # batch size per device during training
    per_device_eval_batch_size      = 16,       # batch size for evaluation
    gradient_accumulation_steps     = 1,        # number of steps before performing a backward/update pass
    gradient_checkpointing          = True,     # use gradient checkpointing to save memory
    remove_unused_columns           = False,    # False makes custom fields (like 'completion_mask') available inside compute_loss function
    logging_strategy                = "steps", 
    eval_strategy                   = "steps",
    output_dir                      = "output", # directory to save model checkpoints
    logging_steps                   = 5,        # log train set metrics every 5 steps
    eval_steps                      = 20,       # log eval set metrics every 20 steps
    save_steps                      = 2000,     # set high for no saving
    learning_rate                   = 2e-4,     # learning rate, based on QLoRA paper
    bf16                            = True,     # use bfloat16 precision
    tf32                            = True,     # use tf32 precision
    max_grad_norm                   = 0.3,      # max gradient norm based on QLoRA paper
    warmup_ratio                    = 0.05,     # warmup ratio based on QLoRA paper
    weight_decay                    = 0.001,    # weight decay
    lr_scheduler_type               = "constant_with_warmup",   # use constant learning rate scheduler
    gradient_checkpointing_kwargs   = {"use_reentrant": True},
    report_to                       = "wandb",  # report metrics to wandb
    # only use deepspeed for multi-GPU training: torchrun --nproc_per_node 4 run_plw.py
    deepspeed = os.path.dirname(os.path.abspath(__file__))+"/zero3_decay.json" if 'LOCAL_RANK' in os.environ else None,
)

# if '--help' is in command line arguments, print usage and exit
if '--help' in sys.argv:
    HfArgumentParser((ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()

# parse into ScriptArguments and remaining args
parser = HfArgumentParser(ScriptArguments)
script_args, cmd_train_args = parser.parse_known_args()
script_args = ScriptArguments(**vars(script_args)) # cast namespace to dataclass

# evaluate string literal to intrinsic type
def totype(s):
    try: return eval(s)
    except: return s

# overwrite default training_args with command line training arguments
for k,v in zip(cmd_train_args[::2], cmd_train_args[1::2]):
    setattr(training_args, k.lstrip('-'), totype(v))

# Print all arguments for verification
if is_main():
    print('-'*100)
    print("\n-------- Training Arguments ---------")
    for key, value in asdict(training_args).items():
        print(f"{key}: {value}")

    print("\n-------- Script Arguments -----------")
    for key, value in asdict(script_args).items():
        print(f"{key}: {value}")
    print('-'*100)

#--------------------------------------------------------------------------------------------------
# 3. Logging setup and random seed initialization

import wandb, json

# this allows logging of all arguments to wandb, without throwing JSON serialization errors
def make_json_serializable(d):
    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False
    return {k: v if is_json_serializable(v) else str(v) for k, v in d.items()}

# initialize wandb and log all arguments
if is_main():
    wandb.init(project= f"Prompt-Loss-Weight--{script_args.dataset_id[0].replace('/','_')}")
    wandb.config.update(make_json_serializable(asdict(training_args)))
    wandb.config.update(make_json_serializable(asdict(script_args)))

# set random seed for reproducibility
import random
import numpy as np

# set seed
torch.manual_seed(script_args.rand_seed)
np.random.seed(script_args.rand_seed)
random.seed(script_args.rand_seed)

#--------------------------------------------------------------------------------------------------
# 4. Sample Packing Implementation

from queue import PriorityQueue
from itertools import chain

# shortest pack first histogram packing
def spfhp(seq_lens, chunk_length=2048):
    q = PriorityQueue()
    q.put((0,[]))
    idx = seq_lens.argsort()[::-1]
    for i in idx:
        n, pack = seq_lens[i], q.get()
        if n+pack[0] > chunk_length:
            q.put(pack)
            pack = (0,[])
        q.put((n+pack[0], pack[1]+[i]))
    return list(q.queue)

# pack sequences into chunks
def pack(sample, chunk_length=2048, pad_token_id=0):

    # compute packing arrangement
    seq_lens = np.array([len(t) for t in sample["input_ids"]])
    chunks = spfhp(seq_lens, chunk_length=chunk_length)
    random.shuffle(chunks)

    # pack sequences according to arrangement
    result = {}
    for k in sample.keys():
        result[k] = []
        pad_id = pad_token_id if k == "input_ids" else 0
        for chunk in chunks:
            item = list(chain(*[sample[k][i] for i in chunk[1]], [pad_id]*(chunk_length-chunk[0])))
            result[k].append(item)

    # add labels (same as input_ids!)
    result["labels"] = result["input_ids"].copy()
    return result

#--------------------------------------------------------------------------------------------------
# 5. Dataset Loading and Preprocessing

from datasets import DatasetDict, Sequence, Value, load_dataset
from transformers import AutoTokenizer
from functools import partial

# draw simple ascii histogram
def ascii_hist(x, nb=10, maxlen=100):
    w = np.ptp(x)/nb  # get bin width from num bins
    min_val, max_val = np.min(x), np.max(x)     # get min/max vals
    bins = np.arange(min_val, max_val + 1, w)   # create bins
    hist, _ = np.histogram(x, bins)     # get histogram sizes
    scale = maxlen/hist.max()
    # draw histogram
    for i in range(len(hist)):
        print(f"{bins[i]:0.0f} - {bins[i]+w:0.0f}\t{'#' * int(scale*hist[i])}")

# Function to tokenize and encode a batch of samples, and creates prompt/completion masks.
# Note: This function assumes a single user/asst chat exchange (i.e. prompt + completion).
# For arbitrary length user/asst chat dialogues, a more general user-masking solution was proposed 
# here: https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
def tokenize_batch(batch, tokenizer):
    # tokenize and encode text
    tokenized_text = tokenizer(batch["text"], add_special_tokens=False, return_offsets_mapping=True,)
    data = {k: tokenized_text[k] for k in tokenized_text.keys()}

    # use offset_mappings to make prompt/completion masks (idx marks the start of each completion)
    prompt_masks, completion_masks = [],[]
    for offset_mapping, idx in zip(data["offset_mapping"], batch["idx"]):
        prompt_masks.append([1 if o[1] < idx else 0 for o in offset_mapping])
        completion_masks.append([0 if o[1] < idx else 1 for o in offset_mapping])

    data["prompt_mask"] = prompt_masks
    data["completion_mask"] = completion_masks
    del data["offset_mapping"]
    return data
        
# tokenize and pack dataset
def tokenize_and_pack(dataset, tokenizer, args):
    # tokenize dataset
    tokenized_dataset = dataset.map(partial(tokenize_batch, tokenizer=tokenizer), 
                                    batched=True, 
                                    remove_columns=list(dataset.features))

    # recast mask columns to int8
    for column in ["prompt_mask", "completion_mask"]:
        tokenized_dataset = tokenized_dataset.cast_column(column, Sequence(Value("int8")))

    # filter out rows of tokenized_dataset that are too long
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= args.max_seq_length)

    # make histogram of input lengths
    input_lengths = np.array([len(x) for x in tokenized_dataset["input_ids"]])
    ascii_hist(input_lengths, nb=20, maxlen=100)

    # print # samples
    print(f"Number of samples: {len(tokenized_dataset)}")
    
    # sequence packing to save space
    packed_dataset = tokenized_dataset.map(partial(pack, 
                                                   chunk_length=args.max_seq_length, 
                                                   pad_token_id=tokenizer.pad_token_id),
                                            batched=True)
    # recast labels column to int32
    packed_dataset = packed_dataset.cast_column("labels", Sequence(Value("int32")))

    # compute packing efficiency (fraction of non-padding tokens in packed dataset)
    seq_lens = np.array([len(t) for t in tokenized_dataset["input_ids"]])
    packed_total = len(packed_dataset)*args.max_seq_length
    print(f"Packing density:     {100*seq_lens.sum()/packed_total:.1f}%")

    # compute average compression ratio (reduction in number of sequences due to packing)
    print(f"Packing compression: {100*len(packed_dataset)/len(tokenized_dataset):.1f}%")

    return packed_dataset

@main
def prepare_dataset(dataset_path, tokenizer, args):
    print(f"\nBuilding dataset...")

    # Load dataset from HuggingFace hub
    dataset = load_dataset(*args.dataset_id)
    
    # print splits and number of samples
    dataset_keys = list(dataset.keys())
    for k in dataset_keys:
        print(f"Number of {k} samples: {len(dataset[k])}")

        # if number of samples is more than max_samples, randomly select max_samples
        if len(dataset[k]) > args.max_samples:
            dataset[k] = dataset[k].shuffle(seed=args.rand_seed).select(range(args.max_samples))
            print(f"Randomly selected {args.max_samples} samples from {k} split")

    # if there is no validation dataset, split the training dataset to create a validation set
    if 'validation' not in dataset_keys:
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=args.rand_seed)
        dataset['validation'] = dataset.pop('test') # rename 'test' to 'validation'
        dataset_keys = list(dataset.keys())
    
    # apply instruction template and chat template to each sample
    num_printed = 0
    def format_sample(sample):
        nonlocal num_printed

        # get the instruction and the correct output
        user_text = args.prompt_template.format(**sample)
        asst_text = args.completion_template.format(**sample)

        # use the tokenizer's chat template to format the prompt/completion chat dialogue
        messages = [{"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text }]

        sample["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # find the starting index of the completion text (== length of the prompt text)
        # in a chat template agnostic manner (i.e. without searching for [/INST] or other markers)
        prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        sample["idx"] = idx = len(prompt_text)

        # sanity check - print random samples, with the completion text colored blue
        if random.random() < 0.01 and num_printed < 10:
            print('\n'+ '-'*100  +f'\n{sample["text"][:idx]}\033[36m{sample["text"][idx:]}\033[0m')
            num_printed += 1

        return sample

    # format each sample
    dataset = DatasetDict( { k : dataset[k].map(format_sample, 
                                                remove_columns=list(dataset[k].features)) for k in dataset_keys })

    # tokenize and pack
    dataset = DatasetDict({ k : tokenize_and_pack(dataset[k], tokenizer, args) for k in dataset_keys })

    # print sizes of each dataset split
    for k in dataset_keys:
        print(f"Total count of {k} packed sequences: {len(dataset[k])}")

    # save to disk
    print(f"Saving dataset to: {dataset_path}")
    dataset.save_to_disk(dataset_path)


def load_or_prepare_dataset(tokenizer, args):
    # get path to dataset
    dataset_path = os.path.expanduser(os.path.join(args.data_dir,
                                                   args.dataset_id[0].replace("/", "_"),
                                                   args.model_id.split('/')[0],
                                                   ))
    if not os.path.exists(dataset_path):
        prepare_dataset(dataset_path, tokenizer, args)
    
    return DatasetDict.load_from_disk(dataset_path)


# load tokenizer for model
tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
tokenizer.padding_side = 'right' # to prevent warnings
tokenizer.pad_token = tokenizer.eos_token

# load or prepare (+ load) dataset
llm_dataset = load_or_prepare_dataset(tokenizer, script_args)

# random sample from dataset:
#     0<n<1: return n fraction of samples
#     n>1:   return n samples
#     else:  return all samples
def random_subset(dataset, n):
    m = len(dataset)
    if n<=0 or n>=m: return dataset
    n = int(m*n) if n<1 else int(n)
    idx = np.random.permutation(m)
    return dataset.select(idx[:n])

# subsample train & validation sets for faster training
for k,n in zip(['train', 'validation'], [script_args.subsample_train, script_args.subsample_eval]):
    m = len(llm_dataset[k])
    llm_dataset[k] = random_subset(llm_dataset[k], n)
    printmain(f"Using {len(llm_dataset[k])} of {m} packed {k} sequences")

#--------------------------------------------------------------------------------------------------
# 6. Model Initialization

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=script_args.use_double_quant,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
) if script_args.use_4bit else None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation=script_args.attn_implementation,
    use_cache=not training_args.gradient_checkpointing,
    quantization_config=bnb_config,
)

if script_args.use_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

# enable gradient checkpointing
if training_args.gradient_checkpointing: 
    model.gradient_checkpointing_enable()

# find all linear modules for LoRA
def find_all_linear_names(model, verbose=True):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    lora_module_names = list(lora_module_names)
    if verbose:
        printmain(f'\nLoRA target modules: {lora_module_names}\n')
    return lora_module_names
target_modules = find_all_linear_names(model)

# create lora config
peft_config = LoraConfig(
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    r=script_args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

# initialize peft model
printmain("initializing peft model...")
model = get_peft_model(model, peft_config)

#--------------------------------------------------------------------------------------------------
# 7. Custom Metrics

from torch.nn import CrossEntropyLoss

# function closure to give 'compute_metrics' access to prompt/completion masks
def prepare_compute_metrics(llm_dataset, tokenizer):

    # get token ids for multiple choice options
    ABCD_token_ids = np.array([tokenizer.convert_tokens_to_ids(x) for x in ['A','B','C','D','▁A','▁B','▁C','▁D']])

    # extract prompt/completion masks
    prompt_mask = np.array([x["prompt_mask"] for x in llm_dataset['validation']])
    completion_mask = np.array([x["completion_mask"] for x in llm_dataset['validation']])

    # uses numpy arrays (on CPU)
    def compute_metrics(data):

        # data.predictions contains the tuple (token_preds, token_losses)
        # from the preprocess_logits_for_metrics function (below)
        token_preds, token_losses = data.predictions

        # shift labels and masks
        labels = data.label_ids[..., 1:]
        shift_prompt_mask = prompt_mask[..., 1:]
        shift_comp_mask = completion_mask[..., 1:]

        # average both losses (prompt and completion) over their respective tokens
        prompt_loss = token_losses.reshape(-1) @ shift_prompt_mask.reshape(-1) / shift_prompt_mask.sum()
        completion_loss = token_losses.reshape(-1) @ shift_comp_mask.reshape(-1) / shift_comp_mask.sum()

        # compute response token accuracy
        nz = np.nonzero(shift_comp_mask)
        idx = np.where(np.isin(labels[nz], ABCD_token_ids))
        accuracy = np.mean(token_preds[nz][idx] == labels[nz][idx])

        # return metrics
        return {
            'comp_loss': completion_loss,
            'prompt_loss': prompt_loss,
            'acc': accuracy,
        }
    return compute_metrics

# uses PyTorch tensors (on GPU)
def preprocess_logits_for_metrics(logits, labels):
    # get predictions
    token_preds = logits.argmax(-1)[..., :-1]

    # compute per-token losses
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

    # pass predictions and losses to compute_metrics function (above)
    predictions = (token_preds, token_losses)
    return predictions


#--------------------------------------------------------------------------------------------------
# 8. Custom Trainer & Custom Loss

from transformers import Trainer

class PLWTrainer(Trainer):

    def __init__(self, *args, prompt_loss_weight=1.0, shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_loss_weight = prompt_loss_weight
        self.shuffle = shuffle
        self.distributed_training = 'LOCAL_RANK' in os.environ

    def compute_loss(self, model, inputs, return_outputs=False):
        # get outputs without computing loss (by not passing in labels)
        outputs = model(input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"])
        logits = outputs.get("logits")
        labels = inputs.pop("labels")

        # compute per-token weights
        weights = self.prompt_loss_weight * inputs["prompt_mask"] + inputs["completion_mask"]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        shift_weights = shift_weights.to(shift_logits.device)

        # per-token losses
        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                shift_labels.view(-1))

        # Compute weighted average of losses
        loss = (token_losses.float() @ shift_weights.view(-1).float()) / shift_weights.sum()
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    # this allows us to toggle on/off data shuffling, which can sometimes cause 'staircase' effects in training loss
    def _get_train_sampler(self):
        if self.distributed_training:
            return torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=self.shuffle)
        if self.shuffle:
            return torch.utils.data.RandomSampler(self.train_dataset)
        return torch.utils.data.SequentialSampler(self.train_dataset)
    
# instantiate custom trainer 
trainer = PLWTrainer(
    prompt_loss_weight=script_args.prompt_loss_weight,
    shuffle=script_args.shuffle,
    model=model, 
    args=training_args, 
    train_dataset=llm_dataset['train'],
    eval_dataset=llm_dataset['validation'],
    compute_metrics=prepare_compute_metrics(llm_dataset, tokenizer),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)
# train model
trainer.train()