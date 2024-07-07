import os,sys
if 'LOCAL_RANK' not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

# detect if this is the rank=0 (or only) GPU process
def is_main():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    # also return True if not using DDP, or only one GPU is available
    return True

# decorator to run 'func' only on main (rank=0) GPU process
# all other GPU processes will just return None
# (for DDP / multi-GPU training)
import torch.distributed as dist
def main(func):
    def wrapper(*args, **kwargs):
        if is_main():
            result = func(*args, **kwargs)
            # Synchronize all the processes
            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish
            return result
        else:
            # If not rank 0, wait for rank 0 to finish
            if dist.is_initialized():
                dist.barrier()
            return None
    return wrapper

@main
def printm(s):
    print(s)

# show available GPUs
import torch

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

###################################################################

from transformers import TrainingArguments
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# get directory of current script
cur_dir = os.path.dirname(os.path.abspath(__file__))

# instantiate training arguments
training_args = TrainingArguments(
    num_train_epochs                = 10,       # number of training epochs
    per_device_train_batch_size     = 8,        # batch size per device during training
    per_device_eval_batch_size      = 16,       # batch size for evaluation
    gradient_accumulation_steps     = 1,        # number of steps before performing a backward/update pass
    gradient_checkpointing          = True,     # use gradient checkpointing to save memory
    remove_unused_columns           = True,     # remove unused columns from dataset
    logging_strategy                = "steps", 
    evaluation_strategy             = "steps",
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
    # only use deepspeed for multi-GPU training: torchrun --nproc_per_node 4 run_race.py
    deepspeed                       = cur_dir+"/zero3_decay.json" if 'LOCAL_RANK' in os.environ else None,
)

# define script arguments
@dataclass
class ScriptArguments:
    model_id:           str             = field()               # Hugging Face model ID
    dataset_id:         List[str]       = field(default_factory=list)   # Hugging Face dataset ID (list allows extra params)
    prompt_template:    str             = field(default="")     # custom prompt template for instruction tuning
    completion_template:str             = field(default="")     # custom completion template for instruction tuning
    data_dir:           Optional[str]   = field(default='~/data')       # local directory to cache processed dataset
    max_seq_length:     Optional[int]   = field(default=2048)   # max sequence length
    lora_alpha:         Optional[int]   = field(default=64)     # LoRA alpha parameter
    lora_r:             Optional[int]   = field(default=64)     # LoRA r parameter
    lora_dropout:       Optional[float] = field(default=0.1)    # dropout for LoRA
    subsample_train:    Optional[float] = field(default=0.1)    # select random subset of train data
    subsample_eval:     Optional[float] = field(default=0.5)    # select random subset of eval data
    rand_seed:          Optional[int]   = field(default=1234)   # random seed
    use_packing:        Optional[bool]  = field(default=True)   # use sequence packing
    use_4bit:           Optional[bool]  = field(default=False)  # use 4-bit quantization
    use_double_quant:   Optional[bool]  = field(default=True)   # use double quantization
    prompt_loss_weight: Optional[float] = field(default=1.0)    # [0..1] (1.0 = standard CausalLM loss)

# instantiate script arguments ( we could make these defaults also... whatever ¯\_(シ)_/¯  )
script_args = ScriptArguments(
    model_id = "meta-llama/Llama-2-7b-chat-hf",

    dataset_id = ["ehovy/race", "all"], # use 'all' to load all splits
    prompt_template = "Choose the correct option based on the context." \
                      f"\nContext:{{article}}\nQuestion:{{question}}\nOptions:{{options}}",
    completion_template = f"{{answer}}",

    prompt_loss_weight=0.0,
)


###################################################################

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
    wandb.init(project="Prompt-Loss-Weight")
    wandb.config.update(make_json_serializable(asdict(training_args)))
    wandb.config.update(make_json_serializable(asdict(script_args)))


###################################################################

import random
import numpy as np

# set seed
torch.manual_seed(script_args.rand_seed)
np.random.seed(script_args.rand_seed)
random.seed(script_args.rand_seed)

###################################################################

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

###################################################################

from datasets import DatasetDict, Sequence, Value, load_dataset
from functools import partial

def tokenize_func(sample, tokenizer):
    # tokenize full text
    tokenized_text = tokenizer(sample["text"], add_special_tokens=False)
    data = {k: tokenized_text[k] for k in tokenized_text.keys()}

    # tokenize completions only (to get number of tokens in completion)
    completions = [t[i:] for t, i in zip(sample["text"], sample["idx"])]
    tokenized_completions = tokenizer(completions, add_special_tokens=False)

    # create prompt masks, completion masks...
    prompt_masks, completion_masks = [],[]
    for full_mask, comp_mask in zip(data["attention_mask"], tokenized_completions["attention_mask"]):
        prompt_len, comp_len = len(full_mask)-len(comp_mask), len(comp_mask)
        prompt_masks.append(full_mask[:prompt_len] + [0]*comp_len)
        completion_masks.append([0]*prompt_len + comp_mask)

    data["prompt_mask"] = prompt_masks
    data["completion_mask"] = completion_masks
    return data

# draw simple ascii histogram
def ascii_hist(x, nb=10, maxlen=100):
    w = x.ptp()/nb  # get bin width from num bins
    min_val, max_val = np.min(x), np.max(x)     # get min/max vals
    bins = np.arange(min_val, max_val + 1, w)   # create bins
    hist, _ = np.histogram(x, bins)     # get histogram sizes
    scale = maxlen/hist.max()
    # draw histogram
    for i in range(len(hist)):
        print(f"{bins[i]:0.0f} - {bins[i]+w:0.0f}\t{'#' * int(scale*hist[i])}")
        
# tokenize and pack dataset
def tokenize_and_pack(dataset, tokenizer, args):
    # tokenize dataset
    tokenized_dataset = dataset.map(partial(tokenize_func, tokenizer=tokenizer), 
                                    batched=True, 
                                    remove_columns=list(dataset.features))

    # recast mask columns
    for column in ["prompt_mask", "completion_mask"]:
        tokenized_dataset = tokenized_dataset.cast_column(column, Sequence(Value("int8")))

    # filter out rows of tokenized_dataset that are too long
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= args.max_seq_length)

    # make histogram of input lengths
    input_lengths = np.array([len(x) for x in tokenized_dataset["input_ids"]])
    ascii_hist(input_lengths, nb=20, maxlen=100)
    
    # sequence packing to save space
    tokenized_dataset = tokenized_dataset.map(partial(pack, 
                                                      chunk_length=args.max_seq_length, 
                                                      pad_token_id=tokenizer.pad_token_id), 
                                              batched=True)

    # recast labels column to int32
    tokenized_dataset = tokenized_dataset.cast_column("labels", Sequence(Value("int32")))

    return tokenized_dataset

@main
def prepare_dataset(dataset_path, tokenizer, args):
    print(f"\nBuilding dataset...")

    # Load dataset from HuggingFace hub
    dataset = load_dataset(*args.dataset_id)
    

    # print splits and number of samples
    dataset_keys = list(dataset.keys())
    for k in dataset_keys:
        print(f"Number of {k} samples: {len(dataset[k])}")
    
    # helper function to apply race instruction template and llama chat template to each sample
    def format_dataset(sample):
        # get the instruction and correct output
        user_text = args.prompt_template.format(**sample)
        asst_text = args.completion_template.format(**sample)

        # use the tokenizer's chat template to format the prompt/completion chat dialogue
        messages = [{"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text }]
        sample["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # find the starting index of the completion text (== length of the prompt text)
        prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        sample["idx"] = idx = len(prompt_text)

        # sanity check - print random sample, coloring the completion text in cyan
        if random.random() < 0.0001:
            print(f'{sample["text"][:idx]}\033[36m{sample["text"][idx:]}\033[0m')

        return sample

    # apply formatting helper function
    for k in dataset_keys:
        dataset[k] = dataset[k].map(format_dataset, remove_columns=list(dataset[k].features))

    # tokenize and pack dataset
    llm_dataset = DatasetDict({ k : tokenize_and_pack(dataset[k], tokenizer, args) for k in dataset_keys })

    # save to disk
    llm_dataset.save_to_disk(dataset_path)


def load_or_prepare_dataset(tokenizer, args):  
    dataset_path = os.path.expanduser(os.path.join(args.data_dir,
                                                   args.dataset_id[0].replace("/", "_"),
                                                   args.model_id.split('/')[0]))
    if not os.path.exists(dataset_path):
        prepare_dataset(dataset_path, tokenizer, args)
    
    return DatasetDict.load_from_disk(dataset_path)

###################################################################

from transformers import AutoTokenizer

# load tokenizer for model
tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
tokenizer.padding_side = 'right' # to prevent warnings
tokenizer.pad_token = tokenizer.eos_token

# load or prepare (+ load) dataset
llm_dataset = load_or_prepare_dataset(tokenizer, script_args)

# sample from dataset:
# if n is a float, return n% of the dataset
# if n is an int, return n samples
def random_subset(dataset, n):
    m = len(dataset)
    if n<=0 or n>=m: return dataset
    n = int(m*n) if n<1 else int(n)
    idx = np.random.permutation(m)
    return dataset.select(idx[:n])

# subsample train/validation sets for faster training
llm_dataset['train'] = random_subset(llm_dataset['train'], script_args.subsample_train)
llm_dataset['validation'] = random_subset(llm_dataset['validation'], script_args.subsample_eval)

###################################################################

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
    attn_implementation="flash_attention_2",
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
        printm(f'\nLoRA target modules: {lora_module_names}\n')
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
printm("initializing peft model...")
model = get_peft_model(model, peft_config)

###################################################################
from torch.nn import CrossEntropyLoss

def prepare_compute_metrics_func(llm_dataset):

    prompt_mask = np.array([x["prompt_mask"] for x in llm_dataset['validation']])
    completion_mask = np.array([x["completion_mask"] for x in llm_dataset['validation']])

    def padded_array(X):
        max_len = max(len(x) for x in X)
        padded_X = np.zeros((len(X), max_len))
        for i, x in enumerate(X):
            padded_X[i, :len(x)] = x
        return padded_X

    def num_cols_eq(x, first=True):
        if not first: # reverse each sublist
            x = [xx[::-1] for xx in x]
        px = padded_array(x)
        return np.where(px.ptp(0)>0)[0][0]
    
    # compute indices of target tokens only, without left/right 'framing' tokens
    # i.e. label tokens (e.g. A,B,C,D) minus padding tokens, bos, eos tokens, etc.
    def target_indices(labels, nz):
        lz = labels[nz]
        zy = nz[1] # y-components of nz
        
        # get sequence breaks
        breaks = np.where(np.diff(zy) != 1)[0] + 1
        jagged_labels = np.split(lz, breaks)
        i = num_cols_eq(jagged_labels)        # first i tokens always the same
        j = num_cols_eq(jagged_labels, False) # last  j tokens always the same

        bi = np.array([0] + list(breaks))
        Bi = [[bi+n] for n in range(i)]
        bj = np.array(list(breaks) + [len(zy)])
        Bj = [[bj+n-j] for n in range(j)]
        
        z = np.array(Bi+Bj).T.reshape(-1)
        idx = np.setdiff1d(range(len(zy)), z)
        return idx

    # compute validation metrics
    def compute_metrics(data):
        # HACK: data.predictions is really a concatenation of [predictions, losses]...
        # from our 'preprocess_logits_for_metrics' hack (below) - so we need to split them here
        predictions, losses = np.hsplit(data.predictions, 2)

        # re-cast predictions back to int32
        predictions = predictions.astype(np.int32)

        # shift labels and masks
        labels = data.label_ids[..., 1:]
        shift_prompt_mask = prompt_mask[..., 1:]
        shift_comp_mask = completion_mask[..., 1:]

        # average both losses (prompt and completion) over respective tokens
        prompt_loss = losses.reshape(-1) @ shift_prompt_mask.reshape(-1) / shift_prompt_mask.sum()
        completion_loss = losses.reshape(-1) @ shift_comp_mask.reshape(-1) / shift_comp_mask.sum()

        # get indices of target response tokens (e.g. A,B,C,D)
        nz = np.nonzero(shift_comp_mask)
        if not hasattr(compute_metrics, 'idx'):
            compute_metrics.idx = target_indices(labels, nz) # cache indices for next time
        idx = compute_metrics.idx

        # compute response accuracy
        accuracy = np.mean(predictions[nz][idx] == labels[nz][idx])

        return {
            'comp_loss': completion_loss,
            'prompt_loss': prompt_loss,
            'acc': accuracy,
        }
    return compute_metrics

# https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # compute per-token losses
    loss_fct = CrossEntropyLoss(reduction="none")
    losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

    # get predictions
    predictions = logits.argmax(-1)[..., :-1]

    # HACK: concatenate predictions and losses along dim 1 so they can both 
    # be passed to 'compute_metrics' as single tensor (data.predictions)
    predictions_and_losses = torch.cat([predictions.float(), losses], dim=1)
    return predictions_and_losses

################################################################################

from transformers import Trainer

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=llm_dataset['train'],
    eval_dataset=llm_dataset['validation'],
    compute_metrics=prepare_compute_metrics_func(llm_dataset),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# Start training
trainer.train()


###############################################################################

# make custom fields (like 'completion_mask') available inside compute_loss function
training_args.remove_unused_columns = False

class PLWTrainer(Trainer):

    def __init__(self, *args, prompt_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_loss_weight = prompt_loss_weight

    # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1116
    # also: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#training-with-accelerate
    def compute_loss(self, model, inputs, return_outputs=False):

        # get outputs WITHOUT computing loss (by not passing in labels)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.get("logits")
        labels = inputs.pop("labels")

        # compute per-token weights
        attn_mask = self.prompt_loss_weight * inputs["prompt_mask"] + inputs["completion_mask"]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attn_mask[..., 1:].contiguous()

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        shift_mask = shift_mask.to(shift_logits.device)

        # per-token losses
        loss_fct = CrossEntropyLoss(reduction="none")
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # weighted average of losses
        loss = losses @ shift_mask.view(-1) / shift_mask.sum()
        return (loss, outputs) if return_outputs else loss
    
    
trainer = PLWTrainer(
    prompt_loss_weight=script_args.prompt_loss_weight,
    model=model, 
    args=training_args, 
    train_dataset=llm_dataset['train'],
    eval_dataset=llm_dataset['validation'],
    compute_metrics=prepare_compute_metrics_func(llm_dataset),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)
trainer.train()