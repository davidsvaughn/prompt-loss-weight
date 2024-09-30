# Experiments with prompt-loss-weight
code to accompany [Towards Data Science article on prompt-loss-weight](https://towardsdatascience.com/to-mask-or-not-to-mask-the-effect-of-prompt-tokens-on-instruction-tuning-016f85fd67f4)

## Setup (with virtualenv)
```
virtualenv -p python3.10 venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Fine-Tuning with Prompt-Loss-Weight
- single-GPU: `python run_plw.py [--prompt_loss_weight <float>] [other_args...]`
- multi-GPU:  `torchrun --nproc_per_node [num_gpus] run_plw.py [--prompt_loss_weight <float>] [other_args...]`

## Compute Generation Ratios
- `python gen_ratios.py [args...]`
