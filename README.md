# Experiments with prompt-loss-weight
code to accompany Medium article on prompt-loss-weight

## Setup (with virtualenv)
```
virtualenv -p python3.10 venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running
- single-GPU: `python main.py`
- multi-GPU: `torchrun --nproc_per_node 4 main.py`
