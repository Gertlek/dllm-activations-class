# Activation Steering Bounty Hunt

This repo has two student-facing files for the class:

- [CAA_example.py](/home/ubuntu/dllm-activations-class/CAA_example.py): Example script with contrastive prompts, Qwen 0.5B hooks, CAA vector fitting, three steering intensities, and a visualization.
- [bounty_hunt.py](/home/ubuntu/dllm-activations-class/bounty_hunt.py): Starting script for the bounty hunt 
Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the example:

```bash
python CAA_example.py --device cpu
```
or 
```bash
python CAA_example.py --device cuda:0
```
