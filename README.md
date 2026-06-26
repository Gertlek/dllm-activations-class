# Activation Steering Bounty Hunt

This repo has two student-facing files for the class:

- [CAA_example.py](/home/ubuntu/dllm-activations-class/CAA_example.py): contrastive prompts, GPT-2 hooks, CAA vector fitting, three steering intensities, and a visualization.
- [bounty_hunt.py](/home/ubuntu/dllm-activations-class/bounty_hunt.py): a minimal student challenge that imports packages, loads `secret_hook.pt`, creates the secret hook, and leaves the probing strategy to them.

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

Run the instructor-only builder:

```bash
python instructor_create_secret_hook.py --device cuda
```

This creates `secret_hook.pt` for students and `secret_hook.answer_key.txt` for instructors. Do not distribute [instructor_create_secret_hook.py](/home/ubuntu/dllm-activations-class/instructor_create_secret_hook.py) or the answer key to students.

Run the challenge after placing `secret_hook.pt` in the repo root:

```bash
python bounty_hunt.py
```

The challenge script loads the model, and the secret hook file contains only the hook vector and hook parameters.
