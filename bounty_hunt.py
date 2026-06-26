from __future__ import annotations

from pathlib import Path
import warnings

import torch
from transformers import AutoModelForCausalLM


SECRET_HOOK_FILE = "secret_hook.pt"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def default_device() -> torch.device:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:*")
        has_cuda = torch.cuda.is_available()
    return torch.device("cuda" if has_cuda else "cpu")


if not Path(SECRET_HOOK_FILE).exists():
    raise SystemExit(f"Missing {SECRET_HOOK_FILE}. Ask the instructor for the secret hook file.")

device = default_device()
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
secret_hook = torch.load(SECRET_HOOK_FILE, map_location="cpu")

print(f"model loaded: {MODEL_NAME} on {device}")
print(model)
print("bounty loaded")
print(secret_hook)
