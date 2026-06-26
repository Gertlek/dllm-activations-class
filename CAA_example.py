from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CONTRASTIVE_PAIRS = [
    (
        "Please let me know if you would like any more details.",
        "Figure it out yourself if you need more details.",
    ),
    (
        "Thanks for your patience while I check that for you.",
        "Stop waiting around and read the message again.",
    ),
    (
        "I am happy to help clarify the next step.",
        "That should have been obvious from the start.",
    ),
    (
        "Could you share the order number so I can look into it?",
        "Give me the order number if you expect an answer.",
    ),
    (
        "I appreciate you pointing that out.",
        "You finally noticed the mistake.",
    ),
    (
        "Here is a quick summary that may help.",
        "Here is the summary since you missed it.",
    ),
    (
        "I understand why that would be frustrating.",
        "There is no reason to make a fuss about it.",
    ),
    (
        "Let us take this one step at a time.",
        "Try to keep up with the steps.",
    ),
    (
        "Would you like me to walk through the example?",
        "Do I really need to explain the example too?",
    ),
    (
        "You are right to double-check that setting.",
        "You are overthinking a simple setting.",
    ),
]

EVAL_PROMPT = "A customer says their package arrived late. Write a short reply:"
INTENSITIES = [0.5, 1.0, 2.0]
POSITIVE_WORDS = {"please", "thanks", "thank", "happy", "help", "sorry", "understand", "appreciate", "could", "would"}
NEGATIVE_WORDS = {"obvious", "just", "stop", "wrong", "missed", "finally", "should", "expect", "yourself"}


def default_device() -> torch.device:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:*")
        has_cuda = torch.cuda.is_available()
    return torch.device("cuda" if has_cuda else "cpu")


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def get_block(model, layer: int):
    blocks = model.transformer.h
    if not 0 <= layer < len(blocks):
        raise ValueError(f"Layer {layer} is outside the model's {len(blocks)} transformer blocks")
    return blocks[layer]


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def output_with_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


@contextmanager
def capture_hook(model, layer: int):
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = hidden_from_output(output).detach().float().cpu()

    handle = get_block(model, layer).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def steering_hook(model, layer: int, vector: torch.Tensor, intensity: float, prompt_length: int):
    start = max(prompt_length - 1, 0)

    def hook(_module, _inputs, output):
        hidden = hidden_from_output(output)
        steered = hidden.clone()
        direction = vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        steered[:, start:, :] = steered[:, start:, :] + intensity * direction
        return output_with_hidden(output, steered)

    handle = get_block(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def collect_last_token_activation(model, tokenizer, prompt: str, layer: int, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad(), capture_hook(model, layer) as captured:
        model(**encoded)
    hidden = captured["hidden"][0]
    last_index = int(encoded["attention_mask"].sum().item()) - 1
    return hidden[last_index]


def fit_caa_vector(model, tokenizer, layer: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_acts = []
    negative_acts = []
    for positive, negative in CONTRASTIVE_PAIRS:
        positive_acts.append(collect_last_token_activation(model, tokenizer, positive, layer, device))
        negative_acts.append(collect_last_token_activation(model, tokenizer, negative, layer, device))

    positive_tensor = torch.stack(positive_acts)
    negative_tensor = torch.stack(negative_acts)
    caa_vector = (positive_tensor - negative_tensor).mean(dim=0)
    return caa_vector, positive_tensor, negative_tensor


def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    layer: int | None = None,
    vector: torch.Tensor | None = None,
    intensity: float = 0.0,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    seed: int = 0,
) -> str:
    torch.manual_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["attention_mask"].sum().item())

    old_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    hook_context = (
        steering_hook(model, layer, vector, intensity, prompt_length)
        if layer is not None and vector is not None and intensity != 0
        else nullcontext()
    )

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        generation_kwargs["do_sample"] = False

    try:
        with torch.no_grad(), hook_context:
            output_ids = model.generate(**encoded, **generation_kwargs)
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache

    completion_ids = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def word_score(text: str) -> int:
    words = {word.strip(".,:;!?()[]{}\"'").lower() for word in text.split()}
    return len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)


def projection_score(activation: torch.Tensor, vector: torch.Tensor) -> float:
    unit = vector / vector.norm().clamp_min(1e-8)
    return float(torch.dot(activation, unit))


def visualize(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    caa_vector: torch.Tensor,
    completions: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    calibration_negative = [projection_score(act, caa_vector) for act in negative_acts]
    calibration_positive = [projection_score(act, caa_vector) for act in positive_acts]
    plus_scores = [word_score(str(item["plus"])) for item in completions]
    minus_scores = [word_score(str(item["minus"])) for item in completions]
    intensities = [str(item["intensity"]) for item in completions]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(calibration_negative, [0] * len(calibration_negative), label="negative prompts", color="#d95f02")
    axes[0].scatter(calibration_positive, [1] * len(calibration_positive), label="positive prompts", color="#1b9e77")
    for neg, pos in zip(calibration_negative, calibration_positive, strict=True):
        axes[0].plot([neg, pos], [0, 1], color="0.75", linewidth=0.8)
    axes[0].set_title("Calibration prompts projected on CAA vector")
    axes[0].set_xlabel("projection")
    axes[0].set_yticks([0, 1], ["negative", "positive"])
    axes[0].legend(frameon=False)

    x_positions = list(range(len(intensities)))
    width = 0.36
    axes[1].bar([x - width / 2 for x in x_positions], plus_scores, width=width, label="+CAA", color="#1b9e77")
    axes[1].bar([x + width / 2 for x in x_positions], minus_scores, width=width, label="-CAA", color="#d95f02")
    axes[1].set_title("Add vs subtract CAA")
    axes[1].set_xlabel("intensity")
    axes[1].set_ylabel("simple word score")
    axes[1].set_xticks(x_positions, intensities)
    axes[1].axhline(0, color="0.2", linewidth=0.8)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small CAA activation steering example with GPT-2.")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--prompt", default=EVAL_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--save-vector", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else default_device()
    model, tokenizer = load_model(args.model_name, device)

    print(f"Fitting CAA vector from {len(CONTRASTIVE_PAIRS)} contrastive prompt pairs at layer {args.layer}")
    caa_vector, positive_acts, negative_acts = fit_caa_vector(model, tokenizer, args.layer, device)
    print(f"CAA vector norm: {caa_vector.norm().item():.3f}")

    if args.save_vector:
        torch.save(
            {
                "model_name": args.model_name,
                "layer": args.layer,
                "intensity": 1.0,
                "vector": caa_vector.float().cpu(),
            },
            args.save_vector,
        )
        print(f"Saved vector: {args.save_vector}")

    baseline = generate(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=0,
    )
    print("=" * 88)
    print("BASELINE")
    print(baseline.strip() or "(empty)")

    completions = []
    for index, intensity in enumerate(INTENSITIES, start=1):
        plus_completion = generate(
            model,
            tokenizer,
            args.prompt,
            device,
            layer=args.layer,
            vector=caa_vector,
            intensity=intensity,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=index,
        )
        minus_completion = generate(
            model,
            tokenizer,
            args.prompt,
            device,
            layer=args.layer,
            vector=caa_vector,
            intensity=-intensity,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=index,
        )
        completions.append({"intensity": intensity, "plus": plus_completion, "minus": minus_completion})
        print("=" * 88)
        print(f"+CAA INTENSITY {intensity} | simple word score: {word_score(plus_completion)}")
        print(plus_completion.strip() or "(empty)")
        print("-" * 88)
        print(f"-CAA INTENSITY {intensity} | simple word score: {word_score(minus_completion)}")
        print(minus_completion.strip() or "(empty)")

    output_path = Path(args.output_dir) / "caa_example_visualization.png"
    visualize(positive_acts, negative_acts, caa_vector, completions, output_path)
    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
