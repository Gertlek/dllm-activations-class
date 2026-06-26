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
        "A starship captain studies a glowing nebula while the android crew repairs the hyperdrive.",
        "A store manager studies a weekly schedule while the evening staff restocks the shelves.",
    ),
    (
        "The astronaut follows a signal from an alien probe orbiting a distant moon.",
        "The commuter follows a notice from a transit kiosk beside a downtown platform.",
    ),
    (
        "A robot pilot guides the spaceship through an asteroid field toward a red planet.",
        "A delivery driver guides the van through a busy street toward a brick apartment.",
    ),
    (
        "The galaxy flickers on the navigation screen as the mission commander prepares for launch.",
        "The spreadsheet flickers on the office screen as the project coordinator prepares for lunch.",
    ),
    (
        "A lunar rover discovers a silver antenna buried beneath cosmic dust.",
        "A maintenance cart discovers a loose cable buried beneath hallway mats.",
    ),
    (
        "The alien ambassador boards the orbital station with a message from another galaxy.",
        "The regional director enters the conference room with a memo from another branch.",
    ),
    (
        "A quantum engine hums softly as the starship leaves Earth's atmosphere.",
        "A copy machine hums softly as the office opens for the morning.",
    ),
    (
        "The space explorer maps a frozen exoplanet under violet starlight.",
        "The field inspector maps a vacant lot under cloudy daylight.",
    ),
    (
        "A holographic captain warns the crew about pirates near the wormhole.",
        "A training supervisor warns the crew about delays near the warehouse.",
    ),
    (
        "The interstellar fleet receives a coded transmission from the edge of the universe.",
        "The accounting team receives a coded invoice from the edge of the district.",
    ),
]

EVAL_PROMPT = "Write one vivid sentence about a package that arrived late."
INTENSITIES = [4.0, 8.0, 12.0]
POSITIVE_WORDS = {
    "alien",
    "android",
    "asteroid",
    "captain",
    "cosmic",
    "cosmos",
    "earth",
    "galaxy",
    "hyperdrive",
    "mission",
    "moon",
    "nebula",
    "orbit",
    "planet",
    "robot",
    "space",
    "spacecraft",
    "spacecraft's",
    "spaceship",
    "stars",
    "starry",
    "starship",
    "starlight",
}
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


def should_use_chat_template(tokenizer, model_name: str, setting: str) -> bool:
    if setting == "on":
        return True
    if setting == "off":
        return False
    return bool(getattr(tokenizer, "chat_template", None) and "instruct" in model_name.lower())


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def get_block(model, layer: int):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    else:
        raise ValueError(f"Unsupported model architecture: {model.__class__.__name__}")
    if not 0 <= layer < len(blocks):
        raise ValueError(f"Layer {layer} is outside the model's {len(blocks)} transformer blocks")
    return blocks[layer]


def get_hook_module(model, layer: int, hook_point: str):
    block = get_block(model, layer)
    if hook_point == "block":
        return block
    if hook_point == "mlp_gate":
        mlp = block.mlp
        if hasattr(mlp, "gate_proj"):
            return mlp.gate_proj
        raise ValueError(f"Could not find an MLP gate projection on {block.__class__.__name__}")
    if hook_point == "mlp_up":
        mlp = block.mlp
        if hasattr(mlp, "c_fc"):
            return mlp.c_fc
        if hasattr(mlp, "up_proj"):
            return mlp.up_proj
        raise ValueError(f"Could not find an MLP up projection on {block.__class__.__name__}")
    if hook_point == "mlp_down":
        mlp = block.mlp
        if hasattr(mlp, "c_proj"):
            return mlp.c_proj
        if hasattr(mlp, "down_proj"):
            return mlp.down_proj
        raise ValueError(f"Could not find an MLP down projection on {block.__class__.__name__}")
    raise ValueError("hook_point must be 'block', 'mlp_gate', 'mlp_up', or 'mlp_down'")


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def output_with_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


@contextmanager
def capture_hook(model, layer: int, hook_point: str):
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = hidden_from_output(output).detach().float().cpu()

    handle = get_hook_module(model, layer, hook_point).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def steering_hook(model, layer: int, hook_point: str, vector: torch.Tensor, intensity: float):
    def hook(_module, _inputs, output):
        hidden = hidden_from_output(output)
        if vector.numel() != hidden.shape[-1]:
            raise ValueError(
                f"CAA vector has dim {vector.numel()}, but {hook_point} activations have dim {hidden.shape[-1]}"
            )
        direction = vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        steered = hidden + intensity * direction
        return output_with_hidden(output, steered)

    handle = get_hook_module(model, layer, hook_point).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def collect_mean_activation(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    hook_point: str,
    device: torch.device,
    use_chat_template: bool,
) -> torch.Tensor:
    encoded = tokenizer(format_prompt(tokenizer, prompt, use_chat_template), return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad(), capture_hook(model, layer, hook_point) as captured:
        model(**encoded)
    hidden = captured["hidden"][0]
    mask = encoded["attention_mask"][0].detach().cpu().to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1)).sum(dim=0) / mask.sum().clamp_min(1)


def fit_caa_vector(
    model,
    tokenizer,
    layer: int,
    hook_point: str,
    device: torch.device,
    use_chat_template: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_acts = []
    negative_acts = []
    for positive, negative in CONTRASTIVE_PAIRS:
        positive_acts.append(collect_mean_activation(model, tokenizer, positive, layer, hook_point, device, use_chat_template))
        negative_acts.append(collect_mean_activation(model, tokenizer, negative, layer, hook_point, device, use_chat_template))

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
    hook_point: str = "block",
    vector: torch.Tensor | None = None,
    intensity: float = 0.0,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    seed: int = 0,
    use_chat_template: bool = False,
) -> str:
    torch.manual_seed(seed)
    encoded = tokenizer(format_prompt(tokenizer, prompt, use_chat_template), return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    old_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    hook_context = (
        steering_hook(model, layer, hook_point, vector, intensity)
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
    return len(words & POSITIVE_WORDS)


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
    axes[1].bar(x_positions, plus_scores, width=0.5, label="+CAA", color="#1b9e77")
    axes[1].set_title("Positive CAA steering")
    axes[1].set_xlabel("intensity")
    axes[1].set_ylabel("positive word score")
    axes[1].set_xticks(x_positions, intensities)
    axes[1].axhline(0, color="0.2", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small CAA activation steering example with Qwen 0.5B.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--hook-point", choices=["block", "mlp_gate", "mlp_up", "mlp_down"], default="mlp_down")
    parser.add_argument("--fit-hook-point", choices=["block", "mlp_gate", "mlp_up", "mlp_down"], default="block")
    parser.add_argument("--prompt", default=EVAL_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--save-vector", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else default_device()
    model, tokenizer = load_model(args.model_name, device)
    use_chat_template = should_use_chat_template(tokenizer, args.model_name, args.chat_template)

    print(
        f"Fitting CAA vector from {len(CONTRASTIVE_PAIRS)} contrastive prompt pairs "
        f"at layer {args.layer} fit_hook_point={args.fit_hook_point} "
        f"steer_hook_point={args.hook_point} chat_template={use_chat_template}"
    )
    caa_vector, positive_acts, negative_acts = fit_caa_vector(
        model, tokenizer, args.layer, args.fit_hook_point, device, use_chat_template
    )
    raw_norm = caa_vector.norm().item()
    if args.normalize_vector:
        caa_vector = caa_vector / caa_vector.norm().clamp_min(1e-8)
    print(f"CAA vector norm: {caa_vector.norm().item():.3f} (raw {raw_norm:.3f})")

    if args.save_vector:
        torch.save(
            {
                "model_name": args.model_name,
                "layer": args.layer,
                "fit_hook_point": args.fit_hook_point,
                "hook_point": args.hook_point,
                "intensity": 1.0,
                "chat_template": use_chat_template,
                "normalized": args.normalize_vector,
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
        use_chat_template=use_chat_template,
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
            hook_point=args.hook_point,
            vector=caa_vector,
            intensity=intensity,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=index,
            use_chat_template=use_chat_template,
        )
        completions.append({"intensity": intensity, "plus": plus_completion})
        print("=" * 88)
        print(f"CAA INTENSITY {intensity} | positive word score: {word_score(plus_completion)}")
        print(plus_completion.strip() or "(empty)")

    output_path = Path(args.output_dir) / "caa_example_visualization.png"
    visualize(positive_acts, negative_acts, caa_vector, completions, output_path)
    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
