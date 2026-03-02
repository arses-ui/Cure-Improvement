#!/usr/bin/env python3
"""
CURE-DiT Demo: Concept erasure on Stable Diffusion 3.

Erases concepts from SD3 by editing text-stream projections (add_k_proj / add_v_proj)
in each JointTransformerBlock, using CURE's spectral projection method.

Usage:
    python demo_sd3.py --concept "car" --device cuda
    python demo_sd3.py --concept "Van Gogh" --alpha 2.0 --device mps
    python demo_sd3.py --concept "nudity" --alpha 5.0 --device cuda

Requirements:
    - Access to an SD3 model (stabilityai/stable-diffusion-3.5-medium or similar)
    - HuggingFace token with gated model access (huggingface-cli login)
"""

import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusion3Pipeline

from cure_dit import SD3CURE


# Default forget prompts for common concepts
DEFAULT_PROMPTS = {
    "car": ["a photo of a car", "car", "automobile", "vehicle driving"],
    "dog": ["a photo of a dog", "dog", "puppy", "canine"],
    "cat": ["a photo of a cat", "cat", "kitten", "feline"],
    "Van Gogh": ["painting by Van Gogh", "Van Gogh style", "starry night Van Gogh"],
    "nudity": ["nudity", "naked", "nude person", "nsfw", "explicit content"],
}


def get_prompts(concept: str) -> list:
    """Get forget prompts for a concept."""
    if concept in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[concept]
    return [concept, f"a photo of {concept}", f"an image of {concept}"]


def main():
    parser = argparse.ArgumentParser(description="CURE-DiT: Concept erasure for SD3")
    parser.add_argument("--concept", type=str, default="car",
                        help="Concept to erase")
    parser.add_argument("--model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium",
                        help="SD3 model ID")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Spectral expansion parameter")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=28,
                        help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/demo_sd3")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading {args.model}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
    )

    eraser = SD3CURE(pipe, device=args.device)
    print(f"Initialized: {eraser}")

    # ── Generate BEFORE ─────────────────────────────────────────────────
    test_prompt = f"a photo of a {args.concept}"
    print(f"\n[BEFORE] Generating: '{test_prompt}'")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    before_imgs = eraser.generate(
        prompt=test_prompt,
        num_inference_steps=args.steps,
        generator=gen,
    )
    before_path = out / f"before_{args.concept.replace(' ', '_')}.png"
    before_imgs[0].save(str(before_path))
    print(f"  Saved: {before_path}")

    # ── Erase concept ───────────────────────────────────────────────────
    forget_prompts = get_prompts(args.concept)
    print(f"\n[ERASING] Concept: '{args.concept}' with {len(forget_prompts)} prompts")
    stats = eraser.erase_concept(
        forget_prompts=forget_prompts,
        alpha=args.alpha,
        concept_name=args.concept,
    )

    # ── Generate AFTER ──────────────────────────────────────────────────
    print(f"\n[AFTER] Generating: '{test_prompt}'")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    after_imgs = eraser.generate(
        prompt=test_prompt,
        num_inference_steps=args.steps,
        generator=gen,
    )
    after_path = out / f"after_{args.concept.replace(' ', '_')}.png"
    after_imgs[0].save(str(after_path))
    print(f"  Saved: {after_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CURE-DiT Erasure Summary")
    print(f"{'='*60}")
    print(f"  Model:        {args.model}")
    print(f"  Concept:      {args.concept}")
    print(f"  Alpha:        {args.alpha}")
    print(f"  Context dim:  {stats['context_dim']}")
    print(f"  Layers edited:{stats['n_layers_edited']}")
    print(f"  Time:         {stats['elapsed_s']:.2f}s")
    print(f"\n  Before: {before_path}")
    print(f"  After:  {after_path}")
    print(f"  Compare these images to verify erasure worked!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
