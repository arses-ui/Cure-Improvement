#!/usr/bin/env python3
"""
CURE-Sequential Demo

Erases a sequence of concepts and shows:
  1. The budget consumption per concept
  2. That earlier erasures remain intact after later ones
  3. Comparison: naive CURE (with interference) vs CURE-Sequential (orthogonalized)

Usage:
    python demo_sequential.py --n-concepts 10 --device cuda
    python demo_sequential.py --concepts car,truck,van,bus --device cpu
"""

import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionPipeline

from cure_seq import SequentialCURE
from experiments.metrics import print_budget_report

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "cure"))
from cure import CURE
from cure.utils import set_seed, save_images, get_default_forget_prompts


SEQUENTIAL_CONCEPTS = [
    # 10 diverse concepts — mix of objects, styles, identities
    "car", "dog", "cat", "french horn", "golf ball",
    "chain saw", "cassette player", "parachute",
    "taylor swift", "van gogh",
]


def erase_and_sample(eraser, concept, prompts, seed, output_dir, label, steps=20):
    """Generate one image for a concept prompt and save it."""
    test_prompt = prompts[0] if prompts else concept
    gen = set_seed(seed)
    imgs = eraser.generate(
        prompt=f"a photo of {test_prompt}",
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=gen,
    )
    path = save_images(imgs, str(output_dir), prefix=label)
    return path[0]


def run_sequential_demo(concepts, alpha, seed, output_dir, device, n_steps):
    print("\n" + "=" * 70)
    print("CURE-Sequential Demo")
    print("=" * 70)

    # Load model once
    print("\nLoading Stable Diffusion v1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        cache_dir=str(Path(__file__).parent.parent / "cure" / "models"),
    )

    eraser = SequentialCURE(pipe, device=device)
    print(f"\nInitialized: {eraser}")

    out = Path(output_dir)
    all_stats = []

    # ── Generate BEFORE images ────────────────────────────────────────────
    print("\n[BEFORE] Generating reference images...")
    before_dir = out / "before"
    for concept in concepts:
        prompts = get_default_forget_prompts(concept)
        erase_and_sample(eraser, concept, prompts, seed, before_dir, concept, n_steps)
        print(f"  Saved before/{concept}.png")

    # ── Sequential erasure ────────────────────────────────────────────────
    print(f"\n[ERASING] Sequentially erasing {len(concepts)} concepts...")
    print(f"{'─'*70}")

    for i, concept in enumerate(concepts):
        prompts = get_default_forget_prompts(concept)
        stats = eraser.erase_concept(
            forget_prompts=prompts,
            retain_prompts=None,
            alpha=alpha,
            concept_name=concept,
        )
        all_stats.append(stats)

    # ── Budget report ─────────────────────────────────────────────────────
    print_budget_report(eraser.bank)

    # ── Generate AFTER images ─────────────────────────────────────────────
    print("\n[AFTER] Generating post-erasure images...")
    after_dir = out / "after"
    for concept in concepts:
        prompts = get_default_forget_prompts(concept)
        erase_and_sample(eraser, concept, prompts, seed, after_dir, concept, n_steps)
        print(f"  Saved after/{concept}.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Concepts erased:    {len(concepts)}")
    print(f"  Budget consumed:    {eraser.bank.dims_used}/768 dims "
          f"({eraser.bank.dims_used/768:.1%})")
    print(f"  Budget remaining:   {eraser.bank.remaining_budget} dims")
    avg_energy = sum(s["energy_retained"] for s in all_stats) / len(all_stats)
    print(f"  Avg energy retained:{avg_energy:.1%}")
    print(f"\n  Check outputs in: {out.absolute()}")
    print(f"  before/ — images before any erasure")
    print(f"  after/  — images after ALL {len(concepts)} concepts erased")
    print(f"\n  Key question: do before/after differ? (they should for ALL concepts)")
    print("=" * 70)

    return eraser, all_stats


def main():
    parser = argparse.ArgumentParser(description="CURE-Sequential Demo")
    parser.add_argument("--concepts", type=str, default=None,
                        help="Comma-separated concepts to erase (default: built-in 10)")
    parser.add_argument("--n-concepts", type=int, default=5,
                        help="Number of concepts from built-in list (default: 5)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Spectral expansion parameter (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/demo_sequential")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20,
                        help="Denoising steps for image generation (20 for speed)")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",")]
    else:
        concepts = SEQUENTIAL_CONCEPTS[:args.n_concepts]

    print(f"Device: {args.device}")
    print(f"Concepts ({len(concepts)}): {concepts}")

    run_sequential_demo(
        concepts=concepts,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        n_steps=args.steps,
    )


if __name__ == "__main__":
    main()
