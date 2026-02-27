#!/usr/bin/env python3
"""
Baseline: Reproduce CURE Figure 6 interference degradation.

This experiment shows WHY we need CURE-Sequential by demonstrating the
cross-term interference in naive sequential CURE application.

We erase N concepts sequentially using plain CURE (no orthogonalization),
then check whether the FIRST erased concept is still properly erased.
This replicates Figure 6 from the CURE paper.

Then we run the same sequence with CURE-Sequential and show SIS stays near 0.

Usage:
    python experiments/baseline_naive.py --n-concepts 20 --device cuda
"""

import sys
import torch
import argparse
import json
from pathlib import Path
from diffusers import StableDiffusionPipeline

# Make sure we can import from parent project
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cure"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cure import CURE
from cure.utils import set_seed, save_images, get_default_forget_prompts
from cure_seq import SequentialCURE
from experiments.metrics import print_budget_report


# Imagenette concepts matching CURE paper Table 4
IMAGENETTE_CONCEPTS = [
    "cassette player", "chain saw", "french horn", "golf ball",
    "garbage truck", "gas pump", "parachute", "tench",
    "english springer", "church",
]

# Additional concepts for longer sequences
EXTENDED_CONCEPTS = IMAGENETTE_CONCEPTS + [
    "car", "dog", "cat", "bicycle", "motorcycle",
    "airplane", "ship", "truck", "bird", "horse",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
]


def load_pipeline(device, cache_dir):
    print(f"Loading Stable Diffusion v1.4 on {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=cache_dir,
    )
    return pipe


def measure_concept_presence(eraser, concept, n_images=10, seed=0, steps=20):
    """
    Rough proxy for concept accuracy: does the model generate the concept?
    Returns list of images for manual inspection (no classifier needed for baseline).
    For proper eval, plug in a ResNet-50 classifier.
    """
    prompts = get_default_forget_prompts(concept)
    test_prompt = prompts[0] if prompts else concept
    images = []
    for i in range(n_images):
        gen = set_seed(seed + i)
        imgs = eraser.generate(
            prompt=f"a photo of a {test_prompt}",
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=gen,
        )
        images.extend(imgs)
    return images


def run_naive_baseline(concepts, alpha, seed, output_dir, device, steps, cache_dir):
    """
    Naive sequential CURE (no orthogonalization).
    Erase all concepts, then check whether concept[0] is still erased.
    This should show degradation matching CURE Figure 6.
    """
    print("\n" + "=" * 70)
    print("NAIVE CURE BASELINE (sequential, no orthogonalization)")
    print("=" * 70)

    pipe = load_pipeline(device, cache_dir)
    eraser = CURE(pipe, device=device)

    out = Path(output_dir) / "naive"

    # Erase all concepts sequentially
    stats = []
    for i, concept in enumerate(concepts):
        prompts = get_default_forget_prompts(concept)
        import time
        t0 = time.time()
        eraser.erase_concept(forget_prompts=prompts, retain_prompts=None, alpha=alpha,
                             save_original=(i == 0))
        elapsed = time.time() - t0
        stats.append({"concept": concept, "step": i + 1, "elapsed_s": elapsed})
        print(f"[{i+1:3d}/{len(concepts)}] Erased '{concept}' in {elapsed:.2f}s")

        # After every 10 concepts, check if concept[0] is still erased
        if (i + 1) % 10 == 0 or i == len(concepts) - 1:
            imgs = measure_concept_presence(eraser, concepts[0], n_images=3, seed=seed, steps=steps)
            ck_dir = out / f"checkpoint_{i+1}" / concepts[0].replace(" ", "_")
            save_images(imgs, str(ck_dir), prefix=f"step{i+1}")
            print(f"  → Saved concept[0]='{concepts[0]}' images at checkpoint {i+1}")

    # Final check on all concepts
    print(f"\nFinal state: erased {len(concepts)} concepts with naive CURE")
    return stats


def run_sequential_orth(concepts, alpha, seed, output_dir, device, steps, cache_dir):
    """
    CURE-Sequential (with orthogonalization).
    Same concept order, same alpha — should maintain erasure quality throughout.
    """
    print("\n" + "=" * 70)
    print("CURE-SEQUENTIAL (orthogonalized)")
    print("=" * 70)

    pipe = load_pipeline(device, cache_dir)
    eraser = SequentialCURE(pipe, device=device)

    out = Path(output_dir) / "sequential_orth"
    all_stats = []

    for i, concept in enumerate(concepts):
        prompts = get_default_forget_prompts(concept)
        stats = eraser.erase_concept(
            forget_prompts=prompts,
            retain_prompts=None,
            alpha=alpha,
            concept_name=concept,
        )
        all_stats.append(stats)

        if (i + 1) % 10 == 0 or i == len(concepts) - 1:
            imgs = measure_concept_presence(eraser, concepts[0], n_images=3, seed=seed, steps=steps)
            ck_dir = out / f"checkpoint_{i+1}" / concepts[0].replace(" ", "_")
            save_images(imgs, str(ck_dir), prefix=f"step{i+1}")
            print(f"  → Saved concept[0]='{concepts[0]}' images at checkpoint {i+1}")

    print_budget_report(eraser.bank)
    return all_stats, eraser.bank


def main():
    parser = argparse.ArgumentParser(description="Naive CURE baseline vs CURE-Sequential")
    parser.add_argument("--n-concepts", type=int, default=10,
                        help="Number of concepts to erase sequentially")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cache-dir", type=str,
                        default=str(Path(__file__).parent.parent.parent / "cure" / "models"))
    parser.add_argument("--naive-only", action="store_true",
                        help="Run only the naive baseline (skip sequential)")
    parser.add_argument("--orth-only", action="store_true",
                        help="Run only the orthogonalized version")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    concepts = IMAGENETTE_CONCEPTS[:args.n_concepts]
    if args.n_concepts > len(IMAGENETTE_CONCEPTS):
        concepts = EXTENDED_CONCEPTS[:args.n_concepts]

    print(f"Device:   {args.device}")
    print(f"Concepts: {concepts}")
    print(f"Alpha:    {args.alpha}")

    results = {}

    if not args.orth_only:
        naive_stats = run_naive_baseline(
            concepts, args.alpha, args.seed,
            args.output_dir, args.device, args.steps, args.cache_dir
        )
        results["naive"] = naive_stats

    if not args.naive_only:
        orth_stats, bank = run_sequential_orth(
            concepts, args.alpha, args.seed,
            args.output_dir, args.device, args.steps, args.cache_dir
        )
        results["sequential_orth"] = orth_stats

    # Save stats
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stats.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nStats saved to {out / 'stats.json'}")

    print("\n" + "=" * 70)
    print("WHAT TO LOOK FOR:")
    print("  naive/checkpoint_N/concept[0]/ — check if concept[0] reappears")
    print("  sequential_orth/checkpoint_N/  — should stay erased throughout")
    print("  If naive shows leakage after N>50 but orth doesn't → paper result!")
    print("=" * 70)


if __name__ == "__main__":
    main()
