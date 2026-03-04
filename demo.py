#!/usr/bin/env python3
"""
CURE Demo: Concept Unlearning via Orthogonal Representation Editing

This script demonstrates erasing the "car" concept from Stable Diffusion v1.4.
After erasure, prompts asking for cars will generate alternative content while
unrelated concepts (like bicycles) remain intact.
"""

import torch
import argparse
from pathlib import Path

from diffusers import StableDiffusionPipeline
from cure import CURE
from cure.utils import set_seed, save_images, create_image_grid, get_default_forget_prompts


def main():
    parser = argparse.ArgumentParser(description="CURE Demo - Erase concepts from Stable Diffusion")
    parser.add_argument("--concept", type=str, default="car", help="Concept to erase")
    parser.add_argument("--alpha", type=float, default=5.0, help="Spectral expansion parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Model ID")
    parser.add_argument("--cache-dir", type=str, default="./models",
                        help="Directory to cache downloaded models (default: ./models)")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Concept to erase: {args.concept}")
    print(f"Alpha parameter: {args.alpha}")
    print()

    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion v1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # Disable for demo purposes
        cache_dir=args.cache_dir,
    )

    # Initialize CURE
    cure = CURE(pipe, device=device)
    print(f"Initialized: {cure}")
    print()

    # Define prompts for concept erasure
    forget_prompts = get_default_forget_prompts(args.concept)
    print(f"Forget prompts ({len(forget_prompts)} total):")
    for i, p in enumerate(forget_prompts):
        print(f"  [{i}] {p}")
    print()

    # NO retain prompts (testing pure removal)
    retain_prompts = None
    print("Retain prompts: None (testing pure concept removal)")
    print()

    # Test prompts - focused on the concept
    test_prompts = [
        f"a red {args.concept} on the street",
        f"a {args.concept} parked in front of a house",
        f"a {args.concept} driving down the road",
        f"a blue {args.concept}",
    ]
    print(f"Test prompts ({len(test_prompts)} total):")
    for i, p in enumerate(test_prompts):
        print(f"  [{i}] {p}")
    print()

    # Generate BEFORE erasure
    print("=" * 50)
    print("Generating images BEFORE concept erasure...")
    print("=" * 50)

    generator = set_seed(args.seed)
    before_images = []

    for prompt in test_prompts:
        print(f"  Generating: {prompt}")
        images = cure.generate(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        )
        before_images.extend(images)
        generator = set_seed(args.seed)  # Reset for consistency

    # Apply concept erasure
    print()
    print("=" * 50)
    print(f"Erasing concept: {args.concept}")
    print("=" * 50)

    import time
    start_time = time.time()

    cure.erase_concept(
        forget_prompts=forget_prompts,
        retain_prompts=retain_prompts,
        alpha=args.alpha
    )

    elapsed = time.time() - start_time
    print(f"Erasure completed in {elapsed:.2f} seconds")
    print()

    # Generate AFTER erasure
    print("=" * 50)
    print("Generating images AFTER concept erasure...")
    print("=" * 50)

    generator = set_seed(args.seed)
    after_images = []

    for prompt in test_prompts:
        print(f"  Generating: {prompt}")
        images = cure.generate(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        )
        after_images.extend(images)
        generator = set_seed(args.seed)  # Reset for consistency

    # Save results
    print()
    print("=" * 50)
    print("Saving results...")
    print("=" * 50)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save individual images
    before_paths = save_images(before_images, str(output_dir / "before"), "before")
    after_paths = save_images(after_images, str(output_dir / "after"), "after")

    print(f"Saved {len(before_paths)} 'before' images to {output_dir / 'before'}")
    print(f"Saved {len(after_paths)} 'after' images to {output_dir / 'after'}")

    # Create comparison grid if we have 4 images
    if len(before_images) == 4 and len(after_images) == 4:
        # Interleave before/after for side-by-side comparison
        comparison = []
        for b, a in zip(before_images, after_images):
            comparison.extend([b, a])

        grid = create_image_grid(comparison, rows=4, cols=2, padding=10)
        grid_path = output_dir / "comparison_grid.png"
        grid.save(grid_path)
        print(f"Saved comparison grid to {grid_path}")

    print()
    print("=" * 50)
    print("Demo complete!")
    print("=" * 50)
    print()
    print("Expected results:")
    print(f"  - Images with '{args.concept}' prompts should show alternative content")
    print("  - Images with unrelated prompts (bicycle, sunset) should be unchanged")
    print()


if __name__ == "__main__":
    main()
