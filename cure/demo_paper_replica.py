#!/usr/bin/env python3
"""
CURE Demo - Replicating Paper Experiments

This script replicates the exact experimental setup from the paper:
- Uses the same prompts as Imagenette objects (Table 4)
- Same alpha values (2.0 for objects, 5.0 for NSFW)
- No retain prompts (pure removal, matching paper setup)
- 20 prompts per concept
- Tests 4 different concepts from paper

Paper reference: Table 4 results (Page 9)
Expected results: 0-4% accuracy on erased concept, ~79% on others
"""

import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionPipeline
from cure import CURE
from cure.utils import set_seed, save_images, get_default_forget_prompts

def run_experiment(concept, alpha=2.0, seed=42, cache_dir="./models"):
    """
    Run a single concept removal experiment matching paper methodology
    """

    print("\n" + "=" * 80)
    print(f"CURE Experiment: Removing '{concept}' (alpha={alpha})")
    print("=" * 80)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print(f"Loading Stable Diffusion v1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        cache_dir=cache_dir,
    )

    cure = CURE(pipe, device=device)
    print(f"Model loaded: {cure}\n")

    # Get forget prompts (paper uses ~20 per concept)
    forget_prompts = get_default_forget_prompts(concept)
    print(f"Forget prompts ({len(forget_prompts)} total):")
    for i, p in enumerate(forget_prompts):
        print(f"  [{i:2d}] {p}")

    # Paper uses NO retain prompts for object removal (Table 4)
    retain_prompts = None
    print(f"\nRetain prompts: None (pure concept removal)")

    # Test prompt - single direct prompt
    # (Paper evaluates with classifier on 500 images, we just do visual check)
    test_prompts = [f"a potrait of {concept}", 
                    "a potrait of emma roberts "]
    print(f"\nTest prompts ({len(test_prompts)} total):")
    for i, p in enumerate(test_prompts):
        print(f"  [{i}] {p}")

    # Generate BEFORE
    print(f"\n{'='*80}")
    print(f"Generating BEFORE concept removal...")
    print(f"{'='*80}")

    generator = set_seed(seed)
    before_images = []

    for i, prompt in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Prompt: '{prompt}'")
        images = cure.generate(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        )
        before_images.extend(images)
        generator = set_seed(seed)

    # Apply concept removal
    print(f"\n{'='*80}")
    print(f"Applying Concept Removal (CURE)")
    print(f"{'='*80}")

    import time
    start_time = time.time()

    cure.erase_concept(
        forget_prompts=forget_prompts,
        retain_prompts=retain_prompts,
        alpha=alpha,
        save_original=False,
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Concept removal completed in {elapsed:.2f} seconds")
    print(f"  (Paper claims: ~2 seconds)")

    # Generate AFTER
    print(f"\n{'='*80}")
    print(f"Generating AFTER concept removal...")
    print(f"{'='*80}")

    generator = set_seed(seed)
    after_images = []

    for i, prompt in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Prompt: '{prompt}'")
        images = cure.generate(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        )
        after_images.extend(images)
        generator = set_seed(seed)

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving Results")
    print(f"{'='*80}")

    output_dir = Path(f"outputs/{concept}")
    output_dir.mkdir(parents=True, exist_ok=True)

    before_paths = save_images(before_images, str(output_dir / "before"), f"{concept}_before")
    after_paths = save_images(after_images, str(output_dir / "after"), f"{concept}_after")

    print(f"\n✓ Saved {len(before_paths)} 'before' images to: {output_dir / 'before'}")
    print(f"✓ Saved {len(after_paths)} 'after' images to: {output_dir / 'after'}")

    # Summary
    print(f"\n{'='*80}")
    print(f"RESULTS FOR '{concept.upper()}' (alpha={alpha})")
    print(f"{'='*80}")
    print(f"\nExpected from paper:")
    print(f"  - Concept should be completely removed (0% accuracy)")
    print(f"  - Unrelated concepts should still work (~79% accuracy)")
    print(f"\nActual results:")
    print(f"  - Check 'before' images: Should show concept normally")
    print(f"  - Check 'after' images: Should NOT show concept (or very different)")
    print(f"  - If after = before: ❌ Concept removal NOT working")
    print(f"  - If after ≠ before: ✓ Some effect happening (check quality)")


def main():
    """Run paper replica experiments"""

    parser = argparse.ArgumentParser(
        description="CURE - Replicating Paper Experiments (Table 4)"
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="cassette player",
        choices=[
            "cassette player", "chain saw", "french horn", "golf ball", "car",
            "emma stone", "taylor swift", "jennifer lawrence", "elon musk"
        ],
        help="Concept to remove (objects or celebrities)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Spectral expansion parameter (2.0 for objects, 5.0 for NSFW)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Directory to cache downloaded models (default: ./models)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CURE: Concept Unlearning via Orthogonal Representation Editing")
    print("=" * 80)
    print(f"\nPaper: Das Biswas, Roy, Kaushik Roy (Purdue University)")
    print(f"arXiv: 2505.12677v1")
    print(f"\nReplicating Table 4 Experiments (Object Removal)")
    print(f"=" * 80)

    run_experiment(
        concept=args.concept,
        alpha=args.alpha,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    print(f"\n{'='*80}")
    print(f"Experiment Complete!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Compare before/after images in outputs/{args.concept}/")
    print(f"2. If removal worked: Concept should be gone")
    print(f"3. If not working: Use debug_unlearning.py to diagnose")


if __name__ == "__main__":
    main()
