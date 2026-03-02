"""
Evaluation metrics for CURE-DiT.

Same metrics as CURE (concept accuracy, FID, CLIP score) adapted for SD3/Flux.
"""

import torch
from typing import List, Dict
from PIL import Image


def generate_concept_images(
    eraser,
    prompt: str,
    n_images: int,
    seed: int = 42,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
) -> List[Image.Image]:
    """Generate n images for a concept prompt with deterministic seeds."""
    images = []
    for i in range(n_images):
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        imgs = eraser.generate(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )
        images.extend(imgs)
    return images


def concept_accuracy(
    images: List[Image.Image],
    classifier,
    target_class_idx: int,
    transform,
) -> float:
    """
    Top-1 accuracy of a pretrained classifier on generated images.
    Lower = better erasure.
    """
    device = next(classifier.parameters()).device
    correct = 0
    for img in images:
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(x)
        pred = logits.argmax(dim=1).item()
        if pred == target_class_idx:
            correct += 1
    return correct / len(images)


def erasure_report(stats_list: List[Dict]) -> None:
    """Print a summary of all erasures performed."""
    print(f"\n{'='*60}")
    print(f"CURE-DiT ERASURE REPORT")
    print(f"{'='*60}")
    print(f"  Total concepts erased: {len(stats_list)}")
    print(f"\n  {'Concept':<25} {'Layers':>7} {'Time':>8}")
    print(f"  {'-'*45}")
    for s in stats_list:
        print(f"  {s['concept']:<25} {s['n_layers_edited']:>7} {s['elapsed_s']:>7.2f}s")
    total_time = sum(s["elapsed_s"] for s in stats_list)
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"{'='*60}")
