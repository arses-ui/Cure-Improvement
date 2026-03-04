"""
Evaluation metrics for CURE-Sequential.

New metric: Sequential Interference Score (SIS)
    Measures how much concept C1's erasure has degraded after erasing C1, C2, ..., Cn.
    Ideal: SIS = 0 (orthogonal method should maintain this).
    CURE (naive): SIS > 0 after ~50 concepts.

Standard metrics (from CURE paper):
    - Concept accuracy via ResNet-50 classifier (↓ = better erasure)
    - Retained class accuracy (↑ = better preservation)
    - FID on COCO-30k (↓ = better generation quality)
    - CLIP score (↑ = better prompt alignment)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
from PIL import Image


def generate_concept_images(
    eraser,
    prompt_template: str,
    n_images: int,
    seed: int = 42,
    num_inference_steps: int = 20,   # fast for eval
    guidance_scale: float = 7.5,
) -> List[Image.Image]:
    """Generate n images for a concept prompt with fixed seed."""
    images = []
    for i in range(n_images):
        gen = torch.Generator()
        gen.manual_seed(seed + i)
        imgs = eraser.generate(
            prompt=prompt_template,
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
    Lower = better erasure (concept is not generated).
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


def sequential_interference_score(
    concept_name: str,
    prompt_template: str,
    eraser_checkpoint_1: object,    # eraser after only concept_1 was erased
    eraser_checkpoint_n: object,    # eraser after concept_1..n were erased
    classifier,
    target_class_idx: int,
    transform,
    n_images: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Sequential Interference Score (SIS).

    Generates concept images from both checkpoint erasers and computes
    the change in concept accuracy. Ideally = 0 (orthogonal method),
    but positive (concept leaking back) for naive sequential CURE.

    Args:
        concept_name: Human label for the concept
        prompt_template: e.g. "a photo of a {car}"
        eraser_checkpoint_1: Model state after erasing ONLY this concept
        eraser_checkpoint_n: Model state after erasing this concept + n-1 others
        classifier: Pretrained ResNet-50 (or similar)
        target_class_idx: ImageNet class index for this concept
        transform: Image preprocessing for classifier
        n_images: Number of images to generate per checkpoint
        seed: Random seed

    Returns:
        Dict with:
          acc_after_1: Concept accuracy after only 1 erasure (should be low)
          acc_after_n: Concept accuracy after n erasures (should stay low)
          sis: acc_after_n - acc_after_1 (ideal: 0, bad: positive)
    """
    imgs_1 = generate_concept_images(eraser_checkpoint_1, prompt_template, n_images, seed)
    imgs_n = generate_concept_images(eraser_checkpoint_n, prompt_template, n_images, seed)

    acc_1 = concept_accuracy(imgs_1, classifier, target_class_idx, transform)
    acc_n = concept_accuracy(imgs_n, classifier, target_class_idx, transform)

    return {
        "concept": concept_name,
        "acc_after_1_erasure": acc_1,
        "acc_after_n_erasures": acc_n,
        "sis": acc_n - acc_1,  # positive = concept leaking back = bad
    }


def budget_analysis(bank) -> Dict:
    """
    Analyze subspace budget consumption across sequential erasures.
    Returns per-concept dim consumption and cumulative budget usage.
    """
    records = []
    cumulative = 0
    for entry in bank.concepts:
        cumulative += entry.n_dims
        records.append({
            "concept": entry.name,
            "dims_consumed": entry.n_dims,
            "cumulative_dims": cumulative,
            "budget_fraction": cumulative / bank.hidden_dim,
            "energy_retained": entry.energy_retained,
        })
    return {
        "records": records,
        "total_dims_used": bank.dims_used,
        "total_concepts": len(bank.concepts),
        "avg_dims_per_concept": bank.dims_used / max(len(bank.concepts), 1),
        "budget_remaining": bank.remaining_budget,
        "projected_capacity": int(bank.hidden_dim / max(
            bank.dims_used / max(len(bank.concepts), 1), 1
        )),
    }


def print_budget_report(bank) -> None:
    """Pretty-print the budget analysis."""
    report = budget_analysis(bank)
    print(f"\n{'='*60}")
    print(f"SUBSPACE BUDGET REPORT")
    print(f"{'='*60}")
    print(f"  Concepts erased:      {report['total_concepts']}")
    print(f"  Dims used:            {report['total_dims_used']}/{bank.hidden_dim} "
          f"({report['total_dims_used']/bank.hidden_dim:.1%})")
    print(f"  Avg dims/concept:     {report['avg_dims_per_concept']:.1f}")
    print(f"  Projected capacity:   ~{report['projected_capacity']} concepts")
    print(f"  Budget remaining:     {report['budget_remaining']} dims")
    print(f"\n  Per-concept:")
    print(f"  {'Concept':<25} {'Dims':>6} {'Cumul':>7} {'Energy':>8}")
    print(f"  {'-'*50}")
    for r in report["records"]:
        print(
            f"  {r['concept']:<25} {r['dims_consumed']:>6} "
            f"{r['cumulative_dims']:>7} {r['energy_retained']:>7.1%}"
        )
    print(f"{'='*60}\n")
