#!/usr/bin/env python3
"""
Debug script to verify the spectral expansion (regularization) mechanism
"""

import torch
from diffusers import StableDiffusionPipeline
from cure import CURE
from cure.spectral import compute_svd, spectral_expansion, build_projector
from cure.utils import get_default_forget_prompts

def debug_regularization():
    """Debug the spectral expansion/regularization term"""

    print("=" * 70)
    print("CURE DEBUG: Spectral Expansion (Regularization) Term")
    print("=" * 70)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    cure = CURE(pipe, device=device)

    # Get embeddings
    concept = "car"
    forget_prompts = get_default_forget_prompts(concept)
    print(f"\n2. Getting embeddings for concept '{concept}'...")
    forget_embeddings = cure.get_text_embeddings(forget_prompts)
    print(f"   Shape: {forget_embeddings.shape}")
    print(f"   Expected: [10, 768] (10 prompts, 768 dims)")

    # Compute SVD
    print(f"\n3. Computing SVD on embeddings...")
    U, S, Vh = compute_svd(forget_embeddings)
    print(f"   U shape: {U.shape} (left singular vectors)")
    print(f"   S shape: {S.shape} (singular values)")
    print(f"   Vh shape: {Vh.shape} (right singular vectors)")

    print(f"\n   Singular values: {S}")
    print(f"   Sum of singular values: {S.sum():.4f}")

    # Check spectral expansion WITHOUT regularization (alpha=1)
    print(f"\n4. Spectral expansion function f(ri; α):")
    print(f"   This is the Tikhonov-inspired regularization term")

    alpha_values = [1.0, 2.0, 5.0]
    for alpha in alpha_values:
        print(f"\n   Alpha = {alpha}:")
        f_r = spectral_expansion(S, alpha)
        print(f"   f(r; α) = {f_r}")
        print(f"   Sum: {f_r.sum():.4f}")
        print(f"   Min: {f_r.min():.4f}, Max: {f_r.max():.4f}")
        print(f"   Mean: {f_r.mean():.4f}")

    # Compare projectors at different alpha values
    print(f"\n5. Building projectors at different alpha values:")
    V = Vh.T  # Convert to [768, k] format

    projectors = {}
    for alpha in alpha_values:
        projector = build_projector(V, S, alpha)
        projectors[alpha] = projector

        print(f"\n   Alpha = {alpha}:")
        print(f"   Projector shape: {projector.shape}")
        print(f"   Projector norm: {torch.norm(projector):.4f}")
        print(f"   Projector min: {projector.min():.6f}, max: {projector.max():.6f}")
        print(f"   Rank (approx): {torch.linalg.matrix_rank(projector)}")

    # Check the effect on weights
    print(f"\n6. Comparing weight updates at different alphas:")

    # Get a sample weight matrix
    sample_layer = next(cure.pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_k.parameters())
    Wk = sample_layer.clone()
    print(f"   Original Wk norm: {torch.norm(Wk):.4f}")

    for alpha in alpha_values:
        projector = projectors[alpha]
        Wk_new = Wk - Wk @ projector
        weight_change = torch.norm(Wk_new - Wk)
        print(f"\n   Alpha = {alpha}:")
        print(f"   Weight change magnitude: {weight_change:.4f}")
        print(f"   Percentage of original: {(weight_change / torch.norm(Wk) * 100):.2f}%")

    # Check if regularization is actually working
    print(f"\n7. Regularization Strength Analysis:")

    # The regularization should amplify smaller singular values
    print(f"\n   Comparing f(r; α=1) vs f(r; α=2):")
    f_r_1 = spectral_expansion(S, 1.0)
    f_r_2 = spectral_expansion(S, 2.0)

    print(f"   α=1: {f_r_1}")
    print(f"   α=2: {f_r_2}")

    ratio = f_r_2 / (f_r_1 + 1e-10)
    print(f"   Ratio (α=2 / α=1): {ratio}")
    print(f"   Amplification of small values: {ratio[-1]:.4f}x for smallest value")

    print(f"\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)

    # Recommendations
    print(f"\nRecommendations:")
    print(f"1. If projector norms are similar across alphas → regularization may be weak")
    print(f"2. If weight changes are < 0.5% → projector is too small")
    print(f"3. If singular values are very uneven → may need higher alpha")
    print(f"4. Try alpha=5.0 for stronger erasure (currently using 2.0)")

if __name__ == "__main__":
    debug_regularization()
