#!/usr/bin/env python3
"""
Test both transpose conventions to see which one actually works for concept removal.
Generate images with BOTH weight update formulas and compare.
"""

import torch
from diffusers import StableDiffusionPipeline
from cure import CURE
from cure.spectral import compute_discriminative_projector
from cure.utils import get_default_forget_prompts, set_seed

def test_both_formulas():
    """Test W_new = W - W@Pdis vs W_new = W - W@Pdis.T"""

    print("=" * 80)
    print("Testing Weight Update Formula: With vs Without Transpose")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model twice (we'll test both formulas)
    print("\nLoading models...")
    pipe1 = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe2 = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    cure1 = CURE(pipe1, device=device)
    cure2 = CURE(pipe2, device=device)

    # Get concept
    concept = "car"
    forget_prompts = get_default_forget_prompts(concept)
    retain_prompts = ["bicycle", "motorcycle", "bus", "road", "street", "building"]
    alpha = 2.0

    # Get embeddings
    forget_embeddings = cure1.get_text_embeddings(forget_prompts)
    retain_embeddings = cure1.get_text_embeddings(retain_prompts)

    # Compute projector
    projector = compute_discriminative_projector(forget_embeddings, retain_embeddings, alpha)

    print(f"\nProjector computed:")
    print(f"  Shape: {projector.shape}")
    print(f"  Norm: {torch.norm(projector):.4f}")

    # ============================================================
    # FORMULA 1: W_new = W - W @ Pdis (NO transpose)
    # ============================================================
    print("\n" + "=" * 80)
    print("FORMULA 1: W_new = W - W @ Pdis (NO transpose)")
    print("=" * 80)

    def apply_update_formula1(pipe, proj):
        """Apply W_new = W - W @ Pdis"""
        from cure.attention import get_cross_attention_layers

        proj = proj.to(device=pipe.device, dtype=next(pipe.unet.parameters()).dtype)

        count = 0
        for layer in get_cross_attention_layers(pipe.unet):
            Wk = layer.to_k.weight.data
            Wv = layer.to_v.weight.data

            # NO transpose
            layer.to_k.weight.data = Wk - Wk @ proj
            layer.to_v.weight.data = Wv - Wv @ proj
            count += 1

        return count

    apply_update_formula1(pipe1, projector)
    print(f"Updated {648} cross-attention layers (formula 1)")

    # ============================================================
    # FORMULA 2: W_new = W - W @ Pdis.T (WITH transpose)
    # ============================================================
    print("\n" + "=" * 80)
    print("FORMULA 2: W_new = W - W @ Pdis.T (WITH transpose)")
    print("=" * 80)

    def apply_update_formula2(pipe, proj):
        """Apply W_new = W - W @ Pdis.T"""
        from cure.attention import get_cross_attention_layers

        proj = proj.to(device=pipe.device, dtype=next(pipe.unet.parameters()).dtype)
        proj_t = proj.T

        count = 0
        for layer in get_cross_attention_layers(pipe.unet):
            Wk = layer.to_k.weight.data
            Wv = layer.to_v.weight.data

            # WITH transpose
            layer.to_k.weight.data = Wk - Wk @ proj_t
            layer.to_v.weight.data = Wv - Wv @ proj_t
            count += 1

        return count

    apply_update_formula2(pipe2, projector)
    print(f"Updated {648} cross-attention layers (formula 2)")

    # ============================================================
    # TEST: Generate with both formulas
    # ============================================================
    test_prompts = [
        "a red car on the street",
        "a car parked in front of a house",
        "a blue car driving",
    ]

    print("\n" + "=" * 80)
    print("GENERATING WITH BOTH FORMULAS")
    print("=" * 80)

    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}: '{prompt}'")

        generator1 = set_seed(42)
        generator2 = set_seed(42)

        print(f"  Generating with Formula 1 (W - W@Pdis)...")
        images1 = pipe1(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator1,
        ).images

        print(f"  Generating with Formula 2 (W - W@Pdis.T)...")
        images2 = pipe2(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator2,
        ).images

        # Save for comparison
        images1[0].save(f"formula1_test{i+1}.png")
        images2[0].save(f"formula2_test{i+1}.png")

        print(f"  Saved formula1_test{i+1}.png and formula2_test{i+1}.png")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("""
Now compare the generated images visually:
  formula1_test*.png: Using W_new = W - W @ Pdis
  formula2_test*.png: Using W_new = W - W @ Pdis.T

Questions to ask:
1. Which formula removes "car" better?
2. Are the images different between formulas?
3. Does either formula work at all?

If formula 1 works better: Remove the .T ✓ (already done)
If formula 2 works better: Add back the .T ✗ (need to revert)
If neither works: Problem is deeper than transpose
""")

if __name__ == "__main__":
    test_both_formulas()
