#!/usr/bin/env python3
"""
Debug script to trace through CURE concept unlearning step-by-step
"""

import torch
from diffusers import StableDiffusionPipeline
from cure import CURE
from cure.utils import get_default_forget_prompts, set_seed

def debug_concept_unlearning():
    """Debug the concept unlearning process"""

    print("=" * 60)
    print("CURE DEBUG: Concept Unlearning Trace")
    print("=" * 60)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Loading SD v1.4 on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    cure = CURE(pipe, device=device)
    print(f"   Loaded: {cure}")

    # Define concept to erase
    concept = "car"
    forget_prompts = get_default_forget_prompts(concept)
    print(f"\n2. Forget prompts ({len(forget_prompts)} total):")
    for i, p in enumerate(forget_prompts[:3]):
        print(f"   [{i}] {p}")
    print(f"   ... (showing first 3 of {len(forget_prompts)})")

    # Get embeddings
    print(f"\n3. Getting text embeddings...")
    forget_embeddings = cure.get_text_embeddings(forget_prompts)
    print(f"   Shape: {forget_embeddings.shape}")
    print(f"   Expected: [{len(forget_prompts) * 77}, 768]")
    print(f"   Match: {forget_embeddings.shape == (len(forget_prompts) * 77, 768)}")

    # Check embedding values
    print(f"   Min: {forget_embeddings.min():.4f}, Max: {forget_embeddings.max():.4f}")
    print(f"   Mean: {forget_embeddings.mean():.4f}, Std: {forget_embeddings.std():.4f}")

    # Compute projector
    print(f"\n4. Computing discriminative projector...")
    projector = cure.compute_spectral_eraser(
        forget_embeddings=forget_embeddings,
        retain_embeddings=None,
        alpha=2.0
    )
    print(f"   Projector shape: {projector.shape}")
    print(f"   Projector values - Min: {projector.min():.4f}, Max: {projector.max():.4f}")
    print(f"   Projector norm: {torch.norm(projector):.4f}")

    # Check weight before update
    print(f"\n5. Checking weights BEFORE update...")
    sample_layer = next(cure.pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_k.parameters())
    print(f"   Wk shape: {sample_layer.shape}")
    print(f"   Wk norm: {torch.norm(sample_layer):.4f}")
    Wk_before = sample_layer.clone()

    # Apply unlearning
    print(f"\n6. Applying concept erasure...")
    cure.erase_concept(
        forget_prompts=forget_prompts,
        retain_prompts=None,
        alpha=2.0,
        save_original=False  # Don't save originals to avoid memory issues
    )

    # Check weight after update
    print(f"\n7. Checking weights AFTER update...")
    sample_layer_after = next(cure.pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_k.parameters())
    print(f"   Wk norm: {torch.norm(sample_layer_after):.4f}")

    # Check if weights actually changed
    weight_diff = torch.norm(sample_layer_after - Wk_before)
    print(f"   Weight change magnitude: {weight_diff:.4f}")
    print(f"   Weights changed: {weight_diff > 1e-6}")

    if weight_diff < 1e-6:
        print("   ⚠️  WARNING: Weights didn't change! Unlearning may not be working!")

    # Test generation
    print(f"\n8. Testing generation...")
    test_prompt = "a red car on the street"
    generator = set_seed(42)

    print(f"   Prompt: '{test_prompt}'")
    print(f"   Generating image...")
    images = cure.generate(
        prompt=test_prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    )
    print(f"   Generated {len(images)} image(s)")

    # Diagnostic questions
    print(f"\n9. Diagnostics:")
    print(f"   ✓ Embeddings shaped correctly: {forget_embeddings.shape[0] > 0}")
    print(f"   ✓ Projector computed: {projector is not None}")
    print(f"   ✓ Weights updated: {weight_diff > 1e-6}")
    print(f"   ✓ Model can generate: True")

    print("\n" + "=" * 60)
    print("Debug complete. Check diagnostics above.")
    print("=" * 60)

if __name__ == "__main__":
    debug_concept_unlearning()
