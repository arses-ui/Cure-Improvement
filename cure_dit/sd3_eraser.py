"""
SD3CURE: CURE concept erasure for Stable Diffusion 3.

Ports CURE's closed-form spectral erasure to SD3's MM-DiT architecture.
Instead of editing UNet cross-attention Wk/Wv (as in SD1.4), we edit the
text-stream projections (add_k_proj / add_v_proj) in each JointTransformerBlock.

Architecture flow:
    T5 text encoder → [batch, seq_len, 4096]
    context_embedder → [batch, seq_len, 1152]     ← we compute Pdis in this space
    JointTransformerBlock × 24:
        add_k_proj: [1152] → [inner_kv_dim]       ← CURE edit target
        add_v_proj: [1152] → [inner_kv_dim]       ← CURE edit target

Usage:
    from diffusers import StableDiffusion3Pipeline
    from cure_dit import SD3CURE

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.float16,
    )
    eraser = SD3CURE(pipe)
    eraser.erase_concept(["a photo of a car", "car", "automobile"])
"""

import torch
import time
from typing import List, Optional, Union
from PIL import Image

from .spectral import compute_discriminative_projector
from .attention_sd3 import (
    get_joint_attention_layers,
    apply_weight_update_sd3,
    count_joint_attention_layers,
    ensure_unfused,
    get_context_dim,
)


class SD3CURE:
    """
    CURE concept erasure for Stable Diffusion 3.

    Computes a discriminative projector from T5 text embeddings (projected to
    context_dim=1152) and applies it to all JointTransformerBlock text-stream
    key/value projections.
    """

    def __init__(
        self,
        pipe,
        device: Optional[str] = None,
    ):
        self.pipe = pipe

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.pipe = self.pipe.to(self.device)

        # Ensure text-stream projections are not fused (need separate k/v)
        ensure_unfused(self.pipe.transformer)

        # Detect context dimension
        self.context_dim = get_context_dim(self.pipe.transformer)

        self.erased_concepts: List[str] = []
        self._original_weights: Optional[list] = None

    # ── Text embeddings ─────────────────────────────────────────────────────

    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """
        Extract text embeddings in the context space that feeds add_k_proj/add_v_proj.

        SD3 pipeline: T5 encoder → [batch, seq_len, 4096] → context_embedder → [batch, seq_len, 1152]

        We replicate this path to get the actual features that the attention
        layers see, so our Pdis operates in the correct space.

        Returns:
            [n_prompts * seq_len, context_dim] tensor (context_dim=1152 for SD3 medium)
        """
        # Use T5 (text_encoder_3) as the primary encoder for SD3
        # T5 carries the bulk of the semantic information
        tokenizer = self.pipe.tokenizer_3
        text_encoder = self.pipe.text_encoder_3

        if tokenizer is None or text_encoder is None:
            raise ValueError(
                "T5 tokenizer/encoder not found. SD3 pipeline must include "
                "text_encoder_3 (T5-XXL). Load with: "
                "StableDiffusion3Pipeline.from_pretrained(...)"
            )

        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)

        with torch.no_grad():
            # T5 forward pass → [batch, seq_len, 4096]
            t5_output = text_encoder(input_ids).last_hidden_state

            # Project through context_embedder → [batch, seq_len, context_dim]
            context_embedder = self.pipe.transformer.context_embedder
            projected = context_embedder(t5_output.to(context_embedder.weight.dtype))

        batch_size, seq_len, context_dim = projected.shape
        return projected.reshape(batch_size * seq_len, context_dim).float()

    # ── Weight save/restore ─────────────────────────────────────────────────

    def save_original_weights(self) -> None:
        """Save original add_k_proj / add_v_proj weights for restoration."""
        self._original_weights = []
        for layer in get_joint_attention_layers(self.pipe.transformer):
            entry = {}
            if layer.add_k_proj is not None:
                entry["Wk"] = layer.add_k_proj.weight.data.clone()
            if layer.add_v_proj is not None:
                entry["Wv"] = layer.add_v_proj.weight.data.clone()
            self._original_weights.append(entry)

    def restore_original_weights(self) -> None:
        """Restore saved weights and clear erasure history."""
        if self._original_weights is None:
            raise ValueError("No weights saved. Call save_original_weights() first.")
        layers = get_joint_attention_layers(self.pipe.transformer)
        for layer, w in zip(layers, self._original_weights):
            if "Wk" in w and layer.add_k_proj is not None:
                layer.add_k_proj.weight.data = w["Wk"].clone()
            if "Wv" in w and layer.add_v_proj is not None:
                layer.add_v_proj.weight.data = w["Wv"].clone()
        self.erased_concepts = []
        print("Weights restored.")

    # ── Main erasure ────────────────────────────────────────────────────────

    def erase_concept(
        self,
        forget_prompts: List[str],
        retain_prompts: Optional[List[str]] = None,
        alpha: float = 2.0,
        concept_name: Optional[str] = None,
        save_original: bool = True,
    ) -> dict:
        """
        Erase a concept from the SD3 model using CURE's spectral projection.

        Args:
            forget_prompts: Prompts describing the concept to erase
            retain_prompts: Optional prompts for concepts to preserve
            alpha: Spectral expansion parameter (2.0 for objects, 5.0 for NSFW)
            concept_name: Label for tracking
            save_original: Save weights before first erasure

        Returns:
            Dict with erasure stats
        """
        if concept_name is None:
            concept_name = forget_prompts[0]

        if save_original and self._original_weights is None:
            self.save_original_weights()

        start = time.time()

        # Step 1: Get projected text embeddings (1152-dim)
        forget_emb = self.get_text_embeddings(forget_prompts)
        retain_emb = self.get_text_embeddings(retain_prompts) if retain_prompts else None

        # Step 2: Compute discriminative projector [context_dim, context_dim]
        Pdis = compute_discriminative_projector(
            forget_embeddings=forget_emb,
            retain_embeddings=retain_emb,
            alpha=alpha,
        )

        # Step 3: Apply to all JointTransformerBlock text-stream projections
        n_layers = 0
        for layer in get_joint_attention_layers(self.pipe.transformer):
            apply_weight_update_sd3(layer, Pdis, device=self.device)
            n_layers += 1

        elapsed = time.time() - start
        self.erased_concepts.append(concept_name)

        stats = {
            "concept": concept_name,
            "n_layers_edited": n_layers,
            "context_dim": self.context_dim,
            "elapsed_s": elapsed,
            "total_erased": len(self.erased_concepts),
        }

        print(
            f"[{len(self.erased_concepts):3d}] Erased '{concept_name}' | "
            f"layers={n_layers} | context_dim={self.context_dim} | "
            f"t={elapsed:.2f}s"
        )

        return stats

    # ── Generation ──────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images using the (possibly edited) SD3 pipeline."""
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )
        return output.images

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_layers = count_joint_attention_layers(self.pipe.transformer)
        return (
            f"SD3CURE("
            f"device={self.device}, "
            f"layers={n_layers}, "
            f"context_dim={self.context_dim}, "
            f"erased={len(self.erased_concepts)})"
        )
