"""
SD3 attention layer utilities for CURE-DiT.

SD3 uses MM-DiT (Multi-Modal Diffusion Transformer) with JointTransformerBlocks.
Each block has joint text+image attention where:
  - Image stream: attn.to_q, attn.to_k, attn.to_v  (leave alone)
  - Text stream:  attn.add_q_proj, attn.add_k_proj, attn.add_v_proj  (CURE targets)

The text features enter each block as 1152-dim vectors (after context_embedder
projects T5's 4096-dim down to caption_projection_dim=1152).

Weight shapes:
  add_k_proj.weight: [inner_kv_dim, 1152]  (output × input)
  add_v_proj.weight: [inner_kv_dim, 1152]

CURE edit:
  W_new = W - W @ Pdis   where Pdis is [1152, 1152]
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def get_joint_attention_layers(transformer) -> List:
    """
    Get all JointTransformerBlock attention modules from an SD3 transformer.

    Args:
        transformer: SD3Transformer2DModel instance

    Returns:
        List of attention modules (each has add_k_proj, add_v_proj)
    """
    layers = []
    for block in transformer.transformer_blocks:
        attn = block.attn
        # Verify this block has text-stream projections
        if hasattr(attn, "add_k_proj") and attn.add_k_proj is not None:
            layers.append(attn)
    return layers


def ensure_unfused(transformer) -> None:
    """
    Ensure QKV projections are not fused (separate add_k_proj / add_v_proj).

    Diffusers can optionally fuse add_q/k/v_proj into add_qkv for speed.
    We need them separate to edit k and v independently.
    """
    if hasattr(transformer, "unfuse_qkv_projections"):
        transformer.unfuse_qkv_projections()


def apply_weight_update_sd3(
    attn_layer,
    Pdis: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Apply CURE weight edit to an SD3 attention layer's text-stream projections.

    Edits add_k_proj and add_v_proj:
        W_new = W - W @ Pdis

    Args:
        attn_layer: Attention module with add_k_proj, add_v_proj
        Pdis: Discriminative projector [context_dim, context_dim]
        device: Device for computation
    """
    Pdis_dev = Pdis.to(device=device)

    for proj_name in ["add_k_proj", "add_v_proj"]:
        proj = getattr(attn_layer, proj_name)
        if proj is None:
            continue

        W = proj.weight.data  # [out_dim, context_dim]
        dtype_orig = W.dtype

        # Compute in float32 for precision
        W_f32 = W.float()
        Pdis_f32 = Pdis_dev.float()

        W_new = W_f32 - W_f32 @ Pdis_f32  # [out_dim, context_dim]
        proj.weight.data = W_new.to(dtype=dtype_orig)


def count_joint_attention_layers(transformer) -> int:
    """Count how many JointTransformerBlocks have text-stream projections."""
    return len(get_joint_attention_layers(transformer))


def get_context_dim(transformer) -> int:
    """Get the context embedding dimension (caption_projection_dim, typically 1152)."""
    if hasattr(transformer, "context_embedder"):
        return transformer.context_embedder.out_features
    # Fallback: inspect first block's add_k_proj input features
    for block in transformer.transformer_blocks:
        if hasattr(block.attn, "add_k_proj") and block.attn.add_k_proj is not None:
            return block.attn.add_k_proj.in_features
    raise ValueError("Could not determine context dimension from transformer")
