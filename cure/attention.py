"""
Cross-attention layer manipulation utilities for Stable Diffusion UNet.
"""

import torch
from torch import nn
from typing import List, Generator, Tuple


def get_cross_attention_layers(unet: nn.Module) -> Generator[nn.Module, None, None]:
    """
    Extract all cross-attention modules from Stable Diffusion UNet.

    Target layers in SD v1.4 UNet:
    - down_blocks[*].attentions[*].transformer_blocks[*].attn2
    - mid_block.attentions[*].transformer_blocks[*].attn2
    - up_blocks[*].attentions[*].transformer_blocks[*].attn2

    Args:
        unet: The UNet model from StableDiffusionPipeline

    Yields:
        Cross-attention modules (attn2 layers)
    """
    # Down blocks
    for down_block in unet.down_blocks:
        if hasattr(down_block, 'attentions'):
            for attention in down_block.attentions:
                for transformer_block in attention.transformer_blocks:
                    if hasattr(transformer_block, 'attn2'):
                        yield transformer_block.attn2

    # Mid block
    if hasattr(unet, 'mid_block') and hasattr(unet.mid_block, 'attentions'):
        for attention in unet.mid_block.attentions:
            for transformer_block in attention.transformer_blocks:
                if hasattr(transformer_block, 'attn2'):
                    yield transformer_block.attn2

    # Up blocks
    for up_block in unet.up_blocks:
        if hasattr(up_block, 'attentions'):
            for attention in up_block.attentions:
                for transformer_block in attention.transformer_blocks:
                    if hasattr(transformer_block, 'attn2'):
                        yield transformer_block.attn2


def get_projection_matrices(attn_layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the key (Wk) and value (Wv) projection matrices from a cross-attention layer.

    Args:
        attn_layer: A cross-attention module

    Returns:
        Tuple of (Wk, Wv) weight tensors
    """
    Wk = attn_layer.to_k.weight.data
    Wv = attn_layer.to_v.weight.data
    return Wk, Wv


def apply_weight_update(
    attn_layer: nn.Module,
    projector: torch.Tensor,
    device: torch.device = None
) -> None:
    """
    Apply the discriminative projector to modify Wk and Wv weights in-place.

    W_new = W_old - W_old @ Pdis.T

    This projects out the forget concept direction from the weight matrices.

    Args:
        attn_layer: A cross-attention module
        projector: The discriminative projector matrix [hidden_dim, hidden_dim]
        device: Target device for computation
    """
    if device is None:
        device = attn_layer.to_k.weight.device

    projector = projector.to(device=device, dtype=attn_layer.to_k.weight.dtype)

    # Get current weights
    Wk = attn_layer.to_k.weight.data
    Wv = attn_layer.to_v.weight.data

    # Apply update: W_new = W_old - W_old @ Pdis (Paper Eq. 7)
    # Wk shape: [out_features, in_features] where in_features = text_hidden_dim
    Wk_new = Wk - Wk @ projector
    Wv_new = Wv - Wv @ projector

    # Update weights in-place
    attn_layer.to_k.weight.data = Wk_new
    attn_layer.to_v.weight.data = Wv_new


def count_cross_attention_layers(unet: nn.Module) -> int:
    """
    Count the number of cross-attention layers in the UNet.

    Args:
        unet: The UNet model

    Returns:
        Number of cross-attention layers
    """
    return sum(1 for _ in get_cross_attention_layers(unet))
