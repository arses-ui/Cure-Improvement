"""
Spectral operations for CURE-DiT.

Dimension-agnostic reuse of CURE's core spectral functions.
Works for any hidden_dim (768 for SD1.4/CLIP, 1152 for SD3 context, 4096 for T5).
"""

import torch
from typing import Optional, Tuple


def compute_svd(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD on embedding matrix. Returns U [n,k], S [k], Vh [k,d]."""
    return torch.linalg.svd(embeddings, full_matrices=False)


def spectral_expansion(singular_values: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Tikhonov-inspired spectral expansion (CURE Equation 4).
    f(ri; α) = α*ri / ((α-1)*ri + 1)   where ri = σi² / Σ σj²
    """
    sigma_sq = singular_values ** 2
    total_energy = sigma_sq.sum()
    r = sigma_sq / (total_energy + 1e-10)
    return (alpha * r) / ((alpha - 1) * r + 1)


def build_projector(U: torch.Tensor, singular_values: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Build energy-scaled projection matrix (CURE Equation 3).
    P = U @ diag(f(ri;α)) @ U.T

    Args:
        U: Column basis [hidden_dim, k]  (i.e. Vh.T)
        singular_values: [k]
        alpha: Spectral expansion parameter
    Returns:
        Projector [hidden_dim, hidden_dim]
    """
    lambda_diag = spectral_expansion(singular_values, alpha)
    scaled_U = U * lambda_diag.unsqueeze(0)   # [hidden_dim, k]
    return scaled_U @ U.T                      # [hidden_dim, hidden_dim]


def compute_discriminative_projector(
    forget_embeddings: torch.Tensor,
    retain_embeddings: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """
    CURE discriminative projector. Pdis = Pf - Pf @ Pr.

    Args:
        forget_embeddings: [n, hidden_dim]
        retain_embeddings: [m, hidden_dim] or None
        alpha: Spectral expansion parameter
    Returns:
        Pdis [hidden_dim, hidden_dim]
    """
    _, Sf, Vhf = compute_svd(forget_embeddings)
    Pf = build_projector(Vhf.T, Sf, alpha)

    if retain_embeddings is not None and retain_embeddings.shape[0] > 0:
        _, Sr, Vhr = compute_svd(retain_embeddings)
        Pr = build_projector(Vhr.T, Sr, alpha)
        return Pf - Pf @ Pr

    return Pf
