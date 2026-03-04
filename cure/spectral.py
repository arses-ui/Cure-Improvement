"""
Spectral operations for CURE: SVD decomposition and spectral expansion.
"""

import torch
from typing import Tuple


def compute_svd(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD on embedding matrix.

    Args:
        embeddings: Tensor of shape [n_samples, hidden_dim]

    Returns:
        U: Left singular vectors [n_samples, k]
        S: Singular values [k]
        Vh: Right singular vectors [k, hidden_dim]
    """
    U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
    return U, S, Vh


def spectral_expansion(singular_values: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Tikhonov-inspired spectral expansion function (Equation 4 from paper).

    f(ri; α) = αri / ((α-1)ri + 1)

    where ri = σi² / Σj σj² is the normalized energy of each singular value.

    This function amplifies smaller singular values relative to larger ones,
    allowing the projector to capture more of the semantic subspace.

    Args:
        singular_values: Tensor of singular values [k]
        alpha: Expansion parameter (2 for objects/artists, 5 for NSFW)

    Returns:
        Diagonal scaling factors [k]
    """
    sigma_sq = singular_values ** 2
    total_energy = sigma_sq.sum()

    # Normalized energy distribution
    r = sigma_sq / (total_energy + 1e-10)

    # Spectral expansion function
    f_r = (alpha * r) / ((alpha - 1) * r + 1)

    return f_r


def build_projector(
    U: torch.Tensor,
    singular_values: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Build energy-scaled projection matrix (Equation 3 from paper).

    P = U @ Λ @ U.T

    where Λ = diag(f(ri; α)) contains the spectral expansion weights.

    Args:
        U: Left singular vectors [n_samples, k]
        singular_values: Singular values [k]
        alpha: Spectral expansion parameter

    Returns:
        Projection matrix [n_samples, n_samples]
    """
    # Get spectral expansion weights
    lambda_diag = spectral_expansion(singular_values, alpha)

    # Build projector: P = U @ diag(λ) @ U.T
    # Efficient computation: (U * λ) @ U.T
    scaled_U = U * lambda_diag.unsqueeze(0)
    P = scaled_U @ U.T

    return P


def compute_discriminative_projector(
    forget_embeddings: torch.Tensor,
    retain_embeddings: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Compute the discriminative projector that isolates forget-only subspace.

    Pdis = Pf - Pf @ Pr

    This projects onto the forget subspace, then removes any component
    that overlaps with the retain subspace.

    Args:
        forget_embeddings: Embeddings for concepts to forget [n_forget, hidden_dim]
        retain_embeddings: Embeddings for concepts to retain [n_retain, hidden_dim]
        alpha: Spectral expansion parameter

    Returns:
        Discriminative projector [hidden_dim, hidden_dim]
    """
    # SVD on forget embeddings
    Uf, Sf, Vhf = compute_svd(forget_embeddings)

    # Build forget projector in embedding space
    Pf = build_projector(Vhf.T, Sf, alpha)

    if retain_embeddings is not None and retain_embeddings.shape[0] > 0:
        # SVD on retain embeddings
        Ur, Sr, Vhr = compute_svd(retain_embeddings)

        # Build retain projector in embedding space
        Pr = build_projector(Vhr.T, Sr, alpha)

        # Discriminative projector: remove retain component from forget
        Pdis = Pf - Pf @ Pr
    else:
        Pdis = Pf

    return Pdis
