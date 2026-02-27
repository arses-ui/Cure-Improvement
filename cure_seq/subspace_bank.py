"""
SubspaceBank: Tracks the cumulative orthonormal basis of all erased concept subspaces.

This is the core new data structure for CURE-Sequential. It allows each new concept's
discriminative projector to be orthogonalized against all previously erased subspaces,
guaranteeing zero cross-term interference in sequential unlearning.

Math:
    If P1_orth, P2_orth, ... are projectors built from mutually orthogonal subspaces,
    then Pi @ Pj = 0 for i != j, so sequential edits compose with zero interference:
        W_n = W0 @ (I - P1_orth - P2_orth - ... - Pn_orth)
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ConceptEntry:
    name: str
    n_dims: int           # How many dimensions this concept consumed
    energy_retained: float  # Fraction of original energy that survived orthogonalization
    bank_start_idx: int   # Where in the bank this concept's dims begin


class SubspaceBank:
    """
    Maintains a growing orthonormal basis of all erased concept subspaces.

    The bank stores a matrix B of shape [m, hidden_dim] where:
    - m = total dimensions consumed across all erased concepts
    - hidden_dim = CLIP embedding dimension (768 for ViT-L/14)
    - B has orthonormal rows: B @ B.T = I_m

    When orthogonalizing a new concept's Vhf [k, hidden_dim]:
        Vhf_orth = Vhf @ (I - B.T @ B)   (project onto complement of span(B))
    Then QR-renormalize Vhf_orth and add to B.
    """

    def __init__(self, hidden_dim: int = 768, orth_threshold: float = 1e-3):
        """
        Args:
            hidden_dim: Dimension of the embedding space (768 for CLIP ViT-L/14)
            orth_threshold: Rows with norm below this after orthogonalization are dropped
        """
        self.hidden_dim = hidden_dim
        self.orth_threshold = orth_threshold
        self.basis: Optional[torch.Tensor] = None   # [m, hidden_dim] orthonormal
        self.concepts: list[ConceptEntry] = []

    def orthogonalize(
        self,
        Vhf: torch.Tensor,
        forget_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project Vhf onto the orthogonal complement of the current bank,
        re-orthonormalize, and compute effective singular values.

        Args:
            Vhf: Right singular vectors from SVD [k, hidden_dim]
            forget_embeddings: Original forget embeddings [n_samples, hidden_dim]
                               Used to recompute effective singular values post-orthogonalization

        Returns:
            Vhf_orth: Orthogonalized and renormalized basis [k', hidden_dim], k' <= k
            S_eff:    Effective singular values [k'] measuring remaining concept energy
        """
        if self.basis is None or self.basis.shape[0] == 0:
            # Nothing to orthogonalize against — use Vhf as-is
            E_proj = forget_embeddings @ Vhf.T          # [n, k]
            S_eff = E_proj.pow(2).sum(0).sqrt()         # [k]
            return Vhf, S_eff

        # Move bank to same device/dtype as Vhf (bank is stored on CPU)
        B = self.basis.to(device=Vhf.device, dtype=Vhf.dtype)  # [m, hidden_dim]

        # Project out the bank: Vhf_orth = Vhf @ (I - B.T @ B)
        # Efficient: overlap = Vhf @ B.T, then subtract overlap @ B
        overlap = Vhf @ B.T          # [k, m]
        Vhf_orth = Vhf - overlap @ B # [k, hidden_dim]

        # Drop rows whose norm is below threshold (fully consumed by bank)
        norms = torch.norm(Vhf_orth, dim=1)  # [k]
        valid_mask = norms > self.orth_threshold
        Vhf_orth = Vhf_orth[valid_mask]      # [k', hidden_dim]

        if Vhf_orth.shape[0] == 0:
            return Vhf_orth, torch.zeros(0, device=Vhf.device)

        # Re-orthonormalize via QR (gives orthonormal column basis → transpose for rows)
        # QR is not implemented on MPS; fall back to CPU explicitly and move back
        dev = Vhf_orth.device
        Q, _ = torch.linalg.qr(Vhf_orth.T.cpu())  # Q: [hidden_dim, k'], orthonormal columns
        Vhf_orth = Q.T.to(dev)                      # [k', hidden_dim], orthonormal rows

        # Recompute effective singular values: energy of each orthogonalized direction
        # in terms of the original concept's embedding variance
        E_proj = forget_embeddings @ Vhf_orth.T  # [n, k']
        S_eff = E_proj.pow(2).sum(0).sqrt()       # [k']

        return Vhf_orth, S_eff

    def add_concept(
        self,
        concept_name: str,
        Vhf_orth: torch.Tensor,
        energy_retained: float,
        lambda_diag: Optional[torch.Tensor] = None,
        lambda_threshold: float = 0.01,
    ) -> None:
        """
        Register an orthogonalized concept subspace into the bank.

        Only registers directions with significant spectral expansion weight.
        Without filtering, a single concept with 20 prompts would consume all 768
        dims (since 20*77=1540 samples span the full 768-dim CLIP space). In
        practice, only ~5-20 directions have non-negligible spectral weight.

        Args:
            concept_name: Human-readable name for tracking
            Vhf_orth: Orthogonalized basis [k', hidden_dim] (from self.orthogonalize)
            energy_retained: Fraction of original concept energy that survived
            lambda_diag: Spectral expansion weights [k']. If provided, only
                         directions with weight > lambda_threshold are registered.
            lambda_threshold: Minimum spectral weight to register (default: 0.01)
        """
        # Filter to only significant directions
        if lambda_diag is not None and Vhf_orth.shape[0] > 0:
            significant = lambda_diag > lambda_threshold
            Vhf_orth = Vhf_orth[significant]

        if Vhf_orth.shape[0] == 0:
            self.concepts.append(ConceptEntry(
                name=concept_name,
                n_dims=0,
                energy_retained=energy_retained,
                bank_start_idx=self.dims_used,
            ))
            return

        start_idx = self.dims_used
        Vhf_cpu = Vhf_orth.cpu().float()

        if self.basis is None:
            self.basis = Vhf_cpu
        else:
            self.basis = torch.cat([self.basis, Vhf_cpu], dim=0)

        self.concepts.append(ConceptEntry(
            name=concept_name,
            n_dims=Vhf_orth.shape[0],
            energy_retained=energy_retained,
            bank_start_idx=start_idx,
        ))

    @property
    def dims_used(self) -> int:
        return self.basis.shape[0] if self.basis is not None else 0

    @property
    def remaining_budget(self) -> int:
        """How many orthogonal dimensions remain in the embedding space."""
        return self.hidden_dim - self.dims_used

    @property
    def budget_fraction_used(self) -> float:
        return self.dims_used / self.hidden_dim

    def compute_energy_retained(
        self,
        Vhf_original: torch.Tensor,
        Vhf_orth: torch.Tensor,
        forget_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute what fraction of the concept's variance survived orthogonalization.

        Args:
            Vhf_original: Pre-orthogonalization basis [k, hidden_dim]
            Vhf_orth: Post-orthogonalization basis [k', hidden_dim]
            forget_embeddings: Forget embeddings [n, hidden_dim]

        Returns:
            Fraction in [0, 1]. 1.0 = no energy lost, 0.0 = fully consumed by bank.
        """
        energy_before = (forget_embeddings @ Vhf_original.T).pow(2).sum().item()
        if Vhf_orth.shape[0] == 0:
            return 0.0
        energy_after = (forget_embeddings @ Vhf_orth.T).pow(2).sum().item()
        return energy_after / (energy_before + 1e-10)

    def summary(self) -> str:
        lines = [
            f"SubspaceBank: {self.dims_used}/{self.hidden_dim} dims used "
            f"({self.budget_fraction_used:.1%} of budget)",
            f"  Concepts erased: {len(self.concepts)}",
        ]
        for c in self.concepts[-5:]:  # show last 5
            lines.append(
                f"  [{c.name}] dims={c.n_dims}, energy_retained={c.energy_retained:.2%}"
            )
        if len(self.concepts) > 5:
            lines.insert(2, f"  ... ({len(self.concepts) - 5} earlier) ...")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SubspaceBank(hidden_dim={self.hidden_dim}, "
            f"dims_used={self.dims_used}, "
            f"n_concepts={len(self.concepts)})"
        )
