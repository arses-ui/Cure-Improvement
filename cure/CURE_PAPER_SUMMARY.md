# CURE Paper Summary

> **CURE: Concept Unlearning via Orthogonal Representation Editing**
> Shristi Das Biswas\*, Arani Roy\*, Kaushik Roy — Purdue University
> arXiv:2505.12677 | CVPR-adjacent, May 2025

---

## Overview

CURE is a **training-free** method for erasing specific concepts from text-to-image diffusion models (Stable Diffusion v1.4). Instead of fine-tuning, it performs a closed-form edit to the cross-attention weight matrices using SVD of CLIP text embeddings. Erasure takes ~2 seconds and modifies only 2.23% of model parameters.

**Core idea:** A concept's semantic footprint in the model is captured by the subspace spanned by the CLIP embeddings of prompts describing that concept. Projecting the cross-attention key/value weights away from this subspace removes the concept's influence on generation.

---

## Problem Statement

Given a pretrained text-to-image model, erase a target concept (e.g., "Van Gogh style", "nudity", "cassette player") such that:
1. The model can no longer generate the concept from any prompt
2. Generation of unrelated concepts is unaffected
3. No retraining or fine-tuning is required

---

## The Algorithm

### Step 1 — Collect Embeddings

Tokenize and encode a set of forget prompts `{pf}` and optionally retain prompts `{pr}` through CLIP's text encoder:

```
F = TextEncoder({pf})  →  [n * 77, 768]    # forget embeddings
R = TextEncoder({pr})  →  [m * 77, 768]    # retain embeddings (optional)
```

Each prompt produces 77 token embeddings of dimension 768 (CLIP ViT-L/14). The matrix is formed by stacking all tokens from all prompts.

### Step 2 — SVD

Compute the singular value decomposition of the forget embedding matrix:

```
F = U @ diag(S) @ Vh
```

The **right singular vectors** `Vh` (rows are basis vectors of the forget subspace) are what matter — they span the concept's representation in embedding space.

### Step 3 — Spectral Expansion (Equation 4)

Raw singular values weight directions by their variance, but CURE uses a nonlinear spectral expansion to sharpen the erasure signal:

```
ri = σi² / Σ σj²          (normalized energy per direction)

f(ri; α) = α·ri / ((α-1)·ri + 1)
```

This is a Tikhonov-inspired function that:
- Amplifies high-energy directions (those most characteristic of the concept)
- Suppresses low-energy directions (noise / incidental correlations)
- α controls sharpness: α=1 → identity, α→∞ → hard projection onto top singular vector

**Default values:** α=2 for objects/artists, α=5 for NSFW content.

### Step 4 — Build the Forget Projector (Equation 3)

```
Pf = Vh.T @ diag(f(ri; α)) @ Vh     [768 × 768]
```

`Pf` is an energy-weighted projection onto the concept subspace.

### Step 5 — Discriminative Projector

If retain prompts are provided, subtract their projection to protect unrelated concepts:

```
Pr = Vhr.T @ diag(f(ri; α)) @ Vhr   [768 × 768]

Pdis = Pf - Pf @ Pr
```

This removes directions shared with the retain set, making the edit concept-specific rather than a blanket erasure of that region of embedding space.

### Step 6 — Weight Edit (Equation 2)

Apply to every cross-attention key and value projection matrix in the UNet:

```
Wk_new = Wk - Wk @ Pdis
Wv_new = Wv - Wv @ Pdis
```

Intuition: `Wk @ Pdis` extracts the component of the key projection that responds to the concept subspace. Subtracting it makes the attention keys blind to concept-related queries.

Only `Wk` and `Wv` are modified — not `Wq`, not self-attention, not feed-forward layers. This is 16 layers × 2 matrices = 32 weight matrices total, or ~2.23% of the model.

---

## Why Cross-Attention?

In Stable Diffusion, the UNet conditions on text via cross-attention:
- **Queries (Q):** come from the image (spatial features)
- **Keys (K) and Values (V):** come from text embeddings

The attention score `softmax(Q @ K.T / sqrt(d))` determines how much each image region attends to each text token. By modifying `Wk` and `Wv`, CURE breaks the mapping from concept text tokens to spatial features — the image can no longer be "steered" toward the concept by the text.

---

## Key Design Choices

| Choice | Alternative | Why CURE's choice |
|---|---|---|
| Edit Wk and Wv only | Edit all attention matrices | Wq edits affect query-side (image features), riskier for quality |
| Closed-form weight edit | Fine-tuning | No training, ~2s, no data required |
| SVD on CLIP embeddings | SVD on generated images | Text embeddings are deterministic and cheap to compute |
| Spectral expansion (nonlinear) | Hard top-k projection | Smooth weighting preserves model stability |
| Retain prompts | Concept-only erasure | Prevents collateral damage to related concepts |

---

## Experimental Results

### Table 4 — Imagenette (10 classes)

CURE achieves concept-level erasure comparable to or better than fine-tuning-based methods, while maintaining retained class accuracy and FID:

| Method | Erased Acc ↓ | Retained Acc ↑ | FID ↓ |
|---|---|---|---|
| ESD | ~5% | ~72% | ~18 |
| FMN | ~8% | ~75% | ~16 |
| UCE | ~6% | ~78% | ~15 |
| **CURE** | **~4%** | **~80%** | **~14** |

CURE achieves the lowest erased accuracy (best erasure) while preserving the highest retained accuracy (least collateral damage).

### Figure 5 — Artist Erasure

Qualitative results on erasing 5 artists (Van Gogh, Monet, Picasso, Rembrandt, Warhol). CURE produces images that lose the target style while maintaining image quality for non-target styles. ESD and FMN show more collateral damage.

### Figure 6 — Sequential Erasure Degradation

**This is the main limitation CURE-Sequential addresses.** When CURE is applied sequentially to erase N concepts, LPIPS between erased and non-erased outputs increases with N. After ~50 concepts, the erasures begin to interfere with each other — earlier erasures partially degrade as later ones are applied.

The paper attributes this to cross-term interference but does not propose a fix.

---

## Limitations (per the paper)

1. **Sequential degradation** — cross-term interference after ~50 erasures (Figure 6)
2. **CLIP-space boundary** — cannot erase concepts that CLIP's text encoder doesn't distinguish (e.g., subtle visual textures without good textual descriptions)
3. **Inversion attacks** — a determined adversary can potentially reconstruct the concept via textual inversion or prompt optimization on the edited model
4. **No quality metric for erasure completeness** — the paper uses classifier accuracy as a proxy but this is imperfect

---

## Technical Details

### Model
- Stable Diffusion v1.4 (CompVis)
- UNet: 16 cross-attention layers, each with Wk and Wv of shape [768, 768]
- CLIP: ViT-L/14, produces 768-dim embeddings, 77 tokens per prompt

### Compute
- Erasure time: ~2 seconds on GPU (dominant cost: text encoder forward pass + SVD)
- Memory: no additional memory beyond the model itself
- Parameters modified: 32 matrices × 768 × 768 = ~19M parameters out of ~860M total

### Hyperparameters
- α (spectral expansion): 2.0 for standard concepts, 5.0 for NSFW
- Number of forget prompts: typically 3-10 (e.g., "a painting by Van Gogh", "Van Gogh style", "Vincent van Gogh artwork")
- Number of retain prompts: optional, typically 5-20 describing adjacent concepts to preserve

---

## Related Work

| Paper | Method | Training-free? | Multi-concept? |
|---|---|---|---|
| ESD (arXiv:2303.07345) | Fine-tune to erase | No | One at a time |
| FMN (arXiv:2303.09625) | Fine-tune to forget | No | One at a time |
| UCE (arXiv:2308.14463) | Closed-form, multi-concept | Yes | Yes (batch) |
| MACE (arXiv:2403.06135) | LoRA + orthogonalization | No | Yes |
| ScaPre (arXiv:2601.06162) | Spectral trace regularizer | Yes | Yes |
| **CURE** | SVD + spectral projection | **Yes** | One at a time (sequentially degrades) |

---

## Key Equations (Reference)

```
# Spectral expansion (Eq. 4)
f(ri; α) = α·ri / ((α-1)·ri + 1)    where ri = σi² / Σ σj²

# Forget projector (Eq. 3)
Pf = Vh_f.T @ diag(f(ri; α)) @ Vh_f

# Discriminative projector
Pdis = Pf - Pf @ Pr

# Weight edit (Eq. 2)
W_new = W - W @ Pdis
```

---

## What CURE-Sequential Adds

CURE-Sequential (this repo) extends CURE with **orthogonal projector composition**:

Before computing `Pdis_n` for concept `n`, the forget subspace basis `Vh_f` is orthogonalized against the cumulative basis of all previously erased subspaces (stored in `SubspaceBank`). This guarantees `Pi @ Pj = 0` for all `i ≠ j`, making sequential edits compose with zero interference:

```
W_n = W0 @ (I - P1_orth - P2_orth - ... - Pn_orth)
```

See `README.md` for implementation details.
