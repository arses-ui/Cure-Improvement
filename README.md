# CURE-Sequential

**Interference-free sequential concept unlearning for Stable Diffusion.**

An extension of [CURE](https://arxiv.org/abs/2505.12677) that fixes the sequential erasure degradation problem (Figure 6 of the paper) through orthogonal projector composition. Erase hundreds of concepts one at a time with zero cross-term interference between erasures.

---

## The Problem

CURE is a training-free method for erasing concepts from Stable Diffusion by editing cross-attention weights via SVD of CLIP text embeddings. It works great for a single concept. But apply it sequentially to erase concept B after concept A, and the math reveals a problem:

```
W1 = W0 - W0 @ P1              # Erase A
W2 = W1 - W1 @ P2              # Erase B
   = W0 - W0@P1 - W0@P2 + W0@(P1@P2)
                         ^^^^^^^^^^^
                         INTERFERENCE TERM
```

The cross-term `W0 @ P1 @ P2` is nonzero whenever the two concept subspaces overlap in CLIP's 768-dim embedding space — which they do for any semantically related concepts. The CURE paper's own Figure 6 shows this empirically: after ~50 sequential erasures, quality degrades.

## The Fix

Before computing the projector for concept `n`, orthogonalize its forget subspace against the cumulative basis of all previously erased subspaces (tracked in a `SubspaceBank`). This guarantees:

```
Pi @ Pj = 0   for all i ≠ j
```

So sequential edits compose exactly:

```
W_n = W0 @ (I - P1 - P2 - ... - Pn)
```

A single clean projection. No cross-terms. No interference. The total edit after `n` concepts is mathematically equivalent to a single batch erasure with the joint projector.

## Results (5 concepts, MPS)

```
[  1] Erased 'car'          | dims=21 | energy=100.0% | budget left=747/768 | t=0.27s
[  2] Erased 'dog'          | dims=27 | energy=100.0% | budget left=720/768 | t=0.23s
[  3] Erased 'cat'          | dims=24 | energy= 83.1% | budget left=696/768 | t=0.31s
[  4] Erased 'french horn'  | dims=38 | energy= 82.6% | budget left=658/768 | t=0.28s
[  5] Erased 'golf ball'    | dims=23 | energy= 80.6% | budget left=635/768 | t=0.10s

Budget consumed: 133/768 dims (17.3%) — ~0.25s per concept, no training
```

When concepts share semantic subspace (cat/dog), adaptive alpha compensation maintains erasure strength.

## Setup

This project depends on the original CURE codebase. Clone both into the same parent directory:

```bash
# Clone CURE (original)
git clone https://github.com/<your-org>/cure  # or your local copy

# Clone this repo
git clone https://github.com/<your-org>/cure-sequential

# Install dependencies
cd cure-sequential
pip install -r requirements.txt
```

The `cure_seq` package imports `attention.py` and `utils.py` directly from the sibling `cure/` directory at runtime.

## Usage

```python
from diffusers import StableDiffusionPipeline
from cure_seq import SequentialCURE

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
eraser = SequentialCURE(pipe)

# Erase concepts one by one — zero interference
eraser.erase_concept(["car", "automobile", "vehicle"], concept_name="car")
eraser.erase_concept(["dog", "puppy"], concept_name="dog")
eraser.erase_concept(["Van Gogh", "van gogh style"], concept_name="van_gogh")

# Check budget
print(eraser.bank.summary())
```

**Run the demo:**

```bash
python demo_sequential.py --n-concepts 5 --device mps
python demo_sequential.py --n-concepts 10 --device cuda
python demo_sequential.py --concepts "car,truck,bus,van" --device cpu
```

**Run the baseline comparison (naive CURE vs orthogonalized):**

```bash
python experiments/baseline_naive.py --n-concepts 20 --device cuda
```

## How It Works

### SubspaceBank

The core new data structure. Maintains a growing orthonormal basis `B` of shape `[m, 768]` covering all erased concept subspaces.

When orthogonalizing a new concept's singular vectors `Vhf`:
```
Vhf_orth = Vhf - (Vhf @ B.T) @ B    # project out the bank
```
Then QR-renormalize and append `Vhf_orth` to `B`.

### Lambda Filtering

A concept's SVD technically spans the full 768-dim space. We only register directions where the spectral expansion weight `f(ri; α) > 0.01` — in practice ~20-40 directions per concept. This keeps the budget realistic.

### Adaptive Alpha

When orthogonalization removes concept energy (due to overlap with prior concepts), alpha is boosted:
```python
alpha_effective = min(alpha / max(energy_retained, 0.1), 10.0)
```

## Project Structure

```
cure-sequential/
├── cure_seq/
│   ├── __init__.py
│   ├── sequential_eraser.py   # SequentialCURE class
│   ├── spectral.py            # Orthogonalization-aware projector computation
│   └── subspace_bank.py       # SubspaceBank — tracks cumulative erased subspace
├── experiments/
│   ├── metrics.py             # Sequential Interference Score (SIS) + budget analysis
│   └── baseline_naive.py      # Naive CURE vs CURE-Sequential comparison
├── demo_sequential.py
└── requirements.txt
```

## Metrics

**Sequential Interference Score (SIS):** After erasing concepts C1...Cn, measure how much C1's erasure has degraded.
```
SIS(n) = concept_accuracy(C1, model_after_n) - concept_accuracy(C1, model_after_1)
```
- Ideal: SIS = 0 (orthogonal method)
- Naive CURE: SIS > 0 after ~50 concepts

## Dimensionality Budget

CLIP's embedding space is 768-dimensional. Each concept consumes ~20-40 dims (after lambda filtering). Theoretical capacity: **~20-40 concepts** before the budget tightens, vs. CURE's empirical degradation at ~50.

The budget is finite but principled — you always know exactly how much space remains and can trade off erasure precision against capacity via `lambda_threshold`.

## Comparison Baselines

| Method | Sequential? | Interference-free? | Training-free? |
|---|---|---|---|
| CURE (naive sequential) | Yes | No (~50 concept limit) | Yes |
| CURE batch | No | Yes (by construction) | Yes |
| MACE | Yes | Partial | No |
| ScaPre (arXiv:2601.06162) | Yes | Partial | Yes |
| **CURE-Sequential (ours)** | **Yes** | **Yes** | **Yes** |

## Reference

Built on top of:
> CURE: Concept Unlearning via Orthogonal Representation Editing
> Shristi Das Biswas, Arani Roy, Kaushik Roy — Purdue University
> arXiv:2505.12677
