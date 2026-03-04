# CURE Implementation - Paper vs Code Analysis

## Paper: "CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models"
**Authors:** Das Biswas, Roy, Kaushik Roy (Purdue University)
**arXiv:** 2505.12677v1

---

## Math Comparison

### ✅ SVD Decomposition (Paper Eq. 2)
**Paper:** `Ef = UfΣf V⊤f`, `Er = UrΣrV⊤r`

**Code:** (`spectral.py:9-22`)
```python
U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
```
✓ **MATCH** - Correctly computes SVD decomposition

---

### ✅ Spectral Expansion Function (Paper Eq. 4)
**Paper:**
```
f(ri; α) = αri / ((α-1)ri + 1)
where ri = σ²i / Σj σ²j
```

**Code:** (`spectral.py:25-52`)
```python
sigma_sq = singular_values ** 2
total_energy = sigma_sq.sum()
r = sigma_sq / (total_energy + 1e-10)
f_r = (alpha * r) / ((alpha - 1) * r + 1)
```
✓ **MATCH** - Exactly implements the expansion function with normalization

---

### ✅ Energy-Scaled Projectors (Paper Eq. 5)
**Paper:**
```
Pf = UfΛf U⊤f , Pr = UrΛr U⊤r
where Λf = diag(f(r(f)i; α))
```

**Code:** (`spectral.py:55-83`)
```python
lambda_diag = spectral_expansion(singular_values, alpha)
scaled_U = U * lambda_diag.unsqueeze(0)
P = scaled_U @ U.T
```
✓ **MATCH** - Implements P = U @ Λ @ U.T correctly

**Note on paper notation:** Paper uses U but stores E as [d, n] (columns are samples).
In code, embeddings are [n, d] (rows are samples), so the paper's U corresponds to
the code's Vh.T (right singular vectors transposed). This is mathematically equivalent.

---

### ✅ Discriminative Projector (Paper Eq. 6)
**Paper:**
```
Punlearn := I − Pdis
where Pdis = Pf − Pf @ Pr
```

**Code:** (`spectral.py:86-125`)
```python
Pf = build_projector(Vhf.T, Sf, alpha)
Pr = build_projector(Vhr.T, Sr, alpha)
Pdis = Pf - Pf @ Pr
```
✓ **MATCH** - Correctly computes the discriminative projector

---

### ✅ Weight Update (Paper Eq. 7) — FIXED
**Paper:**
```
W_new_k = Wk @ Punlearn = Wk @ (I - Pdis) = Wk - Wk @ Pdis
W_new_v = Wv @ Punlearn = Wv @ (I - Pdis) = Wv - Wv @ Pdis
```

**Code:** (`attention.py:92-93`)
```python
Wk_new = Wk - Wk @ projector
Wv_new = Wv - Wv @ projector
```
✓ **MATCH** - Transpose removed. Code now matches paper formula directly.

**History:** Original code had `projector.T` which was incorrect. The paper's Eq. 7
uses right-multiplication `Wk @ Punlearn`, which translates directly to
`Wk - Wk @ Pdis` without any transpose.

---

## Key Parameter Settings

| Concept Type | Paper α value | Code |
|--------------|---------------|------|
| Objects/Artists | α = 2 | ✓ Default in demo_paper_replica.py |
| NSFW Content | α = 5 | ✓ Default in demo.py |

---

## Implementation Completeness

| Component | Status | Location |
|-----------|--------|----------|
| SVD decomposition | ✅ | spectral.py:9-22 |
| Spectral expansion | ✅ | spectral.py:25-52 |
| Projector construction | ✅ | spectral.py:55-83 |
| Discriminative projector | ✅ | spectral.py:86-125 |
| Cross-attention extraction | ✅ | attention.py:10-46 |
| Weight projection extraction | ✅ | attention.py:49-61 |
| Weight updates | ✅ | attention.py:64-97 |
| Text embeddings | ✅ | cure.py:57-99 |
| Model cache dir support | ✅ | demo.py, demo_paper_replica.py |
| Utilities / prompts | ✅ | utils.py |

---

## Issues Found & Fixed

### ✅ Fixed: No-op embedding assignment
**File:** `cure/cure.py`
**Issue:** Original code had `embeddings = embeddings` (no-op), passing 3D tensor to SVD.
**Fix:** Added proper reshaping of embeddings from [batch, 77, 768] to [batch*77, 768].

### ✅ Fixed: Transpose in weight update
**File:** `cure/attention.py:92-93`
**Issue:** Code used `projector.T` but paper formula uses direct multiplication.
**Fix:** Removed `.T`. Now `Wk_new = Wk - Wk @ projector` matches paper Eq. 7.

### ✅ Fixed: float16 SVD crash on CUDA
**File:** `cure/cure.py:97`
**Issue:** `torch.linalg.svd` does not support float16 on CUDA (`svd_cuda_gesvdj` error).
**Fix:** Added `.float()` cast after reshaping embeddings.

### ✅ Verified: Projector space computation
**File:** `cure/spectral.py:111`
**Status:** Using right-singular vectors (Vh.T) is CORRECT for embedding space projectors.
Paper stores E as [d, n], so paper's U = code's Vh.T.

### ✅ Added: Model cache directory support
**Files:** `demo.py`, `demo_paper_replica.py`
**Issue:** Models downloaded to ~/.cache/huggingface/ which may have quota limits.
**Fix:** Added `--cache-dir` flag (defaults to `./models/`) so weights stay in project dir.

---

## Embedding Aggregation: Reshape vs Mean Pooling

**Current approach:** Reshape [batch, 77, 768] → [batch*77, 768]

**Trade-off:**
| | Reshape (current) | Mean Pooling |
|--|---|---|
| SVD input | [n*77, 768] | [n, 768] |
| Projector rank | Up to 768 (full) | Up to n (~20) |
| Concept selectivity | Lower (covers all directions) | Higher (only concept directions) |
| Data for SVD | More (better estimation) | Less (but more targeted) |

**Known issue:** With reshape, when n*77 > 768, SVD produces a full-rank projector
covering the entire 768-dim space. With high alpha, this can dilute erasure by
modifying all directions instead of targeting concept-specific ones. Mean pooling
produces a low-rank projector (rank = n_prompts) that only removes concept-specific
directions. If erasure is too weak, switching to mean pooling may help:
```python
embeddings = embeddings.mean(dim=1).float()  # [batch, 768]
```

---

## Running the Demos

```bash
# Object removal (alpha=2.0)
python demo_paper_replica.py --concept car --alpha 2.0 --cache-dir ./models

# Stronger erasure (alpha=5.0)
python demo.py --concept car --alpha 5.0 --cache-dir ./models

# Celebrity removal
python demo_paper_replica.py --concept "emma stone" --alpha 2.0 --cache-dir ./models
```

Output images saved to `outputs/<concept>/before/` and `outputs/<concept>/after/`.
