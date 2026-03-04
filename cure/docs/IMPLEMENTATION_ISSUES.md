# CURE Implementation Issues - Status Tracker

## Resolved Issues

### ✅ Bug #1: No-op embedding assignment
**File:** `cure/cure.py`
**Was:** `embeddings = embeddings` (did nothing, left 3D tensor)
**Fix:** Reshape to [batch*77, 768] with `.float()` for CUDA SVD compatibility.

### ✅ Bug #2: Transpose in weight update
**File:** `cure/attention.py:92-93`
**Was:** `Wk_new = Wk - Wk @ projector.T`
**Fix:** Removed `.T`. Paper Eq. 7 is `W_new = Wk @ (I - Pdis) = Wk - Wk @ Pdis` — no transpose.

### ✅ Bug #3: float16 SVD crash on CUDA
**File:** `cure/cure.py:97`
**Was:** SVD fails with `RuntimeError: "svd_cuda_gesvdj" not implemented for 'Half'`
**Fix:** Added `.float()` to cast embeddings to float32 before SVD.

### ✅ Verified: Projector uses correct singular vectors
**File:** `cure/spectral.py:111`
Code uses `Vhf.T` (right singular vectors). Paper uses `U` but with E stored as
[d, n] (column convention). These are equivalent — code is correct.

---

## Open Investigation: Embedding Aggregation Strategy

**File:** `cure/cure.py:92-97`
**Current:** Reshape [batch, 77, 768] → [batch*77, 768]

**Concern:** With 20 prompts, reshape gives [1540, 768] which produces a full-rank
(768) projector. This may dilute erasure by spreading it across all directions
instead of targeting concept-specific ones.

**Alternative:** Mean pooling gives [20, 768] → rank-20 projector targeting only
the 20 most concept-specific directions. This is more selective.

**Recommendation:** If erasure results are weak, switch to mean pooling:
```python
embeddings = embeddings.mean(dim=1).float()  # [batch, 768]
```

See `EMBEDDING_AGGREGATION_FIX.md` for detailed analysis.

---

## Testing Checklist

- [x] SVD runs without dimension errors
- [x] SVD runs on CUDA without float16 crash
- [x] Cross-attention weights are modified (before != after)
- [x] Demo scripts run end-to-end
- [ ] Concept erasure quality matches paper (0% accuracy on erased class)
- [ ] Unrelated concepts preserved (~79% accuracy on other classes)
- [ ] Performance: ~2 second erasure time
- [ ] Red-teaming robustness (adversarial prompts)
