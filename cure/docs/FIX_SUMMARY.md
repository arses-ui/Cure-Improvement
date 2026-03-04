# CURE Implementation - Fix Summary

## All Changes Made

### Fix 1: Embedding dimension bug (cure.py)
**Problem:** Original code had `embeddings = embeddings` (no-op). SVD received a
3D tensor [batch, 77, 768] and crashed.

**Fix:** Reshape to 2D: `[batch, 77, 768] → [batch*77, 768]` with `.float()` cast
for CUDA compatibility.

```python
batch_size, seq_len, hidden_dim = embeddings.shape
embeddings = embeddings.reshape(batch_size * seq_len, hidden_dim).float()
```

### Fix 2: Weight update transpose removed (attention.py)
**Problem:** Code used `Wk - Wk @ projector.T` but paper Eq. 7 says
`W_new = Wk @ Punlearn = Wk - Wk @ Pdis` (no transpose).

**Fix:** Removed `.T`:
```python
Wk_new = Wk - Wk @ projector
Wv_new = Wv - Wv @ projector
```

### Fix 3: float16 SVD crash on CUDA (cure.py)
**Problem:** `torch.linalg.svd` on CUDA doesn't support float16. Model loads in
float16 for GPU memory efficiency, causing `svd_cuda_gesvdj` error.

**Fix:** Added `.float()` cast after reshape.

### Fix 4: Model cache directory (demo.py, demo_paper_replica.py)
**Problem:** HuggingFace models download to `~/.cache/huggingface/` by default,
which may exceed user home directory quota on shared servers.

**Fix:** Added `--cache-dir` CLI argument defaulting to `./models/` so weights
stay in the project directory. Added `models/` to `.gitignore`.

### Fix 5: Expanded forget prompts (utils.py)
**Added:** More comprehensive synonym lists for concept removal:
- Imagenette objects (cassette player, chain saw, french horn, golf ball)
- Celebrity identities (20 prompts each: Taylor Swift, Elon Musk, Jennifer Lawrence, Emma Stone)
- Expanded existing concepts (car, dog, cat, person, nudity)

---

## Open Question: Reshape vs Mean Pooling

Current code uses reshape. Mean pooling may give better targeted erasure for some
concepts because it produces a low-rank projector. See `IMPLEMENTATION_ISSUES.md`.
