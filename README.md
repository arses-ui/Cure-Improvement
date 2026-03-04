# CURE Improvements

Extensions of [CURE: Concept Unlearning via Orthogonal Representation Editing](https://arxiv.org/abs/2505.12677) (Biswas, Roy, Roy — Purdue University).

This repo contains the original CURE implementation and two extensions that address its key limitations.

---

## Repository Structure

```
├── cure/           # Original CURE implementation (SD v1.4)
├── cure_seq/       # CURE-Sequential: interference-free sequential erasure
├── cure_dit/       # CURE-DiT: concept erasure for SD3 (MM-DiT)
```

### `cure/` — Original Implementation

The base CURE method: training-free concept erasure for Stable Diffusion v1.4 via SVD of CLIP text embeddings and closed-form weight edits to cross-attention Wk/Wv. Erases a concept in ~2 seconds.

### `cure_seq/` — CURE-Sequential

**Problem:** Applying CURE sequentially to erase multiple concepts causes cross-term interference — earlier erasures degrade after ~50 concepts (Figure 6).

**Solution:** Orthogonalize each new concept's projector against all previously erased subspaces, guaranteeing `Pi @ Pj = 0` for all `i ≠ j`. Sequential edits compose cleanly with zero interference.

See [`cure_seq/README.md`](cure_seq/README.md) for details.

### `cure_dit/` — CURE-DiT

**Problem:** CURE only works on SD v1.4's UNet architecture. SD3 and Flux use MM-DiT (Multi-Modal Diffusion Transformer) with joint attention — no cross-attention layers to target.

**Solution:** Identify the analogous text-stream projections (`add_k_proj`/`add_v_proj`) in SD3's `JointTransformerBlock` and apply CURE's spectral projection in the 1152-dim context space.

See [`cure_dit/README.md`](cure_dit/README.md) for details.

---

## Paper Summary

See [`CURE_PAPER_SUMMARY.md`](CURE_PAPER_SUMMARY.md) for a detailed breakdown of the original CURE paper including the algorithm, equations, experimental results, and limitations.
