# CLAUDE.md — CURE-Sequential

## Status

**Implementation complete and running.** All core components are built and tested on MPS.

## What's Implemented

- `cure_seq/subspace_bank.py` — `SubspaceBank`: tracks cumulative orthonormal basis `B [m, 768]`
- `cure_seq/spectral.py` — `compute_discriminative_projector_orth()`: SVD → orthogonalize → build projector
- `cure_seq/sequential_eraser.py` — `SequentialCURE`: wraps the pipeline, calls bank, applies weight edit
- `experiments/metrics.py` — `Sequential Interference Score (SIS)`, `budget_analysis()`, `print_budget_report()`
- `experiments/baseline_naive.py` — naive CURE sequential vs CURE-Sequential comparison experiment
- `demo_sequential.py` — end-to-end demo

## Reference Codebase

Original CURE lives at `/Users/arses/Desktop/cure/`. We import from it at runtime via `sys.path.insert`:
- `cure/attention.py` — `get_cross_attention_layers()`, `apply_weight_update()` — used as-is
- `cure/utils.py` — `set_seed()`, `save_images()`, `get_default_forget_prompts()` — used as-is

## Core Math

```
After erasing C1:  W1 = W0 - W0 @ P1
After erasing C2:  W2 = W0 - W0@P1 - W0@P2 + W0@(P1@P2)   ← interference
Our fix:           Pi_orth @ Pj_orth = 0  ∀ i≠j
Result:            W_n = W0 @ (I - P1_orth - ... - Pn_orth)  ← clean
```

## Known MPS Gotchas (already fixed)

- `torch.linalg.svd` — not on MPS, falls back to CPU automatically (just a warning, harmless)
- `torch.linalg.qr` — **NOT on MPS**, does NOT auto-fallback. Fixed by explicitly doing QR on CPU: `Q, _ = torch.linalg.qr(Vhf_orth.T.cpu()); Vhf_orth = Q.T.to(dev)`
- `SubspaceBank.basis` is stored on CPU. In `orthogonalize()`, move `B` to `Vhf.device` before matmul: `B = self.basis.to(device=Vhf.device, dtype=Vhf.dtype)`

## Observed Budget (Real CLIP Embeddings)

With real CLIP embeddings (not synthetic), alpha=2.0, lambda_threshold=0.01:
- ~20-40 dims per concept (vs ~5 with synthetic low-rank embeddings)
- Adaptive alpha kicks in when semantically related concepts overlap (e.g., cat after dog)
- Projected capacity: ~20-40 concepts before budget tightens

This is less than originally estimated. The lambda_threshold=0.01 is conservative — tuning it upward reduces dims-per-concept but may weaken erasure for low-energy directions.

## Key Parameters

| Param | Default | Effect |
|---|---|---|
| `alpha` | 2.0 | Spectral sharpness. Higher = stronger erasure, more aggressive |
| `lambda_threshold` | 0.01 | Min spectral weight to register in bank. Higher = fewer dims consumed per concept |
| `orth_threshold` | 1e-3 | Min row norm after orthogonalization before dropping. Rarely needs changing |
| `adaptive_alpha` | True | Boosts alpha when energy loss > 0. Keeps erasure strong despite overlap |

## Next Experiments

1. **Reproduce Figure 6**: run `experiments/baseline_naive.py --n-concepts 50` and compare SIS curves
2. **Budget calibration**: sweep `lambda_threshold` (0.001, 0.01, 0.05, 0.1) and measure dims/concept vs erasure quality tradeoff
3. **Order invariance test**: erase [car, dog] vs [dog, car] and verify identical final weights
4. **Stress test**: 50+ concepts sequential on CUDA

## Session Resumption

1. Read this file
2. Read `README.md` for the math overview
3. Read `CURE_PAPER_SUMMARY.md` for paper context
4. Run `demo_sequential.py --n-concepts 5 --device mps` to verify everything still works
5. Key files to check before editing: `subspace_bank.py` (device handling), `spectral.py` (returns 4-tuple now)
