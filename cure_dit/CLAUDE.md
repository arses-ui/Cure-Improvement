# CLAUDE.md — CURE-DiT Project

## Status

**Core implementation complete (SD3). Not yet tested — requires gated model access.**

## What's Implemented

- `cure_dit/spectral.py` — Dimension-agnostic CURE spectral functions (SVD, spectral expansion, discriminative projector)
- `cure_dit/attention_sd3.py` — SD3 JointTransformerBlock layer extraction, QKV unfusion handling, weight update
- `cure_dit/sd3_eraser.py` — `SD3CURE` class: text embedding extraction (T5 → context_embedder → 1152-dim), erasure, generation
- `experiments/metrics.py` — Concept accuracy, erasure report
- `demo_sd3.py` — End-to-end demo (before/after images)

## SD3 Architecture (Verified from diffusers 0.36.0 source)

```
SD3Transformer2DModel:
    context_embedder = nn.Linear(4096, 1152)   ← projects T5 features
    transformer_blocks = [JointTransformerBlock × 24]:
        attn.add_k_proj: nn.Linear(1152, inner_kv_dim)  ← CURE edit target
        attn.add_v_proj: nn.Linear(1152, inner_kv_dim)  ← CURE edit target
        attn.add_q_proj: nn.Linear(1152, inner_dim)     ← leave alone
        attn.to_k, to_v, to_q                           ← image stream, leave alone
```

- `inner_dim = num_attention_heads × attention_head_dim = 18 × 64 = 1152` (for SD3 medium)
- `caption_projection_dim = 1152`
- `joint_attention_dim = 4096` (T5-XXL output dim)

## Embedding Pipeline

```
T5-XXL text encoder:  prompts → [batch, seq_len, 4096]
context_embedder:     [batch, seq_len, 4096] → [batch, seq_len, 1152]
SVD on:               [batch * seq_len, 1152] → Pdis [1152, 1152]
Apply to:             add_k_proj.weight [1152, 1152] -= W @ Pdis
```

## Text Encoders in SD3

- `pipe.tokenizer_3` / `pipe.text_encoder_3` = T5-XXL (4096-dim, dominant for semantics)
- `pipe.tokenizer` / `pipe.text_encoder` = CLIP-L (768-dim)
- `pipe.tokenizer_2` / `pipe.text_encoder_2` = CLIP-G/OpenCLIP (1280-dim)

We use T5 only for computing Pdis. All three encoders are used during generation.

## QKV Fusion Gotcha

Diffusers can fuse `add_q/k/v_proj` into `add_qkv` for performance. We call
`transformer.unfuse_qkv_projections()` at init to ensure separate projections.

## MPS Gotchas (from CURE-Sequential experience)

- `torch.linalg.svd` falls back to CPU automatically on MPS (just a warning)
- `torch.linalg.qr` is NOT supported on MPS — explicit CPU fallback needed if used
- Device mismatches: always `.to(device)` before matmuls

## Untested / Open Questions

1. Does the approach actually work? Need model access to test
2. Is T5 alone sufficient, or do we need joint CLIP-L + CLIP-G + T5 embeddings?
3. What alpha values work for SD3? (SD1.4 uses α=2 for objects, α=5 for NSFW)
4. Does editing only text-stream projections preserve generation quality?
5. SD3's last JointTransformerBlock has `context_pre_only=True` (no text output) — does editing it matter?

## Model Access

SD3 models are gated. Need:
1. HuggingFace account
2. Accept license at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
3. `huggingface-cli login`

## Next Steps

1. Get model access and download SD3
2. Run `demo_sd3.py` — verify erasure works
3. If T5-only is insufficient, add CLIP-L/CLIP-G embedding support
4. Tune alpha for SD3
5. Add Flux support (double-stream blocks have same `add_k_proj`/`add_v_proj`)

## Session Resumption

1. Read this file
2. Read `README.md` for architecture mapping
3. Key: Pdis is [1152, 1152] operating on context_embedder output space
4. Check diffusers version — API may change between versions
