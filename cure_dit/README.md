# CURE-DiT

**Training-free concept erasure for Stable Diffusion 3 and Flux.**

Ports [CURE](https://arxiv.org/abs/2505.12677) (Concept Unlearning via Orthogonal Representation Editing) from SD v1.4's UNet to MM-DiT (Multi-Modal Diffusion Transformer) architectures. Same closed-form spectral projection, new architecture target.

---

## Why This Is Needed

CURE was designed for Stable Diffusion v1.4, which uses a UNet with explicit cross-attention layers: text embeddings go through `Wk`/`Wv` projections, and CURE edits those projections to erase concepts.

SD3 and Flux use a fundamentally different architecture — **MM-DiT** — where there are no dedicated cross-attention layers. Instead, text and image features are processed through **joint attention blocks** where both modalities share a single attention operation. The text features enter through separate projections (`add_k_proj`, `add_v_proj`) but there's no `attn2` or explicit cross-attention module.

This means CURE's original code can't find any layers to edit. CURE-DiT identifies the correct edit targets in MM-DiT and applies the same spectral projection math.

## Architecture Mapping

| | SD v1.4 (CURE) | SD3 (CURE-DiT) |
|---|---|---|
| Architecture | UNet | MM-DiT (Transformer) |
| Text conditioning | Cross-attention | Joint attention |
| Edit targets | `attn2.to_k`, `attn2.to_v` | `attn.add_k_proj`, `attn.add_v_proj` |
| Text encoder | CLIP ViT-L/14 | T5-XXL + CLIP-L + CLIP-G |
| Embedding dim | 768 | 1152 (after context_embedder) |
| Num layers | 16 | 24 (SD3 medium) |

### Architecture Differences

**SD v1.4 (original CURE target):**
```
Cross-attention:
  k = Wk @ text_embedding      <- CURE targets
  v = Wv @ text_embedding      <- CURE targets
  q = Wq @ image_features
  output = softmax(q @ k.T / sqrt(d)) @ v
```

**SD3 (MM-DiT double-stream):**
```
JointTransformerBlock:
  k_txt = add_k_proj @ txt_tokens    <- CURE-DiT targets
  v_txt = add_v_proj @ txt_tokens    <- CURE-DiT targets

  k_img = to_k @ img_tokens         (untouched)
  v_img = to_v @ img_tokens         (untouched)

  joint_attn = softmax([q_txt, q_img] @ [k_txt, k_img].T) @ [v_txt, v_img]
```

The key insight: SD3's `context_embedder` projects T5's 4096-dim features down to 1152-dim before they enter the transformer blocks. The `add_k_proj`/`add_v_proj` in each `JointTransformerBlock` operate in this 1152-dim space — that's where we compute and apply CURE's discriminative projector.

## Setup

```bash
pip install -r requirements.txt

# Login to HuggingFace for gated model access
huggingface-cli login
```

SD3 models are gated — you need to accept the license on [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) first.

## Usage

```python
from diffusers import StableDiffusion3Pipeline
from cure_dit import SD3CURE
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
)

eraser = SD3CURE(pipe, device="cuda")

# Erase a concept
eraser.erase_concept(
    forget_prompts=["a photo of a car", "car", "automobile"],
    alpha=2.0,
    concept_name="car",
)

# Generate — the concept should be gone
images = eraser.generate("a photo of a car")
```

**Run the demo:**

```bash
python demo_sd3.py --concept "car" --device cuda
python demo_sd3.py --concept "Van Gogh" --alpha 2.0 --device cuda
python demo_sd3.py --concept "nudity" --alpha 5.0 --device cuda
```

## How It Works

Same math as CURE, different target:

1. **Extract T5 embeddings** for forget/retain prompts → project through `context_embedder` → `[n × seq_len, 1152]`
2. **SVD** on the forget embedding matrix → right singular vectors `Vh` span the concept subspace
3. **Spectral expansion** (Tikhonov-inspired) → energy-weighted projector `Pf`
4. **Discriminative projector** → `Pdis = Pf - Pf @ Pr` (subtract retain subspace)
5. **Weight edit** → for each of 24 JointTransformerBlocks:
   ```
   add_k_proj.weight = add_k_proj.weight - add_k_proj.weight @ Pdis
   add_v_proj.weight = add_v_proj.weight - add_v_proj.weight @ Pdis
   ```

## Project Structure

```
cure-dit/
├── cure_dit/
│   ├── __init__.py
│   ├── sd3_eraser.py         # SD3CURE class
│   ├── spectral.py           # SVD + spectral expansion (from CURE, dimension-agnostic)
│   └── attention_sd3.py      # SD3 layer extraction + weight update
├── experiments/
│   ├── __init__.py
│   └── metrics.py            # Concept accuracy, erasure report
├── demo_sd3.py
└── requirements.txt
```

## Hardware Requirements

| Model | VRAM (fp16) | Notes |
|---|---|---|
| SD3 Medium | ~12 GB | T5-XXL is ~9 GB alone |
| SD3.5 Medium | ~12 GB | |
| SD3.5 Large | ~18 GB | |

T5-XXL can be loaded in 8-bit to reduce memory. On Apple Silicon (MPS), models use shared memory.

## Evaluation Plan

Same metrics as the CURE paper:
- **Erasure**: Concept accuracy via ResNet-50 classifier on generated images (lower = better)
- **Retention**: Accuracy on non-erased classes (higher = better)
- **Quality**: FID on COCO-30k, CLIP score
- **Efficiency**: Time per erasure, % model modified

Comparison baselines:
- CURE on SD v1.4 (reference)
- ESD fine-tuned on SD3
- UCE adapted to SD3
- CURE-DiT (ours)

## Key Papers

- CURE (arXiv:2505.12677) — original method
- Knowledge Localization in DiTs (arXiv:2505.18832) — where knowledge lives in Flux/SANA/PixArt
- ConceptAttention (arXiv:2502.04320) — interpretability of MM-DiT features
- ScaPre (arXiv:2601.06162) — multi-concept spectral regularization
- SD3 paper (Stability AI, 2024)

## Reference

Built on top of:
> CURE: Concept Unlearning via Orthogonal Representation Editing
> Shristi Das Biswas, Arani Roy, Kaushik Roy — Purdue University
> arXiv:2505.12677
