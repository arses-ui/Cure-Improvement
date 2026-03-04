# CURE: Concept Unlearning via Orthogonal Representation Editing

Implementation of the **Spectral Eraser** algorithm from the CURE paper (Das Biswas, Roy, Kaushik Roy — Purdue University, arXiv:2505.12677v1).

Removes targeted concepts (e.g. "car", "Van Gogh", celebrity identities) from Stable Diffusion by editing cross-attention weights — no fine-tuning needed, runs in ~2 seconds.

## How it works

1. Encode forget/retain prompts into CLIP text embeddings
2. Compute a discriminative projector via SVD that isolates the target concept's subspace
3. Modify cross-attention Wk and Wv weights to project out that subspace

The result: prompts mentioning the erased concept produce alternative content, while unrelated concepts are preserved.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (match your CUDA version — check with nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Quick start

```bash
# Object removal
python demo_paper_replica.py --concept car --alpha 2.0

# Celebrity removal
python demo_paper_replica.py --concept "emma stone" --alpha 2.0

# Stronger erasure (e.g. NSFW)
python demo.py --concept nudity --alpha 5.0
```

Models download automatically to `./models/` on first run (~5GB). Use `--cache-dir` to change the location.

## Usage

```python
from diffusers import StableDiffusionPipeline
from cure import CURE

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    cache_dir="./models"
)
cure = CURE(pipe)
cure.erase_concept(["car", "automobile", "vehicle"], alpha=2.0)
image = cure.generate("a red car on the street")  # concept erased
```

## Alpha parameter

| Concept type | Recommended α |
|---|---|
| Objects / Artists | 2.0 |
| NSFW / stronger erasure | 5.0 |

Higher α flattens spectral weights, making erasure more aggressive.

## Output

Results saved to `outputs/<concept>/before/` and `outputs/<concept>/after/` for visual comparison. Before/after images use the same seed for structural consistency.

## Paper reference

```
CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models
Shristi Das Biswas*, Arani Roy*, Kaushik Roy
Purdue University
arXiv: 2505.12677v1
```
