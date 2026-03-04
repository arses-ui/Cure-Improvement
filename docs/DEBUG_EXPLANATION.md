# CURE Debugging Script - Step-by-Step Explanation

## Overview
The debug script traces through the entire concept unlearning pipeline to identify WHERE the issue is happening.

Think of it like a **medical diagnostic**:
- Instead of saying "the patient is sick", we check each vital sign
- Similarly, we check each stage of unlearning to find the problem

---

## Step-by-Step Breakdown

### Step 1: Load the Model
```python
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
cure = CURE(pipe, device=device)
```

**What it does:**
- Loads Stable Diffusion v1.4 pipeline
- Wraps it with CURE class

**What it checks:**
- ✓ Model loads without errors
- ✓ CURE initializes correctly

**Output:**
```
CURE(device=cuda, cross_attention_layers=648)
```

---

### Step 2: Get Forget Prompts
```python
forget_prompts = get_default_forget_prompts("car")
```

**What it does:**
- Gets list of prompts describing the concept to forget
- For "car": ["car", "automobile", "vehicle", "sedan", "coupe", ...]

**What it checks:**
- ✓ We have multiple prompts (10+ for robustness)
- ✓ Prompts cover the concept space

**Output:**
```
Forget prompts (10 total):
[0] car
[1] automobile
[2] vehicle
...
```

---

### Step 3: Get Text Embeddings ⭐ CRITICAL
```python
forget_embeddings = cure.get_text_embeddings(forget_prompts)
print(f"Shape: {forget_embeddings.shape}")
```

**What it does:**
- Tokenizes each prompt into 77 tokens
- Gets CLIP embeddings for each token
- Reshapes from [10, 77, 768] → [770, 768]

**What it checks:**
- ✓ Embedding shape is [n_prompts * 77, 768]
- ✓ Embeddings contain actual values (not zeros)
- ✓ Embeddings are normalized (reasonable min/max)

**Expected output:**
```
Shape: [770, 768]           ← 10 prompts × 77 tokens = 770 rows
Expected: [770, 768]        ← Should match!
Min: -2.5000, Max: 2.5000   ← Reasonable CLIP range
Mean: 0.0000, Std: 0.1000   ← Zero-centered
```

**If wrong here:**
- ❌ Shape mismatch → Embedding reshaping broken
- ❌ All zeros → Text encoder not working
- ❌ NaN values → Numerical problem

---

### Step 4: Compute Discriminative Projector ⭐ CRITICAL
```python
projector = cure.compute_spectral_eraser(
    forget_embeddings=forget_embeddings,
    alpha=2.0
)
```

**What it does:**
1. **SVD** on [770, 768] embeddings
   - U: [770, k] - sample vectors
   - S: [k] - singular values
   - V: [768, k] - basis vectors (these matter!)

2. **Spectral expansion** - apply the expansion function f(r; α)

3. **Build projector** P = V @ Λ @ V.T where Λ has expanded singular values

4. **Result:** Pdis = Pf - Pf @ Pr (forget minus shared)

**What it checks:**
- ✓ Projector shape is [768, 768] (embedding space)
- ✓ Projector has reasonable values (not all zeros/ones)
- ✓ Projector norm is non-zero

**Expected output:**
```
Projector shape: [768, 768]
Min: -0.5000, Max: 0.8000   ← Values in reasonable range
Norm: 45.3400               ← Has actual magnitude
```

**If wrong here:**
- ❌ All zeros → SVD failed
- ❌ Identity matrix → Projector not computed
- ❌ NaN → Numerical instability in SVD

**What the projector represents:**
- A [768, 768] matrix that "points in the direction of cars"
- When we do `Wk - Wk @ projector`, we're removing the "car direction"

---

### Step 5: Check Weights BEFORE Update
```python
Wk_before = sample_layer.clone()
print(f"Wk norm: {torch.norm(Wk_before):.4f}")
```

**What it does:**
- Saves a copy of the weight matrix before we change it
- Computes its norm (magnitude)

**What it checks:**
- ✓ Weights exist and have values
- ✓ We have a baseline to compare against

**Expected output:**
```
Wk shape: [768, 768]
Wk norm: 12.3456            ← Reasonable magnitude
```

---

### Step 6: Apply Concept Erasure
```python
cure.erase_concept(
    forget_prompts=forget_prompts,
    alpha=2.0
)
```

**What it does:**
- For each of 648 cross-attention layers in the UNet:
  - Get the Wk and Wv weights
  - Apply: `Wk_new = Wk - Wk @ projector`
  - Update the weights in-place

**What it should do:**
- Modify 648 layer pairs (Wk and Wv) = 1296 weight matrices modified

---

### Step 7: Check Weights AFTER Update ⭐ CRITICAL
```python
Wk_after = sample_layer_after
weight_diff = torch.norm(Wk_after - Wk_before)
print(f"Weight change: {weight_diff:.4f}")
print(f"Weights changed: {weight_diff > 1e-6}")
```

**What it does:**
- Compares weights before and after
- Computes the difference magnitude

**What it checks:**
- ✓ Weights actually changed (not a no-op)
- ✓ Change is substantial (> 1e-6 is threshold)

**Expected output:**
```
Weight change: 2.3456       ← Should be non-zero!
Weights changed: True       ← ✓ Good sign
```

**If wrong here:**
```
Weight change: 0.00000      ← ❌ PROBLEM!
Weights changed: False      ← Unlearning didn't happen!
```

**This is likely your issue!** If weights don't change, unlearning doesn't happen.

---

### Step 8: Test Generation
```python
images = cure.generate("a red car on the street")
```

**What it does:**
- Generates an image with the modified model
- This is the real test of whether unlearning worked

**What it checks:**
- ✓ Model can still generate (no crashes)
- ✓ Image actually reflects concept removal

---

### Step 9: Diagnostics Summary
```python
print(f"✓ Embeddings shaped correctly: {True}")
print(f"✓ Projector computed: {projector is not None}")
print(f"✓ Weights updated: {weight_diff > 1e-6}")
```

**Quick checklist of what worked:**
1. Do embeddings have correct shape?
2. Is projector computed?
3. Did weights actually change?

---

## What Each Output Tells You

### If Step 3 (Embeddings) is wrong:
```
❌ Shape: [10, 77, 768] (NOT reshaped!)
   Problem: Reshaping code not running
   Solution: Check get_text_embeddings implementation
```

### If Step 4 (Projector) is wrong:
```
❌ Projector shape: [768, 768]
   Min: 0.0000, Max: 0.0000    ← All zeros!
   Problem: SVD computed zero projector
   Solution: Check embeddings or spectral expansion
```

### If Step 7 (Weights) is wrong:
```
❌ Weight change: 0.00000
   Weights changed: False
   Problem: apply_weight_update not working
   Solution: Check weight update code
```

---

## Running the Debug Script

```bash
cd /Users/arses/Desktop/cure
python debug_unlearning.py 2>&1 | tee debug_output.txt
```

This will:
1. Run all debug steps
2. Save output to `debug_output.txt` for inspection

Then look at the output and tell me:
- ✓ Which step first shows an issue
- ✓ What the values are
- ✓ What you expected vs. what you got

---

## Common Issues & Solutions

| Issue | Symptom | Likely Cause | Fix |
|-------|---------|--------------|-----|
| Embeddings wrong | Shape [10, 77, 768] | Reshape not running | Check reshape code |
| Projector zero | All zeros in projector | SVD failed | Check embedding quality |
| Weights don't change | weight_diff = 0 | apply_weight_update broken | Check matrix multiplication |
| Still generating concept | Image shows car | Weights didn't change | Trace through steps 1-7 |

---

## Summary

The debug script is like a **circuit tester**:
- Step 1-2: ✓ Input valid?
- Step 3: ✓ Embeddings correct?
- Step 4: ✓ Projector computed?
- Step 5-7: ✓ Weights changed?
- Step 8: ✓ Output changed?

If Step 7 shows "Weights changed: False", **that's your problem** - the concept removal is never happening at all!
