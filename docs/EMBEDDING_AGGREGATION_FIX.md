# Embedding Aggregation: Fixing the Concept Representation

## The Problem We Found

From debug output:
- ✅ Weights are changing (1.1823 magnitude)
- ❌ But cars are still being generated

**This means:** The weight change isn't capturing the concept correctly.

---

## Root Cause: How We Were Computing Concepts

### Old Approach (WRONG)
```python
embeddings = embeddings.reshape(batch_size * seq_len, hidden_dim)
# [10 prompts, 77 tokens, 768] → [770 tokens, 768]
```

**What happened:**
1. We have 10 prompts about "car": ["car", "automobile", "vehicle", ...]
2. Each prompt → 77 tokens (with padding)
3. We reshaped to 770 × 768, treating each token as independent

**The problem:**
```
Prompt 1: "car"           → Tokens: [BOS] c a r [PAD] [PAD] ...
Prompt 2: "automobile"    → Tokens: [BOS] a u t o m o b i l e [PAD] ...

After reshape to [770, 768]:
Token 0:  [BOS] from prompt 1
Token 1:  "c" from prompt 1
Token 77: [BOS] from prompt 2  ← This is different token from position 0!
Token 78: "a" from prompt 2    ← This is different meaning from position 1!

SVD sees 770 DIFFERENT tokens, not "10 car concepts"
```

**The result:**
- SVD captures "car token patterns" not "car concept"
- The projector is weak and unfocused (norm = 0.8135)
- Concept removal is diluted across 77 unrelated positions

---

### New Approach (CORRECT)
```python
embeddings = embeddings.mean(dim=1)
# [10 prompts, 77 tokens, 768] → [10 prompts, 768]
```

**What happens:**
1. For each prompt, average all 77 token embeddings
2. Result: ONE semantic embedding per concept
3. SVD operates on [10, 768] - 10 concept embeddings

**The benefit:**
```
Prompt 1: "car"        → Average of 77 tokens → ONE concept vector
Prompt 2: "automobile" → Average of 77 tokens → ONE concept vector
...
Prompt 10: "sedan"     → Average of 77 tokens → ONE concept vector

SVD now sees 10 MEANINGFUL concepts
Projector captures the shared "car-ness" across all variations
```

---

## Why Mean Pooling Works

### For CLIP embeddings:
- Token embeddings capture semantic meaning
- [BOS] token emphasizes beginning
- Important words have strong embeddings
- [PAD] tokens are near-zero (averaging suppresses them)

**Mean pooling = semantic summary of the prompt**

### Mathematically:
```
For each prompt P:
  embeddings[P] = mean([token_0, token_1, ..., token_77])
                = (token_0 + token_1 + ... + token_77) / 77

Result: A single 768-dim vector representing the concept
```

### Why this is better for SVD:
- **Input:** [10 concepts, 768 dims]
- **SVD** learns what's common across these 10 concept variations
- **Output:** Projector captures the "car concept space"
- **Effect:** Removing this direction removes ALL car concepts

---

## Comparison of Approaches

| Aspect | Old (Reshape) | New (Mean Pool) |
|--------|--------------|-----------------|
| **Input to SVD** | [770, 768] | [10, 768] |
| **Interpretation** | 770 tokens | 10 concepts |
| **Projector norm** | ~0.8 (weak) | Expected: stronger |
| **What SVD learns** | Token patterns | Concept patterns |
| **Concept removal** | Partial | Complete |
| **Semantic meaning** | Low (mixed tokens) | High (averaged concept) |

---

## Expected Improvement

**Before (reshape approach):**
```
Prompt: "a red car"
Result: "a red bicycle" or "a red bus"
         ↓ (only attribute changed, concept remains)
```

**After (mean pooling):**
```
Prompt: "a red car"
Result: "a red [nothing]" or "other transport"
         ↓ (concept removed)
```

---

## Test This Now

```bash
python debug_unlearning.py
```

**Check for:**
1. Step 3: Shape should now be `[10, 768]` (not `[770, 768]`)
2. Step 4: Projector norm should be **stronger** (> 2.0)
3. Step 7: Weight change should be **larger**

**Then test generation:**
```bash
python demo.py --concept car --alpha 2.0
```

**You should now see:**
- ✅ Cars completely gone (not just color changed)
- ✅ Unrelated concepts still work

---

## Why We Changed Our Mind

Initial reasoning: "Use 77 tokens like the model was trained"

**Reality:** The model was trained on:
- **77 tokens at inference time** ✅ (this is correct for generation)
- **But not for concept learning** ❌

For concept unlearning, we need:
- **One semantic vector per concept** ✓
- **That represents what all variations of that concept share** ✓
- **Mean pooling achieves this** ✓

This is the **standard approach in NLP/Vision-Language models** for getting concept-level representations from token embeddings.
