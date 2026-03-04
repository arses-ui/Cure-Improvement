# What Does "Regularization" Mean in CURE?

## Quick Answer
In CURE, **regularization is NOT about preventing overfitting** (like L1/L2 in typical ML).

Instead, it's about **controlling which singular values contribute to concept removal** - making the removal more or less aggressive depending on the α (alpha) parameter.

---

## The Problem It Solves

### Without Regularization (Naive Approach)
If we just use the first few singular values:
```
Pf = U1 @ diag(σ1, σ2, σ3) @ U1.T
```

**Issue:** Only the TOP singular values (largest ones) get used
- σ1 = 72.2 (dominant)
- σ2 = 20.9 (much smaller)
- σ3 = 18.5 (much smaller)
- ...
- σ10 = 9.4 (tiny)

**Result:** The projector is heavily biased toward capturing only the MOST dominant "car concept" direction. Misses other variations of "car" (sports cars, old cars, etc.) that are encoded in the smaller singular values.

---

## How Regularization Fixes It

### The Spectral Expansion Function (Tikhonov-Inspired)

```
f(ri; α) = αri / ((α-1)ri + 1)

where ri = σ²i / Σj σ²j   (normalized energy)
```

This function **amplifies smaller singular values relative to larger ones**.

### Visualization: Effect of Different α Values

```
Without regularization (α→0):
σ²: [72², 20.9², 18.5², 16.8², ...]
    [5184, 437, 342, 282, ...]
    Very skewed! Only first one matters

With α=1 (light regularization):
f(r): [0.73, 0.062, 0.048, 0.040, ...]
      Still skewed, but less extreme

With α=2 (moderate regularization):
f(r): [0.84, 0.116, 0.092, 0.077, ...]
      More balanced! Smaller values amplified ~2x

With α=5 (strong regularization):
f(r): [0.93, 0.247, 0.202, 0.172, ...]
      Much more balanced! All singular values contribute
```

---

## Why This Matters for Concept Unlearning

### Example: Removing "Car" Concept

**What are the singular values capturing?**
```
σ1 = 72.2  → The dominant "car-ness" (shared across all prompts)
           Examples: "shape of a car", "has wheels", "is vehicle"

σ2 = 20.9  → Car variations (old vs new)
           Examples: "vintage" vs "modern"

σ3 = 18.5  → Car types (sedan vs sports)
           Examples: "sports car" vs "family sedan"

σ4 = 16.8  → Size variations
           Examples: "large SUV" vs "compact car"

...and so on
```

### Without Regularization
- Only σ1 gets removed → Only removes "core car concept"
- Other singular values (σ2, σ3, etc.) still influence generation
- Result: Model still generates cars, just different types

### With Regularization (α=2)
- All singular values get weighted and removed
- Removes σ1 (core), σ2 (variations), σ3 (types), etc.
- Result: Removes all types of cars, not just the dominant type

---

## Mathematical Interpretation

### Connection to Tikhonov Regularization (Inverse Problems)

In inverse problems, Tikhonov regularization solves:
```
min ||Ax - b||² + λ||x||²
```

Where:
- First term: fit the data
- Second term: regularize (keep solution small)
- λ controls the trade-off

**In CURE:** The spectral expansion function has a similar structure:
```
f(ri; α) = αri / ((α-1)ri + 1)
```

This can be rewritten (approximately) as:
```
f(ri; α) ≈ σ²i / (σ²i + regularization_term)
```

Where the regularization term prevents σ²i from dominating.

### What's Being Regularized?
- NOT the loss function (like standard ML)
- Instead: The contribution of each singular value to the projector
- Purpose: Balance between capturing dominant and subtle concept variations

---

## Practical Effect: Controlling Erasure Strength

### α = 1.0 (Minimal Regularization)
```
Behavior: Uses singular values proportionally to their energy
Result: Removes mostly dominant concept direction
Effect: Partial erasure (some concept remains)
```

### α = 2.0 (Moderate Regularization - DEFAULT)
```
Behavior: Amplifies smaller singular values ~1.98x
Result: Removes dominant AND some variation directions
Effect: Good erasure for most objects/artists
Paper recommendation: Use for cars, dogs, artists, etc.
```

### α = 5.0 (Strong Regularization)
```
Behavior: Amplifies smaller singular values ~4.75x
Result: Removes all singular values equally
Effect: Aggressive erasure, removes concept completely
Paper recommendation: Use for NSFW/explicit content
```

### α → ∞ (Maximum Regularization)
```
Behavior: f(r) → constant (all r values equal)
Result: Uses all singular values with equal weight
Effect: Uniform removal across all directions
```

---

## Why Use Regularization Instead of Simple Truncation?

### Simple Approach (BAD)
```python
# Just keep top k singular values
P = U[:, :k] @ diag(S[:k]) @ U[:, :k].T
```

Problems:
- Arbitrary cutoff (how to choose k?)
- Wastes information in smaller singular values
- Abrupt/discontinuous

### Regularized Approach (GOOD)
```python
# Weight all singular values, amplify small ones
P = U @ diag(f(r; α)) @ U.T
```

Benefits:
- No arbitrary cutoff
- Uses ALL singular value information
- Smooth transition with α parameter
- Theoretically grounded (Tikhonov regularization)
- Can tune with single parameter α

---

## The Key Insight

**Regularization in CURE is about finding the right balance:**

```
Too little (α small):
  → Removes only dominant direction
  → Concept partially remains
  → Problem: Incomplete erasure

Just right (α=2 for objects):
  → Removes dominant + variation directions
  → Concept mostly removed
  → Problem: None

Too much (α very large):
  → Removes ALL directions equally
  → Concept completely gone
  → Problem: Might remove related concepts too
```

---

## Summary Table

| Aspect | Traditional ML Regularization | CURE Regularization |
|--------|------------------------------|-------------------|
| **What's regularized** | Loss function | Singular value weighting |
| **Purpose** | Prevent overfitting | Control erasure selectivity |
| **Parameter** | λ (lambda) | α (alpha) |
| **Effect** | Makes solution smoother | Amplifies small singular values |
| **Controls** | Generalization | Concept removal completeness |
| **Inspired by** | Ridge/Lasso regression | Tikhonov inverse problems |

---

## The α Parameter in Your Case

```
Your setup: Erasing "car" with α=2.0

Forget prompts (10 total):
  "car", "automobile", "vehicle", "sedan",
  "coupe", "hatchback", "SUV", "sports car",
  "racing car", "vintage car"

Singular values: [72.2, 20.9, 18.5, 16.8, 14.8, 13.2, 11.8, 11.1, 9.8, 9.4]

With α=2.0:
  - σ1 (72.2) → weighted as 0.8449
  - σ2 (20.9) → weighted as 0.1161 (1.88x boost)
  - σ10 (9.4) → weighted as 0.0246 (1.98x boost)

Result: All 10 singular values contribute to concept removal,
        not just the dominant first one.
```

---

## Why It Might Not Be Working

If you're using α=2.0 correctly, but concept removal still isn't working, possible reasons:

1. **Too few forget prompts:** 10 prompts might not capture all "car" variations
2. **Singular values too skewed:** σ1=72 vs σ10=9 ratio is 8x, even with regularization
3. **Projector too small:** 4.36% weight change might not be enough
4. **Model representation issue:** Maybe the model doesn't represent "car" as a separable subspace

**Solution:** Try α=5.0 for stronger regularization of the smaller singular values.

