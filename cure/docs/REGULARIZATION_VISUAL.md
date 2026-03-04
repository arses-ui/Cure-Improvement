# Regularization Visualization

## The Problem: Skewed Singular Values

```
Singular values for "car" concept:
┌─────────────────────────────────────────┐
│ σ₁ = 72.2  ████████████████████ (73%)   │
│ σ₂ = 20.9  █████ (11%)                  │
│ σ₃ = 18.5  ████ (9%)                    │
│ σ₄ = 16.8  ████ (8%)                    │
│ σ₅-₁₀ = ... ▌ (remaining)               │
└─────────────────────────────────────────┘

Issue: First singular value dominates!
```

## How Regularization Solves It

```
WITHOUT Regularization (α=1):
┌──────────────────────────────────────┐
│ Weight of σ₁: 0.7314  ██████████████ │
│ Weight of σ₂: 0.0616  █              │
│ Weight of σ₃: 0.0481  ▌              │
│ Weight of σ₄: 0.0398  ▌              │
│ Weight of σ₅-₁₀: ... ▌ ▌ ▌          │
└──────────────────────────────────────┘
Problem: Still heavily biased to σ₁!

WITH Regularization (α=2):
┌──────────────────────────────────────┐
│ Weight of σ₁: 0.8449  ████████████   │
│ Weight of σ₂: 0.1161  ██             │
│ Weight of σ₃: 0.0919  █              │
│ Weight of σ₄: 0.0765  █              │
│ Weight of σ₅-₁₀: ... ▌ ▌ ▌ ▌        │
└──────────────────────────────────────┘
Better: More balanced!

WITH STRONG Regularization (α=5):
┌──────────────────────────────────────┐
│ Weight of σ₁: 0.9316  ███████████    │
│ Weight of σ₂: 0.2472  ███            │
│ Weight of σ₃: 0.2018  ██             │
│ Weight of σ₄: 0.1715  ██             │
│ Weight of σ₅-₁₀: ... ▌ ▌ ▌ ▌ ▌      │
└──────────────────────────────────────┘
Best: Much more balanced amplification!
```

## Effect on Concept Removal

```
SCENARIO 1: Without Regularization
───────────────────────────────────

Forget concepts: car, automobile, vehicle, sedan, ...
                        ↓
                    [SVD Analysis]
                        ↓
         Finds mainly: "wheelness", "has-4-wheels"
                     (σ₁ dominance)
                        ↓
            [Projector removes σ₁ only]
                        ↓
              ❌ Result: Still generates cars
                        (but different types)


SCENARIO 2: With Regularization (α=2)
──────────────────────────────────────

Forget concepts: car, automobile, vehicle, sedan, ...
                        ↓
                    [SVD Analysis]
                        ↓
    Finds: "wheelness" + "vehicle-shape"
           + "transportation-purpose"
           + "car-variations" + ...
           (balanced across σ₁-σ₁₀)
                        ↓
      [Projector removes ALL singular values]
                        ↓
         ✅ Result: Removes most car concepts


SCENARIO 3: Strong Regularization (α=5)
────────────────────────────────────────

Forget concepts: car, automobile, vehicle, sedan, ...
                        ↓
                    [SVD Analysis]
                        ↓
    Finds ALL directions equally weighted:
    - Core carness
    - All variations (old/new, big/small)
    - All types (sports/family/SUV)
    - Even marginal differences
                        ↓
  [Projector removes all with equal strength]
                        ↓
      ✅✅ Result: Removes almost ALL cars
```

## The Regularization Function Itself

```
The Tikhonov-inspired expansion function:

                    α·r
        f(r;α) = ─────────────
                 (α-1)·r + 1


Graph of f(r;α) for different α values:
(horizontal axis = r, vertical = f(r))

        1.0 │
            │     ╱──── α=5 (strong)
            │    ╱╱
            │   ╱╱
        0.5 │  ╱╱─── α=2 (moderate)
            │ ╱╱
            │╱╱──── α=1 (light)
            │
          0 └────────────────
            0              1

Effect on singular values:
- α=1: Nearly linear (r ≈ f(r))
- α=2: Gently amplifies small r values
- α=5: Strongly amplifies small r values
```

## Real Numbers Example

```
Your "car" concept:
σ = [72.2, 20.9, 18.5, 16.8, 14.8, 13.2, 11.8, 11.1, 9.8, 9.4]

Normalized energy:
r = [0.1326, 0.0110, 0.0086, 0.0072, 0.0055, 0.0044, 0.0035, 0.0031, 0.0024, 0.0022]

With α=2:
f(r) = [0.8449, 0.1161, 0.0919, 0.0765, 0.0602, 0.0480, 0.0387, 0.0345, 0.0271, 0.0246]

Comparison (boost factor):
         Without reg  With α=2   Boost
σ₁       0.7314       0.8449     1.16x
σ₂       0.0616       0.1161     1.88x  ← Small values amplified!
σ₃       0.0481       0.0919     1.91x  ← More amplified
...
σ₁₀      0.0125       0.0246     1.98x  ← Most amplified!

Result: Smaller singular values (representing concept variations)
        get 2x more weight, removing ALL types of cars.
```

## Key Intuition

```
Imagine "car" is encoded in SVD space as:
┌─────────────────────────────────────┐
│ σ₁ direction: "is a car" (common)   │
│ σ₂ direction: "is sedan" (specific) │
│ σ₃ direction: "is fast" (specific)  │
│ σ₄ direction: "is old" (specific)   │
│ ...                                  │
└─────────────────────────────────────┘

Without regularization (α≈1):
  Only remove σ₁ → "is a car"
  Result: Still generates sedans, fast cars, old cars
  ❌ Incomplete!

With regularization (α=2):
  Remove σ₁, σ₂, σ₃, σ₄, ...
  Result: Remove cars AND their variations
  ✅ Complete!
```

## In Code

```python
# Step 1: Compute singular values
sigma_sq = S ** 2
r = sigma_sq / sigma_sq.sum()  # Normalize

# Step 2: Apply regularization function
f_r = (alpha * r) / ((alpha - 1) * r + 1)

# Step 3: Build projector with regularized weights
P = U @ diag(f_r) @ U.T

# Effect:
# - Smaller r values (weaker singular vectors)
#   get boosted by ~1/α factor
# - Creates balanced removal across all directions
# - Higher α = stronger regularization
```

## Summary

**Regularization in CURE = "Don't just remove the obvious concept direction, also remove the subtle variations"**

- **No regularization:** Only removes dominant "car-ness"
- **With regularization (α=2):** Removes cars in all forms
- **Strong regularization (α=5):** Aggressively removes any car-related direction

Think of it like:
- **No regularization:** Remove car engines (obvious)
- **Moderate regularization:** Remove engines, wheels, bodies, shapes (moderate)
- **Strong regularization:** Remove even the idea of transportation (aggressive)
