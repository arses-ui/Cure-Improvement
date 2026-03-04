#!/usr/bin/env python3
"""
Debug the spectral expansion function to verify it's working correctly
"""

import torch
from cure.spectral import spectral_expansion

def test_spectral_expansion():
    """Test the spectral expansion function f(ri; α)"""

    print("=" * 70)
    print("Testing Spectral Expansion Function: f(ri; α) = αri / ((α-1)ri + 1)")
    print("=" * 70)

    # Test case 1: Simple uniform singular values
    print("\nTest 1: Uniform singular values [1, 1, 1, 1, 1]")
    S1 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    for alpha in [1.0, 2.0, 5.0]:
        f_r = spectral_expansion(S1, alpha)
        print(f"\n  α = {alpha}:")
        print(f"  f(ri; α) = {f_r}")
        print(f"  Sum = {f_r.sum():.4f}")

        # For uniform case, all should be equal
        if alpha == 1.0:
            expected = torch.ones(5) / 5  # Should all be 0.2
            print(f"  Expected (α=1): all equal to 0.2")
            print(f"  Actual: all equal to {f_r[0]:.4f}")

    # Test case 2: Skewed singular values (more realistic)
    print("\n\nTest 2: Skewed singular values [10, 5, 2, 1, 0.5]")
    S2 = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])

    for alpha in [1.0, 2.0, 5.0]:
        f_r = spectral_expansion(S2, alpha)
        print(f"\n  α = {alpha}:")
        print(f"  f(ri; α) = {f_r}")
        print(f"  Ratio (smallest/largest) = {f_r[-1] / f_r[0]:.4f}")

        if alpha == 1.0:
            print(f"  (Should favor large singular values)")
        elif alpha == 5.0:
            print(f"  (Should level out the singular values)")

    # Test case 3: Very skewed (what we might see in real data)
    print("\n\nTest 3: Very skewed singular values [100, 50, 10, 5, 1, 0.1, 0.01]")
    S3 = torch.tensor([100.0, 50.0, 10.0, 5.0, 1.0, 0.1, 0.01])

    print(f"\n  Original singular values: {S3}")

    for alpha in [1.0, 2.0, 5.0, 10.0]:
        f_r = spectral_expansion(S3, alpha)
        print(f"\n  α = {alpha}:")
        print(f"  f(ri; α) = {f_r}")
        print(f"  Min/Max ratio = {f_r.min():.6f} / {f_r.max():.6f} = {(f_r.min() / f_r.max()):.6f}")

    # Test case 4: Check mathematical properties
    print("\n\nTest 4: Mathematical Properties")
    print("The expansion function should:")
    print("  1. At α→1: f(r)→r (energy proportional)")
    print("  2. At α→∞: f(r)→constant (all singular values equal)")

    S_test = torch.tensor([0.5, 0.3, 0.15, 0.05])

    print(f"\n  Test values: ri = {S_test}")
    for alpha in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        f_r = spectral_expansion(S_test, alpha)
        print(f"  α={alpha:4.1f}: f(r)={f_r}, ratio(max/min)={f_r.max()/f_r.min():.4f}")

    # Test case 5: Verify formula manually
    print("\n\nTest 5: Manual Verification of Formula")
    print("For α=2, r=0.5:")
    r = 0.5
    alpha = 2.0
    f_r_manual = (alpha * r) / ((alpha - 1) * r + 1)
    print(f"  f(0.5; 2) = (2 * 0.5) / ((2-1) * 0.5 + 1)")
    print(f"            = 1.0 / (0.5 + 1)")
    print(f"            = 1.0 / 1.5")
    print(f"            = {f_r_manual:.4f}")

    S_test2 = torch.tensor([0.5])
    f_r_code = spectral_expansion(S_test2, 2.0)
    print(f"  Code result: {f_r_code[0]:.4f}")
    print(f"  Match: {abs(f_r_manual - f_r_code[0]) < 1e-6}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_spectral_expansion()
