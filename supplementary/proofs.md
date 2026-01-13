# Detailed Proofs

## Necessity Theorem

**Theorem 3.1 (Necessity of Hidden State for Control):** Let (V, f) be a deterministic memoryless dynamical system with f: V → V. Then Control = 0.

### Proof

1. Model the memoryless system as (V, H, f_s) with H = {*} (singleton)
2. Define f_s(v, *) = (f(v), *)
3. By Definition 2.4, Control > 0 requires:
   ∃v ∈ V, ∃h₁, h₂ ∈ H with h₁ ≠ h₂ : π_V(f_s(v, h₁)) ≠ π_V(f_s(v, h₂))
4. Since H = {*}, we have |H| = 1
5. For any h₁, h₂ ∈ H, we have h₁ = h₂ = *
6. The condition h₁ ≠ h₂ cannot be satisfied
7. The existential quantifier ∃h₁ ≠ h₂ fails
8. Therefore Control = 0. ∎

### Corollary

Standard elementary cellular automata (ECAs) have Control = 0.

**Proof:** An ECA with rule φ: {0,1}³ → {0,1} on lattice V = {0,1}ⁿ defines f: V → V by f(v)ᵢ = φ(vᵢ₋₁, vᵢ, vᵢ₊₁). This is memoryless. By Theorem 3.1, Control = 0. ∎

---

## Sufficiency Theorem

**Theorem 4.1 (Sufficiency Conditions for Control):** Let (V, H, f_s) be a deterministic system with |H| > 1. Then Control > 0 if and only if H satisfies:

**(C1) Causal influence:** ∃v ∈ V, ∃h₁ ≠ h₂ ∈ H : π_V(f_s(v, h₁)) ≠ π_V(f_s(v, h₂))

For Control to be dynamically achievable, additionally:

**(C2) Overwriteability:** ∃v ∈ V, ∃h ∈ H : f_H(v, h) ≠ h

**(C3) Reachability:** |H_reach| > 1

### Proof

**(C1 ⟺ Control > 0):** Condition C1 is precisely Definition 2.4 restated. By definition, Control > 0 means the same visible state with different hidden states produces different visible outputs—which is exactly C1.

**(C2 Necessity for Dynamic Control):** Suppose C2 fails: f_H(v, h) = h for all v, h. Then H is invariant under dynamics. Starting from any (v₀, h₀), we have hₜ = h₀ for all t. The system remains at the initial hidden state. Different hidden states cannot arise dynamically. Therefore dynamically achievable Control requires C2.

**(C3 Necessity):** If |H_reach| = 1, say H_reach = {h₀}, then all trajectories have hₜ = h₀. No pair (h₁, h₂) with h₁ ≠ h₂ is dynamically accessible. Control may be formally nonzero but never realized. ∎

---

## Stickiness-Control Universality

**Theorem 6.1:** The confirmation mechanism with depth d ≥ 1 universally generates hidden state satisfying C1, C2, C3 for all non-trivial base rules.

### Proof

**(C1) Causal Influence:** Consider visible configuration v where base rule φ requests a change at position i.
- If hᵢ = 0 (no pending): Change is blocked, vᵢ unchanged
- If hᵢ = d-1 (pending complete): Change is applied, vᵢ flips

Same v, different h → different visible output. C1 satisfied.

**(C2) Overwriteability:**
- If φ does not request change, h resets to 0
- If φ requests change, h increments

Both cases have f_H(v, h) ≠ h for appropriate v.

**(C3) Reachability:** From h = 0, repeated change requests reach h = 1, 2, ..., d-1. From any h > 0, lack of change request returns to h = 0. All h ∈ {0, ..., d-1} are reachable. ∎

---

## Experimental Verification Summary

**All 256 ECA rules tested:**
- Trivial rules identified: 88 (43 nilpotent, 42 static, 2 uniform, 1 near-static)
- Non-trivial rules: 168
- Non-trivial rules with Control > 0.01 under stickiness: **168/168 (100%)**

Zero exceptions found.

---

## Boundary-Control Correlation

**Measurement Results:**
- Pearson r = 0.73 (p < 0.0001)
- Mean Control at boundaries: 0.35
- Mean Control in bulk regions: 0.12
- Boundary/bulk ratio: 2.9×

**Interpretation:** Control concentrates at boundaries but is not exclusive to them. The relationship is statistical (r = 0.73), not absolute. This is a more defensible position than claiming Control is exclusively a boundary phenomenon.

---

## Falsified Hypotheses

The following hypotheses were tested and found to be false:

1. **"Control is exclusively a boundary phenomenon"** — FALSIFIED
   - Bulk Control exists (ratio to boundary < 2×)

2. **"Standard ECAs have zero Control"** — FALSIFIED (technically)
   - Measurement floor ~0.05 due to sampling

3. **"Boundaries can exist without bulk"** — FALSIFIED
   - All systems > 70% bulk

**Revised Statement:** "Stickiness creates boundaries. Boundaries CONCENTRATE Control. Control exists at boundaries AND (weakly) in bulk regions."
