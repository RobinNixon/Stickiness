# What Control is NOT

**Internal Memo**
**Date**: 2026-01-13
**Investigation**: Stickiness-Boundary-Control Relationship, Phase D

---

## Summary

After rigorous negative-result testing (Phase D), we document what Control is NOT, to bound the theory and prevent overstatement.

---

## Control is NOT:

### 1. Exclusively a boundary phenomenon

We attempted to falsify the claim "Control requires boundaries" and found:
- 47/48 test cases violated this claim
- Bulk/boundary Control ratio often < 2x (not >> 1)
- Refractory stickiness produces especially diffuse Control

**Revised claim**: Control is CONCENTRATED at boundaries, not EXCLUSIVE to them.

---

### 2. Zero in standard ECAs (by our measurement)

The perturbation-based Control proxy has a measurement floor:
- Zero stickiness → Control ~0.05 (all rules)
- This is measurement noise, not true Control
- We cannot distinguish "zero" from "very small"

**Limitation acknowledged**: Our method cannot prove Control = 0.

---

### 3. Monotonically increasing with stickiness

Testing stickiness depth 1 → 20:
- Control peaks around depth 2-5
- Control COLLAPSES at depth 20 (activity → 0)
- Frozen systems have no computation

**Optimal range exists**: Too little OR too much stickiness kills Control.

---

### 4. Independent of bulk regions

Attempting to create "all-boundary" systems:
- All configurations remained > 70% bulk
- Cannot eliminate bulk regions
- Boundaries ARE interfaces, requiring two sides

**Definitional constraint**: Boundary without bulk is undefined.

---

### 5. Stationary in space

Tracking Control regions over time:
- 76-87% of Control regions are MOVING
- Mean velocity ~0.3 cells/step
- "Scaffolds" (immobile high-Control) not found

**Control propagates**: It rides boundaries like waves.

---

### 6. Deterministic at collision events

Analyzing boundary collision outcomes:
- Same collision type → variable results (std 0.4-0.7)
- History and hidden state influence outcomes
- Context-dependence is fundamental

**Collision algebra is stochastic**: Not a lookup table.

---

### 7. Reducible to boundary presence

Correlation (r=0.73) does not imply identity:
- Many boundaries have weak Control
- Boundary QUALITY matters (gradient magnitude)
- Necessary ≠ sufficient

**Boundaries enable but do not determine Control.**

---

## Revised Theoretical Statement

**Original (overstatement)**:
> "Stickiness creates boundaries. Boundaries create Control."

**Revised (defensible)**:
> "Stickiness creates boundaries. Boundaries concentrate Control. The relationship is correlational (r=0.73), not exclusive. Control also exists (weakly) in bulk regions."

---

## Implications for Future Work

1. Do not claim "Control is a boundary phenomenon" without qualification
2. Report measurement floor when comparing Control values near zero
3. Acknowledge optimal stickiness range, not "more is better"
4. The theory is statistical, not absolute

---

**This memo documents negative results. These are as important as positive findings for bounding what we actually know.**
