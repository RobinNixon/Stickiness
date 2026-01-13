# Theory Files Index

This folder contains the theoretical foundations for the Hidden State and Control framework.

## Core Theory

| File | Description |
|------|-------------|
| [STICKINESS_CONJECTURE.md](STICKINESS_CONJECTURE.md) | Original conjecture: stickiness universally enables Control |
| [STICKINESS_CONTROL_FINDINGS.md](STICKINESS_CONTROL_FINDINGS.md) | Deep investigation results on Control quality and selectivity |
| [CONTROL_MECHANISM.md](CONTROL_MECHANISM.md) | Hidden state as the mechanism of Control - formal analysis |

## Boundary Theory

| File | Description |
|------|-------------|
| [BOUNDARY_CONTROL_DELIVERABLES.md](BOUNDARY_CONTROL_DELIVERABLES.md) | Boundary taxonomy, invariants, and Control localization |
| [WHAT_CONTROL_IS_NOT.md](WHAT_CONTROL_IS_NOT.md) | Falsified hypotheses and refined theoretical statements |

## Key Results

### Main Theorems (see paper for formal proofs)

1. **Necessity Theorem**: In any deterministic memoryless system f: V → V, Control = 0
2. **Sufficiency Theorem**: Hidden state H enables Control iff (C1) H causally influences V, (C2) H is overwriteable, (C3) Multiple H values are reachable
3. **Universality**: 168/168 non-trivial ECA rules gain Control under stickiness

### Theoretical Chain

```
Stickiness → Hidden State → Context-Dependence → Control
```

This is not merely correlation—necessity is proven (Control impossible without hidden state) and sufficiency is proven (hidden state with C1-C3 guarantees Control).
