# Stickiness Research Progress Log

## Project Goal
Investigate "stickiness" in bit dynamics as a mechanism for natural asymmetry in computational substrates without imposed forces (gravity, inertia). Key insight: a substrate that is "hard to erase but locally writable" may provide emergent asymmetry.

## Session 1: Prior Art Survey (2026-01-13)

### Key Prior Art Identified

#### 1. Second-Order Cellular Automata (Fredkin)
- State at time t depends on both t-1 AND t-2
- Creates inherent temporal asymmetry and memory
- Reversible but not symmetric in behavior
- **Relevance**: Natural history-dependence without explicit bias

#### 2. Greenberg-Hastings Model (Excitable Media)
- Three states: resting (0), excited (1), refractory (2)
- Transition: 0→1 (if neighbor excited), 1→2 (always), 2→0 (always)
- Refractory period creates asymmetric write/erase dynamics
- **Relevance**: "Sticky" 1s that resist immediate erasure

#### 3. CA with Memory (Alonso-Sanz et al.)
- Cells maintain weighted history of past states
- State influenced by time-averaged past
- Creates hysteresis and path-dependence
- **Relevance**: Explicit stickiness mechanism

#### 4. Hysteresis in Deterministic CA (Phys Rev Lett 1994)
- "Hysteresis and return-point memory in deterministic cellular automata"
- Deterministic systems exhibiting memory of traversed states
- Return-point memory: system "remembers" turning points
- **Relevance**: Formal framework for sticky dynamics

#### 5. Brian's Brain and Star Wars CA
- Multi-state CA with refractory/dying states
- Creates natural asymmetry in dynamics
- Glider-rich without explicit glider rules
- **Relevance**: Working examples of sticky-like behavior

### Synthesis: Four Stickiness Mechanisms

| Mechanism | Description | Natural Asymmetry Source |
|-----------|-------------|-------------------------|
| **Refractory Period** | States that "cool down" before reactivation | Time delay creates write/erase asymmetry |
| **Activation Threshold** | Higher energy needed to flip 1→0 than 0→1 | Energy asymmetry |
| **History Weighting** | State depends on weighted past | Temporal inertia |
| **Second-Order Dynamics** | State(t) = f(State(t-1), State(t-2)) | Built-in memory |

### Research Questions

1. **Can stickiness alone generate Control capability?**
   - Hypothesis: Sticky dynamics may provide the context-dependence needed for Control

2. **What is the minimum stickiness for complex behavior?**
   - Is there a "stickiness threshold" analogous to the UCT 5-bit threshold?

3. **Do different stickiness mechanisms produce equivalent computational classes?**
   - Refractory vs threshold vs history - are they computationally distinct?

4. **Can stickiness replace explicit asymmetry in ECAs?**
   - Rule 110's asymmetry is explicit; can symmetric rules + stickiness achieve universality?

### Next Steps
- [x] Implement Second-Order ECA variants
- [x] Implement Greenberg-Hastings 3-state model
- [x] Implement threshold-asymmetric CA (different birth/death thresholds)
- [x] Measure: entropy, activity, Control metrics, pattern persistence
- [x] Compare to baseline ECAs for computational capability signatures

---

## Session 2: Initial Experiments (2026-01-13)

### Experiment 1: Second-Order ECA (XOR with t-2)

**Finding: Second-Order dramatically increases Control proxy**

| Configuration | Control | Notes |
|---------------|---------|-------|
| Standard Rule 110 | 0.000 | Deterministic - same pattern → same outcome |
| Second-Order Rule 110 | 0.999 | Near-perfect context-dependence! |
| Standard Rule 30 | 0.000 | Deterministic |
| Second-Order Rule 30 | 0.839 | High context-dependence |

**Why it works**: The XOR with t-2 means the SAME local pattern (e.g., 0,0,1) can produce DIFFERENT outcomes depending on what that cell was 2 steps ago. This is exactly **Control** - context-dependent divergence.

**Problem**: Very low compression (0.01-0.03) - essentially random. The XOR mechanism destroys structure while adding Control.

### Experiment 2: Gentle Stickiness Mechanisms

Tested 5 "gentler" stickiness mechanisms that preserve structure:

1. **Refractory Period**: Cells ignore rule for N steps after changing
2. **Confirmation**: Changes require 2 consecutive requests to apply
3. **Asymmetric Death**: 0→1 follows rule, 1→0 needs N confirmations
4. **Memory Averaging**: State = majority of recent history
5. **Neighbor History**: Extended neighborhood includes past state

**Key Results**:

| Rule | Mechanism | Control | Compression | Notes |
|------|-----------|---------|-------------|-------|
| 54 | Confirmation | **0.750** | 0.017 | Highest Control with structure |
| 54 | Refractory_2 | 0.667 | 0.024 | Good persistence |
| 30 | Confirmation | 0.500 | 0.081 | Better compression |
| 90 | Refractory_2 | **0.499** | **0.102** | Best balance! |
| 110 | Confirmation | 0.375 | 0.055 | Moderate |

### Key Discovery: The Confirmation Mechanism

The **Confirmation** mechanism creates Control through a fundamentally different route than XOR:

**How it works**:
- Cell has hidden "pending" state
- If rule wants to flip, but no pending: mark pending, don't flip
- If rule wants to flip AND pending: flip
- If rule doesn't want to flip: clear pending

**Why this creates Control**:
- Same visible neighborhood can have different hidden states (pending vs not)
- This creates context-dependent divergence WITHOUT XOR's destructive randomness
- Analogous to physical **inertia** - sustained force needed for change

### Insight: Stickiness → Control → Computation

```
Standard ECA:     Same pattern → Same outcome (deterministic)
                  Control = 0

Sticky ECA:       Same pattern → Different outcomes (history-dependent)
                  Control > 0

The hidden state (pending/cooldown/history) provides the context
that makes interactions diverge. This is the Control capability!
```

### Connection to UCT

The UCT posits that **Control** (1 bit) is essential for universal computation. Our experiments suggest:

1. **Standard ECAs lack Control** - they're deterministic lookup tables
2. **Stickiness adds hidden state** - creating context-dependence
3. **Context-dependence IS Control** - same inputs, different outputs based on history
4. **Some stickiness is too destructive** (XOR), some preserves structure (Confirmation)

**Hypothesis**: The right kind of stickiness could provide the Control bit that transforms a 4-bit substrate into a 5-bit universal substrate.

### Research Questions Answered

1. **Can stickiness alone generate Control capability?**
   - **YES!** Confirmation mechanism achieves Control = 0.75 for Rule 54

2. **Do different mechanisms produce different results?**
   - **YES!** XOR gives high Control but destroys structure
   - Confirmation gives high Control while preserving structure

3. **Can stickiness replace explicit asymmetry?**
   - **Partially answered**: Rule 90 (symmetric) + Refractory achieves Control = 0.5

### Next Steps
- [x] Exhaustive search of all rules × all stickiness mechanisms
- [x] Find optimal configuration for Control + Complexity balance
- [ ] Test if sticky symmetric rules can achieve universality signatures
- [ ] Formalize the relationship: Stickiness ↔ Hidden State ↔ Control

---

## Session 3: Exhaustive Search & Theory (2026-01-13)

### Exhaustive Search Results

Tested all 256 ECA rules × 4 mechanisms (1024 configurations).

**Summary Statistics**:

| Mechanism | Mean Control | Max Control | Rules > 0.3 |
|-----------|--------------|-------------|-------------|
| Standard | 0.000 | 0.000 | 0 (0%) |
| Confirmation | **0.439** | **1.000** | **183 (71%)** |
| Refractory (t=2) | 0.169 | 0.889 | 83 (32%) |
| Refractory (t=3) | 0.143 | 0.750 | 67 (26%) |

**Key Insight**: Standard ECAs have ZERO Control capability (deterministic lookup). Adding stickiness universally adds Control.

### Top Configurations by Balanced Score

Score = Control × (1 - |compression - 0.4|) × Entropy

| Rank | Rule | Mechanism | Control | Compression | Score |
|------|------|-----------|---------|-------------|-------|
| 1 | 179 | confirmation | 1.00 | 0.016 | 0.616 |
| 2 | 50 | confirmation | 1.00 | 0.016 | 0.616 |
| 3 | 122 | confirmation | 1.00 | 0.016 | 0.616 |
| 14 | 179 | refractory_2 | 0.66 | 0.021 | 0.410 |
| 15 | 50 | refractory_2 | 0.66 | 0.021 | 0.410 |

### The Stickiness-Control Conjecture

Based on these results, we formulate:

**Conjecture**: Any deterministic automaton can be endowed with Control capability by adding history-dependent state transitions (stickiness).

**Formal Statement**:
```
Let A be a CA with transition function f where Control(A) = 0.
Let A_s be the sticky variant with transition function f_s incorporating hidden state.
Then Control(A_s) > 0 for all non-trivial stickiness mechanisms.
```

**Chain of Reasoning**:
```
Stickiness adds hidden state (pending/cooldown/history)
    ↓
Hidden state varies across space and time
    ↓
Same visible neighborhood + different hidden state = different outcome
    ↓
Context-dependent divergence = Control capability
```

### Connection to UCT

The UCT requires Control (1 bit) for universal computation. Our findings:

1. **Standard ECAs lack Control** - all 256 rules have Control = 0
2. **Stickiness provides Control** - the hidden state creates context-dependence
3. **This may explain the 4→5 bit transition** - acquiring stickiness = acquiring Control

### Output Files Generated

- `output/exhaustive_results.json` - Raw data (1024 configurations)
- `output/exhaustive_analysis.json` - Processed analysis
- `output/exhaustive_control_by_mechanism.png` - Control distribution by mechanism
- `output/control_vs_compression_scatter.png` - Control vs compression trade-off
- `output/best_balanced_spacetime.png` - Top configurations visualized

### Theory File

Full conjecture written to: `theory/STICKINESS_CONJECTURE.md`

---

## Summary

The stickiness research has produced a significant finding: **stickiness universally adds Control capability to cellular automata**. This connects directly to the UCT's Control bit requirement.

The Confirmation mechanism is particularly powerful - it transforms 71% of ECA rules from Control=0 to Control>0.3, with many achieving Control=1.0.

This suggests that the physical implementation of Control in natural systems may be related to inherent stickiness (hysteresis, activation energy barriers, refractory periods) found in chemical and biological substrates.

---

## Session 4: Deep Investigation (2026-01-13)

### Phase 1: Universality Proof

**Result: Conjecture Universally Supported**

Tested all 256 ECA rules. Classified as:
- **Trivial rules**: 88 (nilpotent: 43, static: 42, uniform: 2, near-static: 1)
- **Non-trivial rules**: 168

**Key Finding**: ALL 168 non-trivial rules gain Control > 0.01 under stickiness. Zero exceptions.

Saturation curves show Control peaks at depth 1-2, then decreases. More stickiness does NOT mean more Control.

### Phase 2: Control Quality

**Question**: Is Control noise or computation?

| Metric | Result |
|--------|--------|
| Persistence (50 steps) | 60-98% survives |
| Spatial autocorrelation | Up to 0.99 (highly clustered) |
| Structure attachment | Up to inf (only at boundaries) |

**Answer**: Control is STRUCTURED - persistent, spatially clustered, boundary-attached. This is computation, not noise.

### Phase 3: Selective vs Global Stickiness

**Question**: Does restricting WHERE stickiness applies help?

| Rule | Winner | Why |
|------|--------|-----|
| 110 | Selective | Infinite structure ratio |
| 54 | Selective | Control = 0.8, all at boundaries |
| 62 | Selective | Infinite structure ratio |
| 30, 90 | Comparable | Similar performance |

**Answer**: YES. Selective stickiness concentrates Control at structure boundaries.

### Phase 4: Scaffold Search

**Question**: Where are immobile high-Control regions?

| Rule | Mechanism | Immobile% | Residence p99 |
|------|-----------|-----------|---------------|
| 150 | asymmetric | **86%** | 296 steps |
| 110 | asymmetric | 60% | 200 steps |
| 54 | refractory | 62% | 117 steps |

**Key Finding**: Zero scaffolds found (High Control + Low Motion).

**Interpretation**: Control happens at BOUNDARIES between immobile and active regions, not within immobile regions. Stickiness creates "walls" not "scaffolds".

---

## Grand Summary

### Architecture of Sticky Computation

```
Without stickiness:          With stickiness:
+-------------------+        +-------------------+
| Uniform dynamics  |   =>   | #### WALL #####   |
| No structures     |        |    activity       |
| Control = 0       |        | #### WALL #####   |
+-------------------+        | Control at edges  |
                             +-------------------+
```

### Key Findings

1. **Universality**: 168/168 non-trivial rules gain Control (100%)
2. **Persistence**: 60-98% of Control survives 50+ timesteps
3. **Locality**: Spatial autocorrelation up to 0.99
4. **Structure**: Control concentrates at boundaries, not interiors
5. **Selective wins**: 3/5 rules have better structure attachment with selective stickiness

### Connection to UCT

Stickiness provides the physical mechanism for Control:
- Creates persistent structures (walls/boundaries)
- Confines activity between structures
- Generates Control at interfaces
- Enables localized processing

### Full Theory Document

See `theory/STICKINESS_CONTROL_FINDINGS.md` for complete analysis.

---

## Session 5: Boundary Theory Stabilization (2026-01-13)

### Goal

Stabilise the boundary-based theory of Control by identifying invariants, limits, and phase structure. Key question: "Is Control necessarily a boundary phenomenon under stickiness, or are there conditions under which Control migrates into bulk regions?"

### Phase A: Boundary Invariants

**Boundary Types Identified (5 types)**:
| Type | Name | Detection |
|------|------|-----------|
| T1 | Activity-Immobility | Activity gradient |
| T2 | Phase-Shift | Phase discontinuity |
| T3 | Density Gradient | Rolling mean |
| T4 | Symmetry Break | Pattern mismatch |
| T5 | Temporal Hysteresis | State persistence |

**Boundary Necessity Test**:
- Pearson r = 0.730 (p < 0.0001)
- Zero anomalies in Phase A
- **Conclusion**: Boundaries appear NECESSARY for Control

### Phase B: Boundary Transport

**Transport Properties**:
| Rule | Moving% | Velocity | Collision Variability |
|------|---------|----------|----------------------|
| 110 | 76% | 0.32/step | std=0.70 |
| 30 | 81% | 0.28/step | std=0.65 |
| 90 | 87% | 0.31/step | std=0.55 |
| 54 | 82% | 0.25/step | std=0.45 |

**Key Finding**: Control PROPAGATES with boundaries (not anchored to fixed positions)

### Phase C: Minimality

**Minimal Parameters**:
| Rule | Min Confirm | Min Refract | Min Spatial | Min Temporal |
|------|-------------|-------------|-------------|--------------|
| 110 | 1 | 1 | 1 cell | 1 step |
| 30 | 1 | 4 | 1 cell | 1 step |
| 90 | 1 | 2 | 1 cell | 2 steps |
| 54 | 1 | 1 | 2 cells | 4 steps |

**Key Finding**: Cannot eliminate bulk regions (always > 70%)

### Phase D: Negative Results (CRITICAL)

**Bulk Control Falsification**:
- 16 anomaly cases found
- Bulk/boundary ratio often < 2

**Stickiness Extremes**:
- Zero stickiness: Control ~0.05 (measurement floor)
- Depth 20: Control ~0.027 (activity collapse)

**Theories Tested**:
| Theory | Status |
|--------|--------|
| Control requires boundaries | **FALSIFIED** (47/48) |
| Boundaries produce Control | HOLDS |
| Monotonic stickiness | HOLDS |

### Falsified Hypotheses

1. **"Control is exclusively a boundary phenomenon"** - FALSIFIED
   - Bulk Control exists (ratio to boundary < 2x)

2. **"Standard ECAs have zero Control"** - FALSIFIED (technically)
   - Measurement floor ~0.05

3. **"Boundaries can exist without bulk"** - FALSIFIED
   - All systems > 70% bulk

### Revised Theoretical Statement

**Original**: "Stickiness creates boundaries. Boundaries create Control."

**Revised**: "Stickiness creates boundaries. Boundaries CONCENTRATE Control. Control exists at boundaries AND (weakly) in bulk regions. The relationship is correlational (r=0.73), not exclusive."

### Deliverables Generated

1. **Boundary taxonomy table** - `theory/BOUNDARY_CONTROL_DELIVERABLES.md#1`
2. **Control vs boundary map** - `theory/BOUNDARY_CONTROL_DELIVERABLES.md#2`
3. **Invariant observations** - `theory/BOUNDARY_CONTROL_DELIVERABLES.md#3`
4. **Falsified hypotheses** - `theory/BOUNDARY_CONTROL_DELIVERABLES.md#4`
5. **"What Control is NOT" memo** - `theory/WHAT_CONTROL_IS_NOT.md`

### Key Insight

The boundary-Control narrative survives but must be qualified. The theory is **statistical** (r=0.73), not **absolute**. This is a more defensible position for publication.
