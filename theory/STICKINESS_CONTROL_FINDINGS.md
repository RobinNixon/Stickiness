# Stickiness-Control Investigation: Complete Findings

## Executive Summary

We conducted a comprehensive investigation into the relationship between stickiness (history-dependent state transitions) and Control capability in cellular automata. The key findings:

1. **Universality Proven**: ALL 168 non-trivial ECA rules gain Control under stickiness (0 exceptions)
2. **Control is Structured**: Persistent (60-90% survives 50+ steps), spatially clustered (autocorr up to 0.99)
3. **Selective > Global**: Selective stickiness produces more structure-bound Control
4. **Boundaries, Not Scaffolds**: Control concentrates at structure boundaries, not within immobile regions

---

## Phase 1: Universality of Stickiness→Control

### Result: Conjecture Universally Supported

| Category | Count | Control Gain Under Stickiness |
|----------|-------|-------------------------------|
| Trivial rules | 88 | N/A (nilpotent, static, uniform) |
| Non-trivial rules | 168 | **168/168 (100%)** gain Control > 0.01 |
| Exceptions | 0 | None |

### Trivial Rule Breakdown
- Nilpotent (converge to all-0 or all-1): 43 rules
- Static fixed point: 42 rules
- Uniform constant: 2 rules
- Near-static: 1 rule

### Saturation Curves

| Rule | Confirmation depth 1 | depth 2 | depth 3 | depth 4 |
|------|---------------------|---------|---------|---------|
| 110 | 0.375 | 0.375 | 0.329 | 0.281 |
| 30 | 0.500 | 0.500 | 0.443 | 0.372 |
| 54 | **0.750** | **0.750** | 0.664 | 0.558 |
| 90 | 0.600 | 0.600 | 0.527 | 0.450 |

**Key insight**: Control peaks at depth 1-2, then decreases. More stickiness ≠ more Control.

---

## Phase 2: Control Quality Metrics

### Question: Is Control noise or computation?

### Persistent Control (survives N timesteps)

| Rule | Mechanism | Basic Control | Persist 10 | Persist 50 | Persist 100 |
|------|-----------|---------------|------------|------------|-------------|
| 110 | confirmation | 0.251 | 55% | **60%** | -- |
| 110 | refractory | 0.382 | 82% | **88%** | -- |
| 62 | confirmation | 0.546 | 83% | **89%** | -- |
| 90 | refractory | 0.452 | 88% | **98%** | -- |

**Finding**: Control is highly persistent (60-98% survives 50+ steps). This is NOT noise.

### Control Locality (spatial autocorrelation)

| Rule | Mechanism | Spatial Autocorr | Interpretation |
|------|-----------|------------------|----------------|
| 62 | confirmation | **0.99** | Extremely clustered |
| 110 | refractory | 0.77 | Highly clustered |
| 30 | confirmation | 0.75 | Highly clustered |
| 90 | confirmation | -0.47 | Anti-clustered (alternating) |

**Finding**: Control spatially clusters in most cases, indicating localized processing.

### Structure Attachment

| Rule | Mechanism | Control at Edges | Control in Interior | Ratio |
|------|-----------|------------------|---------------------|-------|
| 110 | refractory | High | Low | **2.67x** |
| 62 | confirmation | High | Zero | **inf** |
| 30 | confirmation | High | Zero | **inf** |

**Finding**: Control is strongly attached to structure boundaries, not uniformly distributed.

---

## Phase 3: Selective vs Global Stickiness

### Question: Does restricting WHERE stickiness applies improve Control quality?

### Mechanisms Tested
1. **Global confirmation**: Stickiness everywhere
2. **Motif-gated**: Only at edges (where neighbors differ)
3. **Density-gated**: Only in high-activity regions
4. **Asymmetric 1/3**: Easy stick (1 confirm), hard unstick (3 confirms)
5. **Asymmetric 1/5**: Even stickier 1s
6. **Neighborhood-memory**: Only when neighborhood stable

### Results

| Rule | Best Mechanism | Control | Structure Ratio | Winner |
|------|---------------|---------|-----------------|--------|
| 110 | motif_gated | 0.250 | **inf** | Selective |
| 30 | asymmetric_1_3 | 0.463 | 1.36 | Comparable |
| 90 | motif_gated | 0.600 | **inf** | Comparable |
| 54 | density_gated | **0.800** | **inf** | **Selective** |
| 62 | motif_gated | 0.511 | **inf** | Selective |

**Key Finding**: Selective stickiness produces **more structure-bound Control**. For 3/5 rules, selective mechanisms achieve infinite structure ratio (Control ONLY at boundaries).

### Interpretation
- Global stickiness: Control diffuses throughout
- Selective stickiness: Control concentrates at identifiable features
- This distinction matters for computation: structure-bound Control enables localized processing

---

## Phase 4: Scaffold Search

### Question: Where are immobile high-Control regions?

### Immobile Fraction Results

| Rule | Standard | Confirmation | Refractory | Asymmetric |
|------|----------|--------------|------------|------------|
| 110 | 14% | 36% | 14% | **60%** |
| 30 | 4% | 14% | 4% | 15% |
| 90 | 25% | 44% | 6% | 21% |
| 54 | 4% | 14% | **62%** | 14% |
| 150 | 15% | 30% | 65% | **86%** |

### Residence Time (99th percentile)

| Rule | Standard | Confirmation | Refractory | Asymmetric |
|------|----------|--------------|------------|------------|
| 110 | 17 | 148 | 60 | **200** |
| 30 | 8 | 45 | 6 | 60 |
| 150 | 44 | 84 | 119 | **296** |

### Scaffold Candidates
**Zero scaffolds found** (High Control + Low Motion regions)

### Interpretation
Stickiness creates large immobile regions, but Control does NOT concentrate within them. Instead:

**Control happens at the BOUNDARIES between immobile and active regions.**

This is not "scaffolds" but "walls" - persistent structures that constrain and channel activity.

---

## Synthesis: The Architecture of Sticky Computation

### What Stickiness Creates

```
Without stickiness:
┌─────────────────────────────────────┐
│  Uniform dynamics throughout        │
│  No persistent structures           │
│  Control = 0 (deterministic)        │
└─────────────────────────────────────┘

With stickiness:
┌─────────────────────────────────────┐
│  █████    ████   ██████   ████      │ ← Immobile regions (walls)
│       ↑↓↑      ↑↓     ↑↓           │ ← Activity confined between walls
│  █████    ████   ██████   ████      │
│                                     │
│  Control concentrates HERE ────────→│ at wall boundaries
└─────────────────────────────────────┘
```

### The Role of Stickiness in Computation

1. **Creates persistent structures** (walls/boundaries)
2. **Confines activity** between structures
3. **Generates Control at interfaces** where activity meets structure
4. **Enables localized processing** rather than global diffusion

### Connection to UCT

The UCT identifies Control as essential for universal computation. Our findings show:

- **Standard ECAs lack Control** (deterministic lookup)
- **Stickiness provides Control** via hidden state → context-dependence
- **Control quality varies** by mechanism (selective > global)
- **Control localizes at structures** (boundaries, not interiors)

This suggests the physical implementation of Control involves:
1. Persistent structures (enabled by stickiness)
2. Activity confinement (enabled by structures)
3. Context-dependent interactions at boundaries (= Control)

---

## Deliverables

### Figures Generated
1. `phase1/universality_proof.png` - All 256 rules, standard vs confirmation
2. `phase1/saturation_curves.png` - Control vs stickiness depth
3. `phase2/control_quality_comparison.png` - Persistence, locality, structure attachment
4. `phase2/control_maps_rule*.png` - Control distribution visualizations
5. `phase3/selective_vs_global.png` - Mechanism comparison
6. `phase3/mechanisms_rule*.png` - Spacetime diagrams
7. `phase4/scaffold_analysis_rule*.png` - Scaffold detection

### Data Files
1. `phase1/universality_results.json` - Full 256-rule analysis
2. `phase2/control_quality_results.json` - Quality metrics
3. `phase3/selective_results.json` - Selective stickiness comparison
4. `phase4/scaffold_results.json` - Scaffold search data

### Key Tables
1. Universality: 168/168 non-trivial rules gain Control
2. Persistence: 60-98% of Control survives 50+ timesteps
3. Locality: Spatial autocorrelation up to 0.99
4. Selective wins: 3/5 rules have better structure attachment

---

## Open Questions

1. **Can we design rules that maximize boundary Control?**
   - Boundary = where computation happens
   - Can we optimize for boundary quality?

2. **Is there a "scaffold emergence" threshold?**
   - At what stickiness level do scaffolds (vs walls) appear?

3. **Can sticky symmetric rules achieve universality?**
   - Rule 90 (symmetric) + confirmation achieves Control = 0.6
   - Is this enough for universal computation?

4. **Physical substrates**
   - Which materials/chemistries provide optimal stickiness?
   - Is there a "Goldilocks zone" of stickiness for computation?

---

## Conclusion

Stickiness is not just about adding Control - it's about **structuring space** for computation. The key insight:

> **Stickiness creates persistent boundaries. Control emerges at these boundaries. Computation happens in the activity confined between them.**

This provides a architectural model for how physical substrates might implement the Control capability required by the UCT.
