# Exhaustive Enumeration of ECA Rules Under Stickiness

## Methodology

All 256 elementary cellular automaton (ECA) rules were tested with four stickiness mechanisms:

1. **Standard (baseline):** No stickiness
2. **Confirmation (depth 2):** Changes require 2 consecutive requests
3. **Refractory (period 2):** 2-step cooldown after changing
4. **Refractory (period 3):** 3-step cooldown after changing

Total configurations tested: 256 × 4 = 1024

## Rule Classification

### Trivial Rules (88 total)

| Category | Count | Description |
|----------|-------|-------------|
| Nilpotent | 43 | Converge to all-0 or all-1 |
| Static | 42 | Every cell remains unchanged |
| Uniform | 2 | Every cell takes same value |
| Near-static | 1 | Almost no change |

### Non-trivial Rules (168 total)

All remaining rules that exhibit sustained, non-converging dynamics.

## Summary Statistics by Mechanism

| Mechanism | Mean Control | Max Control | Rules > 0.3 |
|-----------|--------------|-------------|-------------|
| Standard | 0.000 | 0.000 | 0 (0%) |
| Confirmation | **0.439** | **1.000** | **183 (71%)** |
| Refractory (t=2) | 0.169 | 0.889 | 83 (32%) |
| Refractory (t=3) | 0.143 | 0.750 | 67 (26%) |

## Key Finding

**100% of non-trivial rules gain Control > 0.01 under stickiness.**

This is the universality result: stickiness is a general mechanism for enabling Control, not dependent on specific rule properties.

## Top Configurations by Balanced Score

Score = Control × (1 - |compression - 0.4|) × Entropy

| Rank | Rule | Mechanism | Control | Compression | Score |
|------|------|-----------|---------|-------------|-------|
| 1 | 179 | confirmation | 1.00 | 0.016 | 0.616 |
| 2 | 50 | confirmation | 1.00 | 0.016 | 0.616 |
| 3 | 122 | confirmation | 1.00 | 0.016 | 0.616 |
| 14 | 179 | refractory_2 | 0.66 | 0.021 | 0.410 |
| 15 | 50 | refractory_2 | 0.66 | 0.021 | 0.410 |

## Representative Rules Detailed Analysis

### Rule 110 (Universal)
- Standard: Control = 0.000
- Confirmation: Control = 0.100
- Refractory: Control = 0.073
- Note: Lower Control than average due to already-structured dynamics

### Rule 30 (Chaotic)
- Standard: Control = 0.000
- Confirmation: Control = 0.400
- Refractory: Control = 0.320

### Rule 90 (Symmetric)
- Standard: Control = 0.000
- Confirmation: Control = 0.210
- Refractory: Control = 0.170

### Rule 54 (Complex)
- Standard: Control = 0.000
- Confirmation: Control = 0.570
- Refractory: Control = 0.430

## Data Files

Complete results available in:
- `data/exhaustive_results.json` — Raw data (1024 configurations)
- `data/exhaustive_analysis.json` — Processed analysis
