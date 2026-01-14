# Hidden State as the Mechanism of Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We present a formal theory establishing that **hidden state is necessary for Control, and sufficient when satisfying three conditions: causal influence on visible state, overwriteability, and dynamic reachability** in deterministic discrete dynamical systems. Control—defined as context-dependent divergence where identical visible configurations produce different outcomes—is proven impossible in memoryless systems and achievable precisely when hidden state causally influences visible updates. We introduce "stickiness" (history-dependent transition resistance) as a natural mechanism for generating hidden state, demonstrate its universality across 168 non-trivial elementary cellular automaton rules, and characterize the boundary-localized structure of the resulting Control.

## Key Results

### Theorem 1 (Necessity)
In any deterministic memoryless system f: V → V, Control = 0.

### Theorem 2 (Sufficiency)
Hidden state H enables Control if and only if:
- **(C1)** H causally influences V
- **(C2)** H is overwriteable
- **(C3)** Multiple H values are dynamically reachable

### Theorem 3 (Universality)
All 168 non-trivial ECA rules gain Control > 0 under stickiness—100% success rate, zero exceptions.

### The Causal Chain

```
Stickiness → Hidden State → Context-Dependence → Control
```

This is not merely correlation. The necessity theorem proves Control is impossible without hidden state. The sufficiency theorem specifies precisely when hidden state produces Control.

## Repository Structure

```
Stickiness/
├── paper/
│   ├── stickiness_control.md           # Full paper (Markdown)
│   ├── stickiness_control.tex          # Full paper (LaTeX)
│   └── references.bib                  # Bibliography
├── figures/
│   ├── fig1.png/pdf                    # Standard ECAs have zero Control
│   ├── fig2.png/pdf                    # Stickiness mechanisms
│   ├── fig3.png/pdf                    # Universality results
│   ├── fig4.png/pdf                    # Control magnitude comparison
│   ├── fig5.png/pdf                    # Boundary-Control correlation
│   ├── fig6.png/pdf                    # Control transport
│   └── fig7.png/pdf                    # Counterfactual measurement
├── code/
│   ├── stickiness_framework.py         # Core stickiness implementation
│   ├── exhaustive_sticky_search.py     # All 256 rules × 4 mechanisms
│   ├── mechanism_hypotheses.py         # Hidden state mechanism tests
│   ├── generate_figures.py             # Generate all paper figures
│   └── phase*.py                       # Detailed experimental phases
├── data/
│   ├── PROGRESS.md                     # Research progress log
│   ├── CHECKPOINT.json                 # Session checkpoint
│   ├── exhaustive_results.json         # Raw data (1024 configurations)
│   └── exhaustive_analysis.json        # Processed analysis
├── theory/
│   ├── INDEX.md                        # Theory file index
│   ├── STICKINESS_CONJECTURE.md        # Original conjecture
│   ├── CONTROL_MECHANISM.md            # Hidden state analysis
│   └── BOUNDARY_CONTROL_DELIVERABLES.md # Boundary theory
├── supplementary/
│   ├── proofs.md                       # Detailed mathematical proofs
│   └── exhaustive_enumeration.md       # Full ECA enumeration results
├── discussion/
│   ├── INDEX.md                        # Discussion file index
│   ├── IMPLICATIONS.md                 # Physical implications
│   ├── FUTURE_WORK.md                  # Open problems
│   └── FAQ.md                          # Common questions
├── LICENSE
├── stickiness_control.pdf              # PDF of the paper
└── README.md                           # This file
```

## Quick Start

### Requirements
- Python 3.10+
- NumPy, Matplotlib, SciPy

### Run Experiments
```bash
cd code
python stickiness_framework.py          # Basic stickiness experiments
python exhaustive_sticky_search.py      # Full enumeration (256 rules)
```

### Generate Figures
```bash
cd code
python generate_figures.py
```

### Build Paper (LaTeX)
```bash
cd paper
pdflatex stickiness_control.tex
bibtex stickiness_control
pdflatex stickiness_control.tex
pdflatex stickiness_control.tex
```

## Key Findings

### 1. Standard ECAs Have Zero Control
All 256 elementary cellular automaton rules, including Rule 110 (universal), have exactly Control = 0 when implemented as standard memoryless systems.

### 2. Stickiness Universally Enables Control
Adding stickiness mechanisms (confirmation or refractory) transforms 100% of non-trivial rules from Control = 0 to Control > 0.

| Mechanism | Mean Control | Rules with Control > 0.3 |
|-----------|--------------|--------------------------|
| Standard | 0.000 | 0% |
| Confirmation | 0.439 | 71% |
| Refractory | 0.169 | 32% |

### 3. Control Concentrates at Boundaries
Control shows strong correlation with boundary presence (r = 0.73, p < 0.0001). Control regions propagate with boundary motion at ~0.3 cells/step.

### 4. Physical Implications
Systems with intrinsic stickiness (activation barriers, refractory periods, hysteresis) should exhibit natural Control capability:

| Physical System | Stickiness Mechanism | Hidden State |
|-----------------|---------------------|--------------|
| Chemical reactions | Activation barriers | Energy levels |
| Neural systems | Refractory periods | Recovery state |
| Magnetic materials | Hysteresis | Magnetization |
| Electronic circuits | Capacitance | Charge |

## Connection to UCT

This work provides a physical mechanism for the "Control bit" in computational threshold theories. The 4-bit to 5-bit transition corresponds to acquiring hidden state. The 5th bit IS the hidden state.

## Citation

```bibtex
@article{stickiness_control_2026,
  title={Hidden State as the Mechanism of Control: A Formal Theory of Stickiness in Discrete Dynamical Systems},
  author={Nixon, Robin},
  journal={[Journal]},
  year={2026},
  note={Preprint}
}
```

## References

1. Fredkin, E. (1990). "Digital Mechanics." *Physica D*.
2. Greenberg, J.M. & Hastings, S.P. (1978). "Spatial Patterns for Discrete Models of Diffusion in Excitable Media." *SIAM J. Appl. Math.*
3. Alonso-Sanz, R. (2009). *Cellular Automata with Memory*. World Scientific.
4. Wolfram, S. (1983). "Statistical Mechanics of Cellular Automata." *Rev. Mod. Phys.*
5. Cook, M. (2004). "Universality in Elementary Cellular Automata." *Complex Systems*.

## Research Series

This paper is part of a seven-paper research series exploring computation, self-organization, and robustness in discrete dynamical systems.

| # | Paper | Repository | Key Contribution |
|---|-------|------------|------------------|
| 1 | The Five-Bit Threshold | [UCT](https://github.com/RobinNixon/UCT) | Minimum complexity for universal computation |
| **2** | **Stickiness and Control** | **[Stickiness](https://github.com/RobinNixon/Stickiness)** | Hidden state as mechanism for Control |
| 3 | Self-Maintenance | [Self-Maintenance](https://github.com/RobinNixon/Self-Maintenance) | 83.7% of ECA rules are life-like under stickiness |
| 4 | Substrate Leakiness | [Leakiness](https://github.com/RobinNixon/Leakiness) | Two-axis predictive framework (R² = 0.96) |
| 5 | Structural Invariants | [Invariants](https://github.com/RobinNixon/Invariants) | Basin criterion for survival under filtering |
| 6 | Anti-Resonance | [Anti-Resonance](https://github.com/RobinNixon/Anti-Resonance) | Carrier-period modulation of visibility |
| 7 | Orthogonal Robustness | [Robustness](https://github.com/RobinNixon/Robustness) | Unified framework: four independent axes |

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions, particularly:
- Extensions to continuous or stochastic systems
- Analysis of higher-dimensional cellular automata
- Physical implementations and experimental validation
- Connections to thermodynamics and information theory

Please open an issue or submit a pull request.
