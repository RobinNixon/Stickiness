# Frequently Asked Questions

## Basic Concepts

### What is "stickiness"?
Stickiness is history-dependent transition resistance. A sticky system "remembers" recent state changes and resists immediate reversal. This can be implemented through:
- **Confirmation:** Multiple requests required before change applies
- **Refractory period:** Cooldown after change during which further changes are blocked
- **Asymmetric thresholds:** Different thresholds for 0→1 vs 1→0 transitions

### What is "Control"?
Control is context-dependent divergence: the same visible configuration can produce different outcomes depending on hidden state. Formally, Control > 0 if there exist v, h₁, h₂ where the same v with different hidden states h₁ ≠ h₂ produces different visible outputs.

### What is "hidden state"?
Hidden state is information that influences system dynamics but is not directly observable in the "visible" output. In our framework, it's the H component in the (V, H, f_s) system representation.

## Main Results

### Why do standard ECAs have Control = 0?
Standard ECAs are memoryless deterministic systems: given the current visible configuration, the next configuration is uniquely determined. There is no hidden state (H is singleton), so the existential quantifier in the Control definition fails.

### Why does stickiness add Control?
Stickiness mechanisms (confirmation, refractory) introduce hidden state:
- **Confirmation:** The "pending counter" is hidden state
- **Refractory:** The "cooldown counter" is hidden state

This hidden state varies across space and time, creating situations where the same visible neighborhood has different hidden states, leading to different outputs.

### Is the stickiness-Control correspondence universal?
Yes, for non-trivial rules. All 168 non-trivial ECA rules gain Control > 0 under stickiness. Trivial rules (nilpotent, static, uniform) may remain at Control = 0 because their dynamics are insufficient to generate hidden state variation.

## Technical Questions

### How is Control measured experimentally?
Counterfactual Control: For each visible configuration v, we measure the fraction of hidden state pairs (h₁, h₂) that produce different visible outputs. The aggregate Control is the average over all v.

### Why is Control concentrated at boundaries?
Boundaries are regions where hidden state varies most. At activity-immobility interfaces, some cells have recently changed (high cooldown/pending) while neighbors haven't. This variation in hidden state creates the context-dependence that IS Control.

### Is Control the same as chaos?
No. Chaos (Lyapunov exponent) measures sensitivity to initial perturbation. Control measures context-dependence due to hidden state. Stickiness can reduce chaos while increasing Control—they are distinct phenomena.

## Comparisons

### How does this relate to the Universal Computation Threshold (UCT)?
The UCT posits that universal computation requires at least 5 bits of descriptive complexity, including 1 bit for "Control." Our framework provides a physical mechanism: the Control bit is the hidden state. Acquiring stickiness = acquiring the Control bit.

### How does this differ from cellular automata with memory?
CA with memory (Alonso-Sanz) explicitly track history in the rule definition. Our framework shows that simple stickiness mechanisms implicitly create sufficient hidden state for Control without requiring explicit memory tracking in the rule.

### What about reversible CA?
Second-order CA (Fredkin) have built-in history dependence. Our XOR mechanism experiments show these achieve high Control but often destroy structure. The confirmation mechanism achieves Control while preserving structure better.

## Practical Questions

### Which stickiness mechanism is best?
Confirmation (depth 2) gives the highest Control with reasonable structure preservation. Refractory is simpler but typically gives lower Control.

### What are the computational costs?
Stickiness adds minimal overhead:
- Memory: 1 integer per cell (counter)
- Computation: 1 comparison per cell per timestep

### Can this be implemented in hardware?
Yes. Physical systems with hysteresis (magnetic materials, electronic circuits with capacitance) naturally exhibit stickiness. The framework suggests these are natural computational substrates.
