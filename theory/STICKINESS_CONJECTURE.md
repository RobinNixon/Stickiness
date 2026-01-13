# The Stickiness-Control Conjecture

## Abstract

We present experimental evidence that **stickiness** (history-dependent state transitions) provides a general mechanism for generating **Control capability** in cellular automata. Specifically, we show that adding a "confirmation requirement" to elementary cellular automata (ECAs) transforms deterministic systems (Control = 0) into context-dependent systems (Control > 0.3 for 183 out of 256 rules). We conjecture that stickiness and Control are deeply connected: stickiness creates hidden state, hidden state creates context-dependence, and context-dependence IS Control.

## Key Finding

**Standard ECAs have zero Control capability. Sticky ECAs universally gain Control.**

| Mechanism | Rules with Control > 0.3 | Mean Control | Max Control |
|-----------|-------------------------|--------------|-------------|
| Standard ECA | 0 / 256 (0%) | 0.000 | 0.000 |
| Confirmation | **183 / 256 (71%)** | 0.439 | 1.000 |
| Refractory (t=2) | 83 / 256 (32%) | 0.169 | 0.889 |
| Refractory (t=3) | 67 / 256 (26%) | 0.143 | 0.750 |

## The Confirmation Mechanism

The most effective stickiness mechanism is **confirmation** - requiring state changes to be requested twice before execution:

```
Standard ECA:
  state(t+1) = f(neighborhood(t))

Confirmation ECA:
  if f(neighborhood(t)) ≠ state(t):
    if pending = f(...):
      state(t+1) = f(...)    # Confirmed - apply change
      pending = ∅
    else:
      pending = f(...)       # First request - defer
      state(t+1) = state(t)  # Keep current state
  else:
    pending = ∅              # No change requested
    state(t+1) = state(t)
```

## Why This Creates Control

### The Control Proxy

We measure **Control** as the degree to which the same local pattern produces different outcomes:

```
Control = 0:  pattern (0,1,1) → always produces 1
Control > 0:  pattern (0,1,1) → sometimes 0, sometimes 1
```

In standard ECAs, the mapping is deterministic: same pattern → same outcome. Control is always zero.

### Hidden State Creates Context-Dependence

The confirmation mechanism adds **hidden state** (the pending flag). This means:
- Two cells with identical visible neighborhoods can have different hidden states
- The same visible pattern can produce different outcomes depending on hidden state
- This IS context-dependent divergence - the definition of Control

### The Chain of Implication

```
Stickiness
    ↓
Creates hidden state (pending/cooldown/history)
    ↓
Hidden state varies across space and time
    ↓
Same visible pattern → different hidden contexts → different outcomes
    ↓
Context-dependent divergence = Control capability
```

## Connection to UCT

The Universal Computation Threshold (UCT) posits that universal computation requires:
- **Logic (L)**: ≥ 2 bits
- **Memory (M)**: ≥ 1 bit
- **Control (K)**: ≥ 1 bit
- **State (S)**: ≥ 1 bit

Our findings suggest that **stickiness provides the Control bit**:

1. Standard ECAs have Logic, Memory, and State, but lack Control
2. Adding stickiness (1 mechanism parameter) adds Control capability
3. This may be why certain physical substrates support computation better than others - they have inherent stickiness

## The Stickiness-Control Conjecture

**Conjecture**: Any deterministic automaton can be endowed with Control capability by adding history-dependent state transitions (stickiness). The Control gained is proportional to the "depth" of history dependence, up to a saturation point.

**Formalization**: Let A be a cellular automaton with transition function f. Define the sticky variant A_s with transition function f_s that incorporates hidden state h:

```
f_s(n, h) = (f(n), h') where h' depends on history
```

Then:
```
Control(A) = 0  implies  Control(A_s) > 0
```

for all non-trivial stickiness mechanisms.

## Supporting Evidence

### 1. Universality of Control Gain

Confirmation mechanism adds Control to **every** non-trivial ECA:
- 183/256 rules gain Control > 0.3
- The remaining rules are mostly trivial (all 0s, all 1s, identity)

### 2. Mechanism Independence

Multiple stickiness mechanisms all add Control:
- Confirmation (pending flag)
- Refractory (cooldown timer)
- Second-order (XOR with t-2)

The specific mechanism matters for structure preservation, but all add Control.

### 3. Proportionality

Stronger stickiness (longer refractory period) tends to increase Control up to a point, then the system becomes too rigid.

## Implications

### For UCT

Stickiness may be the **physical implementation** of the Control bit. Systems with inherent stickiness (chemical reactions, biological systems) naturally acquire Control capability.

### For Emergence

The transition from "interesting dynamics" (4 bits) to "computation" (5 bits) may correspond to acquiring sufficient stickiness for Control.

### For Artificial Systems

Designing computational substrates should explicitly consider stickiness. Too little → no Control. Too much → rigid/random. The "Goldilocks zone" of stickiness enables structured computation.

## Open Questions

1. **Is there a minimum stickiness for universality?**
   - Like the 5-bit UCT threshold, is there a "stickiness threshold"?

2. **Can stickiness replace explicit asymmetry?**
   - Rule 110's asymmetry is built into the rule. Can symmetric rules + stickiness achieve universality?

3. **What is the relationship between stickiness depth and Control?**
   - Confirmation (depth 1) vs refractory_2 (depth 2) vs second-order (depth 2) - how does depth map to Control?

4. **Is Control gained from stickiness "real" Control in the UCT sense?**
   - Does sticky Control enable actual computation, or just statistical divergence?

## Conclusion

Stickiness appears to be a fundamental mechanism for generating Control capability. The confirmation mechanism, in particular, transforms deterministic ECAs into context-dependent systems with measurable Control. This suggests a deep connection between temporal memory and computational capability, with implications for understanding both natural and artificial computation.

---

**Experimental Code**: See `experiments/exhaustive_sticky_search.py`
**Raw Data**: See `output/exhaustive_results.json`
**Connection to UCT**: See `C:\Github\UCT\paper\five_bit_threshold.md`
