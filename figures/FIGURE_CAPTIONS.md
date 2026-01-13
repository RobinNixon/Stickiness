# Figure Captions

## Figure 1: Standard ECAs Have Zero Control

Standard elementary cellular automata (Rules 30, 110, 90) have exactly zero Control. Each panel shows the spacetime evolution from a single-cell initial condition on a lattice of 61 cells over 40 timesteps. Despite exhibiting complex emergent patterns, these systems are deterministic memoryless dynamical systems: given any visible configuration V, the next configuration f(V) is uniquely determined. The same visible state always produces the same output, satisfying the definition of a memoryless system. By Theorem 3.1 (Necessity), Control = 0.000 for all standard ECAs. This figure provides visual evidence for the necessity theorem.

**File:** `figures/fig1.png`

---

## Figure 2: Stickiness Mechanisms Add Hidden State

Stickiness mechanisms introduce hidden state through state transition systems. **(a) Confirmation Mechanism:** A cell maintains a pending counter h ∈ {0, 1, ..., d-1}. When the base rule requests a change, the counter increments; change is only applied when h reaches the threshold d-1. If no change is requested, h resets to 0. This creates context-dependence: the same visible neighborhood can produce different outcomes depending on whether the cell is pending (h > 0) or stable (h = 0). **(b) Refractory Mechanism:** After a cell changes, it enters a cooldown period where it ignores rule requests. The cooldown counter h decrements each timestep until reaching 0. Cells with h > 0 are "blocked" from changing. Both mechanisms satisfy the sufficiency conditions (C1, C2, C3) for Control.

**File:** `figures/fig2.png`

---

## Figure 3: Universality of Stickiness-Control Correspondence

**(a)** Classification of all 256 elementary cellular automaton rules. Of 256 rules, 88 are trivial (43 nilpotent, 42 static, 2 uniform, 3 other) and 168 are non-trivial (exhibiting sustained dynamics). **(b)** Under stickiness (confirmation mechanism, depth 2), all 168 non-trivial rules gain Control > 0.01. This is a 100% success rate with zero exceptions. The universality result demonstrates that stickiness is a general mechanism for enabling Control, not dependent on specific rule properties. Trivial rules may remain at Control = 0 because their dynamics are insufficient to generate hidden state variation.

**File:** `figures/fig3.png`

---

## Figure 4: Stickiness Dramatically Increases Control

**(a)** Counterfactual Control comparison for Rules 30, 54, 90, and 110. Red bars show standard ECA (all exactly 0.000). Green bars show sticky ECA with confirmation mechanism (depth 2): Rule 30 = 0.40, Rule 54 = 0.57, Rule 90 = 0.21, Rule 110 = 0.10. Mean increase across rules: 350×. **(b)** Spacetime comparison for Rule 110. Left half shows standard ECA evolution; right half shows sticky ECA evolution from identical initial conditions. The red dashed line separates the two regimes. While visual patterns differ subtly, the key difference is in Control: standard has Control = 0 (deterministic), sticky has Control > 0 (context-dependent).

**File:** `figures/fig4.png`

---

## Figure 5: Control is Correlated with Boundary Presence

**(a)** Scatter plot of boundary presence vs. Control magnitude across multiple rules and configurations. The correlation is r = 0.73 (p < 0.0001), indicating a strong positive relationship. The red line shows the linear fit. This correlation was measured across 100 sampled configurations from sticky ECAs. **(b)** Heatmap visualization of Control intensity in a sticky Rule 110 system (confirmation, depth 2). Brighter regions indicate higher Control. Control is visibly concentrated at the boundaries between active (changing) and inactive (static) regions, confirming that boundaries serve as loci of context-dependent processing. Note: the relationship is correlational, not exclusive—some Control exists in bulk regions.

**File:** `figures/fig5.png`

---

## Figure 6: Control Propagates with Boundary Motion

**(a)** Bar chart showing the percentage of Control regions that are moving (non-stationary) across four representative rules. Rule 30: 81%, Rule 54: 82%, Rule 90: 87%, Rule 110: 76%. All values exceed 50%, demonstrating that Control is predominantly a moving phenomenon. Annotations show mean velocities (~0.25-0.32 cells/step). **(b)** Spacetime diagram of sticky Rule 110 with trajectory lines overlaid. Red solid lines trace Control region trajectories; blue dashed lines trace boundary trajectories. Control regions move with boundaries at characteristic velocities, rather than remaining anchored to fixed spatial positions. This transport property has implications for how computation propagates through sticky substrates.

**File:** `figures/fig6.png`

---

## Figure 7: Counterfactual Control Measurement Methodology

**Top panel:** Conceptual diagram illustrating the counterfactual Control measurement. A visible state v is fixed. Two different hidden states (h₁ = 0, h₂ = 1) are applied. The update function f_s produces different visible outputs (v' = 0 vs v' = 1). When outputs differ for the same visible input but different hidden states, this constitutes Control by Definition 2.4. **Bottom panels:** Experimental results. Left: Standard ECA shows counterfactual Control = 0.000 for all four rules tested—the same visible state always produces the same output. Middle: Confirmation mechanism (depth 2) produces nonzero Control (0.10-0.57). Right: Refractory mechanism (period 2) produces nonzero Control (0.07-0.52). The dramatic difference between standard (zero) and sticky (nonzero) demonstrates that hidden state is the mechanism enabling Control.

**File:** `figures/fig7.png`

---

## Summary Table

| Figure | Title | Key Message |
|--------|-------|-------------|
| 1 | Standard ECAs Have Zero Control | Necessity theorem visualization |
| 2 | Stickiness Mechanisms | How confirmation and refractory add hidden state |
| 3 | Universality Results | 168/168 non-trivial rules gain Control |
| 4 | Control Magnitude Comparison | 350× increase from stickiness |
| 5 | Boundary-Control Correlation | r = 0.73, Control concentrates at boundaries |
| 6 | Control Transport | 76-87% of Control is moving |
| 7 | Counterfactual Measurement | Methodology and experimental results |
