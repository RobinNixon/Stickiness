"""
Mechanism Hypotheses: Why Does Stickiness Enable Control?
=========================================================

We test four hypotheses for the operative mechanism:

H1: Hidden State Hypothesis
   - Stickiness adds hidden state (pending/cooldown)
   - Same visible pattern + different hidden state -> different outcome
   - Prediction: Control requires hidden state variation

H2: Temporal Memory Hypothesis
   - Stickiness adds memory of past states
   - Breaks Markovian determinism
   - Prediction: Control scales with memory depth

H3: Symmetry Breaking Hypothesis
   - Stickiness creates write/erase asymmetry
   - Time-reversal asymmetry enables irreversibility
   - Prediction: Symmetric stickiness has less Control than asymmetric

H4: Phase Space Expansion Hypothesis
   - More effective states = more trajectories to diverge
   - Prediction: Control scales with effective state count

Experimental Design:
- Each hypothesis gets specific tests designed to isolate its mechanism
- We look for necessary/sufficient conditions
- We measure correlations between mechanism properties and Control
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output/mechanism", exist_ok=True)


def apply_eca_rule(left: int, center: int, right: int, rule: int) -> int:
    """Apply ECA rule to neighborhood."""
    index = int(left) << 2 | int(center) << 1 | int(right)
    return (rule >> index) & 1


# =============================================================================
# EXPERIMENT 1: Hidden State Hypothesis (H1)
# =============================================================================

def test_h1_hidden_state():
    """
    H1: Hidden State Hypothesis

    Test: Is hidden state NECESSARY and SUFFICIENT for Control?

    Experiments:
    1a. Control WITH hidden state (confirmation/refractory) - baseline
    1b. Control WITHOUT hidden state but WITH non-determinism - random tiebreaker
    1c. Hidden state that DOESN'T vary spatially - should have no Control
    1d. Measure: mutual information between hidden state and divergence
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: HIDDEN STATE")
    print("=" * 70)
    print("\nQuestion: Is hidden state necessary and sufficient for Control?")

    results = {}
    width, steps = 60, 80
    test_rules = [30, 54, 90, 110]

    # 1a: Baseline - standard stickiness with hidden state
    print("\n  Test 1a: Stickiness WITH hidden state (baseline)")
    results["1a_with_hidden_state"] = {}

    for rule in test_rules:
        control = measure_control_with_hidden_state(rule, width, steps, "confirmation", 2)
        results["1a_with_hidden_state"][rule] = control
        print(f"    Rule {rule}: Control = {control:.3f}")

    # 1b: Non-determinism WITHOUT hidden state (random perturbation)
    print("\n  Test 1b: Non-determinism WITHOUT hidden state")
    results["1b_random_no_hidden"] = {}

    for rule in test_rules:
        control = measure_control_random_nondeterminism(rule, width, steps, noise_prob=0.05)
        results["1b_random_no_hidden"][rule] = control
        print(f"    Rule {rule}: Control = {control:.3f}")

    # 1c: Hidden state that is UNIFORM (no spatial variation)
    print("\n  Test 1c: Uniform hidden state (no spatial variation)")
    results["1c_uniform_hidden"] = {}

    for rule in test_rules:
        control = measure_control_uniform_hidden_state(rule, width, steps)
        results["1c_uniform_hidden"][rule] = control
        print(f"    Rule {rule}: Control = {control:.3f}")

    # 1d: Mutual information between hidden state and divergence
    print("\n  Test 1d: Mutual information (hidden state <-> divergence)")
    results["1d_mutual_information"] = {}

    for rule in test_rules:
        mi = measure_hidden_state_divergence_mi(rule, width, steps)
        results["1d_mutual_information"][rule] = mi
        print(f"    Rule {rule}: MI = {mi:.3f} bits")

    # Analysis
    print("\n  ANALYSIS:")
    print("  -" * 35)

    with_hidden = np.mean([results["1a_with_hidden_state"][r] for r in test_rules])
    without_hidden = np.mean([results["1b_random_no_hidden"][r] for r in test_rules])
    uniform_hidden = np.mean([results["1c_uniform_hidden"][r] for r in test_rules])
    mean_mi = np.mean([results["1d_mutual_information"][r] for r in test_rules])

    print(f"    Mean Control WITH hidden state: {with_hidden:.3f}")
    print(f"    Mean Control with random noise (no hidden): {without_hidden:.3f}")
    print(f"    Mean Control with uniform hidden state: {uniform_hidden:.3f}")
    print(f"    Mean mutual information: {mean_mi:.3f} bits")

    # Verdict
    hidden_necessary = with_hidden > without_hidden * 1.5
    hidden_sufficient = uniform_hidden < with_hidden * 0.5
    mi_correlates = mean_mi > 0.1

    results["verdict"] = {
        "hidden_state_necessary": hidden_necessary,
        "spatial_variation_necessary": hidden_sufficient,
        "mi_correlates_with_control": mi_correlates
    }

    print(f"\n    Hidden state necessary? {hidden_necessary}")
    print(f"    Spatial variation necessary? {hidden_sufficient}")
    print(f"    MI correlates with Control? {mi_correlates}")

    return results


def measure_control_with_hidden_state(rule: int, width: int, steps: int,
                                       mechanism: str, depth: int) -> float:
    """Measure Control with standard stickiness (has hidden state)."""

    # Run base evolution
    history, hidden_history = run_sticky_eca_with_hidden(rule, width, steps, mechanism, depth)

    # Measure Control via perturbation
    total_divergence = 0
    n_tests = min(20, steps - 10)

    for _ in range(n_tests):
        t = np.random.randint(5, steps - 5)
        pos = np.random.randint(0, width)

        # Get state and hidden state at time t
        state = history[t].copy()
        hidden = hidden_history[t].copy()

        # Perturb visible state
        state[pos] = 1 - state[pos]

        # Continue evolution
        future1, _ = continue_sticky_evolution(rule, state, hidden.copy(), 5, mechanism, depth)

        # Also run unperturbed
        future2, _ = continue_sticky_evolution(rule, history[t].copy(), hidden_history[t].copy(),
                                                5, mechanism, depth)

        divergence = np.mean(future1[-1] != future2[-1])
        total_divergence += divergence

    return total_divergence / n_tests


def measure_control_random_nondeterminism(rule: int, width: int, steps: int,
                                           noise_prob: float) -> float:
    """
    Measure Control with random noise but NO hidden state.

    This tests whether non-determinism alone (without structured hidden state)
    can produce Control.
    """

    def run_noisy_eca(rule, width, steps, noise_prob, seed=None):
        if seed is not None:
            np.random.seed(seed)

        state = np.zeros(width, dtype=np.int8)
        state[width // 2] = 1
        history = [state.copy()]

        for _ in range(steps - 1):
            new_state = np.zeros(width, dtype=np.int8)
            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                output = apply_eca_rule(left, center, right, rule)

                # Add random noise (flip with probability noise_prob)
                if np.random.random() < noise_prob:
                    output = 1 - output

                new_state[i] = output

            state = new_state
            history.append(state.copy())

        return np.array(history)

    # Measure divergence between runs with same seed vs different seeds
    total_divergence = 0
    n_tests = 20

    for test in range(n_tests):
        # Run with same initial seed
        seed1 = test * 1000
        history1 = run_noisy_eca(rule, width, steps, noise_prob, seed=seed1)

        # Perturb at some point
        t = steps // 2
        pos = np.random.randint(0, width)

        perturbed_state = history1[t].copy()
        perturbed_state[pos] = 1 - perturbed_state[pos]

        # Continue both from t (with fresh randomness)
        np.random.seed(seed1 + 500)  # Same noise sequence for both

        # Unperturbed continuation
        state1 = history1[t].copy()
        for _ in range(10):
            new_state = np.zeros(width, dtype=np.int8)
            for i in range(width):
                left = state1[(i - 1) % width]
                center = state1[i]
                right = state1[(i + 1) % width]
                output = apply_eca_rule(left, center, right, rule)
                if np.random.random() < noise_prob:
                    output = 1 - output
                new_state[i] = output
            state1 = new_state

        np.random.seed(seed1 + 500)  # Reset to same noise

        # Perturbed continuation
        state2 = perturbed_state
        for _ in range(10):
            new_state = np.zeros(width, dtype=np.int8)
            for i in range(width):
                left = state2[(i - 1) % width]
                center = state2[i]
                right = state2[(i + 1) % width]
                output = apply_eca_rule(left, center, right, rule)
                if np.random.random() < noise_prob:
                    output = 1 - output
                new_state[i] = output
            state2 = new_state

        divergence = np.mean(state1 != state2)
        total_divergence += divergence

    return total_divergence / n_tests


def measure_control_uniform_hidden_state(rule: int, width: int, steps: int) -> float:
    """
    Measure Control when hidden state is UNIFORM across space.

    If hidden state variation is necessary, uniform hidden state should give no Control.
    We implement this by making ALL cells have the same pending/cooldown value.
    """

    def run_uniform_sticky(rule, width, steps):
        state = np.zeros(width, dtype=np.int8)
        state[width // 2] = 1

        # Uniform hidden state: all cells have same pending status
        global_pending = 0
        global_count = 0
        depth = 2

        history = [state.copy()]

        for _ in range(steps - 1):
            new_state = state.copy()

            # Check if ANY cell wants to change
            any_wants_change = False
            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]
                rule_output = apply_eca_rule(left, center, right, rule)
                if rule_output != center:
                    any_wants_change = True
                    break

            # Update global pending state
            if any_wants_change:
                if global_pending == 1:
                    global_count += 1
                    if global_count >= depth:
                        # Apply changes to ALL cells that want to change
                        for i in range(width):
                            left = state[(i - 1) % width]
                            center = state[i]
                            right = state[(i + 1) % width]
                            rule_output = apply_eca_rule(left, center, right, rule)
                            if rule_output != center:
                                new_state[i] = rule_output
                        global_pending = 0
                        global_count = 0
                else:
                    global_pending = 1
                    global_count = 1
            else:
                global_pending = 0
                global_count = 0

            state = new_state
            history.append(state.copy())

        return np.array(history)

    # Measure Control
    history = run_uniform_sticky(rule, width, steps)

    total_divergence = 0
    n_tests = 20

    for _ in range(n_tests):
        t = np.random.randint(10, steps - 10)
        pos = np.random.randint(0, width)

        # This is tricky - with uniform hidden state, perturbation effect is deterministic
        # We measure if the perturbation spreads

        state1 = history[t].copy()
        state2 = history[t].copy()
        state2[pos] = 1 - state2[pos]

        # Run both forward (standard ECA since hidden state is uniform)
        for _ in range(10):
            new1 = np.zeros(width, dtype=np.int8)
            new2 = np.zeros(width, dtype=np.int8)
            for i in range(width):
                new1[i] = apply_eca_rule(int(state1[(i-1)%width]), int(state1[i]), int(state1[(i+1)%width]), rule)
                new2[i] = apply_eca_rule(int(state2[(i-1)%width]), int(state2[i]), int(state2[(i+1)%width]), rule)
            state1, state2 = new1, new2

        divergence = np.mean(state1 != state2)
        total_divergence += divergence

    return total_divergence / n_tests


def measure_hidden_state_divergence_mi(rule: int, width: int, steps: int) -> float:
    """
    Measure mutual information between hidden state and divergence outcome.

    If hidden state causes divergence, MI(hidden_state, divergence) should be high.
    """

    # Collect samples: (hidden_state_at_position, did_diverge)
    samples = []

    history, hidden_history = run_sticky_eca_with_hidden(rule, width, steps, "confirmation", 2)

    for _ in range(100):
        t = np.random.randint(10, steps - 10)
        pos = np.random.randint(0, width)

        # Get hidden state at this position
        hidden_val = hidden_history[t, pos]

        # Perturb and measure divergence
        state = history[t].copy()
        hidden = hidden_history[t].copy()
        state[pos] = 1 - state[pos]

        future1, _ = continue_sticky_evolution(rule, state, hidden.copy(), 5, "confirmation", 2)
        future2, _ = continue_sticky_evolution(rule, history[t].copy(), hidden_history[t].copy(),
                                                5, "confirmation", 2)

        diverged = int(np.sum(future1[-1] != future2[-1]) > 0)

        samples.append((hidden_val, diverged))

    # Compute mutual information
    # I(X;Y) = H(Y) - H(Y|X)

    samples = np.array(samples)
    hidden_vals = samples[:, 0]
    diverged = samples[:, 1]

    # H(Y) - entropy of divergence
    p_div = np.mean(diverged)
    if p_div == 0 or p_div == 1:
        h_y = 0
    else:
        h_y = -p_div * np.log2(p_div) - (1-p_div) * np.log2(1-p_div)

    # H(Y|X) - conditional entropy
    unique_hidden = np.unique(hidden_vals)
    h_y_given_x = 0

    for h in unique_hidden:
        mask = hidden_vals == h
        p_h = np.mean(mask)

        if p_h > 0:
            p_div_given_h = np.mean(diverged[mask])
            if 0 < p_div_given_h < 1:
                h_y_given_h = -p_div_given_h * np.log2(p_div_given_h) - (1-p_div_given_h) * np.log2(1-p_div_given_h)
            else:
                h_y_given_h = 0
            h_y_given_x += p_h * h_y_given_h

    mi = h_y - h_y_given_x
    return max(0, mi)  # MI should be non-negative


def run_sticky_eca_with_hidden(rule: int, width: int, steps: int,
                                mechanism: str, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run sticky ECA and return both visible and hidden state histories."""

    state = np.zeros(width, dtype=np.int8)
    state[width // 2] = 1

    history = np.zeros((steps, width), dtype=np.int8)
    hidden_history = np.zeros((steps, width), dtype=np.int8)

    history[0] = state

    if mechanism == "confirmation":
        pending = np.zeros(width, dtype=np.int8)
        pending_count = np.zeros(width, dtype=np.int8)
        hidden_history[0] = pending_count

        for t in range(1, steps):
            new_state = state.copy()
            new_pending = pending.copy()
            new_count = pending_count.copy()

            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                rule_output = apply_eca_rule(left, center, right, rule)

                if rule_output != center:
                    if pending[i] == 1:
                        new_count[i] += 1
                        if new_count[i] >= depth:
                            new_state[i] = rule_output
                            new_pending[i] = 0
                            new_count[i] = 0
                    else:
                        new_pending[i] = 1
                        new_count[i] = 1
                else:
                    new_pending[i] = 0
                    new_count[i] = 0

            state = new_state
            pending = new_pending
            pending_count = new_count

            history[t] = state
            hidden_history[t] = pending_count

    return history, hidden_history


def continue_sticky_evolution(rule: int, state: np.ndarray, hidden: np.ndarray,
                               steps: int, mechanism: str, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """Continue sticky evolution from given state and hidden state."""

    width = len(state)
    history = [state.copy()]

    if mechanism == "confirmation":
        pending = (hidden > 0).astype(np.int8)
        pending_count = hidden.copy()

        for _ in range(steps):
            new_state = state.copy()
            new_pending = pending.copy()
            new_count = pending_count.copy()

            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                rule_output = apply_eca_rule(left, center, right, rule)

                if rule_output != center:
                    if pending[i] == 1:
                        new_count[i] += 1
                        if new_count[i] >= depth:
                            new_state[i] = rule_output
                            new_pending[i] = 0
                            new_count[i] = 0
                    else:
                        new_pending[i] = 1
                        new_count[i] = 1
                else:
                    new_pending[i] = 0
                    new_count[i] = 0

            state = new_state
            pending = new_pending
            pending_count = new_count
            history.append(state.copy())

    return np.array(history), pending_count


# =============================================================================
# EXPERIMENT 2: Temporal Memory Hypothesis (H2)
# =============================================================================

def test_h2_temporal_memory():
    """
    H2: Temporal Memory Hypothesis

    Test: Does Control scale with memory depth?

    Experiments:
    2a. Vary confirmation depth (1, 2, 3, 4, 5) - measure Control
    2b. Vary refractory time (1, 2, 3, 4, 5) - measure Control
    2c. Test memory WITHOUT affecting transitions (pure memory, no stickiness)
    2d. Measure temporal autocorrelation of hidden state
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: TEMPORAL MEMORY")
    print("=" * 70)
    print("\nQuestion: Does Control scale with memory depth?")

    results = {}
    width, steps = 60, 80
    test_rules = [30, 54, 90, 110]

    # 2a: Vary confirmation depth
    print("\n  Test 2a: Control vs confirmation depth")
    results["2a_confirmation_depth"] = {}

    for rule in test_rules:
        results["2a_confirmation_depth"][rule] = {}
        controls = []
        for depth in [1, 2, 3, 4, 5]:
            control = measure_control_with_hidden_state(rule, width, steps, "confirmation", depth)
            results["2a_confirmation_depth"][rule][depth] = control
            controls.append(control)
        print(f"    Rule {rule}: depths 1-5 -> Control = {[f'{c:.2f}' for c in controls]}")

    # 2b: Vary refractory time
    print("\n  Test 2b: Control vs refractory time")
    results["2b_refractory_time"] = {}

    for rule in test_rules:
        results["2b_refractory_time"][rule] = {}
        controls = []
        for refract in [1, 2, 3, 4, 5]:
            control = measure_control_refractory(rule, width, steps, refract)
            results["2b_refractory_time"][rule][refract] = control
            controls.append(control)
        print(f"    Rule {rule}: refract 1-5 -> Control = {[f'{c:.2f}' for c in controls]}")

    # 2c: Pure memory without affecting transitions
    print("\n  Test 2c: Memory WITHOUT stickiness effect")
    results["2c_pure_memory"] = {}

    for rule in test_rules:
        control = measure_control_pure_memory(rule, width, steps, memory_depth=3)
        results["2c_pure_memory"][rule] = control
        print(f"    Rule {rule}: Control with pure memory = {control:.3f}")

    # 2d: Temporal autocorrelation of hidden state
    print("\n  Test 2d: Hidden state temporal autocorrelation")
    results["2d_autocorrelation"] = {}

    for rule in test_rules:
        autocorr = measure_hidden_state_autocorrelation(rule, width, steps)
        results["2d_autocorrelation"][rule] = autocorr
        print(f"    Rule {rule}: Hidden state autocorr = {autocorr:.3f}")

    # Analysis
    print("\n  ANALYSIS:")
    print("  -" * 35)

    # Check if Control increases with depth
    monotonic_confirm = 0
    monotonic_refract = 0

    for rule in test_rules:
        depths = [1, 2, 3, 4, 5]
        confirm_controls = [results["2a_confirmation_depth"][rule][d] for d in depths]
        refract_controls = [results["2b_refractory_time"][rule][d] for d in depths]

        # Check monotonicity (allowing for noise)
        if confirm_controls[-1] > confirm_controls[0]:
            monotonic_confirm += 1
        if refract_controls[-1] > refract_controls[0]:
            monotonic_refract += 1

    pure_memory_control = np.mean([results["2c_pure_memory"][r] for r in test_rules])
    mean_autocorr = np.mean([results["2d_autocorrelation"][r] for r in test_rules])

    print(f"    Rules where Control increases with confirm depth: {monotonic_confirm}/4")
    print(f"    Rules where Control increases with refract time: {monotonic_refract}/4")
    print(f"    Mean Control with pure memory (no stickiness): {pure_memory_control:.3f}")
    print(f"    Mean hidden state autocorrelation: {mean_autocorr:.3f}")

    results["verdict"] = {
        "control_scales_with_memory": (monotonic_confirm + monotonic_refract) >= 4,
        "memory_alone_sufficient": pure_memory_control > 0.1,
        "hidden_state_persistent": mean_autocorr > 0.5
    }

    print(f"\n    Control scales with memory? {results['verdict']['control_scales_with_memory']}")
    print(f"    Memory alone sufficient? {results['verdict']['memory_alone_sufficient']}")
    print(f"    Hidden state persistent? {results['verdict']['hidden_state_persistent']}")

    return results


def measure_control_refractory(rule: int, width: int, steps: int, refract_time: int) -> float:
    """Measure Control with refractory mechanism."""

    def run_refractory(rule, width, steps, refract_time, init_state=None):
        if init_state is None:
            state = np.zeros(width, dtype=np.int8)
            state[width // 2] = 1
        else:
            state = init_state.copy()

        cooldown = np.zeros(width, dtype=np.int8)
        history = [state.copy()]

        for _ in range(steps - 1):
            new_state = state.copy()
            new_cooldown = np.maximum(cooldown - 1, 0)

            for i in range(width):
                if cooldown[i] > 0:
                    continue

                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                rule_output = apply_eca_rule(left, center, right, rule)

                if rule_output != center:
                    new_state[i] = rule_output
                    new_cooldown[i] = refract_time

            state = new_state
            cooldown = new_cooldown
            history.append(state.copy())

        return np.array(history)

    # Measure Control
    history = run_refractory(rule, width, steps, refract_time)

    total_divergence = 0
    n_tests = 20

    for _ in range(n_tests):
        t = np.random.randint(10, steps - 15)
        pos = np.random.randint(0, width)

        state1 = history[t].copy()
        state2 = history[t].copy()
        state2[pos] = 1 - state2[pos]

        future1 = run_refractory(rule, width, 10, refract_time, state1)
        future2 = run_refractory(rule, width, 10, refract_time, state2)

        divergence = np.mean(future1[-1] != future2[-1])
        total_divergence += divergence

    return total_divergence / n_tests


def measure_control_pure_memory(rule: int, width: int, steps: int, memory_depth: int) -> float:
    """
    Test memory WITHOUT stickiness effect.

    Cells remember past states but this doesn't affect transitions.
    We add a "memory tag" that varies based on history but rule application is standard.
    Then we test if this memory alone creates Control (it shouldn't, since transitions are deterministic).
    """

    def run_pure_memory(rule, width, steps, memory_depth, init_state=None):
        if init_state is None:
            state = np.zeros(width, dtype=np.int8)
            state[width // 2] = 1
        else:
            state = init_state.copy()

        # Memory stores last `memory_depth` states
        memory = [state.copy() for _ in range(memory_depth)]

        history = [state.copy()]

        for _ in range(steps - 1):
            new_state = np.zeros(width, dtype=np.int8)

            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                # Standard rule application (memory doesn't affect it)
                new_state[i] = apply_eca_rule(left, center, right, rule)

            # Update memory (but it doesn't affect anything)
            memory.pop(0)
            memory.append(state.copy())

            state = new_state
            history.append(state.copy())

        return np.array(history)

    # Measure Control - should be ~0 since transitions are deterministic
    history = run_pure_memory(rule, width, steps, memory_depth)

    total_divergence = 0
    n_tests = 20

    for _ in range(n_tests):
        t = np.random.randint(10, steps - 15)
        pos = np.random.randint(0, width)

        state1 = history[t].copy()
        state2 = history[t].copy()
        state2[pos] = 1 - state2[pos]

        future1 = run_pure_memory(rule, width, 10, memory_depth, state1)
        future2 = run_pure_memory(rule, width, 10, memory_depth, state2)

        divergence = np.mean(future1[-1] != future2[-1])
        total_divergence += divergence

    return total_divergence / n_tests


def measure_hidden_state_autocorrelation(rule: int, width: int, steps: int) -> float:
    """Measure temporal autocorrelation of hidden state."""

    _, hidden_history = run_sticky_eca_with_hidden(rule, width, steps, "confirmation", 2)

    # Flatten and compute lag-1 autocorrelation
    flat = hidden_history.flatten()

    if len(flat) < 2:
        return 0

    mean = np.mean(flat)
    var = np.var(flat)

    if var < 1e-10:
        return 0

    autocorr = np.corrcoef(flat[:-1], flat[1:])[0, 1]

    return autocorr if not np.isnan(autocorr) else 0


# =============================================================================
# EXPERIMENT 3: Symmetry Breaking Hypothesis (H3)
# =============================================================================

def test_h3_symmetry_breaking():
    """
    H3: Symmetry Breaking Hypothesis

    Test: Does asymmetry between write/erase matter?

    Experiments:
    3a. Symmetric stickiness (same resistance for 0->1 and 1->0)
    3b. Asymmetric stickiness (harder to erase than write)
    3c. Reverse asymmetric (harder to write than erase)
    3d. Time-reversal test: run forward and backward, measure asymmetry
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: SYMMETRY BREAKING")
    print("=" * 70)
    print("\nQuestion: Does write/erase asymmetry matter for Control?")

    results = {}
    width, steps = 60, 80
    test_rules = [30, 54, 90, 110]

    # 3a: Symmetric stickiness
    print("\n  Test 3a: Symmetric stickiness (same for 0->1 and 1->0)")
    results["3a_symmetric"] = {}

    for rule in test_rules:
        control = measure_control_symmetric_sticky(rule, width, steps, depth=2)
        results["3a_symmetric"][rule] = control
        print(f"    Rule {rule}: Control = {control:.3f}")

    # 3b: Asymmetric - harder to erase (1->0 needs more confirmation)
    print("\n  Test 3b: Asymmetric - harder to ERASE")
    results["3b_hard_erase"] = {}

    for rule in test_rules:
        control = measure_control_asymmetric_sticky(rule, width, steps,
                                                     write_depth=1, erase_depth=3)
        results["3b_hard_erase"][rule] = control
        print(f"    Rule {rule}: Control = {control:.3f}")

    # 3c: Reverse asymmetric - harder to write
    print("\n  Test 3c: Asymmetric - harder to WRITE")
    results["3c_hard_write"] = {}

    for rule in test_rules:
        control = measure_control_asymmetric_sticky(rule, width, steps,
                                                     write_depth=3, erase_depth=1)
        results["3c_hard_write"][rule] = control
        print(f"    Rule {rule}: Control = {control:.3f}")

    # 3d: Time-reversal asymmetry measure
    print("\n  Test 3d: Time-reversal asymmetry")
    results["3d_time_reversal"] = {}

    for rule in test_rules:
        asymmetry = measure_time_reversal_asymmetry(rule, width, steps)
        results["3d_time_reversal"][rule] = asymmetry
        print(f"    Rule {rule}: Time-reversal asymmetry = {asymmetry:.3f}")

    # Analysis
    print("\n  ANALYSIS:")
    print("  -" * 35)

    mean_symmetric = np.mean([results["3a_symmetric"][r] for r in test_rules])
    mean_hard_erase = np.mean([results["3b_hard_erase"][r] for r in test_rules])
    mean_hard_write = np.mean([results["3c_hard_write"][r] for r in test_rules])
    mean_asymmetry = np.mean([results["3d_time_reversal"][r] for r in test_rules])

    print(f"    Mean Control (symmetric): {mean_symmetric:.3f}")
    print(f"    Mean Control (hard erase): {mean_hard_erase:.3f}")
    print(f"    Mean Control (hard write): {mean_hard_write:.3f}")
    print(f"    Mean time-reversal asymmetry: {mean_asymmetry:.3f}")

    # Determine if asymmetry helps
    asymmetric_better = (mean_hard_erase > mean_symmetric * 1.2) or (mean_hard_write > mean_symmetric * 1.2)
    direction_matters = abs(mean_hard_erase - mean_hard_write) > 0.1

    results["verdict"] = {
        "asymmetry_increases_control": asymmetric_better,
        "direction_matters": direction_matters,
        "best_asymmetry": "hard_erase" if mean_hard_erase > mean_hard_write else "hard_write"
    }

    print(f"\n    Asymmetry increases Control? {asymmetric_better}")
    print(f"    Direction matters? {direction_matters}")
    print(f"    Best asymmetry type: {results['verdict']['best_asymmetry']}")

    return results


def measure_control_symmetric_sticky(rule: int, width: int, steps: int, depth: int) -> float:
    """Symmetric stickiness: same confirmation depth for 0->1 and 1->0."""
    return measure_control_asymmetric_sticky(rule, width, steps, depth, depth)


def measure_control_asymmetric_sticky(rule: int, width: int, steps: int,
                                       write_depth: int, erase_depth: int) -> float:
    """
    Asymmetric stickiness:
    - write_depth: confirmations needed for 0->1
    - erase_depth: confirmations needed for 1->0
    """

    def run_asymmetric(rule, width, steps, write_depth, erase_depth, init_state=None):
        if init_state is None:
            state = np.zeros(width, dtype=np.int8)
            state[width // 2] = 1
        else:
            state = init_state.copy()

        pending = np.zeros(width, dtype=np.int8)
        pending_count = np.zeros(width, dtype=np.int8)

        history = [state.copy()]

        for _ in range(steps - 1):
            new_state = state.copy()
            new_pending = pending.copy()
            new_count = pending_count.copy()

            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                rule_output = apply_eca_rule(left, center, right, rule)

                if rule_output != center:
                    # Determine required depth based on transition type
                    if center == 0 and rule_output == 1:
                        required_depth = write_depth  # 0->1 (write)
                    else:
                        required_depth = erase_depth  # 1->0 (erase)

                    if pending[i] == 1:
                        new_count[i] += 1
                        if new_count[i] >= required_depth:
                            new_state[i] = rule_output
                            new_pending[i] = 0
                            new_count[i] = 0
                    else:
                        new_pending[i] = 1
                        new_count[i] = 1
                else:
                    new_pending[i] = 0
                    new_count[i] = 0

            state = new_state
            pending = new_pending
            pending_count = new_count
            history.append(state.copy())

        return np.array(history)

    # Measure Control
    history = run_asymmetric(rule, width, steps, write_depth, erase_depth)

    total_divergence = 0
    n_tests = 20

    for _ in range(n_tests):
        t = np.random.randint(10, steps - 15)
        pos = np.random.randint(0, width)

        state1 = history[t].copy()
        state2 = history[t].copy()
        state2[pos] = 1 - state2[pos]

        future1 = run_asymmetric(rule, width, 10, write_depth, erase_depth, state1)
        future2 = run_asymmetric(rule, width, 10, write_depth, erase_depth, state2)

        divergence = np.mean(future1[-1] != future2[-1])
        total_divergence += divergence

    return total_divergence / n_tests


def measure_time_reversal_asymmetry(rule: int, width: int, steps: int) -> float:
    """
    Measure time-reversal asymmetry in sticky CA.

    Run forward, then try to run backward. Measure how different the dynamics are.
    """

    # Run forward with stickiness
    history, _ = run_sticky_eca_with_hidden(rule, width, steps, "confirmation", 2)

    # Try to "reverse" by running the inverse rule (if exists) or just measure irreversibility
    # For most rules, there's no true inverse, so we measure entropy increase

    forward_entropy = []
    for t in range(steps):
        p = np.mean(history[t])
        if 0 < p < 1:
            ent = -p * np.log2(p) - (1-p) * np.log2(1-p)
        else:
            ent = 0
        forward_entropy.append(ent)

    # Measure entropy change (should increase for irreversible systems)
    early_entropy = np.mean(forward_entropy[:steps//4])
    late_entropy = np.mean(forward_entropy[-steps//4:])

    asymmetry = late_entropy - early_entropy

    return asymmetry


# =============================================================================
# EXPERIMENT 4: Phase Space Expansion Hypothesis (H4)
# =============================================================================

def test_h4_phase_space():
    """
    H4: Phase Space Expansion Hypothesis

    Test: Does Control scale with effective state count?

    Experiments:
    4a. Count effective states (visible × hidden combinations)
    4b. Correlate effective state count with Control
    4c. Test: artificially expand state space without stickiness
    4d. Test: stickiness with minimal state expansion
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: PHASE SPACE EXPANSION")
    print("=" * 70)
    print("\nQuestion: Does Control scale with effective state count?")

    results = {}
    width, steps = 60, 80
    test_rules = [30, 54, 90, 110]

    # 4a: Count effective states for different stickiness depths
    print("\n  Test 4a: Effective state count vs stickiness depth")
    results["4a_state_counts"] = {}

    for rule in test_rules:
        results["4a_state_counts"][rule] = {}
        for depth in [1, 2, 3, 4]:
            effective_states = count_effective_states(rule, width, steps, depth)
            results["4a_state_counts"][rule][depth] = effective_states
        print(f"    Rule {rule}: depths 1-4 -> states = {[results['4a_state_counts'][rule][d] for d in [1,2,3,4]]}")

    # 4b: Correlate state count with Control
    print("\n  Test 4b: Correlation (state count, Control)")

    state_counts = []
    controls = []

    for rule in test_rules:
        for depth in [1, 2, 3, 4]:
            state_counts.append(results["4a_state_counts"][rule][depth])
            control = measure_control_with_hidden_state(rule, width, steps, "confirmation", depth)
            controls.append(control)

    correlation = np.corrcoef(state_counts, controls)[0, 1]
    results["4b_correlation"] = correlation
    print(f"    Correlation: {correlation:.3f}")

    # 4c: Expand state space WITHOUT stickiness (add dummy states)
    print("\n  Test 4c: State expansion WITHOUT stickiness")
    results["4c_dummy_states"] = {}

    for rule in test_rules:
        control = measure_control_dummy_states(rule, width, steps, n_dummy=4)
        results["4c_dummy_states"][rule] = control
        print(f"    Rule {rule}: Control with dummy states = {control:.3f}")

    # 4d: Minimal state expansion with stickiness
    print("\n  Test 4d: Minimal hidden state (binary pending only)")
    results["4d_minimal_hidden"] = {}

    for rule in test_rules:
        control = measure_control_minimal_hidden(rule, width, steps)
        results["4d_minimal_hidden"][rule] = control
        print(f"    Rule {rule}: Control with minimal hidden = {control:.3f}")

    # Analysis
    print("\n  ANALYSIS:")
    print("  -" * 35)

    mean_dummy = np.mean([results["4c_dummy_states"][r] for r in test_rules])
    mean_minimal = np.mean([results["4d_minimal_hidden"][r] for r in test_rules])

    print(f"    Correlation (state count, Control): {correlation:.3f}")
    print(f"    Mean Control with dummy states (no stickiness): {mean_dummy:.3f}")
    print(f"    Mean Control with minimal hidden state: {mean_minimal:.3f}")

    state_count_matters = correlation > 0.5
    dummy_states_work = mean_dummy > 0.1
    minimal_works = mean_minimal > 0.1

    results["verdict"] = {
        "state_count_correlates": state_count_matters,
        "state_expansion_alone_works": dummy_states_work,
        "minimal_hidden_sufficient": minimal_works
    }

    print(f"\n    State count correlates with Control? {state_count_matters}")
    print(f"    State expansion alone works? {dummy_states_work}")
    print(f"    Minimal hidden state sufficient? {minimal_works}")

    return results


def count_effective_states(rule: int, width: int, steps: int, depth: int) -> int:
    """Count number of distinct (visible, hidden) state combinations observed."""

    history, hidden_history = run_sticky_eca_with_hidden(rule, width, steps, "confirmation", depth)

    # Count unique (visible_pattern, hidden_value) pairs
    seen = set()

    for t in range(steps):
        for i in range(width):
            # Local visible pattern (3 cells)
            left = history[t, (i-1) % width]
            center = history[t, i]
            right = history[t, (i+1) % width]
            visible = (left, center, right)

            # Hidden state at this position
            hidden = hidden_history[t, i]

            seen.add((visible, hidden))

    return len(seen)


def measure_control_dummy_states(rule: int, width: int, steps: int, n_dummy: int) -> float:
    """
    Expand state space with dummy states that don't affect dynamics.

    Each cell has a "color" (0 to n_dummy-1) that cycles but doesn't affect rule application.
    This tests if state space size alone matters.
    """

    def run_with_dummy(rule, width, steps, n_dummy, init_state=None):
        if init_state is None:
            state = np.zeros(width, dtype=np.int8)
            state[width // 2] = 1
        else:
            state = init_state.copy()

        # Dummy state that cycles but doesn't affect dynamics
        dummy = np.zeros(width, dtype=np.int8)

        history = [state.copy()]

        for t in range(steps - 1):
            new_state = np.zeros(width, dtype=np.int8)

            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                # Standard rule (dummy doesn't affect it)
                new_state[i] = apply_eca_rule(left, center, right, rule)

            # Update dummy (cycles)
            dummy = (dummy + 1) % n_dummy

            state = new_state
            history.append(state.copy())

        return np.array(history)

    # Measure Control - should be ~0 since dynamics are deterministic
    history = run_with_dummy(rule, width, steps, n_dummy)

    total_divergence = 0
    n_tests = 20

    for _ in range(n_tests):
        t = np.random.randint(10, steps - 15)
        pos = np.random.randint(0, width)

        state1 = history[t].copy()
        state2 = history[t].copy()
        state2[pos] = 1 - state2[pos]

        future1 = run_with_dummy(rule, width, 10, n_dummy, state1)
        future2 = run_with_dummy(rule, width, 10, n_dummy, state2)

        divergence = np.mean(future1[-1] != future2[-1])
        total_divergence += divergence

    return total_divergence / n_tests


def measure_control_minimal_hidden(rule: int, width: int, steps: int) -> float:
    """
    Test with minimal hidden state: just binary pending (yes/no), no counter.

    This is the smallest possible state expansion (2x states per cell).
    """

    def run_minimal_sticky(rule, width, steps, init_state=None, init_pending=None):
        if init_state is None:
            state = np.zeros(width, dtype=np.int8)
            state[width // 2] = 1
        else:
            state = init_state.copy()

        if init_pending is None:
            pending = np.zeros(width, dtype=np.int8)
        else:
            pending = init_pending.copy()

        history = [state.copy()]

        for _ in range(steps - 1):
            new_state = state.copy()
            new_pending = np.zeros(width, dtype=np.int8)

            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                rule_output = apply_eca_rule(left, center, right, rule)

                if rule_output != center:
                    if pending[i] == 1:
                        # Second request: apply change
                        new_state[i] = rule_output
                        new_pending[i] = 0
                    else:
                        # First request: mark pending
                        new_pending[i] = 1
                else:
                    new_pending[i] = 0

            state = new_state
            pending = new_pending
            history.append(state.copy())

        return np.array(history), pending

    # Measure Control
    history, _ = run_minimal_sticky(rule, width, steps)

    total_divergence = 0
    n_tests = 20

    for _ in range(n_tests):
        t = np.random.randint(10, steps - 15)
        pos = np.random.randint(0, width)

        state1 = history[t].copy()
        state2 = history[t].copy()
        state2[pos] = 1 - state2[pos]

        future1, _ = run_minimal_sticky(rule, width, 10, state1)
        future2, _ = run_minimal_sticky(rule, width, 10, state2)

        divergence = np.mean(future1[-1] != future2[-1])
        total_divergence += divergence

    return total_divergence / n_tests


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_mechanism_visualizations(h1_results: Dict, h2_results: Dict,
                                       h3_results: Dict, h4_results: Dict):
    """Generate visualizations for all hypothesis tests."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    test_rules = [30, 54, 90, 110]

    # H1: Hidden State
    ax1 = axes[0, 0]
    categories = ["With Hidden\nState", "Random\n(no hidden)", "Uniform\nHidden"]
    for i, rule in enumerate(test_rules):
        values = [
            h1_results["1a_with_hidden_state"][rule],
            h1_results["1b_random_no_hidden"][rule],
            h1_results["1c_uniform_hidden"][rule]
        ]
        x = np.arange(len(categories)) + i * 0.2 - 0.3
        ax1.bar(x, values, 0.18, label=f"Rule {rule}")
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("Control")
    ax1.set_title("H1: Hidden State Hypothesis")
    ax1.legend()

    # H2: Temporal Memory
    ax2 = axes[0, 1]
    depths = [1, 2, 3, 4, 5]
    for rule in test_rules:
        controls = [h2_results["2a_confirmation_depth"][rule][d] for d in depths]
        ax2.plot(depths, controls, 'o-', label=f"Rule {rule}")
    ax2.set_xlabel("Confirmation Depth")
    ax2.set_ylabel("Control")
    ax2.set_title("H2: Temporal Memory Hypothesis")
    ax2.legend()

    # H3: Symmetry Breaking
    ax3 = axes[1, 0]
    categories = ["Symmetric", "Hard Erase", "Hard Write"]
    for i, rule in enumerate(test_rules):
        values = [
            h3_results["3a_symmetric"][rule],
            h3_results["3b_hard_erase"][rule],
            h3_results["3c_hard_write"][rule]
        ]
        x = np.arange(len(categories)) + i * 0.2 - 0.3
        ax3.bar(x, values, 0.18, label=f"Rule {rule}")
    ax3.set_xticks(np.arange(len(categories)))
    ax3.set_xticklabels(categories)
    ax3.set_ylabel("Control")
    ax3.set_title("H3: Symmetry Breaking Hypothesis")
    ax3.legend()

    # H4: Phase Space
    ax4 = axes[1, 1]
    # Plot state count vs control
    state_counts = []
    controls = []
    rule_labels = []

    for rule in test_rules:
        for depth in [1, 2, 3, 4]:
            state_counts.append(h4_results["4a_state_counts"][rule][depth])
            # Recompute control for scatter
            ctrl = measure_control_with_hidden_state(rule, 60, 80, "confirmation", depth)
            controls.append(ctrl)
            rule_labels.append(rule)

    colors = {30: 'blue', 54: 'green', 90: 'red', 110: 'purple'}
    for sc, ctrl, rule in zip(state_counts, controls, rule_labels):
        ax4.scatter(sc, ctrl, c=colors[rule], alpha=0.7, s=50)

    # Add legend
    for rule in test_rules:
        ax4.scatter([], [], c=colors[rule], label=f"Rule {rule}")

    ax4.set_xlabel("Effective State Count")
    ax4.set_ylabel("Control")
    ax4.set_title(f"H4: Phase Space Hypothesis (r={h4_results['4b_correlation']:.2f})")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("output/mechanism/hypothesis_tests.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved visualization to output/mechanism/hypothesis_tests.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all mechanism hypothesis tests."""

    print("=" * 70)
    print("MECHANISM HYPOTHESES: Why Does Stickiness Enable Control?")
    print("=" * 70)

    # Run all hypothesis tests
    h1_results = test_h1_hidden_state()
    h2_results = test_h2_temporal_memory()
    h3_results = test_h3_symmetry_breaking()
    h4_results = test_h4_phase_space()

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    generate_mechanism_visualizations(h1_results, h2_results, h3_results, h4_results)

    # Summary
    print("\n" + "=" * 70)
    print("MECHANISM HYPOTHESIS SUMMARY")
    print("=" * 70)

    print("\n  H1 (Hidden State):")
    print(f"    - Hidden state necessary: {h1_results['verdict']['hidden_state_necessary']}")
    print(f"    - Spatial variation necessary: {h1_results['verdict']['spatial_variation_necessary']}")
    print(f"    - MI correlates with Control: {h1_results['verdict']['mi_correlates_with_control']}")

    print("\n  H2 (Temporal Memory):")
    print(f"    - Control scales with memory: {h2_results['verdict']['control_scales_with_memory']}")
    print(f"    - Memory alone sufficient: {h2_results['verdict']['memory_alone_sufficient']}")
    print(f"    - Hidden state persistent: {h2_results['verdict']['hidden_state_persistent']}")

    print("\n  H3 (Symmetry Breaking):")
    print(f"    - Asymmetry increases Control: {h3_results['verdict']['asymmetry_increases_control']}")
    print(f"    - Direction matters: {h3_results['verdict']['direction_matters']}")
    print(f"    - Best asymmetry: {h3_results['verdict']['best_asymmetry']}")

    print("\n  H4 (Phase Space):")
    print(f"    - State count correlates: {h4_results['verdict']['state_count_correlates']}")
    print(f"    - State expansion alone works: {h4_results['verdict']['state_expansion_alone_works']}")
    print(f"    - Minimal hidden sufficient: {h4_results['verdict']['minimal_hidden_sufficient']}")

    # Determine operative mechanism
    print("\n" + "=" * 70)
    print("OPERATIVE MECHANISM IDENTIFICATION")
    print("=" * 70)

    scores = {
        "H1 (Hidden State)": sum([
            h1_results['verdict']['hidden_state_necessary'],
            h1_results['verdict']['spatial_variation_necessary'],
            h1_results['verdict']['mi_correlates_with_control']
        ]),
        "H2 (Temporal Memory)": sum([
            h2_results['verdict']['control_scales_with_memory'],
            not h2_results['verdict']['memory_alone_sufficient'],  # Memory alone NOT sufficient is evidence
            h2_results['verdict']['hidden_state_persistent']
        ]),
        "H3 (Symmetry Breaking)": sum([
            h3_results['verdict']['asymmetry_increases_control'],
            h3_results['verdict']['direction_matters']
        ]),
        "H4 (Phase Space)": sum([
            h4_results['verdict']['state_count_correlates'],
            not h4_results['verdict']['state_expansion_alone_works'],  # Expansion alone NOT working is evidence
            h4_results['verdict']['minimal_hidden_sufficient']
        ])
    }

    print("\n  Evidence scores (higher = more support):")
    for hyp, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {hyp}: {score}/3")

    best_hypothesis = max(scores, key=scores.get)
    print(f"\n  BEST SUPPORTED HYPOTHESIS: {best_hypothesis}")

    # Save results
    all_results = {
        "h1_hidden_state": h1_results,
        "h2_temporal_memory": h2_results,
        "h3_symmetry_breaking": h3_results,
        "h4_phase_space": h4_results,
        "scores": scores,
        "best_hypothesis": best_hypothesis
    }

    # Convert any numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    all_results = convert_numpy(all_results)

    with open("output/mechanism/hypothesis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n  Results saved to output/mechanism/hypothesis_results.json")

    print("\n" + "=" * 70)
    print("MECHANISM HYPOTHESIS TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
