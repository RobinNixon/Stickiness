"""
True Control Mechanism Test
===========================

INSIGHT: Our perturbation-based "Control" metric was measuring the WRONG thing.

TWO DIFFERENT CONCEPTS:
1. Perturbation spreading (Lyapunov): How fast do perturbations propagate?
   - All chaotic ECAs have this
   - Stickiness DAMPENS this (cells resist change)

2. Context-dependence (TRUE Control): Same visible pattern -> different outcomes?
   - This is what we mean by "Control"
   - Requires hidden state that VARIES across the system
   - Stickiness CREATES this

This experiment measures TRUE Control: Given the same visible neighborhood,
how often does the outcome differ based on the hidden state?
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

os.makedirs("output/mechanism", exist_ok=True)


def apply_eca_rule(left: int, center: int, right: int, rule: int) -> int:
    """Apply ECA rule to neighborhood."""
    index = int(left) << 2 | int(center) << 1 | int(right)
    return (rule >> index) & 1


def run_sticky_eca_full(rule: int, width: int, steps: int,
                         mechanism: str, depth: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Run sticky ECA and record:
    - Visible history
    - Hidden state history
    - (visible_pattern, hidden_state, outcome) tuples for analysis
    """
    state = np.zeros(width, dtype=np.int8)
    state[width // 2] = 1

    visible_history = np.zeros((steps, width), dtype=np.int8)
    hidden_history = np.zeros((steps, width), dtype=np.int8)
    transitions = []  # (visible_pattern, hidden_state, outcome)

    visible_history[0] = state

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

                visible_pattern = (int(left), int(center), int(right))
                hidden_val = int(pending_count[i])

                rule_output = apply_eca_rule(left, center, right, rule)

                # Determine actual outcome with stickiness
                if rule_output != center:
                    if pending[i] == 1:
                        new_count[i] += 1
                        if new_count[i] >= depth:
                            actual_outcome = rule_output
                            new_pending[i] = 0
                            new_count[i] = 0
                        else:
                            actual_outcome = center  # blocked
                    else:
                        new_pending[i] = 1
                        new_count[i] = 1
                        actual_outcome = center  # blocked
                else:
                    new_pending[i] = 0
                    new_count[i] = 0
                    actual_outcome = center

                new_state[i] = actual_outcome

                # Record transition
                transitions.append((visible_pattern, hidden_val, int(actual_outcome)))

            state = new_state
            pending = new_pending
            pending_count = new_count

            visible_history[t] = state
            hidden_history[t] = pending_count

    elif mechanism == "refractory":
        cooldown = np.zeros(width, dtype=np.int8)
        hidden_history[0] = cooldown

        for t in range(1, steps):
            new_state = state.copy()
            new_cooldown = np.maximum(cooldown - 1, 0)

            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                visible_pattern = (int(left), int(center), int(right))
                hidden_val = int(cooldown[i])

                if cooldown[i] > 0:
                    actual_outcome = center  # blocked by cooldown
                else:
                    rule_output = apply_eca_rule(left, center, right, rule)
                    if rule_output != center:
                        actual_outcome = rule_output
                        new_cooldown[i] = depth
                    else:
                        actual_outcome = center

                new_state[i] = actual_outcome
                transitions.append((visible_pattern, hidden_val, int(actual_outcome)))

            state = new_state
            cooldown = new_cooldown

            visible_history[t] = state
            hidden_history[t] = cooldown

    else:  # standard
        for t in range(1, steps):
            new_state = np.zeros(width, dtype=np.int8)
            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]

                visible_pattern = (int(left), int(center), int(right))
                rule_output = apply_eca_rule(left, center, right, rule)

                transitions.append((visible_pattern, 0, int(rule_output)))
                new_state[i] = rule_output

            state = new_state
            visible_history[t] = state

    return visible_history, hidden_history, transitions


def measure_true_control(transitions: list) -> Dict:
    """
    Measure TRUE Control: For each visible pattern, how much does the outcome vary?

    TRUE Control = When the same visible pattern produces different outcomes
    based on hidden state.

    Returns:
    - context_dependence: fraction of patterns where outcome depends on hidden state
    - outcome_entropy: average entropy of outcomes given visible pattern
    - hidden_influence: correlation between hidden state and outcome
    """

    # Group by visible pattern
    pattern_outcomes = defaultdict(lambda: defaultdict(list))

    for visible, hidden, outcome in transitions:
        pattern_outcomes[visible][hidden].append(outcome)

    # Analyze each pattern
    patterns_with_variation = 0
    total_patterns = 0
    total_entropy = 0

    for visible_pattern, hidden_outcomes in pattern_outcomes.items():
        total_patterns += 1

        # Get all outcomes for this visible pattern
        all_outcomes = []
        for hidden_val, outcomes in hidden_outcomes.items():
            all_outcomes.extend(outcomes)

        if not all_outcomes:
            continue

        # Check if outcomes vary
        unique_outcomes = set(all_outcomes)
        if len(unique_outcomes) > 1:
            patterns_with_variation += 1

        # Compute entropy
        p0 = all_outcomes.count(0) / len(all_outcomes)
        p1 = all_outcomes.count(1) / len(all_outcomes)
        if 0 < p0 < 1:
            entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
        else:
            entropy = 0
        total_entropy += entropy

    # Check if hidden state predicts outcome variation
    hidden_outcome_pairs = []
    for visible_pattern, hidden_outcomes in pattern_outcomes.items():
        for hidden_val, outcomes in hidden_outcomes.items():
            for outcome in outcomes:
                hidden_outcome_pairs.append((hidden_val, outcome))

    if len(hidden_outcome_pairs) > 10:
        hidden_vals = np.array([p[0] for p in hidden_outcome_pairs])
        outcomes = np.array([p[1] for p in hidden_outcome_pairs])

        # Correlation (handle constant arrays)
        if np.std(hidden_vals) > 0 and np.std(outcomes) > 0:
            correlation = np.corrcoef(hidden_vals, outcomes)[0, 1]
        else:
            correlation = 0
    else:
        correlation = 0

    return {
        "context_dependence": patterns_with_variation / max(1, total_patterns),
        "outcome_entropy": total_entropy / max(1, total_patterns),
        "hidden_outcome_correlation": abs(correlation) if not np.isnan(correlation) else 0,
        "patterns_with_variation": patterns_with_variation,
        "total_patterns": total_patterns
    }


def measure_counterfactual_control(rule: int, width: int, steps: int,
                                    mechanism: str, depth: int) -> float:
    """
    Measure Control via counterfactuals:

    For the SAME visible state, run with DIFFERENT hidden states.
    If outcomes differ, that's TRUE Control.
    """

    # Run the system to get typical states
    visible_history, hidden_history, _ = run_sticky_eca_full(rule, width, steps, mechanism, depth)

    if mechanism == "standard":
        return 0.0  # Standard ECAs have no hidden state

    counterfactual_differences = 0
    total_tests = 0

    # Sample random positions and times
    for _ in range(100):
        t = np.random.randint(10, steps - 5)
        i = np.random.randint(0, width)

        # Get visible state at this point
        visible_state = visible_history[t].copy()

        # Try two different hidden states
        if mechanism == "confirmation":
            # Hidden state 1: no pending
            hidden1 = np.zeros(width, dtype=np.int8)
            # Hidden state 2: pending at this position
            hidden2 = np.zeros(width, dtype=np.int8)
            hidden2[i] = depth - 1  # About to trigger

            # Run one step from each
            outcome1 = run_one_step_confirmation(rule, visible_state, hidden1, depth)[i]
            outcome2 = run_one_step_confirmation(rule, visible_state, hidden2, depth)[i]

        elif mechanism == "refractory":
            # Hidden state 1: no cooldown
            hidden1 = np.zeros(width, dtype=np.int8)
            # Hidden state 2: in cooldown
            hidden2 = np.zeros(width, dtype=np.int8)
            hidden2[i] = depth

            outcome1 = run_one_step_refractory(rule, visible_state, hidden1, depth)[i]
            outcome2 = run_one_step_refractory(rule, visible_state, hidden2, depth)[i]

        else:
            continue

        if outcome1 != outcome2:
            counterfactual_differences += 1
        total_tests += 1

    return counterfactual_differences / max(1, total_tests)


def run_one_step_confirmation(rule: int, state: np.ndarray, pending_count: np.ndarray,
                               depth: int) -> np.ndarray:
    """Run one step with confirmation mechanism."""
    width = len(state)
    new_state = state.copy()
    pending = (pending_count > 0).astype(np.int8)

    for i in range(width):
        left = state[(i - 1) % width]
        center = state[i]
        right = state[(i + 1) % width]

        rule_output = apply_eca_rule(left, center, right, rule)

        if rule_output != center:
            if pending[i] == 1:
                if pending_count[i] + 1 >= depth:
                    new_state[i] = rule_output
            # else: blocked, stays same
        # else: no change requested

    return new_state


def run_one_step_refractory(rule: int, state: np.ndarray, cooldown: np.ndarray,
                             depth: int) -> np.ndarray:
    """Run one step with refractory mechanism."""
    width = len(state)
    new_state = state.copy()

    for i in range(width):
        if cooldown[i] > 0:
            continue  # blocked by cooldown

        left = state[(i - 1) % width]
        center = state[i]
        right = state[(i + 1) % width]

        rule_output = apply_eca_rule(left, center, right, rule)
        new_state[i] = rule_output

    return new_state


def test_true_control_mechanism():
    """
    Test the TRUE mechanism: Does hidden state create context-dependence?
    """
    print("=" * 70)
    print("TRUE CONTROL MECHANISM TEST")
    print("=" * 70)
    print("\nMeasuring ACTUAL context-dependence (same visible -> different outcomes)")

    test_rules = [30, 54, 90, 110]
    width, steps = 80, 100

    results = {}

    # Test 1: Standard ECA (no hidden state)
    print("\n" + "-" * 50)
    print("Test 1: Standard ECA (no hidden state)")
    print("-" * 50)

    results["standard"] = {}
    for rule in test_rules:
        _, _, transitions = run_sticky_eca_full(rule, width, steps, "standard", 0)
        metrics = measure_true_control(transitions)
        counterfactual = measure_counterfactual_control(rule, width, steps, "standard", 0)

        results["standard"][rule] = {
            "context_dependence": metrics["context_dependence"],
            "counterfactual_control": counterfactual
        }

        print(f"  Rule {rule}: context_dep={metrics['context_dependence']:.3f}, "
              f"counterfactual={counterfactual:.3f}")

    # Test 2: Confirmation mechanism
    print("\n" + "-" * 50)
    print("Test 2: Confirmation mechanism (has hidden state)")
    print("-" * 50)

    results["confirmation"] = {}
    for rule in test_rules:
        results["confirmation"][rule] = {}
        for depth in [1, 2, 3]:
            _, _, transitions = run_sticky_eca_full(rule, width, steps, "confirmation", depth)
            metrics = measure_true_control(transitions)
            counterfactual = measure_counterfactual_control(rule, width, steps, "confirmation", depth)

            results["confirmation"][rule][depth] = {
                "context_dependence": metrics["context_dependence"],
                "counterfactual_control": counterfactual,
                "hidden_correlation": metrics["hidden_outcome_correlation"]
            }

            print(f"  Rule {rule}, depth {depth}: context_dep={metrics['context_dependence']:.3f}, "
                  f"counterfactual={counterfactual:.3f}, hidden_corr={metrics['hidden_outcome_correlation']:.3f}")

    # Test 3: Refractory mechanism
    print("\n" + "-" * 50)
    print("Test 3: Refractory mechanism (has hidden state)")
    print("-" * 50)

    results["refractory"] = {}
    for rule in test_rules:
        results["refractory"][rule] = {}
        for depth in [1, 2, 3]:
            _, _, transitions = run_sticky_eca_full(rule, width, steps, "refractory", depth)
            metrics = measure_true_control(transitions)
            counterfactual = measure_counterfactual_control(rule, width, steps, "refractory", depth)

            results["refractory"][rule][depth] = {
                "context_dependence": metrics["context_dependence"],
                "counterfactual_control": counterfactual,
                "hidden_correlation": metrics["hidden_outcome_correlation"]
            }

            print(f"  Rule {rule}, depth {depth}: context_dep={metrics['context_dependence']:.3f}, "
                  f"counterfactual={counterfactual:.3f}, hidden_corr={metrics['hidden_outcome_correlation']:.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: What provides TRUE Control?")
    print("=" * 70)

    # Compare standard vs sticky
    std_context = np.mean([results["standard"][r]["context_dependence"] for r in test_rules])
    std_counterfactual = np.mean([results["standard"][r]["counterfactual_control"] for r in test_rules])

    confirm_context = np.mean([results["confirmation"][r][2]["context_dependence"] for r in test_rules])
    confirm_counterfactual = np.mean([results["confirmation"][r][2]["counterfactual_control"] for r in test_rules])
    confirm_hidden_corr = np.mean([results["confirmation"][r][2]["hidden_correlation"] for r in test_rules])

    refract_context = np.mean([results["refractory"][r][2]["context_dependence"] for r in test_rules])
    refract_counterfactual = np.mean([results["refractory"][r][2]["counterfactual_control"] for r in test_rules])
    refract_hidden_corr = np.mean([results["refractory"][r][2]["hidden_correlation"] for r in test_rules])

    print("\n  SUMMARY (mean across rules):")
    print("  " + "-" * 50)
    print(f"  Standard ECA:")
    print(f"    - Context dependence: {std_context:.3f}")
    print(f"    - Counterfactual control: {std_counterfactual:.3f}")

    print(f"\n  Confirmation (depth=2):")
    print(f"    - Context dependence: {confirm_context:.3f}")
    print(f"    - Counterfactual control: {confirm_counterfactual:.3f}")
    print(f"    - Hidden-outcome correlation: {confirm_hidden_corr:.3f}")

    print(f"\n  Refractory (depth=2):")
    print(f"    - Context dependence: {refract_context:.3f}")
    print(f"    - Counterfactual control: {refract_counterfactual:.3f}")
    print(f"    - Hidden-outcome correlation: {refract_hidden_corr:.3f}")

    # Determine mechanism
    print("\n  VERDICT:")
    print("  " + "-" * 50)

    hidden_provides_control = (confirm_counterfactual > std_counterfactual * 2) or \
                               (refract_counterfactual > std_counterfactual * 2)

    if hidden_provides_control:
        print("  => Hidden state DOES provide TRUE Control")
        print(f"     Counterfactual increase: {max(confirm_counterfactual, refract_counterfactual) / max(0.001, std_counterfactual):.1f}x")

        # What aspect of hidden state?
        if confirm_hidden_corr > 0.1 or refract_hidden_corr > 0.1:
            print("  => Hidden state VALUE correlates with outcome")
        if confirm_context > std_context or refract_context > std_context:
            print("  => Hidden state creates outcome VARIATION")
    else:
        print("  => Hidden state does NOT provide TRUE Control")
        print("     (Need to investigate other mechanisms)")

    results["verdict"] = {
        "hidden_provides_control": hidden_provides_control,
        "standard_counterfactual": std_counterfactual,
        "confirmation_counterfactual": confirm_counterfactual,
        "refractory_counterfactual": refract_counterfactual,
        "control_increase_factor": max(confirm_counterfactual, refract_counterfactual) / max(0.001, std_counterfactual)
    }

    # Visualize
    visualize_true_control(results, test_rules)

    # Save
    def convert_types(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    with open("output/mechanism/true_control_results.json", "w") as f:
        json.dump(convert_types(results), f, indent=2)

    print("\n  Results saved to output/mechanism/true_control_results.json")

    return results


def visualize_true_control(results: Dict, test_rules: List[int]):
    """Visualize TRUE Control results."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Counterfactual Control by mechanism
    ax1 = axes[0]
    mechanisms = ["Standard", "Confirm d=1", "Confirm d=2", "Confirm d=3",
                  "Refract d=1", "Refract d=2", "Refract d=3"]

    for i, rule in enumerate(test_rules):
        values = [
            results["standard"][rule]["counterfactual_control"],
            results["confirmation"][rule][1]["counterfactual_control"],
            results["confirmation"][rule][2]["counterfactual_control"],
            results["confirmation"][rule][3]["counterfactual_control"],
            results["refractory"][rule][1]["counterfactual_control"],
            results["refractory"][rule][2]["counterfactual_control"],
            results["refractory"][rule][3]["counterfactual_control"],
        ]
        x = np.arange(len(mechanisms)) + i * 0.15 - 0.22
        ax1.bar(x, values, 0.12, label=f"Rule {rule}")

    ax1.set_xticks(np.arange(len(mechanisms)))
    ax1.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax1.set_ylabel("Counterfactual Control")
    ax1.set_title("TRUE Control: Same visible, different hidden -> different outcome")
    ax1.legend()

    # Plot 2: Hidden-outcome correlation
    ax2 = axes[1]
    for rule in test_rules:
        depths = [1, 2, 3]
        corrs = [results["confirmation"][rule][d]["hidden_correlation"] for d in depths]
        ax2.plot(depths, corrs, 'o-', label=f"Rule {rule}")
    ax2.set_xlabel("Confirmation Depth")
    ax2.set_ylabel("Hidden-Outcome Correlation")
    ax2.set_title("Does hidden state VALUE predict outcome?")
    ax2.legend()

    # Plot 3: Standard vs Sticky comparison
    ax3 = axes[2]
    std_ctrl = [results["standard"][r]["counterfactual_control"] for r in test_rules]
    sticky_ctrl = [results["confirmation"][r][2]["counterfactual_control"] for r in test_rules]

    x = np.arange(len(test_rules))
    ax3.bar(x - 0.2, std_ctrl, 0.35, label='Standard (no hidden)', color='lightgray')
    ax3.bar(x + 0.2, sticky_ctrl, 0.35, label='Sticky (hidden state)', color='steelblue')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Rule {r}" for r in test_rules])
    ax3.set_ylabel("Counterfactual Control")
    ax3.set_title("Hidden State Creates TRUE Control")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("output/mechanism/true_control.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved visualization to output/mechanism/true_control.png")


if __name__ == "__main__":
    test_true_control_mechanism()
