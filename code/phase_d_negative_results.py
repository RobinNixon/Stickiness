"""
Phase D: Negative Results
=========================
Goal: Identify limits of boundary-Control theory through falsification attempts.

Part 1: Bulk Control Falsification
- Search systematically for Control far from boundaries
- Test whether any rule/mechanism produces "interior Control"

Part 2: Stickiness Extremes
- No stickiness limit (does boundary-Control relationship hold at zero?)
- Infinite stickiness limit (what happens when everything sticks?)

Part 3: Counter-Example Search
- Actively try to break the theory
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Create output directory
os.makedirs("output/phase_d", exist_ok=True)


def apply_eca_rule(left: int, center: int, right: int, rule: int) -> int:
    """Apply ECA rule to neighborhood."""
    index = int(left) << 2 | int(center) << 1 | int(right)
    return (rule >> index) & 1


def run_sticky_eca(rule: int, width: int, steps: int,
                   mechanism: str = "confirmation", depth: int = 1,
                   init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run sticky ECA with given mechanism."""

    if init is None:
        state = np.zeros(width, dtype=np.int8)
        state[width // 2] = 1
    else:
        state = init.copy()

    history = np.zeros((steps, width), dtype=np.int8)
    history[0] = state

    if mechanism == "confirmation":
        pending = np.zeros(width, dtype=np.int8)
        pending_count = np.zeros(width, dtype=np.int8)

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

    elif mechanism == "refractory":
        cooldown = np.zeros(width, dtype=np.int8)

        for t in range(1, steps):
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
                    new_cooldown[i] = depth

            state = new_state
            cooldown = new_cooldown
            history[t] = state

    else:  # standard
        for t in range(1, steps):
            new_state = np.zeros(width, dtype=np.int8)
            for i in range(width):
                left = state[(i - 1) % width]
                center = state[i]
                right = state[(i + 1) % width]
                new_state[i] = apply_eca_rule(left, center, right, rule)
            state = new_state
            history[t] = state

    return history, state


def compute_control_map(rule: int, width: int, steps: int,
                        mechanism: str = "confirmation", depth: int = 1) -> np.ndarray:
    """Compute Control at each position by perturbing and measuring divergence."""

    # Run base evolution
    base_history, _ = run_sticky_eca(rule, width, steps, mechanism, depth)

    control_map = np.zeros((steps - 1, width))

    # Perturb each position at each time and measure future divergence
    for t in range(steps - 1):
        for pos in range(width):
            # Get state at time t
            state = base_history[t].copy()
            # Flip bit at position
            state[pos] = 1 - state[pos]

            # Continue evolution from perturbed state
            perturbed, _ = run_sticky_eca(rule, width, steps - t, mechanism, depth, init=state)

            # Measure divergence at t+1
            if steps - t > 1:
                divergence = np.sum(perturbed[1] != base_history[t + 1]) / width
                control_map[t, pos] = divergence

    return control_map


def detect_boundaries(history: np.ndarray) -> np.ndarray:
    """Detect boundary positions using activity gradient."""
    steps, width = history.shape
    boundary_map = np.zeros((steps - 1, width))

    for t in range(steps - 1):
        # Activity = change from t to t+1
        activity = (history[t] != history[t + 1]).astype(float)

        # Boundary = high gradient in activity
        for i in range(width):
            left_act = activity[(i - 2) % width:(i) % width].mean() if i > 0 else activity[:2].mean()
            right_act = activity[(i + 1) % width:(i + 3) % width].mean() if i < width - 2 else activity[-2:].mean()

            # Simple gradient
            gradient = abs(left_act - right_act)
            boundary_map[t, i] = gradient

    return boundary_map


def measure_bulk_control(control_map: np.ndarray, boundary_map: np.ndarray,
                         boundary_threshold: float = 0.1) -> Dict:
    """Measure Control in bulk (non-boundary) regions."""

    # Classify regions
    is_boundary = boundary_map > boundary_threshold
    is_bulk = ~is_boundary

    # Control in each region type
    bulk_control = control_map[is_bulk].mean() if is_bulk.any() else 0
    boundary_control = control_map[is_boundary].mean() if is_boundary.any() else 0

    # Find maximum Control in bulk
    max_bulk_control = control_map[is_bulk].max() if is_bulk.any() else 0

    # Count high-Control bulk cells
    high_control_threshold = 0.1
    high_bulk_cells = np.sum((control_map > high_control_threshold) & is_bulk)
    total_bulk_cells = np.sum(is_bulk)

    return {
        "bulk_control_mean": float(bulk_control),
        "boundary_control_mean": float(boundary_control),
        "max_bulk_control": float(max_bulk_control),
        "high_bulk_cells": int(high_bulk_cells),
        "total_bulk_cells": int(total_bulk_cells),
        "bulk_control_fraction": float(high_bulk_cells / total_bulk_cells) if total_bulk_cells > 0 else 0,
        "control_ratio": float(boundary_control / bulk_control) if bulk_control > 0 else float('inf')
    }


def bulk_control_falsification(rules: List[int], mechanisms: List[str],
                               width: int = 80, steps: int = 100) -> Dict:
    """
    Part 1: Systematically search for Control in bulk regions.

    If the theory is correct, bulk Control should be negligible.
    We try to FALSIFY this by finding counter-examples.
    """
    print("\n" + "=" * 70)
    print("PART 1: BULK CONTROL FALSIFICATION")
    print("=" * 70)
    print("\nSearching for Control in bulk (non-boundary) regions...")

    results = {}
    anomalies = []

    for rule in rules:
        print(f"\n  Rule {rule}:")
        results[rule] = {}

        for mechanism in mechanisms:
            depths = [1, 2, 3] if mechanism != "standard" else [0]

            for depth in depths:
                key = f"{mechanism}_{depth}" if depth > 0 else "standard"

                # Compute Control map
                control_map = compute_control_map(rule, width, steps, mechanism, depth)

                # Run evolution for boundary detection
                history, _ = run_sticky_eca(rule, width, steps, mechanism, depth)
                boundary_map = detect_boundaries(history)

                # Measure bulk Control
                metrics = measure_bulk_control(control_map, boundary_map)
                results[rule][key] = metrics

                # Check for anomalies (significant bulk Control)
                if metrics["bulk_control_mean"] > 0.05 and metrics["control_ratio"] < 2:
                    anomalies.append({
                        "rule": rule,
                        "mechanism": key,
                        "bulk_control": metrics["bulk_control_mean"],
                        "ratio": metrics["control_ratio"]
                    })

                print(f"    {key}: bulk_ctrl={metrics['bulk_control_mean']:.3f}, "
                      f"boundary_ctrl={metrics['boundary_control_mean']:.3f}, "
                      f"ratio={metrics['control_ratio']:.1f}")

    # Summary
    print(f"\n  Anomalies found: {len(anomalies)}")
    for a in anomalies:
        print(f"    Rule {a['rule']}/{a['mechanism']}: bulk={a['bulk_control']:.3f}, ratio={a['ratio']:.1f}")

    return {
        "results": results,
        "anomalies": anomalies,
        "theory_holds": len(anomalies) == 0
    }


def stickiness_extremes(rules: List[int], width: int = 80, steps: int = 100) -> Dict:
    """
    Part 2: Test stickiness at extreme values.

    - No stickiness (depth=0): Should have zero Control
    - Very high stickiness: What happens?
    """
    print("\n" + "=" * 70)
    print("PART 2: STICKINESS EXTREMES")
    print("=" * 70)

    results = {}

    for rule in rules:
        print(f"\n  Rule {rule}:")
        results[rule] = {
            "zero_stickiness": {},
            "extreme_stickiness": {}
        }

        # Zero stickiness (standard ECA)
        control_map_std = compute_control_map(rule, width, steps, "standard", 0)
        total_control_std = control_map_std.mean()
        results[rule]["zero_stickiness"] = {
            "total_control": float(total_control_std),
            "max_control": float(control_map_std.max())
        }
        print(f"    No stickiness: Control={total_control_std:.4f}")

        # Extreme stickiness (confirmation depth 10)
        for depth in [5, 10, 20]:
            control_map = compute_control_map(rule, width, steps, "confirmation", depth)
            history, _ = run_sticky_eca(rule, width, steps, "confirmation", depth)
            boundary_map = detect_boundaries(history)

            total_control = control_map.mean()
            bulk_metrics = measure_bulk_control(control_map, boundary_map)

            # Check activity
            activity = np.mean(np.diff(history, axis=0) != 0)

            results[rule]["extreme_stickiness"][f"depth_{depth}"] = {
                "total_control": float(total_control),
                "bulk_control": bulk_metrics["bulk_control_mean"],
                "boundary_control": bulk_metrics["boundary_control_mean"],
                "activity": float(activity)
            }
            print(f"    Depth {depth}: Control={total_control:.3f}, activity={activity:.3f}")

    # Analyze trends
    print("\n  ANALYSIS:")
    print("  =========")

    for rule in rules:
        std = results[rule]["zero_stickiness"]["total_control"]
        extreme = results[rule]["extreme_stickiness"]["depth_20"]["total_control"]

        if std < 0.001:
            print(f"    Rule {rule}: Zero Control at no stickiness (as expected)")
        else:
            print(f"    Rule {rule}: Unexpected Control={std:.4f} at no stickiness!")

        if extreme < 0.01:
            print(f"    Rule {rule}: Control collapses at extreme stickiness")
        else:
            print(f"    Rule {rule}: Control persists={extreme:.3f} at extreme stickiness")

    return results


def counter_example_search(width: int = 80, steps: int = 80) -> Dict:
    """
    Part 3: Actively search for theory-breaking counter-examples.

    Theories to test:
    1. "Control requires boundaries" - Find high Control with no boundaries
    2. "Boundaries require stickiness" - Find sticky boundaries without Control
    3. "More stickiness = more Control" - Test monotonicity
    """
    print("\n" + "=" * 70)
    print("PART 3: COUNTER-EXAMPLE SEARCH")
    print("=" * 70)

    results = {
        "theory_1_control_needs_boundaries": {"tested": 0, "violated": 0, "examples": []},
        "theory_2_boundaries_need_stickiness": {"tested": 0, "violated": 0, "examples": []},
        "theory_3_monotonic_stickiness": {"tested": 0, "violated": 0, "examples": []}
    }

    test_rules = [30, 54, 90, 110, 150, 22, 126, 182]

    # Theory 1: Control requires boundaries
    print("\n  Theory 1: 'Control requires boundaries'")
    for rule in test_rules:
        for mechanism in ["confirmation", "refractory"]:
            for depth in [1, 2, 3]:
                results["theory_1_control_needs_boundaries"]["tested"] += 1

                control_map = compute_control_map(rule, width, steps, mechanism, depth)
                history, _ = run_sticky_eca(rule, width, steps, mechanism, depth)
                boundary_map = detect_boundaries(history)

                # Check: high Control in regions with no boundaries?
                no_boundary = boundary_map < 0.05
                high_control = control_map > 0.1

                violation_count = np.sum(no_boundary & high_control)

                if violation_count > width * steps * 0.05:  # > 5% violations
                    results["theory_1_control_needs_boundaries"]["violated"] += 1
                    results["theory_1_control_needs_boundaries"]["examples"].append({
                        "rule": rule,
                        "mechanism": f"{mechanism}_{depth}",
                        "violation_fraction": float(violation_count / (width * steps))
                    })

    print(f"    Tested: {results['theory_1_control_needs_boundaries']['tested']}")
    print(f"    Violations: {results['theory_1_control_needs_boundaries']['violated']}")

    # Theory 2: Boundaries produce Control under stickiness
    print("\n  Theory 2: 'Sticky boundaries produce Control'")
    for rule in test_rules:
        for mechanism in ["confirmation", "refractory"]:
            for depth in [1, 2, 3]:
                results["theory_2_boundaries_need_stickiness"]["tested"] += 1

                control_map = compute_control_map(rule, width, steps, mechanism, depth)
                history, _ = run_sticky_eca(rule, width, steps, mechanism, depth)
                boundary_map = detect_boundaries(history)

                # Check: boundaries with zero Control?
                has_boundary = boundary_map > 0.1
                no_control = control_map < 0.01

                # Fraction of boundaries with no Control
                if has_boundary.any():
                    dead_boundary_fraction = np.sum(has_boundary & no_control) / np.sum(has_boundary)

                    if dead_boundary_fraction > 0.8:  # > 80% dead boundaries
                        results["theory_2_boundaries_need_stickiness"]["violated"] += 1
                        results["theory_2_boundaries_need_stickiness"]["examples"].append({
                            "rule": rule,
                            "mechanism": f"{mechanism}_{depth}",
                            "dead_fraction": float(dead_boundary_fraction)
                        })

    print(f"    Tested: {results['theory_2_boundaries_need_stickiness']['tested']}")
    print(f"    Violations: {results['theory_2_boundaries_need_stickiness']['violated']}")

    # Theory 3: Monotonicity of stickiness
    print("\n  Theory 3: 'More stickiness = more Control' (monotonicity)")
    for rule in test_rules:
        results["theory_3_monotonic_stickiness"]["tested"] += 1

        controls = []
        for depth in [1, 2, 3, 4, 5]:
            control_map = compute_control_map(rule, width, steps, "confirmation", depth)
            controls.append(control_map.mean())

        # Check monotonicity
        monotonic = all(controls[i] <= controls[i+1] for i in range(len(controls)-1))
        anti_monotonic = all(controls[i] >= controls[i+1] for i in range(len(controls)-1))

        if not (monotonic or anti_monotonic):
            results["theory_3_monotonic_stickiness"]["violated"] += 1
            results["theory_3_monotonic_stickiness"]["examples"].append({
                "rule": rule,
                "controls": [float(c) for c in controls]
            })

    print(f"    Tested: {results['theory_3_monotonic_stickiness']['tested']}")
    print(f"    Violations: {results['theory_3_monotonic_stickiness']['violated']}")

    # Summary
    print("\n  COUNTER-EXAMPLE SUMMARY:")
    print("  " + "-" * 40)

    theories_broken = []
    for name, data in results.items():
        status = "HOLDS" if data["violated"] == 0 else f"BROKEN ({data['violated']} cases)"
        print(f"    {name}: {status}")
        if data["violated"] > 0:
            theories_broken.append(name)

    return results


def generate_phase_d_visualizations(bulk_results: Dict, extreme_results: Dict,
                                     counter_results: Dict):
    """Generate visualizations for Phase D findings."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Bulk vs Boundary Control by rule
    ax1 = axes[0, 0]
    rules = list(bulk_results["results"].keys())
    bulk_controls = []
    boundary_controls = []

    for rule in rules:
        # Get confirmation_1 results
        data = bulk_results["results"][rule].get("confirmation_1",
               bulk_results["results"][rule].get("standard", {}))
        bulk_controls.append(data.get("bulk_control_mean", 0))
        boundary_controls.append(data.get("boundary_control_mean", 0))

    x = np.arange(len(rules))
    width_bar = 0.35
    ax1.bar(x - width_bar/2, bulk_controls, width_bar, label='Bulk Control', color='steelblue')
    ax1.bar(x + width_bar/2, boundary_controls, width_bar, label='Boundary Control', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Rule {r}" for r in rules])
    ax1.set_ylabel("Mean Control")
    ax1.set_title("Bulk vs Boundary Control\n(Bulk Falsification Test)")
    ax1.legend()
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Anomaly threshold')

    # Plot 2: Stickiness extremes
    ax2 = axes[0, 1]
    for rule in extreme_results.keys():
        depths = [0, 5, 10, 20]
        controls = [extreme_results[rule]["zero_stickiness"]["total_control"]]
        for d in [5, 10, 20]:
            controls.append(extreme_results[rule]["extreme_stickiness"][f"depth_{d}"]["total_control"])
        ax2.plot(depths, controls, 'o-', label=f"Rule {rule}")
    ax2.set_xlabel("Stickiness Depth")
    ax2.set_ylabel("Total Control")
    ax2.set_title("Control vs Stickiness Depth\n(Extreme Stickiness Test)")
    ax2.legend()
    ax2.set_xscale('symlog', linthresh=1)

    # Plot 3: Counter-example summary
    ax3 = axes[1, 0]
    theory_names = ["Ctrl needs\nboundaries", "Boundaries\nproduce Ctrl", "Monotonic\nstickiness"]
    tested = [counter_results["theory_1_control_needs_boundaries"]["tested"],
              counter_results["theory_2_boundaries_need_stickiness"]["tested"],
              counter_results["theory_3_monotonic_stickiness"]["tested"]]
    violated = [counter_results["theory_1_control_needs_boundaries"]["violated"],
                counter_results["theory_2_boundaries_need_stickiness"]["violated"],
                counter_results["theory_3_monotonic_stickiness"]["violated"]]

    x = np.arange(len(theory_names))
    ax3.bar(x, tested, label='Tested', color='lightgray')
    ax3.bar(x, violated, label='Violated', color='red')
    ax3.set_xticks(x)
    ax3.set_xticklabels(theory_names)
    ax3.set_ylabel("Count")
    ax3.set_title("Theory Falsification Results")
    ax3.legend()

    # Plot 4: Non-monotonicity examples
    ax4 = axes[1, 1]
    non_mono = counter_results["theory_3_monotonic_stickiness"]["examples"]
    if non_mono:
        for ex in non_mono[:4]:
            ax4.plot([1, 2, 3, 4, 5], ex["controls"], 'o-', label=f"Rule {ex['rule']}")
        ax4.set_xlabel("Confirmation Depth")
        ax4.set_ylabel("Mean Control")
        ax4.set_title("Non-Monotonic Control vs Stickiness\n(Falsified Examples)")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No non-monotonic\nexamples found",
                ha='center', va='center', fontsize=14)
        ax4.set_title("Non-Monotonic Examples")

    plt.tight_layout()
    plt.savefig("output/phase_d/negative_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved Phase D visualization to output/phase_d/negative_results.png")


def main():
    """Run Phase D: Negative Results."""

    print("=" * 70, flush=True)
    print("PHASE D: NEGATIVE RESULTS", flush=True)
    print("=" * 70, flush=True)
    print("\nGoal: Identify limits of boundary-Control theory through falsification", flush=True)

    test_rules = [30, 54, 90, 110]
    mechanisms = ["standard", "confirmation", "refractory"]

    # Part 1: Bulk Control Falsification (smaller grid for speed)
    bulk_results = bulk_control_falsification(test_rules, mechanisms, width=40, steps=50)

    # Part 2: Stickiness Extremes (smaller grid)
    extreme_results = stickiness_extremes(test_rules, width=40, steps=50)

    # Part 3: Counter-Example Search (smaller grid)
    counter_results = counter_example_search(width=40, steps=40)

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    generate_phase_d_visualizations(bulk_results, extreme_results, counter_results)

    # Save results
    all_results = {
        "bulk_falsification": bulk_results,
        "stickiness_extremes": extreme_results,
        "counter_examples": counter_results
    }

    # Convert inf to string for JSON
    def convert_inf(obj):
        if isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
            return "inf" if obj > 0 else "-inf"
        elif isinstance(obj, dict):
            return {k: convert_inf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_inf(i) for i in obj]
        return obj

    all_results = convert_inf(all_results)

    with open("output/phase_d/negative_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE D SUMMARY: NEGATIVE RESULTS")
    print("=" * 70)

    print("\n1. BULK CONTROL FALSIFICATION:")
    if bulk_results["theory_holds"]:
        print("   => THEORY HOLDS: No significant bulk Control found")
    else:
        print(f"   => ANOMALIES FOUND: {len(bulk_results['anomalies'])} cases")
        for a in bulk_results["anomalies"]:
            print(f"      Rule {a['rule']}/{a['mechanism']}")

    print("\n2. STICKINESS EXTREMES:")
    print("   Zero stickiness:")
    for rule in test_rules:
        ctrl = extreme_results[rule]["zero_stickiness"]["total_control"]
        print(f"     Rule {rule}: Control = {ctrl:.4f}")
    print("   Extreme stickiness (depth=20):")
    for rule in test_rules:
        ctrl = extreme_results[rule]["extreme_stickiness"]["depth_20"]["total_control"]
        print(f"     Rule {rule}: Control = {ctrl:.3f}")

    print("\n3. THEORIES TESTED:")
    for name, data in counter_results.items():
        status = "HOLDS" if data["violated"] == 0 else f"FALSIFIED ({data['violated']}/{data['tested']})"
        clean_name = name.replace("_", " ").replace("theory ", "")
        print(f"   {clean_name}: {status}")

    print("\n" + "=" * 70)
    print("PHASE D COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
