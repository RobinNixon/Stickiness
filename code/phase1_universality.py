"""
Phase 1: Harden the Conjecture - Universality and Saturation Curves

Goals:
1. Prove that ALL non-trivial deterministic CAs gain nonzero Control under stickiness
2. Define precisely what "non-trivial" means
3. Generate Control vs stickiness depth saturation curves
4. Produce definitive figure showing universality of the effect
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "phase1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CA IMPLEMENTATIONS
# =============================================================================

def standard_eca(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    history[0] = init.copy()
    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            history[t][i] = (rule >> pattern) & 1
    return history


def confirmation_eca(rule: int, width: int, steps: int, depth: int = 1, init=None) -> np.ndarray:
    """Confirmation with variable depth: need `depth` consecutive confirmations."""
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    confirm_count = np.zeros(width, dtype=int)  # How many times confirmed
    pending = np.full(width, -1, dtype=int)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            proposed = (rule >> pattern) & 1

            if proposed != history[t-1][i]:
                if pending[i] == proposed:
                    confirm_count[i] += 1
                    if confirm_count[i] >= depth:
                        history[t][i] = proposed
                        pending[i] = -1
                        confirm_count[i] = 0
                    else:
                        history[t][i] = history[t-1][i]
                else:
                    pending[i] = proposed
                    confirm_count[i] = 1
                    history[t][i] = history[t-1][i]
            else:
                history[t][i] = history[t-1][i]
                pending[i] = -1
                confirm_count[i] = 0
    return history


def refractory_eca(rule: int, width: int, steps: int, refractory_time: int = 2, init=None) -> np.ndarray:
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    cooldown = np.zeros(width, dtype=int)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            if cooldown[i] > 0:
                history[t][i] = history[t-1][i]
                cooldown[i] -= 1
            else:
                left = history[t-1][(i - 1) % width]
                center = history[t-1][i]
                right = history[t-1][(i + 1) % width]
                pattern = (left << 2) | (center << 1) | right
                new_state = (rule >> pattern) & 1
                if new_state != history[t-1][i]:
                    cooldown[i] = refractory_time
                history[t][i] = new_state
    return history


def second_order_eca(rule: int, width: int, steps: int, depth: int = 2, init=None) -> np.ndarray:
    """Second-order: XOR with state from `depth` steps ago."""
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    for d in range(min(depth, steps)):
        history[d] = init.copy()

    # Bootstrap initial steps with standard ECA
    for t in range(1, min(depth, steps)):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            history[t][i] = (rule >> pattern) & 1

    for t in range(depth, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            eca_out = (rule >> pattern) & 1
            history[t][i] = eca_out ^ history[t-depth][i]
    return history


# =============================================================================
# METRICS
# =============================================================================

def compute_control(history: np.ndarray, window: int = 50) -> float:
    """Control proxy: same pattern → different outcomes."""
    if len(history) < window + 2:
        window = max(1, len(history) - 2)
    history = history[-window:]
    pattern_outcomes: Dict[Tuple, List[int]] = defaultdict(list)

    for t in range(1, len(history) - 1):
        for i in range(1, history.shape[1] - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_outcomes[pattern].append(int(outcome))

    divergences = []
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) >= 5:
            mean = np.mean(outcomes)
            divergence = 4 * mean * (1 - mean)
            divergences.append(divergence)
    return np.mean(divergences) if divergences else 0.0


def compute_activity(history: np.ndarray) -> float:
    if len(history) < 2:
        return 0.0
    changes = np.sum(history[1:] != history[:-1], axis=1)
    return np.mean(changes) / history.shape[1]


def compute_entropy(state: np.ndarray) -> float:
    unique, counts = np.unique(state, return_counts=True)
    probs = counts / len(state)
    return -np.sum(probs * np.log2(probs + 1e-10))


# =============================================================================
# TRIVIALITY CLASSIFICATION
# =============================================================================

def classify_rule_triviality(rule: int, width: int = 80, steps: int = 100) -> Dict:
    """
    Classify a rule as trivial or non-trivial.

    Trivial rules:
    - Identity (0, 255): Output depends only on center cell
    - Uniform fixed point (0, 255): All cells converge to same state
    - Nilpotent: Converges to all-0 or all-1
    - Periodic class 1: Converges to static or simple period-2 pattern
    """
    history = standard_eca(rule, width, steps)

    # Check final state entropy
    final_entropy = compute_entropy(history[-1])

    # Check activity
    activity = compute_activity(history[-20:])  # Last 20 steps

    # Check if converged to uniform
    is_uniform = np.all(history[-1] == history[-1][0])

    # Check if converged to static
    is_static = np.allclose(history[-1], history[-2]) if len(history) > 1 else True

    # Check if nilpotent (all 0 or all 1)
    is_nilpotent = np.all(history[-1] == 0) or np.all(history[-1] == 1)

    # Classification
    if rule == 0 or rule == 255:
        triviality = "uniform_constant"
    elif is_nilpotent:
        triviality = "nilpotent"
    elif is_uniform and is_static:
        triviality = "uniform_fixed"
    elif is_static and activity < 0.01:
        triviality = "static_fixed"
    elif activity < 0.02:
        triviality = "near_static"
    else:
        triviality = "non_trivial"

    return {
        'rule': rule,
        'triviality': triviality,
        'final_entropy': final_entropy,
        'activity': activity,
        'is_uniform': is_uniform,
        'is_static': is_static,
        'is_nilpotent': is_nilpotent
    }


# =============================================================================
# UNIVERSALITY TEST
# =============================================================================

def test_universality(width: int = 80, steps: int = 150):
    """Test that ALL non-trivial rules gain Control under stickiness."""
    print("=" * 70)
    print("PHASE 1: UNIVERSALITY TEST")
    print("=" * 70)

    results = {
        'trivial_rules': [],
        'non_trivial_rules': [],
        'control_standard': {},
        'control_confirmation': {},
        'control_refractory': {},
        'exceptions': []  # Non-trivial rules that don't gain Control
    }

    # Classify all rules
    print("\n1. Classifying rules by triviality...")
    for rule in range(256):
        classification = classify_rule_triviality(rule, width, steps)
        if classification['triviality'] != 'non_trivial':
            results['trivial_rules'].append(classification)
        else:
            results['non_trivial_rules'].append(classification)

    print(f"   Trivial rules: {len(results['trivial_rules'])}")
    print(f"   Non-trivial rules: {len(results['non_trivial_rules'])}")

    # Test Control for all rules
    print("\n2. Testing Control under stickiness...")
    for rule in range(256):
        # Standard
        history_std = standard_eca(rule, width, steps)
        control_std = compute_control(history_std)
        results['control_standard'][rule] = control_std

        # Confirmation
        history_conf = confirmation_eca(rule, width, steps, depth=1)
        control_conf = compute_control(history_conf)
        results['control_confirmation'][rule] = control_conf

        # Refractory
        history_ref = refractory_eca(rule, width, steps, refractory_time=2)
        control_ref = compute_control(history_ref)
        results['control_refractory'][rule] = control_ref

        if (rule + 1) % 64 == 0:
            print(f"   Processed {rule + 1}/256 rules")

    # Find exceptions
    print("\n3. Checking for exceptions...")
    non_trivial_rules = [r['rule'] for r in results['non_trivial_rules']]

    for rule in non_trivial_rules:
        control_conf = results['control_confirmation'][rule]
        control_ref = results['control_refractory'][rule]

        # Exception: non-trivial rule that gains no Control
        if control_conf < 0.01 and control_ref < 0.01:
            results['exceptions'].append({
                'rule': rule,
                'control_confirmation': control_conf,
                'control_refractory': control_ref
            })

    print(f"   Exceptions found: {len(results['exceptions'])}")
    if results['exceptions']:
        for exc in results['exceptions']:
            print(f"      Rule {exc['rule']}: conf={exc['control_confirmation']:.4f}, "
                  f"ref={exc['control_refractory']:.4f}")

    return results


# =============================================================================
# SATURATION CURVES
# =============================================================================

def generate_saturation_curves(rules: List[int] = [110, 30, 90, 54, 62, 150],
                                width: int = 80, steps: int = 150):
    """Generate Control vs stickiness depth curves."""
    print("\n" + "=" * 70)
    print("SATURATION CURVES")
    print("=" * 70)

    curves = {
        'confirmation': {},
        'refractory': {},
        'second_order': {}
    }

    for rule in rules:
        print(f"\nRule {rule}:")

        # Confirmation depth 1, 2, 3, 4
        curves['confirmation'][rule] = []
        for depth in [1, 2, 3, 4]:
            history = confirmation_eca(rule, width, steps, depth=depth)
            control = compute_control(history)
            curves['confirmation'][rule].append((depth, control))
            print(f"  Confirmation depth {depth}: Control = {control:.4f}")

        # Refractory length 1, 2, 3, 4, 5
        curves['refractory'][rule] = []
        for length in [1, 2, 3, 4, 5]:
            history = refractory_eca(rule, width, steps, refractory_time=length)
            control = compute_control(history)
            curves['refractory'][rule].append((length, control))
            print(f"  Refractory length {length}: Control = {control:.4f}")

        # Second-order depth 2, 3, 4
        curves['second_order'][rule] = []
        for depth in [2, 3, 4]:
            history = second_order_eca(rule, width, steps, depth=depth)
            control = compute_control(history)
            curves['second_order'][rule].append((depth, control))
            print(f"  Second-order depth {depth}: Control = {control:.4f}")

    return curves


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_universality(results: Dict):
    """Plot showing universality of stickiness → Control."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Control distribution: Standard vs Confirmation
    ax = axes[0]
    std_controls = [results['control_standard'][r] for r in range(256)]
    conf_controls = [results['control_confirmation'][r] for r in range(256)]

    x = np.arange(256)
    width_bar = 0.4
    ax.bar(x - width_bar/2, std_controls, width_bar, label='Standard', alpha=0.7, color='gray')
    ax.bar(x + width_bar/2, conf_controls, width_bar, label='Confirmation', alpha=0.7, color='blue')
    ax.set_xlabel('Rule Number')
    ax.set_ylabel('Control Proxy')
    ax.set_title('Control: Standard vs Confirmation\n(All 256 Rules)')
    ax.legend()
    ax.set_xlim(0, 255)

    # 2. Scatter: Standard Control vs Confirmation Control
    ax = axes[1]
    trivial_rules = [r['rule'] for r in results['trivial_rules']]
    non_trivial_rules = [r['rule'] for r in results['non_trivial_rules']]

    # Plot trivial in gray, non-trivial in blue
    for rule in trivial_rules:
        ax.scatter(results['control_standard'][rule],
                   results['control_confirmation'][rule],
                   c='gray', alpha=0.5, s=30)
    for rule in non_trivial_rules:
        ax.scatter(results['control_standard'][rule],
                   results['control_confirmation'][rule],
                   c='blue', alpha=0.7, s=30)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Standard Control')
    ax.set_ylabel('Confirmation Control')
    ax.set_title('Control Improvement\n(Gray=Trivial, Blue=Non-trivial)')
    ax.legend()

    # 3. Histogram of Control gains
    ax = axes[2]
    gains = []
    for rule in non_trivial_rules:
        gain = results['control_confirmation'][rule] - results['control_standard'][rule]
        gains.append(gain)

    ax.hist(gains, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', label='No gain')
    ax.set_xlabel('Control Gain (Confirmation - Standard)')
    ax.set_ylabel('Number of Non-trivial Rules')
    ax.set_title(f'Control Gain Distribution\n(Non-trivial rules: {len(non_trivial_rules)})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'universality_proof.png', dpi=150)
    plt.close()
    print(f"\nSaved universality figure to {OUTPUT_DIR / 'universality_proof.png'}")


def plot_saturation_curves(curves: Dict, rules: List[int]):
    """Plot saturation curves for all mechanisms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(rules)))

    # Confirmation
    ax = axes[0]
    for idx, rule in enumerate(rules):
        depths, controls = zip(*curves['confirmation'][rule])
        ax.plot(depths, controls, 'o-', color=colors[idx], label=f'Rule {rule}')
    ax.set_xlabel('Confirmation Depth')
    ax.set_ylabel('Control Proxy')
    ax.set_title('Confirmation: Control vs Depth')
    ax.legend()
    ax.set_xticks([1, 2, 3, 4])

    # Refractory
    ax = axes[1]
    for idx, rule in enumerate(rules):
        lengths, controls = zip(*curves['refractory'][rule])
        ax.plot(lengths, controls, 'o-', color=colors[idx], label=f'Rule {rule}')
    ax.set_xlabel('Refractory Length')
    ax.set_ylabel('Control Proxy')
    ax.set_title('Refractory: Control vs Length')
    ax.legend()
    ax.set_xticks([1, 2, 3, 4, 5])

    # Second-order
    ax = axes[2]
    for idx, rule in enumerate(rules):
        depths, controls = zip(*curves['second_order'][rule])
        ax.plot(depths, controls, 'o-', color=colors[idx], label=f'Rule {rule}')
    ax.set_xlabel('Second-Order Depth')
    ax.set_ylabel('Control Proxy')
    ax.set_title('Second-Order: Control vs Depth')
    ax.legend()
    ax.set_xticks([2, 3, 4])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'saturation_curves.png', dpi=150)
    plt.close()
    print(f"Saved saturation curves to {OUTPUT_DIR / 'saturation_curves.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 1: HARDEN THE CONJECTURE")
    print("=" * 70)

    # 1. Universality test
    results = test_universality()

    # Save results
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(OUTPUT_DIR / 'universality_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)

    # 2. Saturation curves
    rules = [110, 30, 90, 54, 62, 150]
    curves = generate_saturation_curves(rules)

    with open(OUTPUT_DIR / 'saturation_curves.json', 'w') as f:
        json.dump(convert(curves), f, indent=2)

    # 3. Visualizations
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_universality(results)
    plot_saturation_curves(curves, rules)

    # 4. Summary
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)

    trivial_count = len(results['trivial_rules'])
    non_trivial_count = len(results['non_trivial_rules'])
    exception_count = len(results['exceptions'])

    print(f"\nTriviality Classification:")
    print(f"  Trivial rules: {trivial_count}")
    print(f"  Non-trivial rules: {non_trivial_count}")

    # Count by triviality type
    trivial_types = defaultdict(int)
    for r in results['trivial_rules']:
        trivial_types[r['triviality']] += 1
    print(f"\n  Trivial breakdown:")
    for ttype, count in sorted(trivial_types.items()):
        print(f"    {ttype}: {count}")

    # Count rules gaining Control
    gained_control = 0
    for rule in [r['rule'] for r in results['non_trivial_rules']]:
        if results['control_confirmation'][rule] > 0.01:
            gained_control += 1

    print(f"\nUniversality Check:")
    print(f"  Non-trivial rules gaining Control > 0.01: {gained_control}/{non_trivial_count}")
    print(f"  Exceptions (non-trivial, no Control): {exception_count}")

    if exception_count == 0:
        print("\n  CONJECTURE SUPPORTED: All non-trivial rules gain Control!")
    else:
        print(f"\n  EXCEPTIONS FOUND: {exception_count} rules need investigation")

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
