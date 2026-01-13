"""
Phase 3: Selective Stickiness

Do NOT increase memory depth. Instead, restrict WHERE stickiness applies.

Question: Does local/selective hysteresis beat global hysteresis for
generating STRUCTURED Control?

Mechanisms:
1. Motif-gated: Confirmation only applies when spatial pattern persists
2. Swap-gated: Stickiness gates swap permission, not state change
3. Asymmetric: Easy to stick (1 confirmation), hard to unstick (3 confirmations)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BASELINE IMPLEMENTATIONS
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


def global_confirmation_eca(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """Global confirmation - applies everywhere equally."""
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
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
                    history[t][i] = proposed
                    pending[i] = -1
                else:
                    pending[i] = proposed
                    history[t][i] = history[t-1][i]
            else:
                history[t][i] = history[t-1][i]
                pending[i] = -1
    return history


# =============================================================================
# SELECTIVE STICKINESS MECHANISMS
# =============================================================================

def motif_gated_confirmation(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """
    Confirmation only applies at edges (where neighbors differ).
    Interior (uniform) regions follow standard rule.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    pending = np.full(width, -1, dtype=int)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            proposed = (rule >> pattern) & 1

            # Check if at edge (neighbors differ)
            is_edge = (left != center) or (center != right)

            if is_edge:
                # Apply confirmation at edges
                if proposed != history[t-1][i]:
                    if pending[i] == proposed:
                        history[t][i] = proposed
                        pending[i] = -1
                    else:
                        pending[i] = proposed
                        history[t][i] = history[t-1][i]
                else:
                    history[t][i] = history[t-1][i]
                    pending[i] = -1
            else:
                # No confirmation in interior - standard rule
                history[t][i] = proposed
                pending[i] = -1

    return history


def density_gated_confirmation(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """
    Confirmation only applies in high-activity regions.
    Quiet regions follow standard rule.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    pending = np.full(width, -1, dtype=int)
    activity = np.zeros(width, dtype=float)  # Rolling activity measure
    history[0] = init.copy()

    alpha = 0.3  # Activity decay rate

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            proposed = (rule >> pattern) & 1

            # Update activity (rolling average of changes)
            if t > 1:
                changed = int(history[t-1][i] != history[t-2][i])
                activity[i] = alpha * changed + (1 - alpha) * activity[i]

            # High activity region: apply confirmation
            if activity[i] > 0.2:
                if proposed != history[t-1][i]:
                    if pending[i] == proposed:
                        history[t][i] = proposed
                        pending[i] = -1
                    else:
                        pending[i] = proposed
                        history[t][i] = history[t-1][i]
                else:
                    history[t][i] = history[t-1][i]
                    pending[i] = -1
            else:
                # Quiet region: standard rule
                history[t][i] = proposed
                pending[i] = -1

    return history


def asymmetric_confirmation(rule: int, width: int, steps: int,
                            stick_confirm: int = 1, unstick_confirm: int = 3,
                            init=None) -> np.ndarray:
    """
    Asymmetric stickiness:
    - 0 -> 1: Easy (stick_confirm confirmations)
    - 1 -> 0: Hard (unstick_confirm confirmations)

    This creates "sticky 1s" - structures that are easy to create
    but hard to destroy.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    pending = np.full(width, -1, dtype=int)
    confirm_count = np.zeros(width, dtype=int)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            proposed = (rule >> pattern) & 1

            current = history[t-1][i]

            if proposed != current:
                # Determine required confirmations based on direction
                if current == 0 and proposed == 1:
                    # Sticking (0 -> 1): easy
                    required = stick_confirm
                else:
                    # Unsticking (1 -> 0): hard
                    required = unstick_confirm

                if pending[i] == proposed:
                    confirm_count[i] += 1
                    if confirm_count[i] >= required:
                        history[t][i] = proposed
                        pending[i] = -1
                        confirm_count[i] = 0
                    else:
                        history[t][i] = current
                else:
                    pending[i] = proposed
                    confirm_count[i] = 1
                    history[t][i] = current
            else:
                history[t][i] = current
                pending[i] = -1
                confirm_count[i] = 0

    return history


def neighborhood_memory_confirmation(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """
    Confirmation applies only when the NEIGHBORHOOD has been stable.
    If neighborhood changed recently, use standard rule.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    pending = np.full(width, -1, dtype=int)
    neighborhood_stable = np.zeros(width, dtype=int)  # Steps neighborhood unchanged
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            proposed = (rule >> pattern) & 1

            # Check if neighborhood changed from t-2 to t-1
            if t > 1:
                prev_left = history[t-2][(i - 1) % width]
                prev_center = history[t-2][i]
                prev_right = history[t-2][(i + 1) % width]
                prev_pattern = (prev_left << 2) | (prev_center << 1) | prev_right

                if pattern == prev_pattern:
                    neighborhood_stable[i] += 1
                else:
                    neighborhood_stable[i] = 0

            # Apply confirmation only if neighborhood has been stable
            if neighborhood_stable[i] >= 2:
                if proposed != history[t-1][i]:
                    if pending[i] == proposed:
                        history[t][i] = proposed
                        pending[i] = -1
                    else:
                        pending[i] = proposed
                        history[t][i] = history[t-1][i]
                else:
                    history[t][i] = history[t-1][i]
                    pending[i] = -1
            else:
                # Neighborhood unstable: standard rule
                history[t][i] = proposed
                pending[i] = -1

    return history


# =============================================================================
# METRICS
# =============================================================================

def compute_control(history: np.ndarray, window: int = 50) -> float:
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


def compute_compression(history: np.ndarray) -> float:
    import zlib
    flat = history.flatten().astype(np.uint8).tobytes()
    compressed = zlib.compress(flat, level=9)
    return len(compressed) / len(flat)


def compute_structure_control_ratio(history: np.ndarray) -> float:
    """Ratio of Control at edges vs interior."""
    T, W = history.shape

    edge_control = []
    interior_control = []

    # Build pattern-outcome map
    pattern_outcomes: Dict[Tuple, List[int]] = defaultdict(list)
    for t in range(1, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_outcomes[pattern].append(int(outcome))

    pattern_divergence = {}
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) >= 5:
            mean = np.mean(outcomes)
            pattern_divergence[pattern] = 4 * mean * (1 - mean)
        else:
            pattern_divergence[pattern] = 0

    # Categorize by edge vs interior
    for t in range(1, T - 1):
        for i in range(1, W - 1):
            left = history[t, i-1]
            center = history[t, i]
            right = history[t, i+1]
            pattern = (left, center, right)

            is_edge = (left != center) or (center != right)
            ctrl = pattern_divergence.get(pattern, 0)

            if is_edge:
                edge_control.append(ctrl)
            else:
                interior_control.append(ctrl)

    edge_mean = np.mean(edge_control) if edge_control else 0
    interior_mean = np.mean(interior_control) if interior_control else 0

    if interior_mean > 0:
        return edge_mean / interior_mean
    elif edge_mean > 0:
        return float('inf')
    else:
        return 1.0


def compute_persistence(history: np.ndarray) -> float:
    """Average time structures persist."""
    if len(history) < 2:
        return 0.0
    changes = np.sum(history[1:] != history[:-1], axis=1)
    activity = np.mean(changes) / history.shape[1]
    return 1.0 / max(activity, 0.001)


# =============================================================================
# COMPARISON EXPERIMENT
# =============================================================================

def run_comparison(rules: List[int] = [110, 30, 90, 54, 62],
                   width: int = 100, steps: int = 200):
    """Compare global vs selective stickiness."""
    print("=" * 70)
    print("PHASE 3: SELECTIVE STICKINESS COMPARISON")
    print("=" * 70)

    mechanisms = {
        'standard': standard_eca,
        'global_confirmation': global_confirmation_eca,
        'motif_gated': motif_gated_confirmation,
        'density_gated': density_gated_confirmation,
        'asymmetric_1_3': lambda r, w, s, i=None: asymmetric_confirmation(r, w, s, 1, 3, i),
        'asymmetric_1_5': lambda r, w, s, i=None: asymmetric_confirmation(r, w, s, 1, 5, i),
        'neighborhood_memory': neighborhood_memory_confirmation,
    }

    results = {}

    for rule in rules:
        print(f"\nRule {rule}:")
        results[rule] = {}

        for mech_name, mech_func in mechanisms.items():
            history = mech_func(rule, width, steps)

            control = compute_control(history)
            compression = compute_compression(history)
            structure_ratio = compute_structure_control_ratio(history)
            persistence = compute_persistence(history)

            results[rule][mech_name] = {
                'control': control,
                'compression': compression,
                'structure_ratio': structure_ratio,
                'persistence': persistence,
                'score': control * (1 - abs(compression - 0.3)) * min(structure_ratio, 5) / 5
            }

            print(f"  {mech_name:25s}: Ctrl={control:.3f}, "
                  f"Comp={compression:.3f}, Struct={structure_ratio:.2f}")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_selective_comparison(results: Dict, rules: List[int]):
    """Compare global vs selective stickiness."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mechanisms = ['global_confirmation', 'motif_gated', 'density_gated',
                  'asymmetric_1_3', 'neighborhood_memory']
    colors = plt.cm.Set2(np.linspace(0, 1, len(mechanisms)))

    x = np.arange(len(rules))
    width_bar = 0.15

    # 1. Control
    ax = axes[0, 0]
    for i, mech in enumerate(mechanisms):
        values = [results[r].get(mech, {}).get('control', 0) for r in rules]
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[i])
    ax.set_xticks(x + width_bar * 2)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Control')
    ax.set_title('Control by Mechanism')
    ax.legend(fontsize=8)

    # 2. Compression
    ax = axes[0, 1]
    for i, mech in enumerate(mechanisms):
        values = [results[r].get(mech, {}).get('compression', 0) for r in rules]
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[i])
    ax.set_xticks(x + width_bar * 2)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Compression')
    ax.set_title('Compression (Structure)')
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # 3. Structure Ratio
    ax = axes[1, 0]
    for i, mech in enumerate(mechanisms):
        values = [min(results[r].get(mech, {}).get('structure_ratio', 1), 5) for r in rules]
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[i])
    ax.set_xticks(x + width_bar * 2)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Edge/Interior Control Ratio')
    ax.set_title('Structure Attachment (>1 = structure-bound)')
    ax.axhline(1, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # 4. Combined Score
    ax = axes[1, 1]
    for i, mech in enumerate(mechanisms):
        values = [results[r].get(mech, {}).get('score', 0) for r in rules]
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[i])
    ax.set_xticks(x + width_bar * 2)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Score')
    ax.set_title('Combined Score (Control x Structure x Complexity)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'selective_vs_global.png', dpi=150)
    plt.close()


def visualize_mechanism_spacetime(rule: int, width: int = 100, steps: int = 150):
    """Spacetime diagrams for all mechanisms."""
    mechanisms = [
        ('Standard', standard_eca(rule, width, steps)),
        ('Global Confirm', global_confirmation_eca(rule, width, steps)),
        ('Motif-Gated', motif_gated_confirmation(rule, width, steps)),
        ('Density-Gated', density_gated_confirmation(rule, width, steps)),
        ('Asymmetric 1/3', asymmetric_confirmation(rule, width, steps, 1, 3)),
        ('Neighborhood Memory', neighborhood_memory_confirmation(rule, width, steps)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, history) in enumerate(mechanisms):
        axes[idx].imshow(history, cmap='binary', aspect='auto')
        ctrl = compute_control(history)
        comp = compute_compression(history)
        axes[idx].set_title(f'{name}\nCtrl={ctrl:.3f}, Comp={comp:.3f}')
        if idx >= 3:
            axes[idx].set_xlabel('Position')
        if idx % 3 == 0:
            axes[idx].set_ylabel('Time')

    plt.suptitle(f'Rule {rule}: Selective Stickiness Mechanisms', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'mechanisms_rule{rule}.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 3: SELECTIVE STICKINESS")
    print("=" * 70)

    rules = [110, 30, 90, 54, 62]
    results = run_comparison(rules)

    # Save results
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if obj == float('inf'):
            return "inf"
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(OUTPUT_DIR / 'selective_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_selective_comparison(results, rules)
    print("  Saved comparison figure")

    for rule in [110, 54]:
        visualize_mechanism_spacetime(rule)
        print(f"  Saved mechanism spacetime for Rule {rule}")

    # Analysis
    print("\n" + "=" * 70)
    print("PHASE 3 ANALYSIS")
    print("=" * 70)

    print("\nDoes selective stickiness beat global stickiness?")

    for rule in rules:
        global_ctrl = results[rule]['global_confirmation']['control']
        global_struct = results[rule]['global_confirmation']['structure_ratio']

        best_selective = None
        best_score = 0

        for mech in ['motif_gated', 'density_gated', 'asymmetric_1_3', 'neighborhood_memory']:
            score = results[rule].get(mech, {}).get('score', 0)
            if score > best_score:
                best_score = score
                best_selective = mech

        if best_selective:
            sel_ctrl = results[rule][best_selective]['control']
            sel_struct = results[rule][best_selective]['structure_ratio']

            print(f"\nRule {rule}:")
            print(f"  Global: Ctrl={global_ctrl:.3f}, Struct={global_struct:.2f}")
            print(f"  Best selective ({best_selective}): Ctrl={sel_ctrl:.3f}, Struct={sel_struct:.2f}")

            if sel_struct > global_struct * 1.2 and sel_ctrl > 0.1:
                print(f"  => SELECTIVE WINS (better structure attachment)")
            elif global_ctrl > sel_ctrl * 1.5:
                print(f"  => GLOBAL WINS (higher Control)")
            else:
                print(f"  => COMPARABLE")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
