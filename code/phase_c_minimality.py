"""
Phase C: Minimality Experiments

Key questions:
1. What is the smallest boundary that sustains Control?
2. What happens when almost everything is boundary?

Part 1: Minimal Boundary - find minimum persistence, stickiness, activity gradient
Part 2: Boundary without Bulk - systems where bulk vanishes
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict
from scipy import ndimage

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "phase_c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CA IMPLEMENTATIONS WITH VARIABLE STICKINESS
# =============================================================================

def variable_confirmation_eca(rule: int, width: int, steps: int,
                               confirm_depth: int = 1, init=None) -> np.ndarray:
    """Confirmation with variable depth."""
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

            if proposed != history[t-1][i]:
                if pending[i] == proposed:
                    confirm_count[i] += 1
                    if confirm_count[i] >= confirm_depth:
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


def variable_refractory_eca(rule: int, width: int, steps: int,
                            refractory_time: int = 2, init=None) -> np.ndarray:
    """Refractory with variable time."""
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


def standard_eca(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """Standard ECA (no stickiness)."""
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


# =============================================================================
# METRICS
# =============================================================================

def compute_control(history: np.ndarray) -> float:
    """Global Control measure."""
    T, W = history.shape
    pattern_outcomes = defaultdict(list)

    for t in range(50, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_outcomes[pattern].append(outcome)

    divergences = []
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) >= 5:
            mean = np.mean(outcomes)
            div = 4 * mean * (1 - mean)
            divergences.append(div)

    return np.mean(divergences) if divergences else 0.0


def compute_boundary_density(history: np.ndarray, window: int = 5) -> float:
    """Fraction of space-time that is boundary."""
    T, W = history.shape
    if T <= window:
        return 0.0

    boundary_map = np.zeros((T - window, W), dtype=float)

    for t in range(window, T):
        local_activity = np.zeros(W)
        for i in range(W):
            changes = np.sum(history[t-window:t, i] != history[t-window+1:t+1, i])
            local_activity[i] = changes / window

        for i in range(W):
            left_act = local_activity[(i-1) % W]
            right_act = local_activity[(i+1) % W]
            gradient = abs(left_act - right_act)
            boundary_map[t - window, i] = gradient

    threshold = np.percentile(boundary_map, 70)
    return np.mean(boundary_map > threshold)


def compute_bulk_fraction(history: np.ndarray, window: int = 5) -> float:
    """Fraction of space-time that is bulk (non-boundary)."""
    return 1.0 - compute_boundary_density(history, window)


def compute_activity(history: np.ndarray) -> float:
    """Fraction of cells changing per timestep."""
    if len(history) < 2:
        return 0.0
    changes = np.sum(history[1:] != history[:-1], axis=1)
    return np.mean(changes) / history.shape[1]


def compute_boundary_persistence(history: np.ndarray, window: int = 5) -> float:
    """Average boundary persistence time."""
    T, W = history.shape
    if T <= window:
        return 0.0

    boundary_map = np.zeros((T - window, W), dtype=float)

    for t in range(window, T):
        local_activity = np.zeros(W)
        for i in range(W):
            changes = np.sum(history[t-window:t, i] != history[t-window+1:t+1, i])
            local_activity[i] = changes / window

        for i in range(W):
            left_act = local_activity[(i-1) % W]
            right_act = local_activity[(i+1) % W]
            boundary_map[t - window, i] = abs(left_act - right_act)

    threshold = np.percentile(boundary_map, 70)
    is_boundary = boundary_map > threshold

    # Measure persistence
    persistence_values = []
    for i in range(W):
        run = 0
        for t in range(len(is_boundary)):
            if is_boundary[t, i]:
                run += 1
            else:
                if run > 0:
                    persistence_values.append(run)
                run = 0
        if run > 0:
            persistence_values.append(run)

    return np.mean(persistence_values) if persistence_values else 0.0


# =============================================================================
# MINIMALITY EXPERIMENTS
# =============================================================================

def find_minimal_stickiness(rule: int, width: int = 100, steps: int = 200) -> Dict:
    """
    Find minimum stickiness needed for Control > threshold.
    """
    print(f"  Searching for minimal stickiness for Rule {rule}...")

    control_threshold = 0.05  # Minimum Control to consider "present"

    results = {
        'confirmation': {},
        'refractory': {}
    }

    # Test confirmation depths 1, 2, 3, 4, 5
    for depth in [1, 2, 3, 4, 5]:
        history = variable_confirmation_eca(rule, width, steps, confirm_depth=depth)
        ctrl = compute_control(history)
        boundary_dens = compute_boundary_density(history)
        results['confirmation'][depth] = {
            'control': ctrl,
            'boundary_density': boundary_dens,
            'has_control': ctrl > control_threshold
        }

    # Test refractory times 1, 2, 3, 4, 5
    for time in [1, 2, 3, 4, 5]:
        history = variable_refractory_eca(rule, width, steps, refractory_time=time)
        ctrl = compute_control(history)
        boundary_dens = compute_boundary_density(history)
        results['refractory'][time] = {
            'control': ctrl,
            'boundary_density': boundary_dens,
            'has_control': ctrl > control_threshold
        }

    # Find minimum
    min_confirm = None
    for depth in sorted(results['confirmation'].keys()):
        if results['confirmation'][depth]['has_control']:
            min_confirm = depth
            break

    min_refract = None
    for time in sorted(results['refractory'].keys()):
        if results['refractory'][time]['has_control']:
            min_refract = time
            break

    results['minimal_confirmation'] = min_confirm
    results['minimal_refractory'] = min_refract

    return results


def find_minimal_boundary(rule: int, width: int = 100, steps: int = 200) -> Dict:
    """
    Find smallest boundary in space/time that sustains Control.
    """
    history = variable_confirmation_eca(rule, width, steps, confirm_depth=1)
    ctrl = compute_control(history)
    boundary_dens = compute_boundary_density(history)
    persistence = compute_boundary_persistence(history)

    # Compute boundary sizes
    T, W = history.shape
    window = 5

    if T <= window:
        return {
            'min_spatial_size': 0,
            'min_temporal_size': 0,
            'control': ctrl,
            'boundary_density': boundary_dens
        }

    boundary_map = np.zeros((T - window, W), dtype=float)
    for t in range(window, T):
        local_activity = np.zeros(W)
        for i in range(W):
            changes = np.sum(history[t-window:t, i] != history[t-window+1:t+1, i])
            local_activity[i] = changes / window

        for i in range(W):
            gradient = abs(local_activity[(i-1) % W] - local_activity[(i+1) % W])
            boundary_map[t - window, i] = gradient

    threshold = np.percentile(boundary_map, 70)
    is_boundary = boundary_map > threshold

    # Find minimum spatial extent of boundaries
    spatial_sizes = []
    for t in range(len(is_boundary)):
        labeled, n_regions = ndimage.label(is_boundary[t])
        for region_id in range(1, n_regions + 1):
            size = np.sum(labeled == region_id)
            spatial_sizes.append(size)

    # Find minimum temporal extent
    temporal_sizes = []
    for i in range(W):
        run = 0
        for t in range(len(is_boundary)):
            if is_boundary[t, i]:
                run += 1
            else:
                if run > 0:
                    temporal_sizes.append(run)
                run = 0
        if run > 0:
            temporal_sizes.append(run)

    return {
        'min_spatial_size': min(spatial_sizes) if spatial_sizes else 0,
        'mean_spatial_size': np.mean(spatial_sizes) if spatial_sizes else 0,
        'min_temporal_size': min(temporal_sizes) if temporal_sizes else 0,
        'mean_temporal_size': np.mean(temporal_sizes) if temporal_sizes else 0,
        'control': ctrl,
        'boundary_density': boundary_dens,
        'persistence': persistence
    }


def test_boundary_only_systems(rules: List[int] = [110, 30, 90, 54],
                                width: int = 100, steps: int = 200) -> Dict:
    """
    Create systems where bulk is minimized (everything is boundary).
    Test if Control saturates, fragments, or collapses.
    """
    print("\n  Testing boundary-dominated systems...")

    results = {}

    for rule in rules:
        # High activity initial conditions (alternating)
        init_alternating = np.array([i % 2 for i in range(width)])

        # Random initial conditions
        init_random = np.random.randint(0, 2, width)

        configs = [
            ('standard_single', variable_confirmation_eca(rule, width, steps, 1)),
            ('alternating_init', variable_confirmation_eca(rule, width, steps, 1, init_alternating)),
            ('random_init', variable_confirmation_eca(rule, width, steps, 1, init_random)),
            ('high_stickiness', variable_confirmation_eca(rule, width, steps, 3)),
        ]

        results[rule] = {}

        for name, history in configs:
            ctrl = compute_control(history)
            boundary_dens = compute_boundary_density(history)
            bulk = compute_bulk_fraction(history)
            activity = compute_activity(history)

            results[rule][name] = {
                'control': ctrl,
                'boundary_density': boundary_dens,
                'bulk_fraction': bulk,
                'activity': activity
            }

            # Check for Control saturation/collapse
            if bulk < 0.1:
                behavior = 'boundary_dominated'
            elif bulk > 0.7:
                behavior = 'bulk_dominated'
            else:
                behavior = 'mixed'

            results[rule][name]['behavior'] = behavior

    return results


def analyze_control_vs_boundary_ratio(rules: List[int] = [110, 30, 90, 54],
                                       width: int = 100, steps: int = 200) -> Dict:
    """
    Systematically vary boundary/bulk ratio and measure Control.
    """
    print("\n  Analyzing Control vs boundary ratio...")

    results = {}

    for rule in rules:
        results[rule] = []

        # Vary stickiness to change boundary density
        for confirm_depth in [1, 2, 3, 4, 5]:
            for init_type in ['single', 'random']:
                if init_type == 'single':
                    init = np.zeros(width, dtype=int)
                    init[width // 2] = 1
                else:
                    init = np.random.randint(0, 2, width)

                history = variable_confirmation_eca(rule, width, steps, confirm_depth, init)

                ctrl = compute_control(history)
                boundary_dens = compute_boundary_density(history)

                results[rule].append({
                    'confirm_depth': confirm_depth,
                    'init_type': init_type,
                    'control': ctrl,
                    'boundary_density': boundary_dens
                })

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_minimal_stickiness(results: Dict, rules: List[int]):
    """Plot Control vs stickiness level."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confirmation
    ax = axes[0]
    for rule in rules:
        if rule in results:
            depths = sorted(results[rule]['confirmation'].keys())
            controls = [results[rule]['confirmation'][d]['control'] for d in depths]
            ax.plot(depths, controls, 'o-', label=f'Rule {rule}')

    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Confirmation Depth')
    ax.set_ylabel('Control')
    ax.set_title('Minimal Confirmation Depth for Control')
    ax.legend()

    # Refractory
    ax = axes[1]
    for rule in rules:
        if rule in results:
            times = sorted(results[rule]['refractory'].keys())
            controls = [results[rule]['refractory'][t]['control'] for t in times]
            ax.plot(times, controls, 'o-', label=f'Rule {rule}')

    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Refractory Time')
    ax.set_ylabel('Control')
    ax.set_title('Minimal Refractory Time for Control')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'minimal_stickiness.png', dpi=150)
    plt.close()


def visualize_control_vs_boundary(results: Dict, rules: List[int]):
    """Plot Control vs boundary density."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(rules)))

    for idx, rule in enumerate(rules):
        if rule in results:
            boundary_densities = [r['boundary_density'] for r in results[rule]]
            controls = [r['control'] for r in results[rule]]
            ax.scatter(boundary_densities, controls, c=[colors[idx]], label=f'Rule {rule}', alpha=0.7, s=50)

    ax.set_xlabel('Boundary Density')
    ax.set_ylabel('Control')
    ax.set_title('Control vs Boundary Density\n(varying stickiness and initial conditions)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'control_vs_boundary_density.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE C: MINIMALITY EXPERIMENTS")
    print("=" * 70)

    rules = [110, 30, 90, 54]

    # Part 1: Find minimal stickiness
    print("\nPart 1: Minimal Stickiness Search")
    stickiness_results = {}
    for rule in rules:
        stickiness_results[rule] = find_minimal_stickiness(rule)
        print(f"  Rule {rule}: min_confirm={stickiness_results[rule]['minimal_confirmation']}, "
              f"min_refract={stickiness_results[rule]['minimal_refractory']}")

    # Part 2: Find minimal boundary size
    print("\nPart 2: Minimal Boundary Size")
    boundary_results = {}
    for rule in rules:
        boundary_results[rule] = find_minimal_boundary(rule)
        print(f"  Rule {rule}: min_spatial={boundary_results[rule]['min_spatial_size']}, "
              f"min_temporal={boundary_results[rule]['min_temporal_size']}")

    # Part 3: Boundary-only systems
    print("\nPart 3: Boundary-Only Systems")
    boundary_only_results = test_boundary_only_systems(rules)

    for rule in rules:
        print(f"\n  Rule {rule}:")
        for config, data in boundary_only_results[rule].items():
            print(f"    {config}: bulk={data['bulk_fraction']:.2f}, ctrl={data['control']:.3f}, "
                  f"behavior={data['behavior']}")

    # Part 4: Control vs boundary ratio
    ratio_results = analyze_control_vs_boundary_ratio(rules)

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
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    all_results = {
        'stickiness': convert(stickiness_results),
        'boundary_size': convert(boundary_results),
        'boundary_only': convert(boundary_only_results),
        'ratio': convert(ratio_results)
    }

    with open(OUTPUT_DIR / 'minimality_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_minimal_stickiness(stickiness_results, rules)
    visualize_control_vs_boundary(ratio_results, rules)
    print("  Saved minimality visualizations")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE C SUMMARY")
    print("=" * 70)

    print("\nMinimal Stickiness for Control:")
    for rule in rules:
        min_c = stickiness_results[rule]['minimal_confirmation']
        min_r = stickiness_results[rule]['minimal_refractory']
        print(f"  Rule {rule}: Confirmation >= {min_c}, Refractory >= {min_r}")

    print("\nMinimal Boundary Size:")
    for rule in rules:
        min_s = boundary_results[rule]['min_spatial_size']
        min_t = boundary_results[rule]['min_temporal_size']
        print(f"  Rule {rule}: Spatial >= {min_s} cells, Temporal >= {min_t} steps")

    print("\nBoundary-Only System Behavior:")
    for rule in rules:
        behaviors = [d['behavior'] for d in boundary_only_results[rule].values()]
        dominant = max(set(behaviors), key=behaviors.count)
        print(f"  Rule {rule}: {dominant}")

    print("\n" + "=" * 70)
    print("PHASE C COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
