"""
Phase A: Boundary Invariants

Primary question: Is Control necessarily a boundary phenomenon?

Part 1: Boundary Definition Sweep
- Enumerate all boundary types
- For each measure: Control magnitude, persistence, width, stability

Part 2: Boundary Necessity Test
- Construct cases with stickiness + motion + NO persistent boundaries
- Test if Control collapses to noise or zero without boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import json
from pathlib import Path
from collections import defaultdict
from scipy import ndimage

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "phase_a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CA IMPLEMENTATIONS
# =============================================================================

def confirmation_eca(rule: int, width: int, steps: int, init=None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (history, pending_history) for boundary analysis."""
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    pending_history = np.zeros((steps, width), dtype=int)  # Track hidden state
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

        pending_history[t] = (pending >= 0).astype(int)

    return history, pending_history


def refractory_eca(rule: int, width: int, steps: int, refractory_time: int = 2, init=None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (history, cooldown_history)."""
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    cooldown_history = np.zeros((steps, width), dtype=int)
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

        cooldown_history[t] = cooldown.copy()

    return history, cooldown_history


# =============================================================================
# BOUNDARY TYPE DETECTION
# =============================================================================

def detect_activity_immobility_boundaries(history: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Type 1: Activity-Immobility boundaries
    Boundary where active region meets static region.
    """
    T, W = history.shape
    boundary_map = np.zeros((T - window, W), dtype=float)

    for t in range(window, T):
        # Compute local activity
        local_activity = np.zeros(W)
        for i in range(W):
            changes = np.sum(history[t-window:t, i] != history[t-window+1:t+1, i])
            local_activity[i] = changes / window

        # Boundary = high gradient in activity
        for i in range(W):
            left_act = local_activity[(i-1) % W]
            right_act = local_activity[(i+1) % W]
            gradient = abs(left_act - right_act)
            boundary_map[t - window, i] = gradient

    return boundary_map


def detect_phase_shift_boundaries(history: np.ndarray) -> np.ndarray:
    """
    Type 2: Phase-shift boundaries
    Boundary between regions with different local patterns/phases.
    """
    T, W = history.shape
    boundary_map = np.zeros((T, W), dtype=float)

    for t in range(1, T):
        for i in range(2, W - 2):
            # Compare local pattern on left vs right
            left_pattern = tuple(history[t, i-2:i+1])
            right_pattern = tuple(history[t, i:i+3])

            # Phase shift = patterns are different
            if left_pattern != right_pattern:
                boundary_map[t, i] = 1.0

    return boundary_map


def detect_density_gradient_boundaries(history: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Type 3: Density gradient boundaries
    Boundary where local density (fraction of 1s) changes sharply.
    """
    T, W = history.shape
    boundary_map = np.zeros((T, W), dtype=float)

    for t in range(T):
        # Compute local density
        local_density = np.zeros(W)
        for i in range(W):
            left = max(0, i - window)
            right = min(W, i + window + 1)
            local_density[i] = np.mean(history[t, left:right])

        # Boundary = high gradient in density
        for i in range(W):
            left_dens = local_density[(i-1) % W]
            right_dens = local_density[(i+1) % W]
            gradient = abs(left_dens - right_dens)
            boundary_map[t, i] = gradient

    return boundary_map


def detect_symmetry_break_boundaries(history: np.ndarray) -> np.ndarray:
    """
    Type 4: Symmetry break boundaries
    Where local left-right symmetry is broken.
    """
    T, W = history.shape
    boundary_map = np.zeros((T, W), dtype=float)

    for t in range(T):
        for i in range(3, W - 3):
            # Check symmetry around position i
            left_side = history[t, i-3:i]
            right_side = history[t, i+1:i+4][::-1]  # Reversed

            asymmetry = np.sum(left_side != right_side) / 3
            boundary_map[t, i] = asymmetry

    return boundary_map


def detect_temporal_hysteresis_boundaries(history: np.ndarray, pending_history: np.ndarray) -> np.ndarray:
    """
    Type 5: Temporal hysteresis edges
    Where pending/cooldown state creates temporal boundary.
    """
    T, W = history.shape
    boundary_map = np.zeros((T-1, W), dtype=float)

    for t in range(1, T):
        for i in range(W):
            # Hysteresis boundary: pending state differs from neighbors
            left_pend = pending_history[t][(i-1) % W]
            center_pend = pending_history[t][i]
            right_pend = pending_history[t][(i+1) % W]

            if center_pend != left_pend or center_pend != right_pend:
                boundary_map[t-1, i] = 1.0

    return boundary_map


# =============================================================================
# CONTROL MEASUREMENT AT BOUNDARIES
# =============================================================================

def compute_local_control(history: np.ndarray, t: int, i: int, window: int = 20) -> float:
    """Compute Control at a specific location."""
    T, W = history.shape

    # Gather pattern-outcome pairs in neighborhood
    pattern_outcomes = defaultdict(list)

    t_start = max(1, t - window)
    t_end = min(T - 1, t + window)

    for tt in range(t_start, t_end):
        for ii in range(max(1, i-2), min(W-1, i+3)):
            pattern = (history[tt, ii-1], history[tt, ii], history[tt, ii+1])
            outcome = history[tt+1, ii]
            pattern_outcomes[pattern].append(outcome)

    # Compute divergence
    divergences = []
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) >= 3:
            mean = np.mean(outcomes)
            div = 4 * mean * (1 - mean)
            divergences.append(div)

    return np.mean(divergences) if divergences else 0.0


def measure_boundary_control(history: np.ndarray, boundary_map: np.ndarray) -> Dict:
    """Measure Control magnitude at boundary vs non-boundary regions."""
    T, W = history.shape

    # Align dimensions
    min_t = min(boundary_map.shape[0], T - 2)
    min_w = min(boundary_map.shape[1], W - 2)

    boundary_threshold = np.percentile(boundary_map[:min_t, :min_w], 80)

    boundary_controls = []
    non_boundary_controls = []

    # Sample positions
    sample_times = np.linspace(20, min_t - 1, min(20, min_t - 21)).astype(int)
    sample_positions = np.linspace(5, min_w - 5, min(20, min_w - 10)).astype(int)

    for t in sample_times:
        for i in sample_positions:
            ctrl = compute_local_control(history, t, i)

            if boundary_map[t, i] > boundary_threshold:
                boundary_controls.append(ctrl)
            else:
                non_boundary_controls.append(ctrl)

    return {
        'boundary_control_mean': np.mean(boundary_controls) if boundary_controls else 0,
        'boundary_control_std': np.std(boundary_controls) if boundary_controls else 0,
        'non_boundary_control_mean': np.mean(non_boundary_controls) if non_boundary_controls else 0,
        'non_boundary_control_std': np.std(non_boundary_controls) if non_boundary_controls else 0,
        'ratio': (np.mean(boundary_controls) / np.mean(non_boundary_controls)
                  if non_boundary_controls and np.mean(non_boundary_controls) > 0 else float('inf')),
        'n_boundary': len(boundary_controls),
        'n_non_boundary': len(non_boundary_controls)
    }


def measure_boundary_persistence(boundary_map: np.ndarray, threshold_percentile: float = 80) -> Dict:
    """Measure how long boundaries persist."""
    threshold = np.percentile(boundary_map, threshold_percentile)
    is_boundary = boundary_map > threshold

    # Track persistence at each position
    persistence_values = []

    for i in range(boundary_map.shape[1]):
        run_length = 0
        for t in range(boundary_map.shape[0]):
            if is_boundary[t, i]:
                run_length += 1
            else:
                if run_length > 0:
                    persistence_values.append(run_length)
                run_length = 0
        if run_length > 0:
            persistence_values.append(run_length)

    if not persistence_values:
        return {'mean': 0, 'max': 0, 'p90': 0}

    return {
        'mean': np.mean(persistence_values),
        'max': np.max(persistence_values),
        'p90': np.percentile(persistence_values, 90),
        'count': len(persistence_values)
    }


def measure_boundary_width(boundary_map: np.ndarray, threshold_percentile: float = 80) -> Dict:
    """Measure spatial width of boundaries."""
    threshold = np.percentile(boundary_map, threshold_percentile)
    is_boundary = boundary_map > threshold

    # For each time step, measure connected boundary widths
    widths = []

    for t in range(boundary_map.shape[0]):
        labeled, n_regions = ndimage.label(is_boundary[t])
        for region_id in range(1, n_regions + 1):
            region_size = np.sum(labeled == region_id)
            widths.append(region_size)

    if not widths:
        return {'mean': 0, 'max': 0, 'mode': 0}

    # Mode (most common width)
    unique, counts = np.unique(widths, return_counts=True)
    mode_width = unique[np.argmax(counts)]

    return {
        'mean': np.mean(widths),
        'max': np.max(widths),
        'mode': mode_width,
        'fraction_single_cell': np.mean(np.array(widths) == 1)
    }


def measure_boundary_stability(history: np.ndarray, boundary_map: np.ndarray,
                                perturbation_prob: float = 0.01) -> Dict:
    """
    Measure stability of boundaries under perturbation.
    Run perturbed version and compare boundary locations.
    """
    T, W = history.shape

    # Create perturbed initial condition
    init_perturbed = history[0].copy()
    perturbation_mask = np.random.random(W) < perturbation_prob
    init_perturbed[perturbation_mask] = 1 - init_perturbed[perturbation_mask]

    # This requires knowing which rule was used - we'll use a proxy
    # by measuring how much boundary_map changes with small history changes

    # Compute boundary map stability via autocorrelation in time
    if boundary_map.shape[0] < 10:
        return {'temporal_stability': 0}

    # Correlation between consecutive boundary maps
    correlations = []
    for t in range(1, min(50, boundary_map.shape[0])):
        corr = np.corrcoef(boundary_map[t-1].flatten(), boundary_map[t].flatten())[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    return {
        'temporal_stability': np.mean(correlations) if correlations else 0,
        'stability_std': np.std(correlations) if correlations else 0
    }


# =============================================================================
# BOUNDARY NECESSITY TEST
# =============================================================================

def create_no_boundary_sticky_system(width: int = 100, steps: int = 200) -> Tuple[np.ndarray, Dict]:
    """
    Attempt to create a system with:
    - Stickiness present
    - Motion present
    - NO persistent boundaries

    Strategy: Use a rule that creates uniform activity + confirmation stickiness
    """
    # Rule 30 is chaotic with uniform activity distribution
    # We'll try to create conditions where boundaries don't persist

    # Random initial condition for uniform activity
    init = np.random.randint(0, 2, width)

    history, pending = confirmation_eca(30, width, steps, init)

    # Measure boundary presence
    activity_boundary = detect_activity_immobility_boundaries(history)
    density_boundary = detect_density_gradient_boundaries(history)

    # Compute overall boundary density
    activity_threshold = np.percentile(activity_boundary, 80)
    density_threshold = np.percentile(density_boundary, 80)

    boundary_fraction = (np.mean(activity_boundary > activity_threshold) +
                         np.mean(density_boundary > density_threshold)) / 2

    # Measure Control
    pattern_outcomes = defaultdict(list)
    for t in range(50, steps - 1):
        for i in range(1, width - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_outcomes[pattern].append(outcome)

    divergences = []
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) >= 5:
            mean = np.mean(outcomes)
            div = 4 * mean * (1 - mean)
            divergences.append(div)

    control = np.mean(divergences) if divergences else 0

    # Measure activity (motion)
    activity = np.mean(history[1:] != history[:-1])

    return history, {
        'boundary_fraction': boundary_fraction,
        'control': control,
        'activity': activity,
        'has_stickiness': True,
        'has_motion': activity > 0.1,
        'has_persistent_boundaries': boundary_fraction > 0.1
    }


def boundary_necessity_sweep(rules: List[int] = [30, 45, 73, 105, 150],
                             width: int = 100, steps: int = 200) -> List[Dict]:
    """
    Sweep rules looking for: stickiness + motion + no persistent boundaries.
    Test if Control collapses without boundaries.
    """
    results = []

    for rule in rules:
        # Random init for maximum mixing
        init = np.random.randint(0, 2, width)

        history, pending = confirmation_eca(rule, width, steps, init)

        # Measure all boundary types
        activity_boundary = detect_activity_immobility_boundaries(history)
        density_boundary = detect_density_gradient_boundaries(history)

        # Combined boundary measure
        combined_boundary = (activity_boundary[:min(len(activity_boundary), len(density_boundary))] +
                            density_boundary[:min(len(activity_boundary), len(density_boundary))]) / 2

        boundary_fraction = np.mean(combined_boundary > np.percentile(combined_boundary, 70))

        # Measure Control
        pattern_outcomes = defaultdict(list)
        for t in range(50, steps - 1):
            for i in range(1, width - 1):
                pattern = (history[t, i-1], history[t, i], history[t, i+1])
                outcome = history[t+1, i]
                pattern_outcomes[pattern].append(outcome)

        divergences = []
        for pattern, outcomes in pattern_outcomes.items():
            if len(outcomes) >= 5:
                mean = np.mean(outcomes)
                div = 4 * mean * (1 - mean)
                divergences.append(div)

        control = np.mean(divergences) if divergences else 0
        activity = np.mean(history[1:] != history[:-1])

        results.append({
            'rule': rule,
            'boundary_fraction': boundary_fraction,
            'control': control,
            'activity': activity,
            'control_per_boundary': control / max(boundary_fraction, 0.001)
        })

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_boundary_definition_sweep(rules: List[int] = [110, 30, 90, 54],
                                   width: int = 100, steps: int = 200):
    """Comprehensive boundary type analysis."""
    print("=" * 70)
    print("PHASE A.1: BOUNDARY DEFINITION SWEEP")
    print("=" * 70)

    all_results = {}

    boundary_types = [
        ('activity_immobility', detect_activity_immobility_boundaries),
        ('phase_shift', detect_phase_shift_boundaries),
        ('density_gradient', detect_density_gradient_boundaries),
        ('symmetry_break', detect_symmetry_break_boundaries),
    ]

    for rule in rules:
        print(f"\nRule {rule}:")
        history, pending = confirmation_eca(rule, width, steps)

        # Add temporal hysteresis (needs pending)
        all_boundary_types = boundary_types + [
            ('temporal_hysteresis', lambda h: detect_temporal_hysteresis_boundaries(h, pending))
        ]

        all_results[rule] = {}

        for btype_name, btype_func in all_boundary_types:
            try:
                boundary_map = btype_func(history)

                control_stats = measure_boundary_control(history, boundary_map)
                persistence = measure_boundary_persistence(boundary_map)
                width_stats = measure_boundary_width(boundary_map)
                stability = measure_boundary_stability(history, boundary_map)

                result = {
                    'control': control_stats,
                    'persistence': persistence,
                    'width': width_stats,
                    'stability': stability
                }

                all_results[rule][btype_name] = result

                print(f"  {btype_name}:")
                print(f"    Control at boundary: {control_stats['boundary_control_mean']:.4f}")
                print(f"    Control ratio (boundary/non): {control_stats['ratio']:.2f}")
                print(f"    Persistence (mean): {persistence['mean']:.1f} steps")
                print(f"    Width (mode): {width_stats['mode']:.1f} cells")
                print(f"    Stability: {stability['temporal_stability']:.3f}")

            except Exception as e:
                print(f"  {btype_name}: ERROR - {e}")
                all_results[rule][btype_name] = {'error': str(e)}

    return all_results


def run_boundary_necessity_test():
    """Test if boundaries are NECESSARY for Control."""
    print("\n" + "=" * 70)
    print("PHASE A.2: BOUNDARY NECESSITY TEST")
    print("=" * 70)
    print("\nQuestion: Can Control exist without persistent boundaries?")

    # Test many rules looking for boundary-free Control
    test_rules = list(range(0, 256, 8))  # Sample every 8th rule

    results = boundary_necessity_sweep(test_rules)

    # Find cases with low boundary but high Control
    print("\nSearching for high Control + low boundary cases...")

    anomalies = []
    for r in results:
        if r['control'] > 0.2 and r['boundary_fraction'] < 0.15:
            anomalies.append(r)
            print(f"  Rule {r['rule']}: Control={r['control']:.3f}, Boundary={r['boundary_fraction']:.3f}")

    if not anomalies:
        print("  No anomalies found - Control appears to require boundaries")

    # Correlation analysis
    controls = [r['control'] for r in results]
    boundaries = [r['boundary_fraction'] for r in results]

    if len(controls) > 5:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(controls, boundaries)
        spearman_r, spearman_p = spearmanr(controls, boundaries)

        print(f"\nCorrelation (Control vs Boundary):")
        print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
        print(f"  Spearman r = {spearman_r:.3f} (p = {spearman_p:.4f})")

    return results, anomalies


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_boundary_types(rule: int, width: int = 100, steps: int = 150):
    """Visualize all boundary types for a rule."""
    history, pending = confirmation_eca(rule, width, steps)

    boundary_maps = {
        'Activity-Immobility': detect_activity_immobility_boundaries(history),
        'Phase Shift': detect_phase_shift_boundaries(history),
        'Density Gradient': detect_density_gradient_boundaries(history),
        'Symmetry Break': detect_symmetry_break_boundaries(history),
        'Temporal Hysteresis': detect_temporal_hysteresis_boundaries(history, pending),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Spacetime diagram
    axes[0].imshow(history, cmap='binary', aspect='auto')
    axes[0].set_title(f'Rule {rule}: Spacetime')
    axes[0].set_ylabel('Time')

    # Boundary maps
    for idx, (name, bmap) in enumerate(boundary_maps.items()):
        ax = axes[idx + 1]
        im = ax.imshow(bmap, cmap='hot', aspect='auto')
        ax.set_title(name)
        if idx >= 3:
            ax.set_xlabel('Position')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'boundary_types_rule{rule}.png', dpi=150)
    plt.close()


def visualize_necessity_test(results: List[Dict]):
    """Plot Control vs Boundary fraction."""
    fig, ax = plt.subplots(figsize=(10, 8))

    controls = [r['control'] for r in results]
    boundaries = [r['boundary_fraction'] for r in results]
    rules = [r['rule'] for r in results]

    scatter = ax.scatter(boundaries, controls, c=rules, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Rule Number')

    ax.set_xlabel('Boundary Fraction')
    ax.set_ylabel('Control')
    ax.set_title('Boundary Necessity Test:\nDoes Control require boundaries?')

    # Add trend line
    if len(controls) > 5:
        z = np.polyfit(boundaries, controls, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(boundaries), max(boundaries), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.2f})')
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boundary_necessity_scatter.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE A: BOUNDARY INVARIANTS")
    print("=" * 70)

    # Part 1: Boundary Definition Sweep
    sweep_results = run_boundary_definition_sweep(rules=[110, 30, 90, 54])

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

    with open(OUTPUT_DIR / 'boundary_sweep_results.json', 'w') as f:
        json.dump(convert(sweep_results), f, indent=2)

    # Part 2: Boundary Necessity Test
    necessity_results, anomalies = run_boundary_necessity_test()

    with open(OUTPUT_DIR / 'boundary_necessity_results.json', 'w') as f:
        json.dump(convert(necessity_results), f, indent=2)

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    for rule in [110, 30, 54]:
        visualize_boundary_types(rule)
        print(f"  Saved boundary types for Rule {rule}")

    visualize_necessity_test(necessity_results)
    print("  Saved necessity scatter plot")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE A SUMMARY")
    print("=" * 70)

    print("\nBoundary Type Analysis:")
    for rule in [110, 30, 90, 54]:
        if rule in sweep_results:
            print(f"\n  Rule {rule}:")
            for btype, data in sweep_results[rule].items():
                if 'error' not in data:
                    ctrl = data['control']['boundary_control_mean']
                    ratio = data['control']['ratio']
                    persist = data['persistence']['mean']
                    print(f"    {btype}: Ctrl={ctrl:.3f}, Ratio={ratio:.2f}, Persist={persist:.1f}")

    print("\nBoundary Necessity Finding:")
    if not anomalies:
        print("  No cases found with high Control and low boundaries.")
        print("  OBSERVATION: Boundaries appear NECESSARY for Control.")
    else:
        print(f"  Found {len(anomalies)} potential anomalies - investigate further.")

    print("\n" + "=" * 70)
    print("PHASE A COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
