"""
Phase 2: Distinguish Constructive Control from Noise

Key question: Does Control remain attached to structures, or does it wash
through everything? This distinguishes computation from noise.

Metrics:
1. Persistent Control: Divergence that survives N timesteps
2. Control Locality: Spatial autocorrelation of Control events
3. Structure-Attached Control: Is Control higher near identifiable structures?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from scipy.stats import pearsonr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "phase2"
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
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1
    history = np.zeros((steps, width), dtype=int)
    confirm_count = np.zeros(width, dtype=int)
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


# =============================================================================
# METRIC 1: PERSISTENT CONTROL
# =============================================================================

def compute_control_events(history: np.ndarray) -> np.ndarray:
    """
    Compute where Control events occur - locations where the same
    pattern led to different outcomes.

    Returns a 2D array of Control "intensity" at each (time, position).
    """
    T, W = history.shape
    control_map = np.zeros((T-2, W-2), dtype=float)

    # For each cell, track what pattern led to what outcome
    pattern_at_location: Dict[Tuple[int, int], List[Tuple[Tuple, int]]] = defaultdict(list)

    for t in range(1, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_at_location[(t, i)].append((pattern, outcome))

    # Build a pattern → outcomes map for each pattern
    pattern_outcomes_global: Dict[Tuple, List[int]] = defaultdict(list)
    for t in range(1, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_outcomes_global[pattern].append(outcome)

    # Compute divergence for each pattern
    pattern_divergence = {}
    for pattern, outcomes in pattern_outcomes_global.items():
        if len(outcomes) >= 5:
            mean = np.mean(outcomes)
            pattern_divergence[pattern] = 4 * mean * (1 - mean)
        else:
            pattern_divergence[pattern] = 0

    # Assign Control intensity to each cell based on its pattern's divergence
    for t in range(1, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            control_map[t-1, i-1] = pattern_divergence.get(pattern, 0)

    return control_map


def compute_persistent_control(history: np.ndarray, persistence_windows: List[int] = [10, 50, 100]) -> Dict[str, float]:
    """
    Measure how much Control persists over time.

    A Control event at time t is "persistent" if the divergent behavior
    continues for N timesteps.
    """
    T, W = history.shape
    results = {}

    # First, identify Control "events" - positions where divergence happens
    control_map = compute_control_events(history)

    for N in persistence_windows:
        if T < N + 5:
            results[f'persistent_{N}'] = 0
            continue

        # Count Control events that persist
        persistent_count = 0
        total_count = 0

        # Slide through time in windows of size N
        for t_start in range(0, len(control_map) - N, N // 2):
            window = control_map[t_start:t_start + N]

            # A position has "persistent Control" if Control is elevated
            # throughout the window
            mean_control_per_position = np.mean(window, axis=0)

            # Count positions with sustained Control
            for i, mean_ctrl in enumerate(mean_control_per_position):
                total_count += 1
                if mean_ctrl > 0.1:  # Threshold for "elevated" Control
                    persistent_count += 1

        results[f'persistent_{N}'] = persistent_count / max(total_count, 1)

    return results


# =============================================================================
# METRIC 2: CONTROL LOCALITY
# =============================================================================

def compute_control_locality(history: np.ndarray) -> Dict[str, float]:
    """
    Measure spatial autocorrelation of Control.

    If Control clusters spatially → computation (localized processing)
    If Control diffuses uniformly → noise (random fluctuation)
    """
    control_map = compute_control_events(history)

    if control_map.size == 0:
        return {'spatial_autocorr': 0, 'cluster_ratio': 0}

    # Flatten to spatial dimension (average over time)
    spatial_control = np.mean(control_map, axis=0)

    # Spatial autocorrelation at lag 1
    if len(spatial_control) < 5:
        autocorr = 0
    else:
        # Pearson correlation between position i and position i+1
        try:
            autocorr, _ = pearsonr(spatial_control[:-1], spatial_control[1:])
            if np.isnan(autocorr):
                autocorr = 0
        except:
            autocorr = 0

    # Cluster ratio: ratio of high-Control regions that are adjacent
    # to other high-Control regions vs isolated
    threshold = np.percentile(spatial_control, 75) if len(spatial_control) > 0 else 0
    high_control = spatial_control > threshold

    if np.sum(high_control) < 2:
        cluster_ratio = 0
    else:
        # Count adjacent pairs
        adjacent_pairs = 0
        total_high = np.sum(high_control)
        for i in range(len(high_control) - 1):
            if high_control[i] and high_control[i + 1]:
                adjacent_pairs += 1

        cluster_ratio = adjacent_pairs / max(total_high - 1, 1)

    return {
        'spatial_autocorr': autocorr,
        'cluster_ratio': cluster_ratio
    }


# =============================================================================
# METRIC 3: STRUCTURE-ATTACHED CONTROL
# =============================================================================

def identify_structures(history: np.ndarray) -> np.ndarray:
    """
    Identify "structures" in the CA - regions of elevated activity.

    Structures are defined as:
    - Edges: boundaries between 0 and 1 regions
    - Activity hotspots: regions with high temporal change rate
    """
    T, W = history.shape
    structure_map = np.zeros((T-1, W), dtype=float)

    for t in range(1, T):
        for i in range(W):
            # Edge detection: XOR with neighbors
            left = history[t][(i - 1) % W]
            right = history[t][(i + 1) % W]
            center = history[t][i]

            is_edge = (center != left) or (center != right)

            # Temporal activity: did this cell change?
            changed = history[t][i] != history[t-1][i]

            # Structure score combines edge and activity
            structure_map[t-1, i] = 0.5 * int(is_edge) + 0.5 * int(changed)

    return structure_map


def compute_structure_attached_control(history: np.ndarray) -> Dict[str, float]:
    """
    Measure whether Control is higher near structures vs empty regions.
    """
    control_map = compute_control_events(history)
    structure_map = identify_structures(history)

    # Align dimensions (structure_map is T-1 x W, control_map is T-2 x W-2)
    # Trim structure_map to match
    if structure_map.shape[0] > control_map.shape[0]:
        structure_map = structure_map[:control_map.shape[0], :]
    if structure_map.shape[1] > control_map.shape[1] + 2:
        structure_map = structure_map[:, 1:-1]
    elif structure_map.shape[1] > control_map.shape[1]:
        diff = structure_map.shape[1] - control_map.shape[1]
        structure_map = structure_map[:, diff//2:-(diff - diff//2) or None]

    if control_map.size == 0 or structure_map.size == 0:
        return {'control_near_structure': 0, 'control_in_empty': 0, 'ratio': 0}

    # Ensure same shape
    min_t = min(control_map.shape[0], structure_map.shape[0])
    min_w = min(control_map.shape[1], structure_map.shape[1])
    control_map = control_map[:min_t, :min_w]
    structure_map = structure_map[:min_t, :min_w]

    # Identify structure vs empty regions
    structure_threshold = np.percentile(structure_map, 70)
    is_structure = structure_map > structure_threshold
    is_empty = structure_map < np.percentile(structure_map, 30)

    # Average Control in each region
    control_near_structure = np.mean(control_map[is_structure]) if np.any(is_structure) else 0
    control_in_empty = np.mean(control_map[is_empty]) if np.any(is_empty) else 0

    # Ratio
    if control_in_empty > 0:
        ratio = control_near_structure / control_in_empty
    elif control_near_structure > 0:
        ratio = float('inf')
    else:
        ratio = 1.0

    return {
        'control_near_structure': control_near_structure,
        'control_in_empty': control_in_empty,
        'ratio': ratio
    }


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

def analyze_control_quality(rule: int, width: int = 100, steps: int = 200,
                             mechanisms: Dict = None) -> Dict:
    """Analyze Control quality for a rule under different mechanisms."""
    if mechanisms is None:
        mechanisms = {
            'standard': lambda: standard_eca(rule, width, steps),
            'confirmation': lambda: confirmation_eca(rule, width, steps),
            'refractory_2': lambda: refractory_eca(rule, width, steps, 2),
        }

    results = {'rule': rule}

    for mech_name, mech_func in mechanisms.items():
        history = mech_func()

        # Basic Control
        control_map = compute_control_events(history)
        basic_control = np.mean(control_map)

        # Persistent Control
        persistent = compute_persistent_control(history)

        # Locality
        locality = compute_control_locality(history)

        # Structure attachment
        structure_attached = compute_structure_attached_control(history)

        results[mech_name] = {
            'basic_control': basic_control,
            **persistent,
            **locality,
            **structure_attached
        }

    return results


def run_phase2_analysis(rules: List[int] = [110, 30, 90, 54, 62, 150],
                        width: int = 100, steps: int = 250):
    """Run Phase 2 analysis on selected rules."""
    print("=" * 70)
    print("PHASE 2: CONTROL QUALITY ANALYSIS")
    print("=" * 70)

    all_results = {}

    for rule in rules:
        print(f"\nAnalyzing Rule {rule}...")
        results = analyze_control_quality(rule, width, steps)
        all_results[rule] = results

        for mech_name in ['standard', 'confirmation', 'refractory_2']:
            m = results[mech_name]
            print(f"  {mech_name}:")
            print(f"    Basic Control: {m['basic_control']:.4f}")
            print(f"    Persistent (10): {m.get('persistent_10', 0):.4f}")
            print(f"    Persistent (50): {m.get('persistent_50', 0):.4f}")
            print(f"    Spatial Autocorr: {m['spatial_autocorr']:.4f}")
            print(f"    Structure Ratio: {m['ratio']:.2f}")

    return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_control_maps(rule: int, width: int = 100, steps: int = 150):
    """Visualize Control distribution in space and time."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    mechanisms = [
        ('Standard', standard_eca(rule, width, steps)),
        ('Confirmation', confirmation_eca(rule, width, steps)),
        ('Refractory', refractory_eca(rule, width, steps, 2)),
    ]

    for idx, (name, history) in enumerate(mechanisms):
        # Top row: spacetime diagram
        axes[0, idx].imshow(history, cmap='binary', aspect='auto')
        axes[0, idx].set_title(f'{name}: Spacetime')
        axes[0, idx].set_ylabel('Time')

        # Bottom row: Control map
        control_map = compute_control_events(history)
        im = axes[1, idx].imshow(control_map, cmap='hot', aspect='auto', vmin=0, vmax=1)
        axes[1, idx].set_title(f'{name}: Control Map')
        axes[1, idx].set_ylabel('Time')
        axes[1, idx].set_xlabel('Position')
        plt.colorbar(im, ax=axes[1, idx], label='Control')

    plt.suptitle(f'Rule {rule}: Control Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'control_maps_rule{rule}.png', dpi=150)
    plt.close()


def visualize_control_quality_comparison(results: Dict):
    """Compare Control quality metrics across rules and mechanisms."""
    rules = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mechanisms = ['standard', 'confirmation', 'refractory_2']
    colors = {'standard': 'gray', 'confirmation': 'blue', 'refractory_2': 'green'}

    # 1. Basic Control
    ax = axes[0, 0]
    x = np.arange(len(rules))
    width_bar = 0.25
    for i, mech in enumerate(mechanisms):
        values = [results[r][mech]['basic_control'] for r in rules]
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[mech])
    ax.set_xticks(x + width_bar)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Basic Control')
    ax.set_title('Basic Control by Rule and Mechanism')
    ax.legend()

    # 2. Persistent Control (N=50)
    ax = axes[0, 1]
    for i, mech in enumerate(mechanisms):
        values = [results[r][mech].get('persistent_50', 0) for r in rules]
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[mech])
    ax.set_xticks(x + width_bar)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Persistent Control (N=50)')
    ax.set_title('Control that Persists 50+ Timesteps')
    ax.legend()

    # 3. Spatial Autocorrelation
    ax = axes[1, 0]
    for i, mech in enumerate(mechanisms):
        values = [results[r][mech]['spatial_autocorr'] for r in rules]
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[mech])
    ax.set_xticks(x + width_bar)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Spatial Autocorrelation')
    ax.set_title('Control Locality (Higher = More Clustered)')
    ax.legend()
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    # 4. Structure Attachment Ratio
    ax = axes[1, 1]
    for i, mech in enumerate(mechanisms):
        values = [min(results[r][mech]['ratio'], 5) for r in rules]  # Cap for visibility
        ax.bar(x + i * width_bar, values, width_bar, label=mech, color=colors[mech])
    ax.set_xticks(x + width_bar)
    ax.set_xticklabels([f'Rule {r}' for r in rules])
    ax.set_ylabel('Control Near Structure / In Empty')
    ax.set_title('Structure Attachment (Higher = More Structure-Bound)')
    ax.legend()
    ax.axhline(1, color='red', linestyle='--', alpha=0.5, label='Uniform')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'control_quality_comparison.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 2: DISTINGUISH CONSTRUCTIVE CONTROL FROM NOISE")
    print("=" * 70)

    # Analyze rules
    rules = [110, 30, 90, 54, 62, 150]
    results = run_phase2_analysis(rules)

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
        if obj == float('inf'):
            return "inf"
        return obj

    with open(OUTPUT_DIR / 'control_quality_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    for rule in [110, 54, 30]:
        visualize_control_maps(rule)
        print(f"  Saved control maps for Rule {rule}")

    visualize_control_quality_comparison(results)
    print("  Saved comparison figure")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)

    print("\nKey Question: Does Control cluster at structures or diffuse everywhere?")
    print("\nFindings by rule:")

    for rule in rules:
        conf = results[rule]['confirmation']
        print(f"\n  Rule {rule}:")
        print(f"    Control: {conf['basic_control']:.3f}")
        print(f"    Persistence (50): {conf.get('persistent_50', 0):.3f}")
        print(f"    Spatial clustering: {conf['spatial_autocorr']:.3f}")
        print(f"    Structure attachment: {conf['ratio']:.2f}x")

        if conf['ratio'] > 1.5 and conf['spatial_autocorr'] > 0.1:
            print(f"    => STRUCTURED CONTROL (clustered, structure-bound)")
        elif conf['ratio'] < 0.8 and conf['spatial_autocorr'] < 0.1:
            print(f"    => DIFFUSE CONTROL (noise-like)")
        else:
            print(f"    => MIXED BEHAVIOR")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
