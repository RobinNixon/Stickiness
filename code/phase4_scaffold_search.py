"""
Phase 4: Search for Scaffolds, Not Gliders

Change the success metric. Stop measuring mobile structures.
Start measuring immobile high-Control regions - this is where "building" lives.

Metrics:
1. Residence time distribution: Look for heavy tails (things that stay)
2. Immobile region emergence: Areas that persist without moving
3. Activity attractors: Structures that attract activity but don't move
4. High Control + Low Motion: Regions where Control is elevated but velocity is zero
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import json
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from scipy.stats import pearsonr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "phase4"
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


def confirmation_eca(rule: int, width: int, steps: int, init=None) -> np.ndarray:
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


def asymmetric_confirmation(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """Easy to stick, hard to unstick."""
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
                required = 1 if (current == 0 and proposed == 1) else 3

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


# =============================================================================
# SCAFFOLD METRICS
# =============================================================================

def compute_residence_times(history: np.ndarray) -> Dict:
    """
    Compute how long each position stays in the same state.
    Heavy-tailed distribution = scaffolds exist.
    """
    T, W = history.shape
    residence_times = []

    for col in range(W):
        current_run = 1
        for t in range(1, T):
            if history[t, col] == history[t-1, col]:
                current_run += 1
            else:
                residence_times.append(current_run)
                current_run = 1
        residence_times.append(current_run)

    residence_times = np.array(residence_times)

    return {
        'mean': np.mean(residence_times),
        'median': np.median(residence_times),
        'max': np.max(residence_times),
        'std': np.std(residence_times),
        'p90': np.percentile(residence_times, 90),
        'p99': np.percentile(residence_times, 99),
        'heavy_tail_ratio': np.sum(residence_times > 10) / len(residence_times),
        'raw': residence_times
    }


def compute_immobile_regions(history: np.ndarray, min_duration: int = 20) -> Dict:
    """
    Find regions that stay still for extended periods.
    """
    T, W = history.shape
    immobile_map = np.zeros((T, W), dtype=int)

    # Track consecutive unchanged steps for each position
    unchanged_count = np.zeros(W, dtype=int)

    for t in range(1, T):
        for i in range(W):
            if history[t, i] == history[t-1, i]:
                unchanged_count[i] += 1
            else:
                unchanged_count[i] = 0

            if unchanged_count[i] >= min_duration:
                immobile_map[t, i] = 1

    # Find connected immobile regions
    labeled, num_regions = ndimage.label(immobile_map)

    # Measure region properties
    region_sizes = []
    region_durations = []

    for region_id in range(1, num_regions + 1):
        region_mask = labeled == region_id
        region_coords = np.where(region_mask)

        if len(region_coords[0]) > 0:
            t_span = np.max(region_coords[0]) - np.min(region_coords[0])
            x_span = np.max(region_coords[1]) - np.min(region_coords[1])
            region_sizes.append(np.sum(region_mask))
            region_durations.append(t_span)

    return {
        'num_regions': num_regions,
        'total_immobile_area': np.sum(immobile_map),
        'immobile_fraction': np.sum(immobile_map) / (T * W),
        'mean_region_size': np.mean(region_sizes) if region_sizes else 0,
        'max_region_size': np.max(region_sizes) if region_sizes else 0,
        'mean_duration': np.mean(region_durations) if region_durations else 0,
        'map': immobile_map
    }


def compute_velocity_field(history: np.ndarray) -> np.ndarray:
    """
    Estimate local "velocity" - how fast structures are moving.
    Use cross-correlation between consecutive windows.
    """
    T, W = history.shape
    window_size = 5
    velocity = np.zeros((T - window_size, W))

    for t in range(window_size, T):
        window_curr = history[t-window_size//2:t+window_size//2+1]
        window_prev = history[t-window_size//2-1:t+window_size//2]

        for i in range(W):
            # Check correlation with shifted versions
            best_shift = 0
            best_corr = -1

            for shift in [-2, -1, 0, 1, 2]:
                i_shifted = (i + shift) % W
                if window_prev.shape == window_curr.shape:
                    corr = np.corrcoef(window_prev[:, i].flatten(),
                                       window_curr[:, i_shifted].flatten())[0, 1]
                    if not np.isnan(corr) and corr > best_corr:
                        best_corr = corr
                        best_shift = shift

            velocity[t - window_size, i] = abs(best_shift)

    return velocity


def compute_control_map(history: np.ndarray) -> np.ndarray:
    """Compute local Control intensity."""
    T, W = history.shape
    control_map = np.zeros((T-2, W-2), dtype=float)

    # Build pattern-outcome map
    pattern_outcomes: Dict[Tuple, List[int]] = defaultdict(list)
    for t in range(1, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_outcomes[pattern].append(int(outcome))

    pattern_divergence = {}
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) >= 3:
            mean = np.mean(outcomes)
            pattern_divergence[pattern] = 4 * mean * (1 - mean)
        else:
            pattern_divergence[pattern] = 0

    for t in range(1, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            control_map[t-1, i-1] = pattern_divergence.get(pattern, 0)

    return control_map


def find_scaffold_candidates(history: np.ndarray) -> Dict:
    """
    Find regions with HIGH Control + LOW Velocity.
    These are scaffold candidates - computational but immobile.
    """
    control_map = compute_control_map(history)
    velocity = compute_velocity_field(history)

    # Align dimensions
    min_t = min(control_map.shape[0], velocity.shape[0])
    min_w = min(control_map.shape[1], velocity.shape[1])

    if min_t <= 0 or min_w <= 0:
        return {
            'num_candidates': 0,
            'total_scaffold_area': 0,
            'scaffold_fraction': 0,
            'candidates': []
        }

    control_aligned = control_map[:min_t, :min_w]
    velocity_aligned = velocity[:min_t, :min_w]

    # Scaffold = high control + low velocity
    control_threshold = np.percentile(control_aligned, 70)
    velocity_threshold = np.percentile(velocity_aligned, 30)

    scaffold_mask = (control_aligned > control_threshold) & (velocity_aligned < velocity_threshold)

    # Find connected scaffold regions
    labeled, num_scaffolds = ndimage.label(scaffold_mask)

    candidates = []
    for scaffold_id in range(1, num_scaffolds + 1):
        region_mask = labeled == scaffold_id
        region_coords = np.where(region_mask)

        if len(region_coords[0]) > 5:  # Minimum size
            center_t = int(np.mean(region_coords[0]))
            center_x = int(np.mean(region_coords[1]))
            size = np.sum(region_mask)
            mean_control = np.mean(control_aligned[region_mask])
            mean_velocity = np.mean(velocity_aligned[region_mask])

            candidates.append({
                'center': (center_t, center_x),
                'size': size,
                'control': mean_control,
                'velocity': mean_velocity
            })

    # Sort by size
    candidates.sort(key=lambda x: -x['size'])

    return {
        'num_candidates': len(candidates),
        'total_scaffold_area': np.sum(scaffold_mask),
        'scaffold_fraction': np.sum(scaffold_mask) / scaffold_mask.size,
        'candidates': candidates[:10],  # Top 10
        'mask': scaffold_mask
    }


def compute_activity_attractor_metric(history: np.ndarray) -> Dict:
    """
    Find positions that attract activity but don't move.
    Measured by: high local activity density but low position shift.
    """
    T, W = history.shape

    # Activity per position (fraction of time changing)
    activity = np.zeros(W)
    for i in range(W):
        changes = np.sum(history[1:, i] != history[:-1, i])
        activity[i] = changes / (T - 1)

    # Find activity "hotspots" that persist in same location
    hotspot_threshold = np.percentile(activity, 80)
    hotspots = activity > hotspot_threshold

    # Measure hotspot stability (do they stay in place?)
    window = 20
    hotspot_persistence = []

    for i in range(W):
        if hotspots[i]:
            # Check if this position remains high-activity over time
            persistence = 0
            for t_start in range(0, T - window, window):
                window_activity = np.sum(
                    history[t_start+1:t_start+window, i] != history[t_start:t_start+window-1, i]
                ) / window
                if window_activity > 0.2:
                    persistence += 1
            hotspot_persistence.append(persistence)

    return {
        'num_hotspots': np.sum(hotspots),
        'mean_activity': np.mean(activity),
        'hotspot_activity': np.mean(activity[hotspots]) if np.any(hotspots) else 0,
        'mean_persistence': np.mean(hotspot_persistence) if hotspot_persistence else 0,
        'activity_profile': activity
    }


# =============================================================================
# COMPREHENSIVE SCAFFOLD ANALYSIS
# =============================================================================

def analyze_scaffolds(rule: int, width: int = 120, steps: int = 300,
                      mechanism: str = 'confirmation') -> Dict:
    """Complete scaffold analysis for a rule."""
    if mechanism == 'standard':
        history = standard_eca(rule, width, steps)
    elif mechanism == 'confirmation':
        history = confirmation_eca(rule, width, steps)
    elif mechanism == 'refractory':
        history = refractory_eca(rule, width, steps, 2)
    elif mechanism == 'asymmetric':
        history = asymmetric_confirmation(rule, width, steps)
    else:
        history = confirmation_eca(rule, width, steps)

    results = {
        'rule': rule,
        'mechanism': mechanism,
        'residence': compute_residence_times(history),
        'immobile': compute_immobile_regions(history),
        'scaffolds': find_scaffold_candidates(history),
        'attractors': compute_activity_attractor_metric(history)
    }

    # Remove large arrays for JSON serialization
    results['residence'].pop('raw', None)
    results['immobile'].pop('map', None)
    results['scaffolds'].pop('mask', None)

    return results


def run_scaffold_search(rules: List[int] = [110, 30, 90, 54, 62, 150],
                        width: int = 120, steps: int = 300):
    """Search for scaffolds across rules and mechanisms."""
    print("=" * 70)
    print("PHASE 4: SCAFFOLD SEARCH")
    print("=" * 70)

    all_results = {}
    scaffold_candidates = []

    mechanisms = ['standard', 'confirmation', 'refractory', 'asymmetric']

    for rule in rules:
        all_results[rule] = {}
        print(f"\nRule {rule}:")

        for mech in mechanisms:
            results = analyze_scaffolds(rule, width, steps, mech)
            all_results[rule][mech] = results

            scaffolds = results['scaffolds']
            immobile = results['immobile']
            residence = results['residence']

            print(f"  {mech:15s}: scaffolds={scaffolds['num_candidates']:3d}, "
                  f"immobile={immobile['immobile_fraction']:.3f}, "
                  f"residence_p99={residence['p99']:.1f}")

            # Track promising candidates
            if scaffolds['num_candidates'] > 0 and immobile['immobile_fraction'] > 0.1:
                scaffold_candidates.append({
                    'rule': rule,
                    'mechanism': mech,
                    'num_scaffolds': scaffolds['num_candidates'],
                    'immobile_fraction': immobile['immobile_fraction'],
                    'scaffold_fraction': scaffolds['scaffold_fraction'],
                    'residence_p99': residence['p99']
                })

    return all_results, scaffold_candidates


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_scaffold_analysis(rule: int, width: int = 120, steps: int = 200):
    """Visualize scaffold detection for a rule."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    mechanisms = [
        ('Standard', standard_eca(rule, width, steps)),
        ('Confirmation', confirmation_eca(rule, width, steps)),
        ('Refractory', refractory_eca(rule, width, steps, 2)),
        ('Asymmetric', asymmetric_confirmation(rule, width, steps)),
    ]

    for col, (name, history) in enumerate(mechanisms):
        # Row 1: Spacetime
        axes[0, col].imshow(history, cmap='binary', aspect='auto')
        axes[0, col].set_title(f'{name}')
        axes[0, col].set_ylabel('Time' if col == 0 else '')

        # Row 2: Control map
        control_map = compute_control_map(history)
        im = axes[1, col].imshow(control_map, cmap='hot', aspect='auto', vmin=0, vmax=1)
        axes[1, col].set_ylabel('Control' if col == 0 else '')

        # Row 3: Scaffold candidates
        scaffolds = find_scaffold_candidates(history)
        if 'mask' in scaffolds and scaffolds['mask'].size > 0:
            axes[2, col].imshow(scaffolds['mask'], cmap='Greens', aspect='auto')
        else:
            axes[2, col].imshow(np.zeros((steps-5, width-2)), cmap='Greens', aspect='auto')
        axes[2, col].set_ylabel('Scaffolds' if col == 0 else '')
        axes[2, col].set_xlabel('Position')
        axes[2, col].set_title(f"n={scaffolds['num_candidates']}")

    plt.suptitle(f'Rule {rule}: Scaffold Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'scaffold_analysis_rule{rule}.png', dpi=150)
    plt.close()


def visualize_residence_distribution(all_results: Dict, rules: List[int]):
    """Compare residence time distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, rule in enumerate(rules[:6]):
        ax = axes[idx]

        for mech in ['standard', 'confirmation', 'asymmetric']:
            if mech in all_results[rule]:
                res = all_results[rule][mech]['residence']
                ax.axvline(res['p99'], linestyle='--', alpha=0.5, label=f'{mech} p99')

        ax.set_xlabel('Residence Time')
        ax.set_ylabel('Cumulative')
        ax.set_title(f'Rule {rule}')
        ax.legend(fontsize=8)

    plt.suptitle('Residence Time Distributions (Heavy Tail = Scaffolds)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'residence_distributions.png', dpi=150)
    plt.close()


def create_scaffold_summary_table(candidates: List[Dict]):
    """Create summary table of scaffold candidates."""
    if not candidates:
        return "No scaffold candidates found."

    # Sort by scaffold fraction
    candidates.sort(key=lambda x: -x['scaffold_fraction'])

    lines = [
        "=" * 70,
        "SCAFFOLD CANDIDATES (High Control + Low Motion)",
        "=" * 70,
        f"{'Rule':<8} {'Mechanism':<15} {'Scaffolds':<10} {'Immobile%':<10} {'Scaffold%':<10} {'Resid p99':<10}",
        "-" * 70
    ]

    for c in candidates[:15]:
        lines.append(
            f"{c['rule']:<8} {c['mechanism']:<15} {c['num_scaffolds']:<10} "
            f"{c['immobile_fraction']*100:<10.1f} {c['scaffold_fraction']*100:<10.1f} "
            f"{c['residence_p99']:<10.1f}"
        )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 4: SCAFFOLD SEARCH")
    print("=" * 70)

    rules = [110, 30, 90, 54, 62, 150]
    all_results, candidates = run_scaffold_search(rules)

    # Save results
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float('inf'):
            return "inf"
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(OUTPUT_DIR / 'scaffold_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    with open(OUTPUT_DIR / 'scaffold_candidates.json', 'w') as f:
        json.dump(convert(candidates), f, indent=2)

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    for rule in [110, 54, 30]:
        visualize_scaffold_analysis(rule)
        print(f"  Saved scaffold analysis for Rule {rule}")

    # Summary
    print("\n" + create_scaffold_summary_table(candidates))

    print("\n" + "=" * 70)
    print("PHASE 4 ANALYSIS")
    print("=" * 70)

    print("\nKey Question: Where do scaffolds (immobile computational regions) emerge?")

    if candidates:
        best = candidates[0]
        print(f"\nBest scaffold candidate:")
        print(f"  Rule {best['rule']} + {best['mechanism']}")
        print(f"  Scaffold fraction: {best['scaffold_fraction']*100:.1f}%")
        print(f"  Immobile fraction: {best['immobile_fraction']*100:.1f}%")

        # Count by mechanism
        mech_counts = defaultdict(int)
        for c in candidates:
            mech_counts[c['mechanism']] += 1

        print(f"\nScaffold candidates by mechanism:")
        for mech, count in sorted(mech_counts.items(), key=lambda x: -x[1]):
            print(f"  {mech}: {count}")
    else:
        print("\nNo strong scaffold candidates found in tested configurations.")

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
