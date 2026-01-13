"""
Phase B: Boundary Transport vs Boundary Computation

Key questions:
1. Does Control move or stay anchored?
2. What happens at boundary-boundary collisions?

Part 1: Track labeled Control regions across time
Part 2: Collision Algebra - observe boundary interactions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import json
from pathlib import Path
from collections import defaultdict
from scipy import ndimage

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "phase_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CA IMPLEMENTATION
# =============================================================================

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


# =============================================================================
# CONTROL REGION DETECTION AND TRACKING
# =============================================================================

def compute_control_map(history: np.ndarray) -> np.ndarray:
    """Compute local Control intensity at each position and time."""
    T, W = history.shape
    control_map = np.zeros((T-2, W), dtype=float)

    # Build global pattern-outcome map
    pattern_outcomes = defaultdict(list)
    for t in range(1, T - 1):
        for i in range(1, W - 1):
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]
            pattern_outcomes[pattern].append(outcome)

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
            control_map[t-1, i] = pattern_divergence.get(pattern, 0)

    return control_map


def detect_control_regions(control_map: np.ndarray, threshold_percentile: float = 70) -> np.ndarray:
    """Label connected high-Control regions."""
    threshold = np.percentile(control_map, threshold_percentile)
    is_high_control = control_map > threshold

    # Label connected regions
    labeled, n_regions = ndimage.label(is_high_control)

    return labeled, n_regions


def track_region_centroids(labeled_map: np.ndarray, n_regions: int) -> Dict[int, List[Tuple[int, int]]]:
    """Track centroid of each labeled region over time."""
    T, W = labeled_map.shape
    region_tracks = defaultdict(list)

    for region_id in range(1, n_regions + 1):
        region_mask = labeled_map == region_id
        if np.any(region_mask):
            coords = np.where(region_mask)
            centroid_t = np.mean(coords[0])
            centroid_x = np.mean(coords[1])
            region_tracks[region_id].append((centroid_t, centroid_x))

    return dict(region_tracks)


def compute_region_velocities(control_map: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute local velocity of Control regions.
    Use cross-correlation to estimate movement.
    """
    T, W = control_map.shape
    velocity_map = np.zeros((T - window, W), dtype=float)

    for t in range(window, T):
        prev_window = control_map[t-window:t-window//2]
        curr_window = control_map[t-window//2:t]

        for i in range(W):
            # Check correlation with shifted versions
            best_shift = 0
            best_corr = -1

            for shift in range(-3, 4):
                i_shifted = (i + shift) % W
                prev_col = prev_window[:, i].flatten()
                curr_col = curr_window[:, i_shifted].flatten()

                if len(prev_col) > 0 and len(curr_col) > 0:
                    if np.std(prev_col) > 0 and np.std(curr_col) > 0:
                        corr = np.corrcoef(prev_col, curr_col)[0, 1]
                        if not np.isnan(corr) and corr > best_corr:
                            best_corr = corr
                            best_shift = shift

            velocity_map[t - window, i] = best_shift / (window // 2)  # Cells per timestep

    return velocity_map


def analyze_control_transport(history: np.ndarray) -> Dict:
    """
    Determine if Control:
    - Propagates with moving boundaries
    - Hops between boundaries
    - Is annihilated/created at collisions
    """
    control_map = compute_control_map(history)
    velocity_map = compute_region_velocities(control_map)

    # Classify Control regions by velocity
    threshold = np.percentile(control_map[:len(velocity_map)], 70)
    is_high_control = control_map[:len(velocity_map)] > threshold

    # Stationary Control (velocity near 0)
    stationary_mask = (np.abs(velocity_map) < 0.1) & is_high_control
    moving_mask = (np.abs(velocity_map) >= 0.1) & is_high_control

    stationary_fraction = np.sum(stationary_mask) / max(np.sum(is_high_control), 1)
    moving_fraction = np.sum(moving_mask) / max(np.sum(is_high_control), 1)

    # Average velocity of Control regions
    if np.any(is_high_control):
        mean_velocity = np.mean(np.abs(velocity_map[is_high_control]))
    else:
        mean_velocity = 0

    return {
        'stationary_fraction': stationary_fraction,
        'moving_fraction': moving_fraction,
        'mean_velocity': mean_velocity,
        'velocity_map': velocity_map
    }


# =============================================================================
# COLLISION ALGEBRA
# =============================================================================

def detect_boundary_map(history: np.ndarray, window: int = 5) -> np.ndarray:
    """Detect boundaries based on activity gradient."""
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


def track_boundaries(boundary_map: np.ndarray, threshold_percentile: float = 80) -> List[Dict]:
    """
    Track individual boundaries across time.
    Returns list of boundary objects with trajectory.
    """
    threshold = np.percentile(boundary_map, threshold_percentile)
    is_boundary = boundary_map > threshold

    T, W = boundary_map.shape
    boundaries = []

    # Simple tracking: for each column, find continuous boundary segments
    for i in range(W):
        in_boundary = False
        start_t = 0

        for t in range(T):
            if is_boundary[t, i] and not in_boundary:
                in_boundary = True
                start_t = t
            elif not is_boundary[t, i] and in_boundary:
                in_boundary = False
                boundaries.append({
                    'position': i,
                    'start_time': start_t,
                    'end_time': t,
                    'duration': t - start_t
                })

        if in_boundary:
            boundaries.append({
                'position': i,
                'start_time': start_t,
                'end_time': T,
                'duration': T - start_t
            })

    return boundaries


def detect_collisions(boundary_map: np.ndarray, control_map: np.ndarray,
                      threshold_percentile: float = 80) -> List[Dict]:
    """
    Detect boundary-boundary collisions.
    A collision is where two boundary regions merge or one terminates.
    """
    threshold = np.percentile(boundary_map, threshold_percentile)
    is_boundary = boundary_map > threshold

    T, W = boundary_map.shape
    collisions = []

    # Detect collision events: where boundary count changes
    for t in range(1, T - 1):
        labeled_prev, n_prev = ndimage.label(is_boundary[t-1])
        labeled_curr, n_curr = ndimage.label(is_boundary[t])
        labeled_next, n_next = ndimage.label(is_boundary[t+1])

        # Merge: n decreases
        if n_curr < n_prev:
            # Find where regions merged
            for i in range(W):
                if labeled_prev[i] > 0 and labeled_curr[i] > 0:
                    # Check if this position was part of a merge
                    neighbors_prev = set()
                    neighbors_curr = set()

                    for di in [-1, 0, 1]:
                        idx = (i + di) % W
                        if labeled_prev[idx] > 0:
                            neighbors_prev.add(labeled_prev[idx])
                        if labeled_curr[idx] > 0:
                            neighbors_curr.add(labeled_curr[idx])

                    if len(neighbors_prev) > len(neighbors_curr):
                        # Merge event
                        control_at_collision = control_map[t, i] if t < len(control_map) and i < control_map.shape[1] else 0
                        collisions.append({
                            'type': 'merge',
                            'time': t,
                            'position': i,
                            'control_before': np.mean(control_map[t-1, max(0,i-2):i+3]) if t > 0 else 0,
                            'control_after': np.mean(control_map[min(t+1, len(control_map)-1), max(0,i-2):i+3]),
                        })
                        break

        # Annihilation: region disappears
        if n_curr > n_next:
            for i in range(W):
                if labeled_curr[i] > 0 and labeled_next[i] == 0:
                    control_at_collision = control_map[t, i] if t < len(control_map) and i < control_map.shape[1] else 0
                    collisions.append({
                        'type': 'annihilate',
                        'time': t,
                        'position': i,
                        'control_before': np.mean(control_map[t, max(0,i-2):i+3]) if t < len(control_map) else 0,
                        'control_after': 0,
                    })
                    break

        # Creation: new region appears
        if n_curr < n_next:
            for i in range(W):
                if labeled_curr[i] == 0 and labeled_next[i] > 0:
                    collisions.append({
                        'type': 'create',
                        'time': t,
                        'position': i,
                        'control_before': 0,
                        'control_after': np.mean(control_map[min(t+1, len(control_map)-1), max(0,i-2):i+3]),
                    })
                    break

    return collisions


def analyze_collision_algebra(collisions: List[Dict]) -> Dict:
    """Analyze collision outcomes: deterministic or context-dependent?"""
    if not collisions:
        return {
            'total_collisions': 0,
            'merge_count': 0,
            'annihilate_count': 0,
            'create_count': 0,
            'control_change_at_collision': 0
        }

    merge_count = sum(1 for c in collisions if c['type'] == 'merge')
    annihilate_count = sum(1 for c in collisions if c['type'] == 'annihilate')
    create_count = sum(1 for c in collisions if c['type'] == 'create')

    # Control change at collisions
    control_changes = []
    for c in collisions:
        before = c.get('control_before', 0)
        after = c.get('control_after', 0)
        if before > 0 or after > 0:
            control_changes.append(after - before)

    return {
        'total_collisions': len(collisions),
        'merge_count': merge_count,
        'annihilate_count': annihilate_count,
        'create_count': create_count,
        'control_change_at_collision': np.mean(control_changes) if control_changes else 0,
        'control_change_std': np.std(control_changes) if control_changes else 0
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_transport_analysis(rules: List[int] = [110, 30, 90, 54],
                           width: int = 120, steps: int = 250):
    """Analyze Control transport for multiple rules."""
    print("=" * 70)
    print("PHASE B.1: CONTROL TRANSPORT ANALYSIS")
    print("=" * 70)
    print("\nQuestion: Does Control move or stay anchored?")

    results = {}

    for rule in rules:
        print(f"\nRule {rule}:")
        history = confirmation_eca(rule, width, steps)
        transport = analyze_control_transport(history)

        results[rule] = {
            'stationary_fraction': transport['stationary_fraction'],
            'moving_fraction': transport['moving_fraction'],
            'mean_velocity': transport['mean_velocity']
        }

        print(f"  Stationary Control: {transport['stationary_fraction']*100:.1f}%")
        print(f"  Moving Control: {transport['moving_fraction']*100:.1f}%")
        print(f"  Mean velocity: {transport['mean_velocity']:.3f} cells/step")

        if transport['stationary_fraction'] > 0.7:
            print(f"  => Control is predominantly ANCHORED")
        elif transport['moving_fraction'] > 0.7:
            print(f"  => Control PROPAGATES with boundaries")
        else:
            print(f"  => Control shows MIXED transport")

    return results


def run_collision_analysis(rules: List[int] = [110, 30, 90, 54],
                           width: int = 120, steps: int = 250):
    """Analyze collision algebra for multiple rules."""
    print("\n" + "=" * 70)
    print("PHASE B.2: COLLISION ALGEBRA")
    print("=" * 70)
    print("\nObserving boundary-boundary interactions...")

    results = {}

    for rule in rules:
        print(f"\nRule {rule}:")
        history = confirmation_eca(rule, width, steps)
        control_map = compute_control_map(history)
        boundary_map = detect_boundary_map(history)

        # Align dimensions
        min_t = min(len(control_map), len(boundary_map))
        control_map = control_map[:min_t]
        boundary_map = boundary_map[:min_t]

        collisions = detect_collisions(boundary_map, control_map)
        algebra = analyze_collision_algebra(collisions)

        results[rule] = algebra

        print(f"  Total collisions: {algebra['total_collisions']}")
        print(f"    Merges: {algebra['merge_count']}")
        print(f"    Annihilations: {algebra['annihilate_count']}")
        print(f"    Creations: {algebra['create_count']}")
        print(f"  Control change at collision: {algebra['control_change_at_collision']:.3f} +/- {algebra['control_change_std']:.3f}")

        # Determine if outcomes are deterministic
        if algebra['control_change_std'] < 0.1:
            print(f"  => Collision outcomes appear DETERMINISTIC")
        else:
            print(f"  => Collision outcomes are CONTEXT-DEPENDENT")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_control_transport(rule: int, width: int = 120, steps: int = 200):
    """Visualize Control transport for a rule."""
    history = confirmation_eca(rule, width, steps)
    control_map = compute_control_map(history)
    transport = analyze_control_transport(history)
    velocity_map = transport['velocity_map']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Spacetime
    axes[0, 0].imshow(history, cmap='binary', aspect='auto')
    axes[0, 0].set_title(f'Rule {rule}: Spacetime')
    axes[0, 0].set_ylabel('Time')

    # Control map
    im1 = axes[0, 1].imshow(control_map, cmap='hot', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('Control Map')
    plt.colorbar(im1, ax=axes[0, 1])

    # Velocity map
    im2 = axes[1, 0].imshow(velocity_map, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('Velocity Map (red=left, blue=right)')
    axes[1, 0].set_ylabel('Time')
    axes[1, 0].set_xlabel('Position')
    plt.colorbar(im2, ax=axes[1, 0])

    # Velocity histogram
    high_control_mask = control_map[:len(velocity_map)] > np.percentile(control_map[:len(velocity_map)], 70)
    velocities = velocity_map[high_control_mask]
    axes[1, 1].hist(velocities, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Velocity (cells/step)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Velocity Distribution of Control Regions\n'
                         f'Stationary: {transport["stationary_fraction"]*100:.0f}%')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'control_transport_rule{rule}.png', dpi=150)
    plt.close()


def visualize_collision_events(rule: int, width: int = 120, steps: int = 200):
    """Visualize collision events."""
    history = confirmation_eca(rule, width, steps)
    control_map = compute_control_map(history)
    boundary_map = detect_boundary_map(history)

    min_t = min(len(control_map), len(boundary_map))
    collisions = detect_collisions(boundary_map[:min_t], control_map[:min_t])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boundary map with collision markers
    axes[0].imshow(boundary_map, cmap='hot', aspect='auto')
    axes[0].set_title(f'Rule {rule}: Boundaries with Collisions')
    axes[0].set_ylabel('Time')
    axes[0].set_xlabel('Position')

    # Mark collisions
    for c in collisions[:50]:  # Limit for visibility
        color = {'merge': 'cyan', 'annihilate': 'red', 'create': 'green'}.get(c['type'], 'white')
        marker = {'merge': 'o', 'annihilate': 'x', 'create': '+'}.get(c['type'], 's')
        axes[0].scatter(c['position'], c['time'], c=color, marker=marker, s=30, alpha=0.8)

    # Control change at collisions
    if collisions:
        changes = [c.get('control_after', 0) - c.get('control_before', 0) for c in collisions]
        types = [c['type'] for c in collisions]

        type_colors = {'merge': 'cyan', 'annihilate': 'red', 'create': 'green'}
        colors = [type_colors.get(t, 'gray') for t in types]

        axes[1].scatter(range(len(changes)), changes, c=colors, alpha=0.6)
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].set_xlabel('Collision Index')
        axes[1].set_ylabel('Control Change (after - before)')
        axes[1].set_title('Control Change at Collisions\ncyan=merge, red=annihilate, green=create')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'collision_events_rule{rule}.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE B: BOUNDARY TRANSPORT VS COMPUTATION")
    print("=" * 70)

    rules = [110, 30, 90, 54]

    # Part 1: Transport analysis
    transport_results = run_transport_analysis(rules)

    # Part 2: Collision analysis
    collision_results = run_collision_analysis(rules)

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

    combined_results = {
        'transport': convert(transport_results),
        'collisions': convert(collision_results)
    }

    with open(OUTPUT_DIR / 'transport_collision_results.json', 'w') as f:
        json.dump(combined_results, f, indent=2)

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    for rule in [110, 54]:
        visualize_control_transport(rule)
        visualize_collision_events(rule)
        print(f"  Saved transport and collision visualizations for Rule {rule}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE B SUMMARY")
    print("=" * 70)

    print("\nControl Transport:")
    for rule in rules:
        t = transport_results[rule]
        print(f"  Rule {rule}: {t['stationary_fraction']*100:.0f}% stationary, "
              f"{t['mean_velocity']:.3f} cells/step mean velocity")

    print("\nCollision Algebra:")
    for rule in rules:
        c = collision_results[rule]
        print(f"  Rule {rule}: {c['total_collisions']} collisions "
              f"(M:{c['merge_count']}, A:{c['annihilate_count']}, C:{c['create_count']})")

    # Key observations
    stationary_dominant = sum(1 for r in transport_results.values()
                              if r['stationary_fraction'] > 0.5)
    print(f"\nOBSERVATION: {stationary_dominant}/{len(rules)} rules have predominantly stationary Control")

    print("\n" + "=" * 70)
    print("PHASE B COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
