"""
Stickiness in Bit Dynamics - Experimental Framework

Tests four mechanisms for "stickiness" that may provide natural asymmetry
and Control capability without explicit rule bias.

Mechanisms:
1. Second-Order ECA: state(t) = f(state(t-1), state(t-2), neighborhood)
2. Greenberg-Hastings: 3-state excitable media with refractory period
3. Threshold Asymmetry: different thresholds for birth vs death
4. History Weighting: state influenced by weighted past

Connection to UCT: We hypothesize that stickiness provides the context-dependent
divergence needed for Control capability (the 5th bit).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict, List
import json
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# METRICS
# =============================================================================

def entropy(state: np.ndarray) -> float:
    """Shannon entropy of a binary/discrete state array."""
    unique, counts = np.unique(state, return_counts=True)
    probs = counts / len(state)
    return -np.sum(probs * np.log2(probs + 1e-10))


def activity(history: np.ndarray) -> float:
    """Fraction of cells that change between consecutive timesteps."""
    if len(history) < 2:
        return 0.0
    changes = np.sum(history[1:] != history[:-1], axis=1)
    return np.mean(changes) / history.shape[1]


def pattern_persistence(history: np.ndarray) -> float:
    """Average duration a cell stays in the same state."""
    if len(history) < 2:
        return float('inf')

    total_runs = 0
    total_length = 0

    for col in range(history.shape[1]):
        column = history[:, col]
        run_length = 1
        for i in range(1, len(column)):
            if column[i] == column[i-1]:
                run_length += 1
            else:
                total_runs += 1
                total_length += run_length
                run_length = 1
        total_runs += 1
        total_length += run_length

    return total_length / total_runs if total_runs > 0 else float('inf')


def asymmetry_index(history: np.ndarray) -> float:
    """Measure asymmetry in 0→1 vs 1→0 transitions."""
    if len(history) < 2:
        return 0.0

    transitions_01 = 0  # 0 to 1
    transitions_10 = 0  # 1 to 0

    for t in range(1, len(history)):
        for i in range(history.shape[1]):
            prev, curr = history[t-1, i], history[t, i]
            # Handle multi-state by checking first bit equivalent
            if prev == 0 and curr > 0:
                transitions_01 += 1
            elif prev > 0 and curr == 0:
                transitions_10 += 1

    total = transitions_01 + transitions_10
    if total == 0:
        return 0.0

    # Asymmetry: 0 means balanced, 1 means completely one-sided
    return abs(transitions_01 - transitions_10) / total


def control_proxy(history: np.ndarray) -> float:
    """
    Proxy for Control capability: context-dependent divergence.
    Measures how much the same local pattern leads to different outcomes.
    """
    if len(history) < 3:
        return 0.0

    # Sample local patterns and their outcomes
    pattern_outcomes: Dict[Tuple, List[int]] = {}

    for t in range(1, len(history) - 1):
        for i in range(1, history.shape[1] - 1):
            # Local pattern: center and neighbors at time t
            pattern = (history[t, i-1], history[t, i], history[t, i+1])
            outcome = history[t+1, i]

            if pattern not in pattern_outcomes:
                pattern_outcomes[pattern] = []
            pattern_outcomes[pattern].append(int(outcome > 0))  # Binarize

    # Calculate divergence: patterns with varied outcomes indicate Control
    divergences = []
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) > 10:  # Enough samples
            mean = np.mean(outcomes)
            # Variance-like measure: max divergence when mean = 0.5
            divergence = 4 * mean * (1 - mean)  # Peaks at 0.5
            divergences.append(divergence)

    return np.mean(divergences) if divergences else 0.0


# =============================================================================
# MECHANISM 1: SECOND-ORDER ECA
# =============================================================================

def second_order_eca(rule: int, width: int, steps: int,
                     init: np.ndarray = None) -> np.ndarray:
    """
    Second-order Elementary Cellular Automaton.
    State at time t depends on state at t-1 XOR state at t-2, plus standard ECA rule.

    This creates inherent memory and path-dependence.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    # Two history layers needed
    history = np.zeros((steps, width), dtype=int)
    history[0] = init.copy()
    history[1] = init.copy()  # Start with same state for t=0 and t=-1

    # Apply standard ECA once to get t=1
    for i in range(width):
        left = history[0][(i - 1) % width]
        center = history[0][i]
        right = history[0][(i + 1) % width]
        pattern = (left << 2) | (center << 1) | right
        history[1][i] = (rule >> pattern) & 1

    # Second-order evolution
    for t in range(2, steps):
        for i in range(width):
            # Standard neighborhood from t-1
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right

            # Standard ECA output
            eca_out = (rule >> pattern) & 1

            # Second-order: XOR with t-2 state (creates stickiness)
            history[t][i] = eca_out ^ history[t-2][i]

    return history


# =============================================================================
# MECHANISM 2: GREENBERG-HASTINGS (Excitable Media)
# =============================================================================

def greenberg_hastings(width: int, steps: int, threshold: int = 1,
                       refractory: int = 2, init: np.ndarray = None) -> np.ndarray:
    """
    Greenberg-Hastings model: 3-state excitable medium.
    States: 0 (resting), 1 (excited), 2...(refractory-1) (refractory)

    Transitions:
    - Resting (0) → Excited (1) if ≥threshold excited neighbors
    - Excited (1) → Refractory (2)
    - Refractory → Refractory-1 → ... → Resting (0)

    The refractory period creates natural "stickiness" - a cell that fired
    cannot immediately fire again.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        # Seed with a few excited cells
        init[width // 2] = 1
        init[width // 3] = 1

    history = np.zeros((steps, width), dtype=int)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            current = history[t-1][i]

            if current == 0:  # Resting
                # Count excited neighbors
                left = history[t-1][(i - 1) % width]
                right = history[t-1][(i + 1) % width]
                excited_neighbors = (left == 1) + (right == 1)

                if excited_neighbors >= threshold:
                    history[t][i] = 1  # Become excited
                else:
                    history[t][i] = 0  # Stay resting

            elif current == 1:  # Excited
                history[t][i] = 2  # Enter refractory

            else:  # Refractory (2 to refractory-1)
                if current >= refractory:
                    history[t][i] = 0  # Return to resting
                else:
                    history[t][i] = current + 1  # Continue refractory

    return history


# =============================================================================
# MECHANISM 3: THRESHOLD ASYMMETRY
# =============================================================================

def threshold_asymmetric_ca(width: int, steps: int,
                            birth_threshold: int = 1,
                            death_threshold: int = 2,
                            init: np.ndarray = None) -> np.ndarray:
    """
    CA with asymmetric birth/death thresholds.
    - Birth: 0→1 if ≥birth_threshold neighbors are 1
    - Death: 1→0 if <death_threshold neighbors are 1

    When death_threshold > birth_threshold, 1s are "sticky" -
    easier to create than destroy.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            right = history[t-1][(i + 1) % width]
            neighbors = left + right
            current = history[t-1][i]

            if current == 0:  # Dead cell
                if neighbors >= birth_threshold:
                    history[t][i] = 1  # Birth
                else:
                    history[t][i] = 0  # Stay dead
            else:  # Live cell
                if neighbors < death_threshold:
                    history[t][i] = 0  # Death
                else:
                    history[t][i] = 1  # Survive (sticky)

    return history


# =============================================================================
# MECHANISM 4: HISTORY WEIGHTING
# =============================================================================

def history_weighted_ca(rule: int, width: int, steps: int,
                        memory_weight: float = 0.3,
                        threshold: float = 0.5,
                        init: np.ndarray = None) -> np.ndarray:
    """
    CA where state is influenced by weighted history.

    Effective state = (1-w)*current + w*past_average
    If effective state > threshold → 1, else → 0

    This creates "inertia" - states tend to persist.
    """
    if init is None:
        init = np.zeros(width, dtype=float)
        init[width // 2] = 1.0

    # Track continuous "potential" and discrete state
    potential = init.astype(float).copy()
    history = np.zeros((steps, width), dtype=int)
    history[0] = (potential > threshold).astype(int)

    for t in range(1, steps):
        new_potential = np.zeros(width)

        for i in range(width):
            # Standard ECA computation on discrete state
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            eca_out = float((rule >> pattern) & 1)

            # Blend with history (stickiness)
            new_potential[i] = (1 - memory_weight) * eca_out + memory_weight * potential[i]

        potential = new_potential
        history[t] = (potential > threshold).astype(int)

    return history


# =============================================================================
# BASELINE: STANDARD ECA
# =============================================================================

def standard_eca(rule: int, width: int, steps: int,
                 init: np.ndarray = None) -> np.ndarray:
    """Standard Elementary Cellular Automaton for comparison."""
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
# EXPERIMENT RUNNER
# =============================================================================

def compute_metrics(history: np.ndarray) -> Dict[str, float]:
    """Compute all metrics for a simulation history."""
    return {
        'entropy': entropy(history[-1]),
        'activity': activity(history),
        'persistence': pattern_persistence(history),
        'asymmetry': asymmetry_index(history),
        'control_proxy': control_proxy(history)
    }


def run_comparison_experiment(width: int = 100, steps: int = 200):
    """
    Compare all stickiness mechanisms against baseline ECAs.
    """
    print("=" * 60)
    print("STICKINESS IN BIT DYNAMICS - Comparison Experiment")
    print("=" * 60)

    results = {}

    # Baseline: Standard ECAs (Rule 110 and Rule 30 for reference)
    print("\n1. BASELINE: Standard ECAs")
    for rule in [110, 30, 90, 184]:
        history = standard_eca(rule, width, steps)
        metrics = compute_metrics(history)
        results[f'ECA_{rule}'] = metrics
        print(f"   Rule {rule}: entropy={metrics['entropy']:.3f}, "
              f"activity={metrics['activity']:.3f}, "
              f"control={metrics['control_proxy']:.3f}")

    # Mechanism 1: Second-Order ECA
    print("\n2. Second-Order ECA (XOR with t-2)")
    for rule in [110, 30, 90]:
        history = second_order_eca(rule, width, steps)
        metrics = compute_metrics(history)
        results[f'SecondOrder_{rule}'] = metrics
        print(f"   Rule {rule}: entropy={metrics['entropy']:.3f}, "
              f"activity={metrics['activity']:.3f}, "
              f"persistence={metrics['persistence']:.3f}, "
              f"control={metrics['control_proxy']:.3f}")

    # Mechanism 2: Greenberg-Hastings
    print("\n3. Greenberg-Hastings (Excitable Media)")
    for refractory in [2, 3, 4, 5]:
        history = greenberg_hastings(width, steps, threshold=1, refractory=refractory)
        metrics = compute_metrics(history)
        results[f'GH_refract{refractory}'] = metrics
        print(f"   Refractory={refractory}: entropy={metrics['entropy']:.3f}, "
              f"activity={metrics['activity']:.3f}, "
              f"persistence={metrics['persistence']:.3f}, "
              f"asymmetry={metrics['asymmetry']:.3f}")

    # Mechanism 3: Threshold Asymmetry
    print("\n4. Threshold Asymmetry (Sticky 1s)")
    for death_thresh in [1, 2, 3]:
        history = threshold_asymmetric_ca(width, steps,
                                          birth_threshold=1,
                                          death_threshold=death_thresh)
        metrics = compute_metrics(history)
        results[f'Thresh_b1_d{death_thresh}'] = metrics
        print(f"   Birth=1, Death={death_thresh}: entropy={metrics['entropy']:.3f}, "
              f"activity={metrics['activity']:.3f}, "
              f"persistence={metrics['persistence']:.3f}")

    # Mechanism 4: History Weighting
    print("\n5. History Weighting (Inertia)")
    for weight in [0.1, 0.3, 0.5, 0.7]:
        history = history_weighted_ca(110, width, steps, memory_weight=weight)
        metrics = compute_metrics(history)
        results[f'HistWeight_{weight}'] = metrics
        print(f"   Weight={weight}: entropy={metrics['entropy']:.3f}, "
              f"activity={metrics['activity']:.3f}, "
              f"persistence={metrics['persistence']:.3f}")

    return results


def visualize_mechanisms(width: int = 100, steps: int = 100):
    """Create visualization comparing all mechanisms."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Standard ECAs
    axes[0, 0].imshow(standard_eca(110, width, steps), cmap='binary', aspect='auto')
    axes[0, 0].set_title('Standard ECA Rule 110')
    axes[0, 0].set_ylabel('Time')

    axes[0, 1].imshow(standard_eca(30, width, steps), cmap='binary', aspect='auto')
    axes[0, 1].set_title('Standard ECA Rule 30')

    axes[0, 2].imshow(standard_eca(90, width, steps), cmap='binary', aspect='auto')
    axes[0, 2].set_title('Standard ECA Rule 90')

    # Second-Order
    axes[1, 0].imshow(second_order_eca(110, width, steps), cmap='binary', aspect='auto')
    axes[1, 0].set_title('Second-Order ECA Rule 110')
    axes[1, 0].set_ylabel('Time')

    # Greenberg-Hastings (use colormap for 3+ states)
    gh_history = greenberg_hastings(width, steps, threshold=1, refractory=3)
    axes[1, 1].imshow(gh_history, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Greenberg-Hastings (refract=3)')

    # Threshold Asymmetry
    axes[1, 2].imshow(threshold_asymmetric_ca(width, steps, birth_threshold=1,
                                               death_threshold=2),
                      cmap='binary', aspect='auto')
    axes[1, 2].set_title('Threshold Asymmetric (sticky 1s)')

    # History Weighting variations
    axes[2, 0].imshow(history_weighted_ca(110, width, steps, memory_weight=0.3),
                      cmap='binary', aspect='auto')
    axes[2, 0].set_title('History Weighted (w=0.3)')
    axes[2, 0].set_ylabel('Time')
    axes[2, 0].set_xlabel('Cell')

    axes[2, 1].imshow(history_weighted_ca(110, width, steps, memory_weight=0.5),
                      cmap='binary', aspect='auto')
    axes[2, 1].set_title('History Weighted (w=0.5)')
    axes[2, 1].set_xlabel('Cell')

    axes[2, 2].imshow(history_weighted_ca(30, width, steps, memory_weight=0.3),
                      cmap='binary', aspect='auto')
    axes[2, 2].set_title('History Weighted Rule 30 (w=0.3)')
    axes[2, 2].set_xlabel('Cell')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stickiness_comparison.png', dpi=150)
    plt.close()
    print(f"\nVisualization saved to {OUTPUT_DIR / 'stickiness_comparison.png'}")


def main():
    """Run all experiments."""
    print("\n" + "=" * 60)
    print("STICKINESS RESEARCH - Initial Experiments")
    print("=" * 60)

    # Run comparison
    results = run_comparison_experiment(width=100, steps=200)

    # Save results
    results_file = OUTPUT_DIR / 'stickiness_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_mechanisms(width=100, steps=100)

    # Summary analysis
    print("\n" + "=" * 60)
    print("SUMMARY ANALYSIS")
    print("=" * 60)

    # Find highest control proxy (best candidate for computation)
    best_control = max(results.items(), key=lambda x: x[1]['control_proxy'])
    print(f"\nHighest Control Proxy: {best_control[0]} = {best_control[1]['control_proxy']:.4f}")

    # Find highest persistence (stickiest)
    finite_persist = {k: v for k, v in results.items() if v['persistence'] < 1000}
    if finite_persist:
        stickiest = max(finite_persist.items(), key=lambda x: x[1]['persistence'])
        print(f"Highest Persistence: {stickiest[0]} = {stickiest[1]['persistence']:.2f}")

    # Compare sticky mechanisms to baseline
    baseline_control = results.get('ECA_110', {}).get('control_proxy', 0)
    print(f"\nBaseline (Rule 110) Control: {baseline_control:.4f}")

    sticky_improvements = []
    for name, metrics in results.items():
        if 'SecondOrder' in name or 'GH' in name or 'Thresh' in name or 'HistWeight' in name:
            improvement = metrics['control_proxy'] - baseline_control
            sticky_improvements.append((name, improvement))

    print("\nControl improvement over baseline:")
    for name, imp in sorted(sticky_improvements, key=lambda x: -x[1])[:5]:
        sign = '+' if imp >= 0 else ''
        print(f"   {name}: {sign}{imp:.4f}")


if __name__ == '__main__':
    main()
