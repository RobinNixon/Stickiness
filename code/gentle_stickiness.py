"""
Gentle Stickiness Mechanisms

The XOR-based second-order mechanism creates high Control but destroys structure
(very low compression). We need "gentler" stickiness that provides Control
while preserving complexity.

Mechanisms to test:
1. Probabilistic stickiness: history influences probability, not determinism
2. Refractory stickiness: cells resist change for N timesteps after flipping
3. Neighborhood memory: cells remember neighbor history, not just own state
4. Majority voting: include past state in neighbor count
5. Asymmetric transitions: 0→1 follows rule, 1→0 requires confirmation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def standard_eca(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """Standard ECA baseline."""
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
# MECHANISM 1: REFRACTORY PERIOD (cells resist change after flipping)
# =============================================================================

def refractory_eca(rule: int, width: int, steps: int,
                   refractory_time: int = 2, init=None) -> np.ndarray:
    """
    ECA where cells have a refractory period after changing state.
    After flipping, a cell ignores the rule for `refractory_time` steps.
    This is "gentle" stickiness - preserves rule structure but adds inertia.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    cooldown = np.zeros(width, dtype=int)  # Refractory countdown
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            if cooldown[i] > 0:
                # In refractory period: keep current state
                history[t][i] = history[t-1][i]
                cooldown[i] -= 1
            else:
                # Normal ECA rule
                left = history[t-1][(i - 1) % width]
                center = history[t-1][i]
                right = history[t-1][(i + 1) % width]
                pattern = (left << 2) | (center << 1) | right
                new_state = (rule >> pattern) & 1

                if new_state != history[t-1][i]:
                    # State changed - enter refractory period
                    cooldown[i] = refractory_time

                history[t][i] = new_state

    return history


# =============================================================================
# MECHANISM 2: CONFIRMATION REQUIRED (changes must be confirmed next step)
# =============================================================================

def confirmation_eca(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """
    ECA where state changes require confirmation.
    A cell only flips if the rule says to flip for TWO consecutive steps.
    This creates "sticky" states that resist single-step perturbations.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    pending = np.zeros(width, dtype=int)  # Pending new state (-1 = none)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            proposed = (rule >> pattern) & 1

            if proposed != history[t-1][i]:
                # Rule wants to change state
                if pending[i] == proposed:
                    # Confirmed! Apply the change
                    history[t][i] = proposed
                    pending[i] = -1
                else:
                    # First request - mark as pending, keep old state
                    pending[i] = proposed
                    history[t][i] = history[t-1][i]
            else:
                # Rule wants to keep state - reset pending
                history[t][i] = history[t-1][i]
                pending[i] = -1

    return history


# =============================================================================
# MECHANISM 3: ASYMMETRIC TRANSITIONS (birth easy, death hard)
# =============================================================================

def asymmetric_transition_eca(rule: int, width: int, steps: int,
                               death_confirm: int = 2, init=None) -> np.ndarray:
    """
    ECA with asymmetric transitions.
    - 0→1: Follows standard rule (easy to create)
    - 1→0: Requires `death_confirm` consecutive requests (hard to destroy)

    This creates "sticky 1s" - structures persist longer.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    death_count = np.zeros(width, dtype=int)  # Consecutive death requests
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            proposed = (rule >> pattern) & 1

            if history[t-1][i] == 0:
                # Dead cell - standard rule applies for birth
                history[t][i] = proposed
                death_count[i] = 0
            else:
                # Live cell - death requires confirmation
                if proposed == 0:
                    death_count[i] += 1
                    if death_count[i] >= death_confirm:
                        history[t][i] = 0  # Finally dies
                        death_count[i] = 0
                    else:
                        history[t][i] = 1  # Survives (sticky)
                else:
                    history[t][i] = 1  # Rule says live
                    death_count[i] = 0

    return history


# =============================================================================
# MECHANISM 4: MEMORY AVERAGING (state = weighted average of history)
# =============================================================================

def memory_averaging_eca(rule: int, width: int, steps: int,
                         memory_depth: int = 3, threshold: float = 0.5,
                         init=None) -> np.ndarray:
    """
    ECA where effective state is majority vote of recent history.
    Cells consider their last `memory_depth` states when applying the rule.
    Creates smooth, inertial dynamics.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    history[0] = init.copy()

    for t in range(1, steps):
        for i in range(width):
            # Compute effective states from memory
            def effective_state(cell_idx, time_idx):
                if time_idx < 0:
                    return history[0][cell_idx]
                start = max(0, time_idx - memory_depth + 1)
                avg = np.mean(history[start:time_idx+1, cell_idx])
                return 1 if avg >= threshold else 0

            left = effective_state((i - 1) % width, t - 1)
            center = effective_state(i, t - 1)
            right = effective_state((i + 1) % width, t - 1)

            pattern = (left << 2) | (center << 1) | right
            history[t][i] = (rule >> pattern) & 1

    return history


# =============================================================================
# MECHANISM 5: NEIGHBOR HISTORY (rule considers neighbor's past too)
# =============================================================================

def neighbor_history_eca(rule: int, width: int, steps: int, init=None) -> np.ndarray:
    """
    Extended ECA where rule input includes previous state of center cell.
    Pattern becomes 4 bits: (left, center, right, center_previous)
    This adds "self-memory" to the standard neighborhood.

    Uses extended rule: 16 possible inputs, but we derive from 8-bit rule.
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    history[0] = init.copy()
    history[1] = init.copy()  # Need two initial states

    # Bootstrap t=1 with standard ECA
    for i in range(width):
        left = history[0][(i - 1) % width]
        center = history[0][i]
        right = history[0][(i + 1) % width]
        pattern = (left << 2) | (center << 1) | right
        history[1][i] = (rule >> pattern) & 1

    for t in range(2, steps):
        for i in range(width):
            left = history[t-1][(i - 1) % width]
            center = history[t-1][i]
            right = history[t-1][(i + 1) % width]
            prev = history[t-2][i]

            # Standard pattern
            pattern = (left << 2) | (center << 1) | right
            eca_out = (rule >> pattern) & 1

            # Modify based on previous state (soft stickiness)
            if prev == center:
                # Stable - follow rule
                history[t][i] = eca_out
            else:
                # Just changed - bias toward stability
                # 50% chance to keep new state vs follow rule
                if prev == eca_out:
                    history[t][i] = eca_out  # Rule and inertia agree
                else:
                    # Rule and inertia disagree - flip a "coin" based on neighborhood
                    # Use parity of extended neighborhood as pseudo-random
                    extended = left + center + right + prev
                    history[t][i] = eca_out if extended % 2 == 0 else center

    return history


# =============================================================================
# METRICS
# =============================================================================

def compute_entropy(state: np.ndarray) -> float:
    unique, counts = np.unique(state, return_counts=True)
    probs = counts / len(state)
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_activity(history: np.ndarray) -> float:
    if len(history) < 2:
        return 0.0
    changes = np.sum(history[1:] != history[:-1], axis=1)
    return np.mean(changes) / history.shape[1]


def compute_control(history: np.ndarray) -> float:
    """Control proxy: divergent outcomes for same pattern."""
    if len(history) < 50:
        return 0.0

    history = history[-50:]
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
    """Compression ratio - proxy for structured complexity."""
    import zlib
    flat = history.flatten().astype(np.uint8).tobytes()
    compressed = zlib.compress(flat, level=9)
    return len(compressed) / len(flat)


def compute_persistence(history: np.ndarray) -> float:
    """Average time cells stay in same state."""
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


def all_metrics(history: np.ndarray) -> Dict[str, float]:
    return {
        'entropy': compute_entropy(history[-1]),
        'activity': compute_activity(history),
        'control': compute_control(history),
        'compression': compute_compression(history),
        'persistence': compute_persistence(history)
    }


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_gentle_stickiness_experiment(rules: List[int] = [110, 30, 90, 54],
                                      width: int = 100, steps: int = 200):
    """Compare gentle stickiness mechanisms across multiple rules."""
    print("=" * 70)
    print("GENTLE STICKINESS MECHANISMS - Comparison")
    print("=" * 70)

    results = {}

    mechanisms = [
        ('Standard', lambda r: standard_eca(r, width, steps)),
        ('Refractory_2', lambda r: refractory_eca(r, width, steps, refractory_time=2)),
        ('Refractory_3', lambda r: refractory_eca(r, width, steps, refractory_time=3)),
        ('Confirmation', lambda r: confirmation_eca(r, width, steps)),
        ('AsymDeath_2', lambda r: asymmetric_transition_eca(r, width, steps, death_confirm=2)),
        ('AsymDeath_3', lambda r: asymmetric_transition_eca(r, width, steps, death_confirm=3)),
        ('MemoryAvg_3', lambda r: memory_averaging_eca(r, width, steps, memory_depth=3)),
        ('NeighborHist', lambda r: neighbor_history_eca(r, width, steps)),
    ]

    for rule in rules:
        print(f"\n{'='*70}")
        print(f"Rule {rule} (binary: {bin(rule)[2:].zfill(8)})")
        print(f"{'='*70}")
        print(f"{'Mechanism':<20} {'Entropy':<10} {'Activity':<10} {'Control':<10} "
              f"{'Compress':<10} {'Persist':<10}")
        print("-" * 70)

        results[rule] = {}
        for name, func in mechanisms:
            try:
                history = func(rule)
                metrics = all_metrics(history)
                results[rule][name] = metrics

                print(f"{name:<20} {metrics['entropy']:<10.4f} {metrics['activity']:<10.4f} "
                      f"{metrics['control']:<10.4f} {metrics['compression']:<10.4f} "
                      f"{metrics['persistence']:<10.2f}")
            except Exception as e:
                print(f"{name:<20} ERROR: {e}")

    return results


def visualize_gentle_mechanisms(rule: int = 110, width: int = 100, steps: int = 150):
    """Visual comparison of gentle stickiness mechanisms."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    mechanisms = [
        ('Standard', standard_eca(rule, width, steps)),
        ('Refractory (t=2)', refractory_eca(rule, width, steps, refractory_time=2)),
        ('Refractory (t=3)', refractory_eca(rule, width, steps, refractory_time=3)),
        ('Confirmation', confirmation_eca(rule, width, steps)),
        ('AsymDeath (n=2)', asymmetric_transition_eca(rule, width, steps, death_confirm=2)),
        ('AsymDeath (n=3)', asymmetric_transition_eca(rule, width, steps, death_confirm=3)),
        ('MemoryAvg (d=3)', memory_averaging_eca(rule, width, steps, memory_depth=3)),
        ('NeighborHist', neighbor_history_eca(rule, width, steps)),
        ('MemoryAvg (d=5)', memory_averaging_eca(rule, width, steps, memory_depth=5)),
    ]

    for idx, (name, history) in enumerate(mechanisms):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(history, cmap='binary', aspect='auto')
        metrics = all_metrics(history)
        ax.set_title(f'{name}\nCtrl={metrics["control"]:.3f}, '
                     f'Comp={metrics["compression"]:.3f}')
        if idx >= 6:
            ax.set_xlabel('Cell')
        if idx % 3 == 0:
            ax.set_ylabel('Time')

    plt.suptitle(f'Gentle Stickiness Mechanisms - Rule {rule}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'gentle_stickiness_rule{rule}.png', dpi=150)
    plt.close()
    print(f"\nSaved visualization to {OUTPUT_DIR / f'gentle_stickiness_rule{rule}.png'}")


def find_best_mechanism():
    """Find the mechanism that best balances Control and structured complexity."""
    print("\n" + "=" * 70)
    print("SEARCHING FOR OPTIMAL STICKINESS")
    print("Goal: Maximize Control while maintaining structured complexity")
    print("=" * 70)

    best_score = 0
    best_config = None
    all_scores = []

    rules = [110, 30, 90, 54, 62, 118, 105, 150]
    width, steps = 100, 200

    for rule in rules:
        for refract in [1, 2, 3, 4]:
            history = refractory_eca(rule, width, steps, refractory_time=refract)
            m = all_metrics(history)

            # Score: control * (1 - |compression - 0.4|)
            # We want control high and compression near 0.3-0.5 (complex but not random)
            comp_score = 1 - abs(m['compression'] - 0.4)
            score = m['control'] * comp_score * m['entropy']

            all_scores.append({
                'rule': rule, 'mechanism': f'Refractory_{refract}',
                'control': m['control'], 'compression': m['compression'],
                'score': score
            })

            if score > best_score:
                best_score = score
                best_config = (rule, f'Refractory_{refract}', m)

        for death_n in [2, 3, 4]:
            history = asymmetric_transition_eca(rule, width, steps, death_confirm=death_n)
            m = all_metrics(history)
            comp_score = 1 - abs(m['compression'] - 0.4)
            score = m['control'] * comp_score * m['entropy']

            all_scores.append({
                'rule': rule, 'mechanism': f'AsymDeath_{death_n}',
                'control': m['control'], 'compression': m['compression'],
                'score': score
            })

            if score > best_score:
                best_score = score
                best_config = (rule, f'AsymDeath_{death_n}', m)

    # Report top configurations
    sorted_scores = sorted(all_scores, key=lambda x: -x['score'])

    print(f"\n{'Rank':<6} {'Rule':<8} {'Mechanism':<15} {'Control':<10} "
          f"{'Compress':<10} {'Score':<10}")
    print("-" * 65)

    for i, s in enumerate(sorted_scores[:15]):
        print(f"{i+1:<6} {s['rule']:<8} {s['mechanism']:<15} {s['control']:<10.4f} "
              f"{s['compression']:<10.4f} {s['score']:<10.4f}")

    if best_config:
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION: Rule {best_config[0]}, {best_config[1]}")
        print(f"Metrics: {best_config[2]}")

    return sorted_scores


def main():
    print("\n" + "=" * 70)
    print("GENTLE STICKINESS EXPERIMENT")
    print("=" * 70)

    # Basic comparison
    results = run_gentle_stickiness_experiment(rules=[110, 30, 90, 54])

    # Save results
    with open(OUTPUT_DIR / 'gentle_stickiness_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Visualizations
    for rule in [110, 30]:
        visualize_gentle_mechanisms(rule)

    # Find optimal
    scores = find_best_mechanism()

    # Save scores
    with open(OUTPUT_DIR / 'stickiness_optimization.json', 'w') as f:
        json.dump(scores, f, indent=2)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
