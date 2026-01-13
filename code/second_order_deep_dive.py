"""
Second-Order ECA Deep Dive

The initial experiment showed that Second-Order ECAs have dramatically higher
Control proxy (0.937 vs 0.0 for standard ECAs). This script investigates:

1. Which second-order rules show complex behavior?
2. Is there a "stickiness threshold" for emergence?
3. How does second-order Rule 110 compare to standard Rule 110 in detail?
4. Can we find universal candidates in second-order space?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def second_order_eca(rule: int, width: int, steps: int,
                     init: np.ndarray = None,
                     xor_mode: bool = True) -> np.ndarray:
    """
    Second-order ECA with configurable memory combination.

    xor_mode=True: state(t) = ECA(t-1) XOR state(t-2)  [original]
    xor_mode=False: state(t) = ECA(t-1) if state(t-2)==0, else NOT ECA(t-1)
    """
    if init is None:
        init = np.zeros(width, dtype=int)
        init[width // 2] = 1

    history = np.zeros((steps, width), dtype=int)
    history[0] = init.copy()
    history[1] = init.copy()

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
            pattern = (left << 2) | (center << 1) | right
            eca_out = (rule >> pattern) & 1

            if xor_mode:
                history[t][i] = eca_out ^ history[t-2][i]
            else:
                # Alternative: conditional flip based on history
                if history[t-2][i] == 0:
                    history[t][i] = eca_out
                else:
                    history[t][i] = 1 - eca_out

    return history


def standard_eca(rule: int, width: int, steps: int,
                 init: np.ndarray = None) -> np.ndarray:
    """Standard ECA for comparison."""
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

def compute_entropy(state: np.ndarray) -> float:
    """Entropy of final state."""
    unique, counts = np.unique(state, return_counts=True)
    probs = counts / len(state)
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_activity(history: np.ndarray) -> float:
    """Average cell change rate."""
    if len(history) < 2:
        return 0.0
    changes = np.sum(history[1:] != history[:-1], axis=1)
    return np.mean(changes) / history.shape[1]


def compute_control_proxy(history: np.ndarray, window: int = 50) -> float:
    """
    Control proxy: how much does the same pattern lead to different outcomes?
    Uses last `window` timesteps for stability.
    """
    if len(history) < window + 2:
        window = len(history) - 2

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
    """
    Compression ratio as proxy for complexity.
    Random = low compression, periodic = high compression, complex = medium.
    """
    flat = history.flatten().astype(np.uint8).tobytes()
    import zlib
    compressed = zlib.compress(flat, level=9)
    return len(compressed) / len(flat)


def compute_lyapunov_proxy(history1: np.ndarray, history2: np.ndarray) -> float:
    """
    Lyapunov exponent proxy: sensitivity to initial conditions.
    Measures how a single-bit perturbation grows over time.
    """
    differences = np.sum(history1 != history2, axis=1)
    # Avoid log(0)
    differences = np.maximum(differences, 1)

    # Linear regression on log(differences) vs time
    times = np.arange(len(differences))
    log_diff = np.log(differences)

    if len(times) > 10:
        # Use later portion for stability
        times = times[10:]
        log_diff = log_diff[10:]

    if np.std(times) > 0:
        slope = np.corrcoef(times, log_diff)[0, 1] * np.std(log_diff) / np.std(times)
        return slope
    return 0.0


# =============================================================================
# EXHAUSTIVE SECOND-ORDER SCAN
# =============================================================================

def scan_all_second_order_rules(width: int = 80, steps: int = 150) -> Dict:
    """Scan all 256 second-order rules and classify them."""
    print("Scanning all 256 second-order ECA rules...")

    results = {}

    for rule in range(256):
        history = second_order_eca(rule, width, steps)

        # Basic metrics
        entropy = compute_entropy(history[-1])
        activity = compute_activity(history)
        control = compute_control_proxy(history)
        compression = compute_compression(history)

        # Compare to standard ECA
        std_history = standard_eca(rule, width, steps)
        std_control = compute_control_proxy(std_history)

        # Perturbation sensitivity
        init_perturbed = np.zeros(width, dtype=int)
        init_perturbed[width // 2] = 1
        init_perturbed[width // 2 + 1] = 1  # One extra bit

        history_perturbed = second_order_eca(rule, width, steps, init_perturbed)
        lyapunov = compute_lyapunov_proxy(history, history_perturbed)

        results[rule] = {
            'entropy': entropy,
            'activity': activity,
            'control': control,
            'control_improvement': control - std_control,
            'compression': compression,
            'lyapunov': lyapunov
        }

        if (rule + 1) % 32 == 0:
            print(f"  Processed {rule + 1}/256 rules")

    return results


def classify_rules(results: Dict) -> Dict[str, List[int]]:
    """Classify rules by behavior type."""
    classes = {
        'high_control': [],      # Control > 0.5
        'high_complexity': [],   # Good compression + entropy
        'chaotic': [],           # High lyapunov
        'periodic': [],          # Low entropy, high compression
        'dead': [],              # Near-zero activity
        'promising': []          # High control + moderate compression (not random)
    }

    for rule, metrics in results.items():
        if metrics['activity'] < 0.01:
            classes['dead'].append(rule)
        elif metrics['entropy'] < 0.5:
            classes['periodic'].append(rule)
        else:
            if metrics['control'] > 0.5:
                classes['high_control'].append(rule)

            if 0.3 < metrics['compression'] < 0.7:
                classes['high_complexity'].append(rule)

            if metrics['lyapunov'] > 0.1:
                classes['chaotic'].append(rule)

            # Promising: high control + not too random
            if metrics['control'] > 0.5 and metrics['compression'] < 0.7:
                classes['promising'].append(rule)

    return classes


# =============================================================================
# DETAILED RULE ANALYSIS
# =============================================================================

def analyze_rule_detail(rule: int, width: int = 120, steps: int = 200):
    """Detailed analysis of a specific second-order rule."""
    print(f"\n{'='*60}")
    print(f"Detailed Analysis: Second-Order Rule {rule}")
    print(f"Base rule binary: {bin(rule)[2:].zfill(8)}")
    print(f"{'='*60}")

    # Run both versions
    so_history = second_order_eca(rule, width, steps)
    std_history = standard_eca(rule, width, steps)

    # Metrics
    print("\nMetrics Comparison:")
    print(f"{'Metric':<20} {'Standard':<15} {'Second-Order':<15}")
    print("-" * 50)

    for name, func in [('Entropy', compute_entropy),
                       ('Activity', compute_activity),
                       ('Control', compute_control_proxy),
                       ('Compression', compute_compression)]:
        if name in ['Entropy']:
            std_val = func(std_history[-1])
            so_val = func(so_history[-1])
        else:
            std_val = func(std_history)
            so_val = func(so_history)
        print(f"{name:<20} {std_val:<15.4f} {so_val:<15.4f}")

    # Pattern analysis
    print("\nPattern-Outcome Analysis (Second-Order):")
    pattern_outcomes: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for t in range(50, len(so_history) - 1):
        for i in range(1, so_history.shape[1] - 1):
            pattern = (so_history[t, i-1], so_history[t, i], so_history[t, i+1])
            outcome = so_history[t+1, i]
            pattern_outcomes[pattern][outcome] += 1

    # Show patterns with divergent outcomes
    divergent_patterns = []
    for pattern, outcomes in pattern_outcomes.items():
        if len(outcomes) > 1:
            total = sum(outcomes.values())
            if total > 20:  # Enough samples
                max_prob = max(outcomes.values()) / total
                if max_prob < 0.9:  # Significant divergence
                    divergent_patterns.append((pattern, outcomes, max_prob))

    if divergent_patterns:
        print(f"\nFound {len(divergent_patterns)} patterns with divergent outcomes:")
        for pattern, outcomes, max_prob in divergent_patterns[:5]:
            print(f"  Pattern {pattern}: outcomes={dict(outcomes)}, max_prob={max_prob:.2f}")
    else:
        print("  No significantly divergent patterns found")

    return so_history, std_history


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_top_rules(results: Dict, top_n: int = 9):
    """Visualize the top N rules by control proxy."""
    # Sort by control
    sorted_rules = sorted(results.items(), key=lambda x: -x[1]['control'])
    top_rules = [r[0] for r in sorted_rules[:top_n]]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, rule in enumerate(top_rules):
        history = second_order_eca(rule, 100, 100)
        axes[idx].imshow(history, cmap='binary', aspect='auto')
        metrics = results[rule]
        axes[idx].set_title(f'Rule {rule}\nCtrl={metrics["control"]:.3f}, '
                           f'Comp={metrics["compression"]:.3f}')
        if idx >= 6:
            axes[idx].set_xlabel('Cell')
        if idx % 3 == 0:
            axes[idx].set_ylabel('Time')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'second_order_top_control.png', dpi=150)
    plt.close()
    print(f"\nSaved top control visualization to {OUTPUT_DIR / 'second_order_top_control.png'}")


def visualize_rule_comparison(rule: int):
    """Side-by-side comparison of standard vs second-order."""
    width, steps = 120, 150

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    std_history = standard_eca(rule, width, steps)
    so_history = second_order_eca(rule, width, steps)

    axes[0].imshow(std_history, cmap='binary', aspect='auto')
    axes[0].set_title(f'Standard ECA Rule {rule}')
    axes[0].set_xlabel('Cell')
    axes[0].set_ylabel('Time')

    axes[1].imshow(so_history, cmap='binary', aspect='auto')
    axes[1].set_title(f'Second-Order ECA Rule {rule}')
    axes[1].set_xlabel('Cell')
    axes[1].set_ylabel('Time')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'rule{rule}_comparison.png', dpi=150)
    plt.close()
    print(f"Saved comparison to {OUTPUT_DIR / f'rule{rule}_comparison.png'}")


def plot_control_distribution(results: Dict):
    """Distribution of control proxy across all rules."""
    controls = [m['control'] for m in results.values()]
    improvements = [m['control_improvement'] for m in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(controls, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.5, color='red', linestyle='--', label='Control = 0.5')
    axes[0].set_xlabel('Control Proxy')
    axes[0].set_ylabel('Number of Rules')
    axes[0].set_title('Distribution of Control Proxy\n(Second-Order ECAs)')
    axes[0].legend()

    axes[1].hist(improvements, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(0, color='red', linestyle='--', label='No improvement')
    axes[1].set_xlabel('Control Improvement over Standard')
    axes[1].set_ylabel('Number of Rules')
    axes[1].set_title('Control Improvement from Second-Order')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'control_distribution.png', dpi=150)
    plt.close()
    print(f"Saved control distribution to {OUTPUT_DIR / 'control_distribution.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("SECOND-ORDER ECA DEEP DIVE")
    print("=" * 60)

    # Full scan
    results = scan_all_second_order_rules()

    # Save results
    with open(OUTPUT_DIR / 'second_order_scan.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'second_order_scan.json'}")

    # Classification
    classes = classify_rules(results)
    print("\n" + "=" * 60)
    print("RULE CLASSIFICATION")
    print("=" * 60)

    for class_name, rules in classes.items():
        print(f"\n{class_name}: {len(rules)} rules")
        if rules and len(rules) <= 20:
            print(f"  Rules: {sorted(rules)}")
        elif rules:
            print(f"  First 10: {sorted(rules)[:10]}...")

    # Top performers
    print("\n" + "=" * 60)
    print("TOP 10 BY CONTROL PROXY")
    print("=" * 60)

    sorted_by_control = sorted(results.items(), key=lambda x: -x[1]['control'])
    print(f"\n{'Rule':<8} {'Control':<10} {'Improve':<10} {'Compress':<10} {'Lyapunov':<10}")
    print("-" * 48)
    for rule, metrics in sorted_by_control[:10]:
        print(f"{rule:<8} {metrics['control']:<10.4f} {metrics['control_improvement']:<10.4f} "
              f"{metrics['compression']:<10.4f} {metrics['lyapunov']:<10.4f}")

    # Promising candidates (high control, medium compression = not random)
    print("\n" + "=" * 60)
    print("PROMISING CANDIDATES (High Control + Medium Compression)")
    print("=" * 60)

    promising = [(r, m) for r, m in results.items()
                 if m['control'] > 0.5 and 0.25 < m['compression'] < 0.65]
    promising.sort(key=lambda x: -x[1]['control'])

    if promising:
        print(f"\n{'Rule':<8} {'Control':<10} {'Compress':<10} {'Activity':<10}")
        print("-" * 38)
        for rule, metrics in promising[:10]:
            print(f"{rule:<8} {metrics['control']:<10.4f} {metrics['compression']:<10.4f} "
                  f"{metrics['activity']:<10.4f}")

    # Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    visualize_top_rules(results)
    plot_control_distribution(results)

    # Detailed analysis of best candidate
    if promising:
        best_rule = promising[0][0]
        analyze_rule_detail(best_rule)
        visualize_rule_comparison(best_rule)

    # Also analyze Rule 110 (known universal)
    analyze_rule_detail(110)
    visualize_rule_comparison(110)

    print("\n" + "=" * 60)
    print("DEEP DIVE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
