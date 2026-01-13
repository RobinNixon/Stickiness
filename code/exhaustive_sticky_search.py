"""
Exhaustive Search: All 256 ECA Rules × Stickiness Mechanisms

Goal: Find optimal configurations that maximize:
- Control (context-dependent divergence)
- Structured complexity (not too random, not too periodic)
- Persistence (stickiness effect)

Focus on most promising mechanisms from initial experiments:
- Confirmation (best Control with structure)
- Refractory (good balance)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# CELLULAR AUTOMATA IMPLEMENTATIONS
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
    """Changes require 2 consecutive requests."""
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


def refractory_eca(rule: int, width: int, steps: int,
                   refractory_time: int = 2, init=None) -> np.ndarray:
    """Cells have cooldown period after changing."""
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


def compute_control(history: np.ndarray, window: int = 50) -> float:
    """Control: same pattern → different outcomes (context-dependent)."""
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


def compute_persistence(history: np.ndarray) -> float:
    """Average run length (how long cells stay in same state)."""
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
# EXHAUSTIVE SEARCH
# =============================================================================

def exhaustive_search(width: int = 80, steps: int = 150):
    """Search all 256 rules × key mechanisms."""
    print("=" * 70)
    print("EXHAUSTIVE STICKINESS SEARCH")
    print(f"256 rules × 4 mechanisms = 1024 configurations")
    print("=" * 70)

    results = {}

    mechanisms = {
        'standard': lambda r, w, s: standard_eca(r, w, s),
        'confirmation': lambda r, w, s: confirmation_eca(r, w, s),
        'refractory_2': lambda r, w, s: refractory_eca(r, w, s, refractory_time=2),
        'refractory_3': lambda r, w, s: refractory_eca(r, w, s, refractory_time=3),
    }

    for rule in range(256):
        results[rule] = {}

        for mech_name, mech_func in mechanisms.items():
            try:
                history = mech_func(rule, width, steps)
                metrics = all_metrics(history)
                results[rule][mech_name] = metrics
            except Exception as e:
                results[rule][mech_name] = {'error': str(e)}

        if (rule + 1) % 32 == 0:
            print(f"  Processed {rule + 1}/256 rules")

    return results


def analyze_results(results: Dict) -> Dict:
    """Find best configurations and patterns."""
    analysis = {
        'best_control': [],
        'best_balance': [],
        'control_improvement': [],
        'by_mechanism': defaultdict(list)
    }

    for rule, mechs in results.items():
        std = mechs.get('standard', {})
        std_control = std.get('control', 0)

        for mech_name, metrics in mechs.items():
            if 'error' in metrics:
                continue

            # Track by mechanism
            analysis['by_mechanism'][mech_name].append({
                'rule': rule,
                **metrics
            })

            # Track control improvement over standard
            if mech_name != 'standard':
                improvement = metrics.get('control', 0) - std_control
                analysis['control_improvement'].append({
                    'rule': rule,
                    'mechanism': mech_name,
                    'improvement': improvement,
                    'final_control': metrics.get('control', 0)
                })

            # Score: Control × (1 - |compression - 0.4|) × entropy
            # Best balance between Control, complexity, and non-triviality
            comp = metrics.get('compression', 0)
            ent = metrics.get('entropy', 0)
            ctrl = metrics.get('control', 0)

            comp_score = 1 - abs(comp - 0.4) if comp > 0 else 0
            balance_score = ctrl * comp_score * ent

            analysis['best_balance'].append({
                'rule': rule,
                'mechanism': mech_name,
                'control': ctrl,
                'compression': comp,
                'entropy': ent,
                'score': balance_score
            })

            # Track high control configurations
            if ctrl > 0.3:
                analysis['best_control'].append({
                    'rule': rule,
                    'mechanism': mech_name,
                    'control': ctrl,
                    'compression': comp
                })

    # Sort
    analysis['best_control'].sort(key=lambda x: -x['control'])
    analysis['best_balance'].sort(key=lambda x: -x['score'])
    analysis['control_improvement'].sort(key=lambda x: -x['improvement'])

    return analysis


def print_analysis(analysis: Dict):
    """Print key findings."""
    print("\n" + "=" * 70)
    print("TOP 20 BY CONTROL")
    print("=" * 70)
    print(f"{'Rule':<8} {'Mechanism':<15} {'Control':<10} {'Compression':<10}")
    print("-" * 50)
    for item in analysis['best_control'][:20]:
        print(f"{item['rule']:<8} {item['mechanism']:<15} {item['control']:<10.4f} "
              f"{item['compression']:<10.4f}")

    print("\n" + "=" * 70)
    print("TOP 20 BY BALANCED SCORE (Control × Complexity × Entropy)")
    print("=" * 70)
    print(f"{'Rule':<8} {'Mechanism':<15} {'Score':<10} {'Control':<10} "
          f"{'Compress':<10} {'Entropy':<10}")
    print("-" * 70)
    for item in analysis['best_balance'][:20]:
        print(f"{item['rule']:<8} {item['mechanism']:<15} {item['score']:<10.4f} "
              f"{item['control']:<10.4f} {item['compression']:<10.4f} "
              f"{item['entropy']:<10.4f}")

    print("\n" + "=" * 70)
    print("BIGGEST CONTROL IMPROVEMENTS FROM STICKINESS")
    print("=" * 70)
    print(f"{'Rule':<8} {'Mechanism':<15} {'Improvement':<12} {'Final Control':<12}")
    print("-" * 50)
    for item in analysis['control_improvement'][:20]:
        print(f"{item['rule']:<8} {item['mechanism']:<15} +{item['improvement']:<11.4f} "
              f"{item['final_control']:<12.4f}")

    # Statistics by mechanism
    print("\n" + "=" * 70)
    print("STATISTICS BY MECHANISM")
    print("=" * 70)
    for mech_name, configs in analysis['by_mechanism'].items():
        controls = [c['control'] for c in configs if 'control' in c]
        compressions = [c['compression'] for c in configs if 'compression' in c]

        if controls:
            print(f"\n{mech_name}:")
            print(f"  Control: mean={np.mean(controls):.4f}, "
                  f"max={np.max(controls):.4f}, "
                  f"rules>0.3: {sum(1 for c in controls if c > 0.3)}")
            print(f"  Compression: mean={np.mean(compressions):.4f}")


def visualize_exhaustive(results: Dict, analysis: Dict):
    """Visualize the exhaustive search results."""

    # 1. Control by rule and mechanism
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    mechanisms = ['standard', 'confirmation', 'refractory_2', 'refractory_3']

    for idx, mech in enumerate(mechanisms):
        ax = axes[idx // 2, idx % 2]
        controls = []
        for rule in range(256):
            ctrl = results[rule].get(mech, {}).get('control', 0)
            controls.append(ctrl)

        ax.bar(range(256), controls, width=1.0, alpha=0.7)
        ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Control=0.3')
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Control=0.5')
        ax.set_xlabel('Rule Number')
        ax.set_ylabel('Control Proxy')
        ax.set_title(f'{mech}: Control by Rule')
        ax.set_xlim(0, 255)
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exhaustive_control_by_mechanism.png', dpi=150)
    plt.close()

    # 2. Control vs Compression scatter
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {'standard': 'gray', 'confirmation': 'blue',
              'refractory_2': 'green', 'refractory_3': 'orange'}

    for mech in mechanisms:
        controls = []
        compressions = []
        for rule in range(256):
            m = results[rule].get(mech, {})
            if 'control' in m and 'compression' in m:
                controls.append(m['control'])
                compressions.append(m['compression'])

        ax.scatter(compressions, controls, alpha=0.5, label=mech,
                   c=colors.get(mech, 'black'), s=20)

    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.axvline(0.4, color='blue', linestyle='--', alpha=0.3)
    ax.set_xlabel('Compression (lower = more random)')
    ax.set_ylabel('Control Proxy')
    ax.set_title('Control vs Compression: All Rules × Mechanisms')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'control_vs_compression_scatter.png', dpi=150)
    plt.close()

    # 3. Best configurations spacetime
    best = analysis['best_balance'][:9]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    mechanisms_map = {
        'standard': standard_eca,
        'confirmation': confirmation_eca,
        'refractory_2': lambda r, w, s: refractory_eca(r, w, s, 2),
        'refractory_3': lambda r, w, s: refractory_eca(r, w, s, 3),
    }

    for idx, config in enumerate(best):
        rule = config['rule']
        mech = config['mechanism']
        func = mechanisms_map.get(mech)

        if func:
            history = func(rule, 80, 100)
            axes[idx].imshow(history, cmap='binary', aspect='auto')
            axes[idx].set_title(f"Rule {rule} + {mech}\n"
                                f"Score={config['score']:.3f}, "
                                f"Ctrl={config['control']:.3f}")
        if idx >= 6:
            axes[idx].set_xlabel('Cell')
        if idx % 3 == 0:
            axes[idx].set_ylabel('Time')

    plt.suptitle('Best Balanced Configurations', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'best_balanced_spacetime.png', dpi=150)
    plt.close()

    print(f"\nVisualizations saved to {OUTPUT_DIR}")


def main():
    print("\n" + "=" * 70)
    print("EXHAUSTIVE STICKINESS SEARCH")
    print("=" * 70)

    # Run search
    results = exhaustive_search(width=80, steps=150)

    # Save raw results
    with open(OUTPUT_DIR / 'exhaustive_results.json', 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(results), f, indent=2)

    # Analyze
    analysis = analyze_results(results)
    print_analysis(analysis)

    # Save analysis
    with open(OUTPUT_DIR / 'exhaustive_analysis.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(dict(analysis)), f, indent=2, default=str)

    # Visualize
    visualize_exhaustive(results, analysis)

    print("\n" + "=" * 70)
    print("EXHAUSTIVE SEARCH COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
