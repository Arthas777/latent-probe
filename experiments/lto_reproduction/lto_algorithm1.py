"""
Experiment A: Reproduce LTO Algorithm 1 on Latent-SFT and CoLaR paradigms.

LTO Algorithm 1 (KL-regularized rejection sampling):
  Input: question x, LRM r(x,z), sampling budget N, required samples M, weight β
  1. Sample N trajectories from π_ref(z|x)
  2. Track r_max across all samples
  3. For each candidate z_i, accept with probability φ_i = exp((r(z_i,x) - r_max) / β)
  4. Repeat until M samples collected
  5. Return answer from accepted sample

We demonstrate that LTO Algorithm 1 fails on compression-based paradigms where
the confidence head encodes problem-level suitability rather than trajectory-level
correctness (within-problem AUC ≈ 0.5).

Three settings:
  Setting 1: LSFT(2) deterministic — all N trajectories identical, LTO degenerates
  Setting 2: LSFT(4) + ε=0.001 noise injection — diversity exists but head can't select
  Setting 3: CoLaR + σ=1000 — diversity exists but head can't select
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lto_algorithm1(rollouts, beta, M=1, max_iterations=1000):
    """
    LTO Algorithm 1: KL-regularized rejection sampling.

    Args:
        rollouts: list of dicts with 'confidence' (reward) and 'correct' (label)
        beta: temperature for acceptance probability
        M: number of required accepted samples
        max_iterations: maximum rejection-sampling iterations

    Returns:
        accepted: list of M accepted rollouts
        n_iterations: total iterations used
    """
    N = len(rollouts)
    if N == 0:
        return [], 0

    rewards = np.array([r['confidence'] for r in rollouts])
    r_max = rewards.max()

    accepted = []
    n_iterations = 0

    while len(accepted) < M and n_iterations < max_iterations:
        n_iterations += 1
        # Uniformly sample one of the N candidates
        idx = np.random.randint(0, N)
        z_i = rollouts[idx]
        r_i = rewards[idx]

        # Acceptance probability: φ_i = exp((r_i - r_max) / β)
        phi_i = np.exp((r_i - r_max) / beta)

        # Accept with probability φ_i
        u = np.random.uniform(0, 1)
        if u < phi_i:
            accepted.append(z_i)

    return accepted, n_iterations


def best_of_n(rollouts):
    """Simple best-of-N: pick the rollout with highest confidence."""
    if not rollouts:
        return None
    return max(rollouts, key=lambda r: r['confidence'])


def majority_vote(rollouts):
    """Majority voting: pick the most common answer."""
    if not rollouts:
        return None
    answer_counts = defaultdict(int)
    for r in rollouts:
        answer_counts[r['answer']] += 1
    best_answer = max(answer_counts, key=answer_counts.get)
    for r in rollouts:
        if r['answer'] == best_answer:
            return r
    return rollouts[0]


def weighted_majority_vote(rollouts):
    """Weighted majority voting using confidence as weight."""
    if not rollouts:
        return None
    answer_weights = defaultdict(float)
    for r in rollouts:
        answer_weights[r['answer']] += r['confidence']
    best_answer = max(answer_weights, key=answer_weights.get)
    for r in rollouts:
        if r['answer'] == best_answer:
            return r
    return rollouts[0]


def evaluate_setting(data, betas, N, M=1, n_trials=50, setting_name=""):
    """
    Evaluate LTO Algorithm 1 across multiple β values.

    Args:
        data: list of per-problem dicts with 'rollouts', 'det_correct', 'det_conf'
        betas: list of β values to scan
        N: number of rollouts per problem (use all available or subsample)
        M: number of accepted samples per problem
        n_trials: number of repeated trials for variance estimation
    """
    results = {
        'setting': setting_name,
        'N': N,
        'M': M,
        'n_problems': len(data),
        'n_trials': n_trials,
    }

    # Baseline: deterministic single-sample accuracy
    det_acc = np.mean([p['det_correct'] for p in data])
    results['det_acc'] = det_acc

    # Oracle: best possible accuracy with N rollouts
    oracle_correct = []
    for p in data:
        rollouts = p['rollouts'][:N]
        oracle_correct.append(any(r['correct'] for r in rollouts))
    oracle_acc = np.mean(oracle_correct)
    results['oracle_acc'] = oracle_acc
    results['oracle_delta'] = oracle_acc - det_acc

    # Best-of-N (β→0 limit of LTO)
    bon_accs = []
    for _ in range(n_trials):
        correct_count = 0
        for p in data:
            rollouts = p['rollouts'][:N]
            if rollouts:
                selected = best_of_n(rollouts)
                correct_count += selected['correct']
        bon_accs.append(correct_count / len(data))
    results['best_of_n_acc'] = np.mean(bon_accs)
    results['best_of_n_std'] = np.std(bon_accs)

    # Majority voting
    mv_correct = 0
    for p in data:
        rollouts = p['rollouts'][:N]
        if rollouts:
            selected = majority_vote(rollouts)
            mv_correct += selected['correct']
    results['majority_vote_acc'] = mv_correct / len(data)

    # Weighted majority voting
    wmv_correct = 0
    for p in data:
        rollouts = p['rollouts'][:N]
        if rollouts:
            selected = weighted_majority_vote(rollouts)
            wmv_correct += selected['correct']
    results['weighted_mv_acc'] = wmv_correct / len(data)

    # LTO Algorithm 1 across β values
    results['lto_results'] = {}
    for beta in betas:
        trial_accs = []
        trial_accept_rates = []

        for trial in range(n_trials):
            correct_count = 0
            total_iterations = 0

            for p in data:
                rollouts = p['rollouts'][:N]
                if not rollouts:
                    continue
                accepted, n_iter = lto_algorithm1(rollouts, beta, M=M)
                total_iterations += n_iter
                if accepted:
                    correct_count += accepted[0]['correct']
                else:
                    # Fallback to random if no sample accepted
                    idx = np.random.randint(0, len(rollouts))
                    correct_count += rollouts[idx]['correct']

            trial_accs.append(correct_count / len(data))
            avg_accept_rate = M * len(data) / total_iterations if total_iterations > 0 else 0
            trial_accept_rates.append(avg_accept_rate)

        results['lto_results'][str(beta)] = {
            'acc_mean': np.mean(trial_accs),
            'acc_std': np.std(trial_accs),
            'delta_vs_det': np.mean(trial_accs) - det_acc,
            'accept_rate_mean': np.mean(trial_accept_rates),
        }

    # Random selection baseline
    random_accs = []
    for _ in range(n_trials):
        correct_count = 0
        for p in data:
            rollouts = p['rollouts'][:N]
            if rollouts:
                idx = np.random.randint(0, len(rollouts))
                correct_count += rollouts[idx]['correct']
        random_accs.append(correct_count / len(data))
    results['random_acc'] = np.mean(random_accs)
    results['random_std'] = np.std(random_accs)

    return results


def run_setting1_deterministic(trajectory_data, betas, n_trials=50):
    """
    Setting 1: Deterministic LSFT(2).
    All N trajectories are identical (greedy argmax), so LTO degenerates to single sample.
    We simulate this by creating N copies of the deterministic result.
    """
    data_for_lto = []
    for p in trajectory_data:
        det_rollout = {
            'correct': p['det_correct'],
            'answer': p.get('det_answer', str(p.get('gt', ''))),
            'confidence': p['det_conf'],
            'T': p.get('det_T', 13),
        }
        # N identical copies
        fake_rollouts = [dict(det_rollout) for _ in range(20)]
        data_for_lto.append({
            'det_correct': p['det_correct'],
            'det_conf': p['det_conf'],
            'rollouts': fake_rollouts,
        })

    return evaluate_setting(
        data_for_lto, betas, N=20, M=1, n_trials=n_trials,
        setting_name="Setting 1: LSFT(2) deterministic (N=20 identical)"
    )


def run_setting2_lsft_noise(per_problem_data, betas, n_trials=50):
    """
    Setting 2: LSFT(4) + ε=0.001 noise injection.
    Uses existing rollout data with diversity.
    """
    return evaluate_setting(
        per_problem_data, betas, N=8, M=1, n_trials=n_trials,
        setting_name="Setting 2: LSFT(4) + ε=0.001 noise (N=8)"
    )


def run_setting3_colar(per_problem_data, betas, n_trials=50):
    """
    Setting 3: CoLaR + σ=1000.
    Uses existing rollout data with diversity.
    """
    return evaluate_setting(
        per_problem_data, betas, N=8, M=1, n_trials=n_trials,
        setting_name="Setting 3: CoLaR + σ=1000 (N=8)"
    )


def print_results_table(all_results):
    """Print formatted results table."""
    print("\n" + "="*90)
    print("LTO Algorithm 1 Reproduction Results")
    print("="*90)

    for res in all_results:
        print(f"\n--- {res['setting']} ---")
        print(f"  N={res['N']}, n_problems={res['n_problems']}, n_trials={res['n_trials']}")
        print(f"  Deterministic acc:  {res['det_acc']:.4f}")
        print(f"  Oracle acc:         {res['oracle_acc']:.4f} (Δ={res['oracle_delta']:.4f})")
        print(f"  Random selection:   {res['random_acc']:.4f} ± {res['random_std']:.4f}")
        print(f"  Majority vote:      {res['majority_vote_acc']:.4f}")
        print(f"  Weighted MV:        {res['weighted_mv_acc']:.4f}")
        print(f"  Best-of-N:          {res['best_of_n_acc']:.4f} ± {res['best_of_n_std']:.4f}")
        print(f"\n  LTO Algorithm 1 (β scan):")
        print(f"  {'β':<10} {'Acc':<12} {'Δ vs det':<12} {'Accept rate':<12}")
        print(f"  {'-'*46}")

        best_beta = None
        best_acc = -1
        for beta_str, r in sorted(res['lto_results'].items(), key=lambda x: float(x[0])):
            marker = ""
            if r['acc_mean'] > best_acc:
                best_acc = r['acc_mean']
                best_beta = beta_str
            print(f"  {beta_str:<10} {r['acc_mean']:.4f}±{r['acc_std']:.4f} "
                  f"{r['delta_vs_det']:+.4f}      {r['accept_rate_mean']:.4f}")

        print(f"\n  Best β={best_beta}: acc={best_acc:.4f}, Δ={best_acc - res['det_acc']:+.4f}")

    # Summary table for paper
    print("\n\n" + "="*90)
    print("Paper Table (§10): LTO Algorithm 1 Paradigm Comparison")
    print("="*90)
    print(f"\n{'Method':<35} {'Paradigm':<20} {'Stochasticity':<15} {'β':<8} {'Acc':<10} {'Δ vs base':<10}")
    print("-"*98)

    for res in all_results:
        # Best LTO result for this setting
        best_beta = max(res['lto_results'].items(), key=lambda x: x[1]['acc_mean'])
        beta_str, best_r = best_beta

        setting_short = res['setting'].split(':')[0].strip()
        paradigm = "LSFT(2)" if "LSFT(2)" in res['setting'] else \
                   "LSFT(4)" if "LSFT" in res['setting'] else "CoLaR"
        stoch = "none" if "deterministic" in res['setting'] else \
                "ε=0.001" if "ε=0.001" in res['setting'] else "σ=1000"

        print(f"{'Pure latent (det.)':<35} {paradigm:<20} {stoch:<15} {'—':<8} "
              f"{res['det_acc']:.4f}    {'—':<10}")
        print(f"{'LTO Alg.1 (best β)':<35} {paradigm:<20} {stoch:<15} {beta_str:<8} "
              f"{best_r['acc_mean']:.4f}    {best_r['delta_vs_det']:+.4f}")

    print("-"*98)
    print(f"{'LTO on Huginn (their paper)':<35} {'Huginn-3.5B':<20} {'native':<15} {'best':<8} "
          f"{'0.385':<10} {'+0.059':<10}")
    print(f"{'Our routing (ours)':<35} {'LSFT(2)':<20} {'none':<15} {'—':<8} "
          f"{'0.525':<10} {'+0.025':<10}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce LTO Algorithm 1 on compression paradigms")
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                        help='β values for KL regularization')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of repeated trials for variance estimation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    base_dir = os.path.join(PROJ_ROOT, 'experiments', 'confidence_head', 'direction1_results')

    # Load Setting 2 data: LSFT(4) + ε=0.001
    lsft_path = os.path.join(base_dir, 'lsft_full_per_problem.json')
    print(f"Loading LSFT data from {lsft_path}...")
    with open(lsft_path) as f:
        lsft_data = json.load(f)
    print(f"  Loaded {len(lsft_data)} problems, {len(lsft_data[0]['rollouts'])} rollouts each")

    # Load Setting 3 data: CoLaR σ=1000 eval data (.pt files with confidence)
    import torch
    colar_eval_paths = [
        os.path.join(base_dir, 'eval_gpu0.pt'),
        os.path.join(base_dir, 'eval_gpu1.pt'),
    ]
    colar_data = None
    if all(os.path.exists(p) for p in colar_eval_paths):
        print(f"Loading CoLaR σ=1000 eval data from .pt files...")
        colar_raw = []
        for p in colar_eval_paths:
            colar_raw.extend(torch.load(p, map_location='cpu', weights_only=False))
        # Convert to same format as LSFT data (need 'answer' field — use dummy)
        colar_data = []
        for item in colar_raw:
            rollouts_formatted = []
            for r in item['rollouts']:
                rollouts_formatted.append({
                    'correct': r['correct'],
                    'confidence': r['confidence'],
                    'T': r['T'],
                    'answer': f"ans_{r['correct']}_{r['T']}",  # dummy
                })
            colar_data.append({
                'det_correct': item['det_correct'],
                'det_conf': item['det_conf'],
                'rollouts': rollouts_formatted,
            })
        print(f"  Loaded {len(colar_data)} problems")
        # Stats
        mixed = sum(1 for p in colar_data
                    if any(r['correct'] for r in p['rollouts'])
                    and not all(r['correct'] for r in p['rollouts']))
        oracle = sum(1 for p in colar_data if any(r['correct'] for r in p['rollouts'])) / len(colar_data)
        det_acc = sum(1 for p in colar_data if p['det_correct']) / len(colar_data)
        print(f"  Det acc: {det_acc:.4f}, Oracle: {oracle:.4f}, Mixed: {mixed}")
    else:
        print(f"[WARNING] CoLaR eval .pt not found at {colar_eval_paths}")

    all_results = []

    # ====== Setting 1: Deterministic ======
    print("\n" + "="*60)
    print("Setting 1: LSFT(2) Deterministic (simulated N=20 identical)")
    print("="*60)
    setting1_results = run_setting1_deterministic(lsft_data, args.betas, args.n_trials)
    all_results.append(setting1_results)

    # ====== Setting 2: LSFT + noise ======
    print("\n" + "="*60)
    print("Setting 2: LSFT(4) + ε=0.001 noise injection")
    print("="*60)
    setting2_results = run_setting2_lsft_noise(lsft_data, args.betas, args.n_trials)
    all_results.append(setting2_results)

    # ====== Setting 3: CoLaR + σ=1000 ======
    if colar_data is not None:
        print("\n" + "="*60)
        print("Setting 3: CoLaR + σ=1000")
        print("="*60)
        setting3_results = run_setting3_colar(colar_data, args.betas, args.n_trials)
        all_results.append(setting3_results)
    else:
        print("\n[WARNING] CoLaR per-problem data not found. Skipping Setting 3.")

    # Print and save
    print_results_table(all_results)

    output_file = os.path.join(args.output_dir, 'lto_reproduction_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
