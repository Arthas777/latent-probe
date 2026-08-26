"""
LENS analysis: compute from multi-rollout data
1. Signal AUC (cross-problem discrimination)
2. Best-of-N selection performance
3. LENS vs baselines comparison

Usage:
  python analyze_lens.py --input results/lens_rollouts_N8_c5_seed42.json
"""

import json
import argparse
import numpy as np
from scipy import stats
from pathlib import Path


def auc_mannwhitney(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative='two-sided')
    return u_stat / (len(pos) * len(neg))


def best_of_n_by_signal(rollouts, signal_key, N):
    """Select the rollout with highest signal, return its correctness."""
    if len(rollouts) < N:
        return None
    selected = rollouts[:N]
    best_idx = max(range(N), key=lambda i: selected[i][signal_key])
    return selected[best_idx]["correct"]


def majority_vote(rollouts, N):
    """Majority vote over N rollouts."""
    if len(rollouts) < N:
        return None
    selected = rollouts[:N]
    answers = [r["pred_answer"] for r in selected]
    from collections import Counter
    vote = Counter(answers).most_common(1)[0][0]
    gt = None
    for r in selected:
        if r["pred_answer"] == vote:
            return r["correct"]
    return 0.0


def oracle_best_of_n(rollouts, N):
    """Oracle: return True if any of N rollouts is correct."""
    if len(rollouts) < N:
        return None
    selected = rollouts[:N]
    return float(any(r["correct"] for r in selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from {args.input}")
    n_rollouts_per_q = [q["n_valid_rollouts"] for q in data]
    print(f"Rollouts per question: min={min(n_rollouts_per_q)}, "
          f"max={max(n_rollouts_per_q)}, mean={np.mean(n_rollouts_per_q):.1f}")

    # Flatten for signal-level AUC
    all_rollouts = []
    for q in data:
        for r in q["rollouts"]:
            all_rollouts.append(r)

    correct = np.array([r["correct"] for r in all_rollouts])
    lens_vals = np.array([r["lens"] for r in all_rollouts])
    h_vals = np.array([r["spectral_entropy"] for r in all_rollouts])
    erank_vals = np.array([r["effective_rank"] for r in all_rollouts])
    cos_sim_vals = np.array([r["mean_cos_sim"] for r in all_rollouts])
    token_ent_vals = np.array([r["mean_token_entropy"] for r in all_rollouts])
    last_conf_vals = np.array([r["last_token_conf"] for r in all_rollouts])
    n_latent_vals = np.array([r["n_latent"] for r in all_rollouts])

    print(f"\nTotal rollouts: {len(all_rollouts)}, Per-rollout acc: {correct.mean():.4f}")

    # ================================================================
    # Part 1: Signal-level AUC
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 1: Signal AUC (predicting correct rollout)")
    print("=" * 70)

    signals = {
        "LENS (H/logT)": lens_vals,
        "Raw H (spectral entropy)": h_vals,
        "Effective Rank": erank_vals,
        "-cos_sim": -cos_sim_vals,
        "Mean token entropy": token_ent_vals,
        "Last-token confidence": last_conf_vals,
        "-Length": -n_latent_vals.astype(float),
    }

    for name, sig in signals.items():
        auc = auc_mannwhitney(correct, sig)
        rho, p = stats.spearmanr(sig, correct)
        print(f"  {name:35s}  AUC={auc:.4f}  rho={rho:+.4f}  p={p:.1e}")

    # ================================================================
    # Part 2: Best-of-N
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 2: Best-of-N Selection Accuracy")
    print("=" * 70)

    N_values = [1, 2, 4, 8]
    max_N = max(q["n_valid_rollouts"] for q in data)

    selection_methods = {
        "LENS": "lens",
        "Raw H": "spectral_entropy",
        "Mean token entropy": "mean_token_entropy",
        "Last-token conf": "last_token_conf",
        "-Length (shortest)": None,  # special handling
    }

    print(f"\n{'Method':<25s}", end="")
    for N in N_values:
        if N <= max_N:
            print(f"  N={N:<4d}", end="")
    print()
    print("-" * 70)

    for method_name, signal_key in selection_methods.items():
        print(f"{method_name:<25s}", end="")
        for N in N_values:
            if N > max_N:
                continue
            accs = []
            for q in data:
                rollouts = q["rollouts"]
                if len(rollouts) < N:
                    continue
                if signal_key is None:
                    # -Length: select shortest
                    selected = rollouts[:N]
                    best_idx = min(range(N), key=lambda i: selected[i]["n_latent"])
                    acc = selected[best_idx]["correct"]
                else:
                    acc = best_of_n_by_signal(rollouts, signal_key, N)
                if acc is not None:
                    accs.append(acc)
            if accs:
                print(f"  {np.mean(accs):.4f}", end="")
            else:
                print(f"  {'N/A':>6s}", end="")
        print()

    # Majority voting
    print(f"{'Majority vote':<25s}", end="")
    for N in N_values:
        if N > max_N:
            continue
        accs = []
        for q in data:
            rollouts = q["rollouts"]
            if len(rollouts) < N:
                continue
            acc = majority_vote(rollouts, N)
            if acc is not None:
                accs.append(acc)
        if accs:
            print(f"  {np.mean(accs):.4f}", end="")
        else:
            print(f"  {'N/A':>6s}", end="")
    print()

    # Oracle
    print(f"{'Oracle (any correct)':<25s}", end="")
    for N in N_values:
        if N > max_N:
            continue
        accs = []
        for q in data:
            rollouts = q["rollouts"]
            if len(rollouts) < N:
                continue
            acc = oracle_best_of_n(rollouts, N)
            if acc is not None:
                accs.append(acc)
        if accs:
            print(f"  {np.mean(accs):.4f}", end="")
        else:
            print(f"  {'N/A':>6s}", end="")
    print()

    # Random baseline
    print(f"{'Random select':<25s}", end="")
    for N in N_values:
        if N > max_N:
            continue
        accs = []
        for q in data:
            rollouts = q["rollouts"]
            if len(rollouts) < N:
                continue
            selected = rollouts[:N]
            acc = np.mean([r["correct"] for r in selected])
            accs.append(acc)
        if accs:
            print(f"  {np.mean(accs):.4f}", end="")
        else:
            print(f"  {'N/A':>6s}", end="")
    print()

    # ================================================================
    # Part 3: Within-question analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 3: Within-question signal (key for Best-of-N)")
    print("=" * 70)
    print("For questions with mixed correct/wrong rollouts:")

    mixed_questions = [
        q for q in data
        if q["n_valid_rollouts"] >= 2
        and 0 < sum(r["correct"] for r in q["rollouts"]) < len(q["rollouts"])
    ]
    print(f"  Mixed questions: {len(mixed_questions)} / {len(data)}")

    if mixed_questions:
        within_aucs = {name: [] for name in signals.keys()}
        for q in mixed_questions:
            rollouts = q["rollouts"]
            c_q = np.array([r["correct"] for r in rollouts])
            if c_q.std() == 0:
                continue
            for name, key in [
                ("LENS (H/logT)", "lens"),
                ("Raw H (spectral entropy)", "spectral_entropy"),
                ("Mean token entropy", "mean_token_entropy"),
                ("Last-token confidence", "last_token_conf"),
                ("-Length", None),
            ]:
                if key is None:
                    vals = -np.array([r["n_latent"] for r in rollouts], dtype=float)
                else:
                    vals = np.array([r[key] for r in rollouts])
                if vals.std() == 0:
                    continue
                auc = auc_mannwhitney(c_q, vals)
                within_aucs[name].append(auc)

        print(f"\n  {'Signal':<35s}  {'Mean AUC':>10s}  {'Median':>8s}  {'N_questions':>12s}")
        print("  " + "-" * 70)
        for name, aucs in within_aucs.items():
            if aucs:
                print(f"  {name:<35s}  {np.mean(aucs):>10.4f}  {np.median(aucs):>8.4f}  {len(aucs):>12d}")

    # ================================================================
    # Part 4: Efficiency analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 4: Compute efficiency (LENS threshold routing)")
    print("=" * 70)

    # For each question, take N=8 rollouts
    # Strategy: if max LENS among first K rollouts > threshold, stop and output
    # Otherwise sample more
    N_full = min(8, max_N)
    questions_with_full = [q for q in data if q["n_valid_rollouts"] >= N_full]
    if questions_with_full:
        # Compute: what if we select by LENS from N_full rollouts
        lens_acc = np.mean([
            best_of_n_by_signal(q["rollouts"], "lens", N_full)
            for q in questions_with_full
            if best_of_n_by_signal(q["rollouts"], "lens", N_full) is not None
        ])
        single_acc = np.mean([q["rollouts"][0]["correct"] for q in questions_with_full])
        print(f"  Single rollout acc: {single_acc:.4f}")
        print(f"  LENS best-of-{N_full} acc: {lens_acc:.4f}")
        print(f"  Gain: +{lens_acc - single_acc:.4f} ({(lens_acc-single_acc)/single_acc*100:.1f}%)")


if __name__ == "__main__":
    main()
