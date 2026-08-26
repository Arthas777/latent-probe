"""
Adaptive Routing for CoLaR: confidence-based routing between CoLaR and CoT-SFT.
Same framework as adaptive_routing.py but using CoLaR as the latent reasoning model.

Requires:
  1. CoLaR per-problem confs (from compute_colar_confs.py)
  2. CoT-SFT per-problem results (from run_cot_sft_inference.py)

Note: CoLaR and CoT-SFT use the same 1319 GSM8k-Aug test problems in the same order.

Usage: python adaptive_routing_colar.py
"""
import os, json, numpy as np

np.random.seed(42)

RESULTS_DIR = './routing_results'


def routing_sweep(latent_data, cot_data, taus):
    N = len(latent_data)
    results = []
    for tau in taus:
        n_latent = 0
        n_correct = 0
        total_tokens = 0
        for i in range(N):
            if latent_data[i]['conf'] > tau:
                n_latent += 1
                n_correct += int(latent_data[i]['correct'])
                total_tokens += latent_data[i]['T']
            else:
                n_correct += int(cot_data[i]['correct'])
                total_tokens += cot_data[i]['total_tokens']
        results.append({
            'tau': tau,
            'latent_frac': n_latent / N,
            'accuracy': n_correct / N,
            'avg_tokens': total_tokens / N,
        })
    return results


def random_routing_baseline(latent_data, cot_data, latent_fracs, n_seeds=20):
    N = len(latent_data)
    results = []
    for frac in latent_fracs:
        n_latent = int(round(frac * N))
        accs = []
        tokens_list = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 1000 + int(frac * 100))
            latent_idx = set(rng.choice(N, size=n_latent, replace=False))
            n_correct = 0
            total_tokens = 0
            for i in range(N):
                if i in latent_idx:
                    n_correct += int(latent_data[i]['correct'])
                    total_tokens += latent_data[i]['T']
                else:
                    n_correct += int(cot_data[i]['correct'])
                    total_tokens += cot_data[i]['total_tokens']
            accs.append(n_correct / N)
            tokens_list.append(total_tokens / N)
        results.append({
            'latent_frac': frac,
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'avg_tokens_mean': float(np.mean(tokens_list)),
            'avg_tokens_std': float(np.std(tokens_list)),
        })
    return results


def fold_cv_threshold(latent_data, cot_data, n_folds=5, criterion='iso_accuracy'):
    N = len(latent_data)
    indices = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    fold_size = N // n_folds
    taus_to_try = np.arange(0.05, 0.95, 0.02).tolist()

    fold_results = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else N
        val_idx = set(indices[val_start:val_end].tolist())
        train_idx = [i for i in range(N) if i not in val_idx]

        cot_acc_train = np.mean([cot_data[i]['correct'] for i in train_idx])
        best_tau = None
        best_metric = -1

        for tau in taus_to_try:
            n_correct = 0
            total_tokens = 0
            n_total = len(train_idx)
            for i in train_idx:
                if latent_data[i]['conf'] > tau:
                    n_correct += int(latent_data[i]['correct'])
                    total_tokens += latent_data[i]['T']
                else:
                    n_correct += int(cot_data[i]['correct'])
                    total_tokens += cot_data[i]['total_tokens']
            acc = n_correct / n_total
            avg_tok = total_tokens / n_total

            if criterion == 'iso_accuracy':
                if acc >= cot_acc_train - 0.005:
                    metric = -avg_tok
                    if metric > best_metric:
                        best_metric = metric
                        best_tau = tau
            elif criterion == 'best_efficiency':
                metric = acc / max(avg_tok, 1)
                if metric > best_metric:
                    best_metric = metric
                    best_tau = tau

        if best_tau is None:
            best_tau = 0.1

        val_list = sorted(val_idx)
        n_correct = 0
        total_tokens = 0
        n_latent = 0
        for i in val_list:
            if latent_data[i]['conf'] > best_tau:
                n_latent += 1
                n_correct += int(latent_data[i]['correct'])
                total_tokens += latent_data[i]['T']
            else:
                n_correct += int(cot_data[i]['correct'])
                total_tokens += cot_data[i]['total_tokens']

        fold_results.append({
            'fold': fold,
            'best_tau': best_tau,
            'val_acc': n_correct / len(val_list),
            'val_avg_tokens': total_tokens / len(val_list),
            'val_latent_frac': n_latent / len(val_list),
        })
    return fold_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load CoLaR data
    print("Loading CoLaR data...")
    with open(os.path.join(RESULTS_DIR, 'colar_deterministic_confs.json')) as f:
        colar_raw = json.load(f)
    latent_data = colar_raw['per_problem']
    N = len(latent_data)
    latent_acc = colar_raw['accuracy']
    latent_avg_T = colar_raw['avg_T']
    print(f"  CoLaR: N={N}, acc={latent_acc:.4f}, avg_T={latent_avg_T:.1f}, pooled_AUC={colar_raw['pooled_auc']:.4f}")

    # Load CoT-SFT data
    print("Loading CoT-SFT data...")
    with open(os.path.join(RESULTS_DIR, 'cot_sft_inference.json')) as f:
        cot_raw = json.load(f)
    cot_data = cot_raw['per_problem']
    cot_acc = cot_raw['accuracy']
    cot_avg_tokens = cot_raw['avg_tokens']
    print(f"  CoT-SFT: N={len(cot_data)}, acc={cot_acc:.4f}, avg_tokens={cot_avg_tokens:.1f}")

    # === Step 1: Routing sweep ===
    print("\n=== CoLaR Confidence-based routing sweep ===")
    taus = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    sweep = routing_sweep(latent_data, cot_data, taus)

    print(f"\n{'τ':>6} | {'Latent%':>8} | {'Accuracy':>9} | {'Avg Tok':>8} | {'vs CoT':>7}")
    print("-" * 55)
    for r in sweep:
        tok_vs_cot = r['avg_tokens'] / cot_avg_tokens * 100
        print(f"{r['tau']:6.2f} | {r['latent_frac']*100:7.1f}% | {r['accuracy']:9.4f} | {r['avg_tokens']:8.1f} | {tok_vs_cot:6.1f}%")

    # Find sweet spots
    iso_acc_point = None
    for r in sweep:
        if r['accuracy'] >= cot_acc and r['latent_frac'] > 0:
            if iso_acc_point is None or r['avg_tokens'] < iso_acc_point['avg_tokens']:
                iso_acc_point = r

    best_efficiency = max(sweep, key=lambda r: r['accuracy'] / max(r['avg_tokens'], 1))

    print(f"\n  Pure CoLaR:     acc={latent_acc:.4f}, tokens={latent_avg_T:.1f}")
    print(f"  Pure CoT-SFT:   acc={cot_acc:.4f}, tokens={cot_avg_tokens:.1f}")
    if iso_acc_point:
        saving = 1 - iso_acc_point['avg_tokens'] / cot_avg_tokens
        print(f"  Iso-accuracy:   τ={iso_acc_point['tau']:.2f}, acc={iso_acc_point['accuracy']:.4f}, "
              f"tokens={iso_acc_point['avg_tokens']:.1f} ({saving*100:.1f}% saving)")
    else:
        print(f"  Iso-accuracy:   NOT FOUND")
    print(f"  Best efficiency: τ={best_efficiency['tau']:.2f}, acc={best_efficiency['accuracy']:.4f}, "
          f"tokens={best_efficiency['avg_tokens']:.1f}")

    # === Step 2: Random baseline ===
    print("\n=== Random routing baseline ===")
    latent_fracs_from_sweep = sorted(set(round(r['latent_frac'], 2) for r in sweep if 0 < r['latent_frac'] < 1))
    if not latent_fracs_from_sweep:
        latent_fracs_from_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    random_baseline = random_routing_baseline(latent_data, cot_data, latent_fracs_from_sweep)

    dominates = True
    for rb in random_baseline:
        conf_point = min(sweep, key=lambda s: abs(s['latent_frac'] - rb['latent_frac']))
        if conf_point['accuracy'] < rb['accuracy_mean'] - 0.001:
            dominates = False
            print(f"  WARNING: random dominates at frac={rb['latent_frac']:.2f}")

    print(f"  Confidence routing dominates random: {'YES' if dominates else 'NO'}")

    # === Step 3: 5-Fold CV ===
    print("\n=== 5-Fold CV threshold selection ===")
    cv_results = fold_cv_threshold(latent_data, cot_data, n_folds=5, criterion='iso_accuracy')

    taus_sel = [r['best_tau'] for r in cv_results]
    accs_sel = [r['val_acc'] for r in cv_results]
    toks_sel = [r['val_avg_tokens'] for r in cv_results]
    fracs_sel = [r['val_latent_frac'] for r in cv_results]
    print(f"  τ* = {np.mean(taus_sel):.3f} ± {np.std(taus_sel):.3f}")
    print(f"  Val acc = {np.mean(accs_sel):.4f} ± {np.std(accs_sel):.4f}")
    print(f"  Val tokens = {np.mean(toks_sel):.1f} ± {np.std(toks_sel):.1f}")
    print(f"  Latent frac = {np.mean(fracs_sel):.3f} ± {np.std(fracs_sel):.3f}")
    print(f"  Token saving vs CoT: {(1 - np.mean(toks_sel)/cot_avg_tokens)*100:.1f}%")

    # === Summary ===
    print("\n" + "=" * 70)
    print("ADAPTIVE ROUTING SUMMARY (CoLaR → CoT-SFT)")
    print("=" * 70)

    criteria_a = iso_acc_point is not None and (1 - iso_acc_point['avg_tokens'] / cot_avg_tokens) >= 0.30
    half_cot = cot_avg_tokens * 0.5
    iso_token_point = min(sweep, key=lambda r: abs(r['avg_tokens'] - half_cot))
    criteria_b = iso_token_point['accuracy'] - cot_acc >= 0.02
    criteria_c = dominates

    print(f"\n  Criterion (a): {'PASS' if criteria_a else 'FAIL'}")
    if iso_acc_point:
        print(f"    → τ={iso_acc_point['tau']:.2f}, saving={(1-iso_acc_point['avg_tokens']/cot_avg_tokens)*100:.1f}%")
    print(f"  Criterion (b): {'PASS' if criteria_b else 'FAIL'}")
    print(f"    → At {iso_token_point['avg_tokens']:.1f} tokens, acc={iso_token_point['accuracy']:.4f} (Δ={iso_token_point['accuracy']-cot_acc:+.4f})")
    print(f"  Criterion (c): {'PASS' if criteria_c else 'FAIL'}")

    if criteria_a or criteria_b:
        go_status = "FULL GO"
    elif criteria_c:
        go_status = "MINIMAL GO"
    else:
        go_status = "NO-GO"
    print(f"\n  >>> Decision: {go_status} <<<")

    # Save
    def convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "baselines": {
            "colar": {"accuracy": float(latent_acc), "avg_tokens": float(latent_avg_T), "pooled_auc": float(colar_raw['pooled_auc'])},
            "cot_sft": {"accuracy": float(cot_acc), "avg_tokens": float(cot_avg_tokens)},
        },
        "routing_sweep": sweep,
        "random_baseline": random_baseline,
        "cv_results": cv_results,
        "sweet_spots": {
            "iso_accuracy": iso_acc_point,
            "best_efficiency": {
                'tau': best_efficiency['tau'],
                'accuracy': best_efficiency['accuracy'],
                'avg_tokens': best_efficiency['avg_tokens'],
                'latent_frac': best_efficiency['latent_frac'],
            },
            "iso_token": {
                'tau': iso_token_point['tau'],
                'accuracy': iso_token_point['accuracy'],
                'avg_tokens': iso_token_point['avg_tokens'],
                'latent_frac': iso_token_point['latent_frac'],
            },
        },
        "go_nogo": {
            "criteria_a": bool(criteria_a),
            "criteria_b": bool(criteria_b),
            "criteria_c": bool(criteria_c),
            "decision": go_status,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'colar_routing_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved to {RESULTS_DIR}/colar_routing_results.json")


if __name__ == "__main__":
    main()
