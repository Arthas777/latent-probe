"""
Tier 2 Task 5: Comparison with calibration baselines.

Implements:
  1. Temperature scaling on last-token confidence (top-1 probability)
  2. Platt scaling (logistic regression) on last-token confidence
  3. Histogram binning calibration
  4. Platt scaling on multiple features (T, entropy, top1)

Compares with trained hidden-state probe to prove:
  "hidden-state probe > calibrated logit"

Usage:
    python calibration_baselines.py --device cuda:0
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from scipy.optimize import minimize_scalar

from train_confidence_head import (
    TrajectoryDataset, AttentionConfidenceHead, roc_auc_score, compute_ece
)
from evaluate_selective import selective_accuracy, compute_aurc, compute_auccr


def extract_features(dataset):
    """Extract per-sample features for calibration."""
    raw_data = dataset.data
    features = {
        'top1_last': [],      # last-token top-1 probability
        'mean_entropy': [],   # mean trajectory entropy
        'T': [],              # trajectory length
        'H_spec': [],         # spectral entropy
        'lens': [],           # LENS score
    }
    labels = []

    for item in raw_data:
        T = item['T']
        ent = item['token_entropies']
        top1 = item['top1_probs']
        spec = item['spectral_features']

        last_conf = top1[-1].item() if hasattr(top1, '__len__') and len(top1) > 0 else 0.5
        features['top1_last'].append(last_conf)

        mean_ent = ent.mean().item() if hasattr(ent, 'mean') and len(ent) > 0 else 0.0
        features['mean_entropy'].append(mean_ent)

        features['T'].append(float(T))
        H_spec = spec.get('H_spec', 0.0)
        features['H_spec'].append(H_spec)
        features['lens'].append(H_spec / np.log(max(T, 2)))

        labels.append(float(item['correct']))

    for k in features:
        features[k] = np.array(features[k])
    labels = np.array(labels)

    return features, labels


# ============================================================
# Temperature Scaling
# ============================================================

def temperature_scaling_cv(confidences, labels, n_folds=5, seed=42):
    """Fit temperature on train folds, evaluate on test folds."""
    n = len(labels)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    all_calibrated = np.zeros(n)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Fit temperature on train
        train_conf = confidences[train_idx]
        train_labels = labels[train_idx]

        # Convert confidence to logit
        train_logits = np.log(train_conf / (1 - train_conf + 1e-10) + 1e-10)

        def nll(T):
            scaled = 1.0 / (1.0 + np.exp(-train_logits / T))
            eps = 1e-10
            loss = -np.mean(train_labels * np.log(scaled + eps) + (1 - train_labels) * np.log(1 - scaled + eps))
            return loss

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        best_T = result.x

        # Apply to test
        test_logits = np.log(confidences[test_idx] / (1 - confidences[test_idx] + 1e-10) + 1e-10)
        all_calibrated[test_idx] = 1.0 / (1.0 + np.exp(-test_logits / best_T))

    return all_calibrated


# ============================================================
# Platt Scaling (logistic regression on single feature)
# ============================================================

def platt_scaling_cv(feature, labels, n_folds=5, seed=42):
    """Fit logistic regression (Platt scaling) on a single feature."""
    n = len(labels)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    all_calibrated = np.zeros(n)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Fit a + b*x logistic regression
        train_x = feature[train_idx]
        train_y = labels[train_idx]

        def nll(params):
            a, b = params
            logits = a + b * train_x
            probs = 1.0 / (1.0 + np.exp(-logits))
            eps = 1e-10
            loss = -np.mean(train_y * np.log(probs + eps) + (1 - train_y) * np.log(1 - probs + eps))
            return loss

        # Simple gradient descent
        a, b = 0.0, 1.0
        lr = 0.01
        for _ in range(1000):
            logits = a + b * train_x
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad_a = np.mean(probs - train_y)
            grad_b = np.mean((probs - train_y) * train_x)
            a -= lr * grad_a
            b -= lr * grad_b

        # Apply to test
        test_logits = a + b * feature[test_idx]
        all_calibrated[test_idx] = 1.0 / (1.0 + np.exp(-test_logits))

    return all_calibrated


# ============================================================
# Multi-feature Platt (logistic regression on multiple features)
# ============================================================

def multi_feature_platt_cv(features_matrix, labels, n_folds=5, seed=42):
    """Logistic regression on multiple features (T, entropy, top1, H_spec)."""
    n = len(labels)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    all_calibrated = np.zeros(n)
    d = features_matrix.shape[1]

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_X = features_matrix[train_idx]
        train_y = labels[train_idx]

        # Normalize
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0) + 1e-8
        train_X_norm = (train_X - mean) / std

        # Fit logistic regression with gradient descent
        w = np.zeros(d)
        bias = 0.0
        lr = 0.01

        for _ in range(2000):
            logits = train_X_norm @ w + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            error = probs - train_y
            grad_w = (error[:, None] * train_X_norm).mean(axis=0) + 0.01 * w
            grad_b = error.mean()
            w -= lr * grad_w
            bias -= lr * grad_b

        # Apply to test
        test_X_norm = (features_matrix[test_idx] - mean) / std
        test_logits = test_X_norm @ w + bias
        all_calibrated[test_idx] = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -30, 30)))

    return all_calibrated


# ============================================================
# Trained Hidden-State Probe (nested CV)
# ============================================================

def hidden_state_probe_cv(dataset, device, hidden_dim, seed=42, n_folds=5):
    """Nested CV for trained attention head (same as in eval_selective_complete)."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    all_probs = np.zeros(n)
    all_labels = np.zeros(n)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_val_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        inner_rng = np.random.default_rng(seed + fold)
        inner_perm = inner_rng.permutation(len(train_val_idx))
        n_inner_val = len(train_val_idx) // 5
        inner_val_idx = train_val_idx[inner_perm[:n_inner_val]]
        inner_train_idx = train_val_idx[inner_perm[n_inner_val:]]

        train_loader = DataLoader(Subset(dataset, inner_train_idx), batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(dataset, inner_val_idx), batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False, num_workers=0)

        model = AttentionConfidenceHead(hidden_dim=hidden_dim, proj_dim=128, n_heads=4, dropout=0.1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None
        patience = 0

        for epoch in range(80):
            model.train()
            for batch in train_loader:
                lab = batch['label'].to(device)
                logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
                loss = criterion(logits, lab)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            vp, vl = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
                    vp.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                    vl.extend(batch['label'].numpy().tolist())
            vauc = roc_auc_score(vl, vp)
            if vauc > best_auc:
                best_auc = vauc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 15:
                    break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        tp, tl = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
                tp.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                tl.extend(batch['label'].numpy().tolist())
        all_probs[test_idx] = np.array(tp)
        all_labels[test_idx] = np.array(tl)

    return all_probs, all_labels


def evaluate_method(name, probs, labels):
    """Compute all metrics for a calibration method."""
    auc = roc_auc_score(labels, probs)
    ece = compute_ece(probs, labels)
    aurc = compute_aurc(probs, labels)
    sel = selective_accuracy(probs, labels, [0.1, 0.3, 0.5, 0.7, 0.9])
    return {
        'method': name,
        'auc': float(auc),
        'ece': float(ece),
        'aurc': float(aurc),
        'acc_at_50': float(sel[0.5]),
    }


def run_paradigm(paradigm_name, trajectory_file, device, hidden_dim):
    """Run calibration comparison for one paradigm."""
    print(f"\n{'#'*70}")
    print(f"  CALIBRATION BASELINES: {paradigm_name}")
    print(f"{'#'*70}")

    dataset = TrajectoryDataset(trajectory_file, max_T=64)
    features, labels = extract_features(dataset)
    overall_acc = labels.mean()
    print(f"  N={len(labels)}, accuracy={overall_acc:.4f}")

    results = []

    # 1. Raw last-token confidence (uncalibrated)
    raw_conf = features['top1_last']
    # For CoLaR, top1_probs are placeholders (all zeros). Use alternative.
    if raw_conf.max() < 1e-5:
        print("  NOTE: top1_probs are placeholder (CoLaR). Using LENS as base confidence.")
        raw_conf = features['lens']
        base_name = 'LENS (raw)'
    else:
        base_name = 'Last-token conf (raw)'

    r = evaluate_method(base_name, raw_conf, labels)
    results.append(r)
    print(f"  {r['method']:<35} AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}  Acc@50%={r['acc_at_50']:.4f}")

    # 2. Temperature scaling on last-token confidence
    if features['top1_last'].max() > 1e-5:
        temp_scaled = temperature_scaling_cv(features['top1_last'], labels)
        r = evaluate_method('Last-token + Temp scaling', temp_scaled, labels)
        results.append(r)
        print(f"  {r['method']:<35} AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}  Acc@50%={r['acc_at_50']:.4f}")

    # 3. Platt scaling on last-token confidence
    if features['top1_last'].max() > 1e-5:
        platt_conf = platt_scaling_cv(features['top1_last'], labels)
        r = evaluate_method('Last-token + Platt scaling', platt_conf, labels)
        results.append(r)
        print(f"  {r['method']:<35} AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}  Acc@50%={r['acc_at_50']:.4f}")

    # 4. Platt scaling on mean entropy
    platt_ent = platt_scaling_cv(-features['mean_entropy'], labels)
    r = evaluate_method('Mean entropy + Platt scaling', platt_ent, labels)
    results.append(r)
    print(f"  {r['method']:<35} AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}  Acc@50%={r['acc_at_50']:.4f}")

    # 5. Platt scaling on LENS
    platt_lens = platt_scaling_cv(features['lens'], labels)
    r = evaluate_method('LENS + Platt scaling', platt_lens, labels)
    results.append(r)
    print(f"  {r['method']:<35} AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}  Acc@50%={r['acc_at_50']:.4f}")

    # 6. Multi-feature logistic regression
    feature_matrix = np.column_stack([
        features['T'],
        features['mean_entropy'],
        features['top1_last'],
        features['H_spec'],
        features['lens'],
    ])
    multi_platt = multi_feature_platt_cv(feature_matrix, labels)
    r = evaluate_method('Multi-feature logistic regression', multi_platt, labels)
    results.append(r)
    print(f"  {r['method']:<35} AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}  Acc@50%={r['acc_at_50']:.4f}")

    # 7. Hidden-state probe (nested CV) — our method
    print(f"\n  Training hidden-state probe (nested CV)...")
    probe_probs, probe_labels = hidden_state_probe_cv(dataset, device, hidden_dim)
    r = evaluate_method('Hidden-state probe (ours)', probe_probs, probe_labels)
    results.append(r)
    print(f"  {r['method']:<35} AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}  Acc@50%={r['acc_at_50']:.4f}")

    return {'paradigm': paradigm_name, 'overall_accuracy': float(overall_acc), 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hidden_dim = 2048
    base_dir = os.path.dirname(os.path.abspath(__file__))

    all_results = []

    # Latent-SFT
    lsft_path = os.path.join(base_dir, 'trajectory_data/trajectories_GSM8k.pt')
    if os.path.exists(lsft_path):
        r = run_paradigm('Latent-SFT', lsft_path, device, hidden_dim)
        all_results.append(r)

    # CoLaR
    colar_path = os.path.join(base_dir, 'trajectory_data_colar/trajectories_GSM8k_colar.pt')
    if os.path.exists(colar_path):
        r = run_paradigm('CoLaR', colar_path, device, hidden_dim)
        all_results.append(r)

    # Summary
    print(f"\n\n{'='*80}")
    print("CALIBRATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Paradigm':<12} {'Method':<35} {'AUC':>6} {'ECE':>6} {'AURC':>6} {'Acc@50%':>8}")
    print("-" * 80)

    for paradigm_result in all_results:
        paradigm = paradigm_result['paradigm']
        for r in paradigm_result['results']:
            print(f"  {paradigm:<10} {r['method']:<35} {r['auc']:>6.3f} {r['ece']:>6.3f} {r['aurc']:>6.3f} {r['acc_at_50']:>8.3f}")
        print()

    # Save
    output_file = os.path.join(args.output_dir, 'calibration_baselines.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
