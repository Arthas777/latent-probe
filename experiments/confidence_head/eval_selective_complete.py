"""
Tier 1 Task 2: Complete selective prediction evaluation.

Cross-paradigm (Latent-SFT + CoLaR) × cross-dataset (GSM8k + SVAMP)
× full coverage (10/30/50/70/90%) × all methods.

For in-distribution (GSM8k): uses nested CV out-of-fold predictions.
For OOD (SVAMP): uses trained head inference on pre-collected trajectories.

Usage:
    python eval_selective_complete.py --device cuda:0
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from train_confidence_head import (
    AttentionConfidenceHead, HybridConfidenceHead, MLPConfidenceHead,
    TrajectoryDataset, roc_auc_score, compute_ece
)
from evaluate_selective import selective_accuracy, compute_aurc, compute_auccr


COVERAGE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def get_baseline_signals(dataset):
    """Extract hand-crafted confidence signals from trajectory data."""
    raw_data = dataset.data if hasattr(dataset, 'data') else dataset

    signals = {}
    labels = []

    lens_scores = []
    ent_neg_scores = []
    length_neg_scores = []
    last_token_scores = []

    for item in raw_data:
        T = item['T']
        ent = item['token_entropies']
        top1 = item['top1_probs']
        spec = item['spectral_features']

        H_spec = spec.get('H_spec', 0.0)
        lens_scores.append(H_spec / np.log(max(T, 2)))

        mean_ent = ent.mean().item() if hasattr(ent, 'mean') and len(ent) > 0 else 0.0
        ent_neg_scores.append(-mean_ent)

        length_neg_scores.append(-float(T))

        last_conf = top1[-1].item() if hasattr(top1, '__len__') and len(top1) > 0 else 0.0
        last_token_scores.append(last_conf)

        labels.append(float(item['correct']))

    signals['LENS'] = np.array(lens_scores)
    signals['mean_entropy_neg'] = np.array(ent_neg_scores)
    signals['length_neg'] = np.array(length_neg_scores)
    signals['last_token_conf'] = np.array(last_token_scores)
    labels = np.array(labels)

    return signals, labels


def nested_cv_predictions(dataset, device, hidden_dim, seed=42, n_folds=5, epochs=80):
    """Train attention head with nested CV, return out-of-fold predictions."""
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None
        patience = 0

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                labels = batch['label'].to(device)
                logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                    val_labels.extend(batch['label'].numpy().tolist())
            val_auc = roc_auc_score(val_labels, val_probs)
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 15:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        test_probs = []
        test_labels = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
                test_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                test_labels.extend(batch['label'].numpy().tolist())

        all_probs[test_idx] = np.array(test_probs)
        all_labels[test_idx] = np.array(test_labels)
        fold_auc = roc_auc_score(test_labels, test_probs)
        print(f"    Fold {fold+1}/{n_folds}: AUC={fold_auc:.4f}")

    overall_auc = roc_auc_score(all_labels, all_probs)
    print(f"    Overall nested CV AUC: {overall_auc:.4f}")
    return all_probs, all_labels


def evaluate_ood_with_head(head_path, trajectory_file, device, hidden_dim):
    """Load trained head, run inference on OOD trajectory, return predictions."""
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    model = AttentionConfidenceHead(hidden_dim=hidden_dim, proj_dim=128, n_heads=4, dropout=0.0).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dataset = TrajectoryDataset(trajectory_file, max_T=64)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(batch['label'].numpy().tolist())

    return np.array(all_probs), np.array(all_labels)


def compute_all_metrics(confidences, labels, method_name):
    """Compute AUC, AURC, AUCCR, selective accuracy at standard coverage levels."""
    result = {'method': method_name}

    if len(np.unique(labels)) >= 2:
        result['auc'] = float(roc_auc_score(labels, confidences))
    else:
        result['auc'] = float('nan')

    result['aurc'] = compute_aurc(confidences, labels)
    result['auccr'] = compute_auccr(confidences, labels)

    sel_acc = selective_accuracy(confidences, labels, COVERAGE_LEVELS)
    result['selective_accuracy'] = {str(k): v for k, v in sel_acc.items()}

    return result


def evaluate_paradigm_dataset(paradigm, dataset_name, trajectory_file, device,
                              hidden_dim, head_path=None, use_nested_cv=True):
    """Full evaluation for one paradigm × dataset combination."""
    print(f"\n{'='*60}")
    print(f"  {paradigm} × {dataset_name}")
    print(f"{'='*60}")

    dataset = TrajectoryDataset(trajectory_file, max_T=64)
    baseline_signals, labels = get_baseline_signals(dataset)
    overall_acc = labels.mean()
    n_samples = len(labels)
    print(f"  N={n_samples}, accuracy={overall_acc:.4f}")

    results = {
        'paradigm': paradigm,
        'dataset': dataset_name,
        'n_samples': n_samples,
        'overall_accuracy': float(overall_acc),
        'methods': {}
    }

    # Baselines
    for name, scores in baseline_signals.items():
        metrics = compute_all_metrics(scores, labels, name)
        results['methods'][name] = metrics
        sel50 = metrics['selective_accuracy'].get('0.5', 0)
        print(f"    {name:<20} AUC={metrics['auc']:.4f}  AURC={metrics['aurc']:.4f}  Acc@50%={sel50:.4f}")

    # Trained head
    if use_nested_cv:
        print(f"\n  Training attention head (nested CV)...")
        probs, cv_labels = nested_cv_predictions(dataset, device, hidden_dim)
        metrics = compute_all_metrics(probs, cv_labels, 'trained_attention_nestedCV')
        results['methods']['trained_attention_nestedCV'] = metrics
        sel50 = metrics['selective_accuracy'].get('0.5', 0)
        print(f"    {'trained_attn(CV)':<20} AUC={metrics['auc']:.4f}  AURC={metrics['aurc']:.4f}  Acc@50%={sel50:.4f}")
    elif head_path is not None:
        print(f"  Using pre-trained head for OOD evaluation...")
        probs, head_labels = evaluate_ood_with_head(head_path, trajectory_file, device, hidden_dim)
        metrics = compute_all_metrics(probs, head_labels, 'trained_attention_transfer')
        results['methods']['trained_attention_transfer'] = metrics
        sel50 = metrics['selective_accuracy'].get('0.5', 0)
        print(f"    {'trained_attn(OOD)':<20} AUC={metrics['auc']:.4f}  AURC={metrics['aurc']:.4f}  Acc@50%={sel50:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--skip_nested_cv', action='store_true',
                        help='Skip nested CV (use pre-computed results if available)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hidden_dim = 2048

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define all evaluation configs
    configs = [
        {
            'paradigm': 'Latent-SFT',
            'dataset': 'GSM8k',
            'trajectory_file': os.path.join(base_dir, 'trajectory_data/trajectories_GSM8k.pt'),
            'use_nested_cv': True,
            'head_path': None,
        },
        {
            'paradigm': 'CoLaR',
            'dataset': 'GSM8k',
            'trajectory_file': os.path.join(base_dir, 'trajectory_data_colar/trajectories_GSM8k_colar.pt'),
            'use_nested_cv': True,
            'head_path': None,
        },
        {
            'paradigm': 'Latent-SFT',
            'dataset': 'SVAMP',
            'trajectory_file': os.path.join(base_dir, 'eval_results/trajectories_Svamp.pt'),
            'use_nested_cv': False,
            'head_path': os.path.join(base_dir, 'trained_heads/confidence_head_attention.pt'),
        },
    ]

    # Check for CoLaR SVAMP trajectory (may not exist yet)
    colar_svamp_path = os.path.join(base_dir, 'trajectory_data_colar/trajectories_Svamp_colar.pt')
    if os.path.exists(colar_svamp_path):
        configs.append({
            'paradigm': 'CoLaR',
            'dataset': 'SVAMP',
            'trajectory_file': colar_svamp_path,
            'use_nested_cv': False,
            'head_path': os.path.join(base_dir, 'trained_heads/confidence_head_attention.pt'),
        })

    all_results = []

    for cfg in configs:
        if not os.path.exists(cfg['trajectory_file']):
            print(f"\n  SKIPPING {cfg['paradigm']} × {cfg['dataset']}: trajectory file not found")
            continue

        if args.skip_nested_cv and cfg['use_nested_cv']:
            print(f"\n  SKIPPING {cfg['paradigm']} × {cfg['dataset']}: --skip_nested_cv")
            continue

        result = evaluate_paradigm_dataset(
            paradigm=cfg['paradigm'],
            dataset_name=cfg['dataset'],
            trajectory_file=cfg['trajectory_file'],
            device=device,
            hidden_dim=hidden_dim,
            head_path=cfg['head_path'],
            use_nested_cv=cfg['use_nested_cv'],
        )
        all_results.append(result)

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n\n{'='*80}")
    print("COMPLETE SELECTIVE PREDICTION RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Paradigm':<12} {'Dataset':<8} {'Method':<25} {'AUC':>6} {'AURC':>6} {'Acc@10%':>8} {'Acc@30%':>8} {'Acc@50%':>8} {'Acc@70%':>8} {'Acc@90%':>8}")
    print("-" * 105)

    for result in all_results:
        paradigm = result['paradigm']
        dataset = result['dataset']
        overall = result['overall_accuracy']

        # Always-answer baseline
        print(f"  {paradigm:<10} {dataset:<8} {'always_answer':<25} {'—':>6} {'—':>6} {overall:>8.3f} {overall:>8.3f} {overall:>8.3f} {overall:>8.3f} {overall:>8.3f}")

        for method_name, metrics in result['methods'].items():
            sel = metrics['selective_accuracy']
            auc_str = f"{metrics['auc']:.3f}" if not np.isnan(metrics['auc']) else "—"
            aurc_str = f"{metrics['aurc']:.3f}"
            acc10 = sel.get('0.1', 0)
            acc30 = sel.get('0.3', 0)
            acc50 = sel.get('0.5', 0)
            acc70 = sel.get('0.7', 0)
            acc90 = sel.get('0.9', 0)
            print(f"  {'':<10} {'':<8} {method_name:<25} {auc_str:>6} {aurc_str:>6} {acc10:>8.3f} {acc30:>8.3f} {acc50:>8.3f} {acc70:>8.3f} {acc90:>8.3f}")
        print()

    # Save
    output_file = os.path.join(args.output_dir, 'selective_prediction_complete.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
