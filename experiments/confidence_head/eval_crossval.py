"""
Cross-validated evaluation of confidence heads.
Uses 5-fold CV to get unbiased estimates of AUC, ECE, and selective prediction.
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from train_confidence_head import (
    MLPConfidenceHead, AttentionConfidenceHead, HybridConfidenceHead,
    TrajectoryDataset, compute_ece, roc_auc_score, train, evaluate
)
from evaluate_selective import selective_accuracy, compute_aurc, compute_auccr


def cross_validate(dataset, model_class, model_kwargs, train_args, device,
                   model_type='attention', n_folds=5, seed=42):
    """Run k-fold cross-validation and return per-sample predictions."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    fold_size = n // n_folds
    all_probs = np.zeros(n)
    all_labels = np.zeros(n)

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=train_args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=train_args.batch_size, shuffle=False, num_workers=0)

        model = model_class(**model_kwargs).to(device)
        history, best_auc = train(model, train_loader, val_loader, device, train_args, model_type=model_type)

        # Get predictions on val fold
        metrics = evaluate(model, val_loader, device, model_type)
        all_probs[val_idx] = metrics['probs']
        all_labels[val_idx] = metrics['labels']

        print(f"  Fold {fold+1}/{n_folds}: val_auc={metrics['auc']:.4f}")

    return all_probs, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_T', type=int, default=64)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading trajectories from {args.trajectory_file}...")
    dataset = TrajectoryDataset(args.trajectory_file, max_T=args.max_T)
    n = len(dataset)
    print(f"Loaded {n} samples.")

    sample = dataset[0]
    hidden_dim = sample['hidden_states'].shape[-1]

    # Get ground truth labels and baseline signals
    labels = np.array([dataset.data[i]['correct'] for i in range(n)], dtype=float)
    overall_acc = labels.mean()
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Baseline signals (no training needed)
    baseline_results = {}
    for name, extractor in [
        ("LENS", lambda d: d['spectral_features']['H_spec'] / np.log(max(d['T'], 2))),
        ("mean_entropy_neg", lambda d: -(d['token_entropies'].mean().item() if len(d['token_entropies']) > 0 else 0)),
        ("length_neg", lambda d: -float(d['T'])),
    ]:
        scores = np.array([extractor(dataset.data[i]) for i in range(n)])
        auc = roc_auc_score(labels, scores)
        sel_acc = selective_accuracy(scores, labels)
        aurc = compute_aurc(scores, labels)
        auccr = compute_auccr(scores, labels)
        baseline_results[name] = {
            "auc": float(auc), "aurc": aurc, "auccr": auccr,
            "selective_accuracy": sel_acc,
        }
        print(f"  Baseline {name:<20}: AUC={auc:.4f} | Acc@50%={sel_acc[0.5]:.4f}")

    # Cross-validated models
    cv_results = {}
    models_to_test = [
        ("mlp", MLPConfidenceHead, {"input_dim": 8, "hidden_dim": 64}),
        ("attention", AttentionConfidenceHead, {"hidden_dim": hidden_dim, "proj_dim": args.proj_dim, "n_heads": 4, "dropout": 0.1}),
        ("hybrid", HybridConfidenceHead, {"hidden_dim": hidden_dim, "proj_dim": args.proj_dim, "n_heads": 4, "n_scalar_features": 8, "dropout": 0.1}),
    ]

    for mt, model_class, model_kwargs in models_to_test:
        print(f"\n{'='*60}")
        print(f"Cross-validating {mt.upper()} ({args.n_folds}-fold)")
        print(f"{'='*60}")

        probs, cv_labels = cross_validate(
            dataset, model_class, model_kwargs, args, device,
            model_type=mt, n_folds=args.n_folds, seed=args.seed
        )

        auc = roc_auc_score(cv_labels, probs)
        ece = compute_ece(probs, cv_labels)
        sel_acc = selective_accuracy(probs, cv_labels)
        aurc = compute_aurc(probs, cv_labels)
        auccr = compute_auccr(probs, cv_labels)
        brier = ((probs - cv_labels) ** 2).mean()

        cv_results[f"trained_{mt}"] = {
            "auc": float(auc), "ece": float(ece), "brier": float(brier),
            "aurc": aurc, "auccr": auccr,
            "selective_accuracy": sel_acc,
        }
        print(f"\n  {mt} CV result: AUC={auc:.4f} | ECE={ece:.4f} | AURC={aurc:.4f} | Acc@50%={sel_acc[0.5]:.4f}")

    # Final comparison
    all_results = {**baseline_results, **cv_results}

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS ({args.n_folds}-FOLD CROSS-VALIDATION)")
    print(f"{'='*80}")
    print(f"Overall accuracy (always answer): {overall_acc:.4f}")
    print(f"\n{'Method':<22} {'AUC':>7} {'ECE':>7} {'AURC↓':>7} {'AUCCR↑':>7} | {'Acc@50%':>8} {'Acc@70%':>8} {'Acc@90%':>8}")
    print("-" * 85)
    for name, res in all_results.items():
        sel = res["selective_accuracy"]
        ece_str = f"{res.get('ece', float('nan')):>7.4f}" if 'ece' in res else "    N/A"
        print(f"  {name:<20} {res['auc']:>7.4f} {ece_str} {res['aurc']:>7.4f} {res['auccr']:>7.4f} | {sel[0.5]:>8.4f} {sel[0.7]:>8.4f} {sel[0.9]:>8.4f}")

    # Improvements over LENS
    lens_auc = baseline_results["LENS"]["auc"]
    print(f"\n--- Improvement over LENS baseline (AUC={lens_auc:.4f}) ---")
    for name, res in cv_results.items():
        delta = res["auc"] - lens_auc
        print(f"  {name:<20}: +{delta:.4f} AUC")

    # Save
    output_file = os.path.join(args.output_dir, "crossval_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "overall_accuracy": float(overall_acc),
            "n_samples": n,
            "n_folds": args.n_folds,
            "methods": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
