"""
Ablation experiments to address non-trivial confounds:
  Risk A: Is the head just learning length-based routing?
  Risk B: Does the head only need the last few steps?
  Risk D: Epoch selection using test fold (optimistic bias)

Usage:
    python ablation_confounds.py \
        --trajectory_file ./trajectory_data/trajectories_GSM8k.pt \
        --output_dir ./ablation_results \
        --device cuda:0
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm

from train_confidence_head import (
    AttentionConfidenceHead, TrajectoryDataset, roc_auc_score,
    compute_ece, evaluate, MLPConfidenceHead
)


# ============================================================
# Risk D fix: Nested CV (inner val for epoch selection)
# ============================================================

def train_with_inner_val(model, train_loader, inner_val_loader, device, args, model_type='attention'):
    """Train with inner validation set for early stopping. Test set never seen."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            labels = batch['label'].to(device)
            if model_type == 'attention':
                logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
            else:
                logits = model(batch['scalar_features'].to(device))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Early stopping on INNER val (not test fold)
        val_metrics = evaluate(model, inner_val_loader, device, model_type)
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


def nested_cv(dataset, model_class, model_kwargs, args, device, model_type='attention', n_folds=5):
    """Proper nested CV: inner val for epoch selection, outer test for evaluation."""
    n = len(dataset)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n)

    fold_size = n // n_folds
    all_probs = np.zeros(n)
    all_labels = np.zeros(n)

    for fold in range(n_folds):
        # Outer split: test fold
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_val_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Inner split: 80% train, 20% inner-val (for early stopping)
        inner_rng = np.random.default_rng(args.seed + fold)
        inner_perm = inner_rng.permutation(len(train_val_idx))
        n_inner_val = len(train_val_idx) // 5
        inner_val_idx = train_val_idx[inner_perm[:n_inner_val]]
        inner_train_idx = train_val_idx[inner_perm[n_inner_val:]]

        train_loader = DataLoader(Subset(dataset, inner_train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
        inner_val_loader = DataLoader(Subset(dataset, inner_val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = model_class(**model_kwargs).to(device)
        train_with_inner_val(model, train_loader, inner_val_loader, device, args, model_type)

        # Evaluate on held-out test fold (never seen during training or epoch selection)
        metrics = evaluate(model, test_loader, device, model_type)
        all_probs[test_idx] = metrics['probs']
        all_labels[test_idx] = metrics['labels']

        print(f"  Fold {fold+1}/{n_folds}: test_auc={metrics['auc']:.4f}")

    return all_probs, all_labels


# ============================================================
# Risk A: Fixed-length ablation
# ============================================================

class FixedLengthDataset(Dataset):
    """Wraps TrajectoryDataset but masks out length information."""

    def __init__(self, base_dataset, fixed_T=None):
        self.base = base_dataset
        # Use median T as fixed length if not specified
        if fixed_T is None:
            Ts = [base_dataset.data[i]['T'] for i in range(len(base_dataset))]
            fixed_T = int(np.median(Ts))
        self.fixed_T = fixed_T

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # Override T with fixed value (hide length from model)
        item = dict(item)  # shallow copy
        item['T'] = torch.tensor(float(self.fixed_T), dtype=torch.float32)
        # Also override scalar features to remove T-related features
        sf = item['scalar_features'].clone()
        sf[0] = float(self.fixed_T)  # T
        sf[1] = np.log(max(self.fixed_T, 1))  # log T
        item['scalar_features'] = sf
        return item


# ============================================================
# Risk B: Last-k steps ablation
# ============================================================

class LastKDataset(Dataset):
    """Only keep last k steps of trajectory."""

    def __init__(self, base_dataset, k=1):
        self.base = base_dataset
        self.k = k

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item = dict(item)

        hs = item['hidden_states']  # [max_T, hidden_dim]
        mask = item['mask']  # [max_T]

        # Find actual length
        actual_T = int(mask.sum().item())
        if actual_T <= self.k:
            return item  # keep as is

        # Zero out everything except last k steps
        new_mask = torch.zeros_like(mask)
        new_hs = torch.zeros_like(hs)
        start = actual_T - self.k
        new_mask[start:actual_T] = 1.0
        new_hs[start:actual_T] = hs[start:actual_T]

        item['hidden_states'] = new_hs
        item['mask'] = new_mask
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./ablation_results')
    parser.add_argument('--max_T', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = TrajectoryDataset(args.trajectory_file, max_T=args.max_T)
    n = len(dataset)
    hidden_dim = dataset[0]['hidden_states'].shape[-1]
    labels = np.array([dataset.data[i]['correct'] for i in range(n)], dtype=float)
    overall_acc = labels.mean()
    print(f"Loaded {n} samples, acc={overall_acc:.4f}, hidden_dim={hidden_dim}")

    results = {}

    # ============================================================
    # Risk D: Nested CV (proper epoch selection)
    # ============================================================
    print(f"\n{'='*60}")
    print("RISK D: Nested CV (inner val for epoch selection)")
    print(f"{'='*60}")

    model_kwargs = {"hidden_dim": hidden_dim, "proj_dim": 128, "n_heads": 4, "dropout": 0.1}
    probs, cv_labels = nested_cv(dataset, AttentionConfidenceHead, model_kwargs, args, device,
                                  model_type='attention', n_folds=5)
    auc_nested = roc_auc_score(cv_labels, probs)
    print(f"\n  Nested CV Attention AUC: {auc_nested:.4f}")
    results['nested_cv_attention'] = float(auc_nested)

    # ============================================================
    # Risk A: Fixed-length ablation
    # ============================================================
    print(f"\n{'='*60}")
    print("RISK A: Fixed-length ablation (hide T from model)")
    print(f"{'='*60}")

    Ts = [dataset.data[i]['T'] for i in range(n)]
    median_T = int(np.median(Ts))
    print(f"  Median T = {median_T}, using as fixed_T")

    fixed_dataset = FixedLengthDataset(dataset, fixed_T=median_T)
    probs_fixed, labels_fixed = nested_cv(fixed_dataset, AttentionConfidenceHead, model_kwargs, args, device,
                                           model_type='attention', n_folds=5)
    auc_fixed = roc_auc_score(labels_fixed, probs_fixed)
    print(f"\n  Fixed-length Attention AUC: {auc_fixed:.4f}")
    print(f"  Delta vs full: {auc_fixed - auc_nested:+.4f}")
    results['fixed_length_attention'] = float(auc_fixed)

    if auc_fixed < auc_nested - 0.05:
        print("  ⚠️ Length IS a significant driver (>0.05 drop)")
    else:
        print("  ✓ Length is NOT the main driver (<0.05 drop)")

    # ============================================================
    # Risk B: Last-k steps ablation
    # ============================================================
    print(f"\n{'='*60}")
    print("RISK B: Last-k steps ablation")
    print(f"{'='*60}")

    for k in [1, 2, 3, 5]:
        lastk_dataset = LastKDataset(dataset, k=k)
        probs_k, labels_k = nested_cv(lastk_dataset, AttentionConfidenceHead, model_kwargs, args, device,
                                       model_type='attention', n_folds=5)
        auc_k = roc_auc_score(labels_k, probs_k)
        print(f"\n  Last-{k} steps AUC: {auc_k:.4f} (delta vs full: {auc_k - auc_nested:+.4f})")
        results[f'last_{k}_steps'] = float(auc_k)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Full trajectory (nested CV): {results['nested_cv_attention']:.4f}")
    print(f"  Fixed length (hide T):       {results['fixed_length_attention']:.4f} (Δ={results['fixed_length_attention']-results['nested_cv_attention']:+.4f})")
    for k in [1, 2, 3, 5]:
        key = f'last_{k}_steps'
        print(f"  Last-{k} steps only:          {results[key]:.4f} (Δ={results[key]-results['nested_cv_attention']:+.4f})")

    print(f"\n  LENS baseline AUC: ~0.713 (Latent-SFT) / ~0.650 (CoLaR)")

    # Save
    output_file = os.path.join(args.output_dir, "ablation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
