"""
Tier 2 Task 4: Mechanism deep-dive — per-step contribution analysis.

Experiments:
  1. Attention weight distribution: which steps does the trained head attend to?
  2. Leave-one-out: mask each step individually, measure AUC drop
  3. Incremental: add steps from end, measure AUC growth
  4. Gradient attribution: gradient of confidence logit w.r.t. each step's hidden state

Runs on both Latent-SFT and CoLaR.

Usage:
    python mechanism_analysis.py --device cuda:0
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm

from train_confidence_head import (
    AttentionConfidenceHead, TrajectoryDataset, roc_auc_score
)


def train_head_for_analysis(dataset, device, hidden_dim, seed=42, epochs=80):
    """Train a single attention head on 80% data, validate on 20%, return model + val set."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = n // 5
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False, num_workers=0)

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
        val_probs, val_labels = [], []
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

    model.load_state_dict(best_state)
    print(f"  Trained head: best val AUC = {best_auc:.4f} (stopped at epoch {epoch+1-patience})")
    return model, val_idx


def get_predictions(model, dataset, device, indices=None):
    """Get predictions on a subset."""
    if indices is not None:
        loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False, num_workers=0)
    else:
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


# ============================================================
# Experiment 1: Attention Weight Analysis
# ============================================================

def attention_weight_analysis(model, dataset, device, indices):
    """Extract attention weights from trained head to see which steps are attended."""
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False, num_workers=0)

    all_attn_weights = []
    all_Ts = []
    all_labels = []

    model.eval()

    # Hook to capture attention weights
    attn_weights_storage = []

    def attn_hook(module, input, output):
        # output is (attn_output, attn_weights) when need_weights=True
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights_storage.append(output[1].detach().cpu())

    # Temporarily modify to get weights
    # We need to modify the forward to use need_weights=True
    original_forward = model.attn.forward

    def forward_with_weights(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True
        return original_forward(*args, **kwargs)

    model.attn.forward = forward_with_weights

    with torch.no_grad():
        for batch in loader:
            attn_weights_storage.clear()
            hs = batch['hidden_states'].to(device)
            mask = batch['mask'].to(device)
            T = batch['T'].to(device)

            # Manual forward to capture attention
            x = model.proj(hs)
            key_padding_mask = ~mask.bool()
            attn_out, attn_w = model.attn(x, x, x, key_padding_mask=key_padding_mask,
                                           need_weights=True, average_attn_weights=True)

            # attn_w: [B, seq_len, seq_len]
            # We want to see which positions get attended (column-wise sum)
            B = attn_w.shape[0]
            for i in range(B):
                actual_T = int(mask[i].sum().item())
                if actual_T > 0:
                    # Average attention received by each position
                    w = attn_w[i, :actual_T, :actual_T].cpu().numpy()
                    # Column mean = how much each position is attended to
                    col_mean = w.mean(axis=0)
                    all_attn_weights.append(col_mean)
                    all_Ts.append(actual_T)
                    all_labels.append(batch['label'][i].item())

    model.attn.forward = original_forward

    return all_attn_weights, all_Ts, all_labels


def summarize_attention_by_position(attn_weights, Ts, labels):
    """Summarize attention patterns by relative position (0=first, 1=last)."""
    n_bins = 10
    correct_profile = np.zeros(n_bins)
    incorrect_profile = np.zeros(n_bins)
    correct_count = np.zeros(n_bins)
    incorrect_count = np.zeros(n_bins)

    for w, T, label in zip(attn_weights, Ts, labels):
        if T < 2:
            continue
        # Normalize position to [0, 1]
        positions = np.linspace(0, 1, T)
        bins = np.clip((positions * n_bins).astype(int), 0, n_bins - 1)

        for pos_idx, bin_idx in enumerate(bins):
            if label > 0.5:
                correct_profile[bin_idx] += w[pos_idx]
                correct_count[bin_idx] += 1
            else:
                incorrect_profile[bin_idx] += w[pos_idx]
                incorrect_count[bin_idx] += 1

    correct_profile = np.divide(correct_profile, correct_count, where=correct_count > 0)
    incorrect_profile = np.divide(incorrect_profile, incorrect_count, where=incorrect_count > 0)

    return correct_profile, incorrect_profile


# ============================================================
# Experiment 2: Leave-One-Out
# ============================================================

class MaskedStepDataset(Dataset):
    """Mask out a specific step (by relative position) from trajectory."""
    def __init__(self, base_dataset, mask_position='last', mask_step_idx=None):
        self.base = base_dataset
        self.mask_position = mask_position
        self.mask_step_idx = mask_step_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item = dict(item)
        hs = item['hidden_states'].clone()
        mask = item['mask'].clone()
        actual_T = int(mask.sum().item())

        if actual_T < 2:
            return item

        if self.mask_step_idx is not None:
            step = self.mask_step_idx
            if step < actual_T:
                hs[step] = 0.0
                mask[step] = 0.0
        elif self.mask_position == 'last':
            hs[actual_T - 1] = 0.0
            mask[actual_T - 1] = 0.0
        elif self.mask_position == 'second_last':
            if actual_T >= 2:
                hs[actual_T - 2] = 0.0
                mask[actual_T - 2] = 0.0
        elif self.mask_position == 'first':
            hs[0] = 0.0
            mask[0] = 0.0

        item['hidden_states'] = hs
        item['mask'] = mask
        return item


def leave_one_out_analysis(model, dataset, device, indices, max_T_for_analysis=10):
    """For each relative position, mask it and measure AUC drop."""
    baseline_probs, baseline_labels = get_predictions(model, dataset, device, indices)
    baseline_auc = roc_auc_score(baseline_labels, baseline_probs)

    results = {'baseline_auc': baseline_auc, 'position_results': []}

    # Analyze by relative position (last, second-last, etc.)
    for offset in range(min(max_T_for_analysis, 8)):
        # Mask the step at position (T - 1 - offset) for each sample
        masked_ds = MaskedStepDataset(dataset, mask_step_idx=None)
        # Need custom approach per sample
        probs_list = []
        labels_list = []

        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)

        model.eval()
        with torch.no_grad():
            for batch in loader:
                hs = batch['hidden_states'].clone().to(device)
                mask = batch['mask'].clone().to(device)
                T_vals = batch['T'].to(device)

                # Mask the step at (actual_T - 1 - offset)
                B = hs.shape[0]
                for i in range(B):
                    actual_T = int(mask[i].sum().item())
                    step_to_mask = actual_T - 1 - offset
                    if 0 <= step_to_mask < actual_T:
                        hs[i, step_to_mask] = 0.0
                        mask[i, step_to_mask] = 0.0

                logits = model(hs, mask, T_vals)
                probs_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels_list.extend(batch['label'].numpy().tolist())

        masked_auc = roc_auc_score(labels_list, probs_list)
        delta = masked_auc - baseline_auc
        results['position_results'].append({
            'offset_from_end': offset,
            'position_label': f'T-{offset+1}' if offset > 0 else 'last (T)',
            'masked_auc': masked_auc,
            'delta': delta,
        })

    return results


# ============================================================
# Experiment 3: Incremental from End
# ============================================================

def incremental_from_end(model, dataset, device, indices, max_k=10):
    """Add steps incrementally from the end, measure AUC growth."""
    results = []

    for k in range(1, min(max_k + 1, 9)):
        # Only keep last-k steps
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)

        probs_list = []
        labels_list = []

        model.eval()
        with torch.no_grad():
            for batch in loader:
                hs = batch['hidden_states'].clone().to(device)
                mask = batch['mask'].clone().to(device)
                T_vals = batch['T'].to(device)

                B = hs.shape[0]
                for i in range(B):
                    actual_T = int(mask[i].sum().item())
                    if actual_T > k:
                        # Zero out everything before (actual_T - k)
                        start = actual_T - k
                        hs[i, :start] = 0.0
                        mask[i, :start] = 0.0

                logits = model(hs, mask, T_vals)
                probs_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels_list.extend(batch['label'].numpy().tolist())

        auc = roc_auc_score(labels_list, probs_list)
        results.append({'k': k, 'auc': auc})

    return results


# ============================================================
# Experiment 4: Gradient Attribution
# ============================================================

def gradient_attribution(model, dataset, device, indices, n_samples=200):
    """Compute gradient of confidence logit w.r.t. each step's hidden state."""
    subset_indices = indices[:min(n_samples, len(indices))]
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    all_grad_norms = []  # per-sample list of [T] gradient norms
    all_Ts = []
    all_labels = []

    model.eval()

    for batch in loader:
        hs = batch['hidden_states'].to(device).requires_grad_(True)
        mask = batch['mask'].to(device)
        T_val = batch['T'].to(device)

        logit = model(hs, mask, T_val)
        logit.backward()

        grad = hs.grad[0]  # [max_T, hidden_dim]
        actual_T = int(mask[0].sum().item())

        if actual_T > 0:
            grad_norms = grad[:actual_T].norm(dim=-1).detach().cpu().numpy()
            all_grad_norms.append(grad_norms)
            all_Ts.append(actual_T)
            all_labels.append(batch['label'][0].item())

        hs.grad = None

    return all_grad_norms, all_Ts, all_labels


def summarize_gradients_by_position(grad_norms, Ts, labels):
    """Summarize gradient attribution by relative position."""
    n_bins = 10
    correct_profile = np.zeros(n_bins)
    incorrect_profile = np.zeros(n_bins)
    correct_count = np.zeros(n_bins)
    incorrect_count = np.zeros(n_bins)

    for grads, T, label in zip(grad_norms, Ts, labels):
        if T < 2:
            continue
        positions = np.linspace(0, 1, T)
        bins = np.clip((positions * n_bins).astype(int), 0, n_bins - 1)
        # Normalize grads
        grads_norm = grads / (grads.sum() + 1e-10)

        for pos_idx, bin_idx in enumerate(bins):
            if label > 0.5:
                correct_profile[bin_idx] += grads_norm[pos_idx]
                correct_count[bin_idx] += 1
            else:
                incorrect_profile[bin_idx] += grads_norm[pos_idx]
                incorrect_count[bin_idx] += 1

    correct_profile = np.divide(correct_profile, correct_count, where=correct_count > 0)
    incorrect_profile = np.divide(incorrect_profile, incorrect_count, where=incorrect_count > 0)

    return correct_profile, incorrect_profile


# ============================================================
# Main
# ============================================================

def run_paradigm(paradigm_name, trajectory_file, device, hidden_dim):
    """Run all mechanism analyses for one paradigm."""
    print(f"\n{'#'*70}")
    print(f"  MECHANISM ANALYSIS: {paradigm_name}")
    print(f"{'#'*70}")

    dataset = TrajectoryDataset(trajectory_file, max_T=64)
    n = len(dataset)
    print(f"  Loaded {n} samples")

    # Train head
    print("\n  [1] Training attention head...")
    model, val_idx = train_head_for_analysis(dataset, device, hidden_dim)

    # Experiment 1: Attention weights
    print("\n  [2] Attention weight analysis...")
    attn_weights, Ts, labels = attention_weight_analysis(model, dataset, device, val_idx)
    correct_attn, incorrect_attn = summarize_attention_by_position(attn_weights, Ts, labels)

    print("    Attention profile (relative position 0=start, 9=end):")
    print(f"    Correct:   {' '.join(f'{v:.3f}' for v in correct_attn)}")
    print(f"    Incorrect: {' '.join(f'{v:.3f}' for v in incorrect_attn)}")
    print(f"    Ratio (end/start): correct={correct_attn[-1]/(correct_attn[0]+1e-10):.2f}, incorrect={incorrect_attn[-1]/(incorrect_attn[0]+1e-10):.2f}")

    # Experiment 2: Leave-one-out
    print("\n  [3] Leave-one-out analysis...")
    loo_results = leave_one_out_analysis(model, dataset, device, val_idx)
    print(f"    Baseline AUC: {loo_results['baseline_auc']:.4f}")
    for r in loo_results['position_results']:
        marker = " ***" if abs(r['delta']) > 0.01 else ""
        print(f"    Mask {r['position_label']:<12}: AUC={r['masked_auc']:.4f} (Δ={r['delta']:+.4f}){marker}")

    # Experiment 3: Incremental from end
    print("\n  [4] Incremental from end...")
    incr_results = incremental_from_end(model, dataset, device, val_idx)
    for r in incr_results:
        print(f"    Last-{r['k']}: AUC={r['auc']:.4f}")

    # Experiment 4: Gradient attribution
    print("\n  [5] Gradient attribution...")
    grad_norms, grad_Ts, grad_labels = gradient_attribution(model, dataset, device, val_idx)
    correct_grad, incorrect_grad = summarize_gradients_by_position(grad_norms, grad_Ts, grad_labels)

    print("    Gradient profile (relative position 0=start, 9=end):")
    print(f"    Correct:   {' '.join(f'{v:.3f}' for v in correct_grad)}")
    print(f"    Incorrect: {' '.join(f'{v:.3f}' for v in incorrect_grad)}")

    # Compute endpoint concentration metric
    end_fraction_correct = correct_grad[-3:].sum() / (correct_grad.sum() + 1e-10)
    end_fraction_incorrect = incorrect_grad[-3:].sum() / (incorrect_grad.sum() + 1e-10)
    print(f"    Last-30% gradient fraction: correct={end_fraction_correct:.3f}, incorrect={end_fraction_incorrect:.3f}")

    return {
        'paradigm': paradigm_name,
        'attention_profile': {'correct': correct_attn.tolist(), 'incorrect': incorrect_attn.tolist()},
        'leave_one_out': loo_results,
        'incremental': incr_results,
        'gradient_profile': {'correct': correct_grad.tolist(), 'incorrect': incorrect_grad.tolist()},
        'endpoint_gradient_fraction': {
            'correct': float(end_fraction_correct),
            'incorrect': float(end_fraction_incorrect),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hidden_dim = 2048

    base_dir = os.path.dirname(os.path.abspath(__file__))

    results = []

    # Latent-SFT
    lsft_path = os.path.join(base_dir, 'trajectory_data/trajectories_GSM8k.pt')
    if os.path.exists(lsft_path):
        r = run_paradigm('Latent-SFT', lsft_path, device, hidden_dim)
        results.append(r)

    # CoLaR
    colar_path = os.path.join(base_dir, 'trajectory_data_colar/trajectories_GSM8k_colar.pt')
    if os.path.exists(colar_path):
        r = run_paradigm('CoLaR', colar_path, device, hidden_dim)
        results.append(r)

    # Summary
    print(f"\n\n{'='*70}")
    print("MECHANISM ANALYSIS SUMMARY")
    print(f"{'='*70}")

    for r in results:
        paradigm = r['paradigm']
        print(f"\n  {paradigm}:")
        print(f"    Attention: end-heavy (last bin / first bin ratio)")
        attn_c = r['attention_profile']['correct']
        print(f"      Correct: {attn_c[-1]/(attn_c[0]+1e-10):.2f}x")

        loo = r['leave_one_out']
        most_important = min(loo['position_results'], key=lambda x: x['masked_auc'])
        print(f"    Most important step (leave-one-out): {most_important['position_label']} (Δ={most_important['delta']:+.4f})")

        incr = r['incremental']
        print(f"    Incremental: last-1={incr[0]['auc']:.4f}, last-3={incr[2]['auc']:.4f}, full={loo['baseline_auc']:.4f}")

        print(f"    Gradient concentration in last 30%: correct={r['endpoint_gradient_fraction']['correct']:.3f}, incorrect={r['endpoint_gradient_fraction']['incorrect']:.3f}")

    # Save
    output_file = os.path.join(args.output_dir, 'mechanism_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
