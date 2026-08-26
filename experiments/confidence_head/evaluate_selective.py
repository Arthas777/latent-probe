"""
Step 3: Evaluate confidence head on selective prediction task.
Computes selective accuracy at various coverage levels and AURC.

Usage:
    python evaluate_selective.py \
        --trajectory_file ./trajectory_data/trajectories_GSM8k.pt \
        --head_dir ./trained_heads \
        --output_dir ./eval_results
"""

import os, json, argparse
import numpy as np
import torch

from train_confidence_head import (
    MLPConfidenceHead, AttentionConfidenceHead, HybridConfidenceHead,
    TrajectoryDataset, compute_ece, roc_auc_score
)


def selective_accuracy(confidences, correctness, coverage_levels=None):
    """
    Compute selective accuracy: answer only the top-k% most confident samples.
    Returns dict of coverage → accuracy.
    """
    if coverage_levels is None:
        coverage_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    confidences = np.array(confidences)
    correctness = np.array(correctness)
    n = len(confidences)

    # Sort by confidence descending
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correctness[sorted_idx]

    results = {}
    for cov in coverage_levels:
        k = max(1, int(n * cov))
        acc = sorted_correct[:k].mean()
        results[cov] = float(acc)

    return results


def compute_aurc(confidences, correctness):
    """
    Area Under the Risk-Coverage Curve (AURC).
    Lower is better — measures how well confidence correlates with correctness.
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    n = len(confidences)

    sorted_idx = np.argsort(-confidences)
    sorted_errors = 1.0 - correctness[sorted_idx]

    # Risk at each coverage level
    cumulative_errors = np.cumsum(sorted_errors)
    coverages = np.arange(1, n + 1) / n
    risks = cumulative_errors / np.arange(1, n + 1)

    # Trapezoidal integration
    aurc = np.trapz(risks, coverages)
    return float(aurc)


def compute_auccr(confidences, correctness):
    """
    Area Under the Cumulative Correct Rate curve.
    Higher is better.
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    n = len(confidences)

    sorted_idx = np.argsort(-confidences)
    sorted_correct = correctness[sorted_idx]

    cumulative_correct = np.cumsum(sorted_correct)
    coverages = np.arange(1, n + 1) / n
    accuracies = cumulative_correct / np.arange(1, n + 1)

    auccr = np.trapz(accuracies, coverages)
    return float(auccr)


def get_confidence_from_head(model, dataset, device, model_type, batch_size=64):
    """Get confidence predictions from trained head."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch['label']

            if model_type == 'mlp':
                logits = model(batch['scalar_features'].to(device))
            elif model_type == 'attention':
                logits = model(
                    batch['hidden_states'].to(device),
                    batch['mask'].to(device),
                    batch['T'].to(device),
                )
            else:
                logits = model(
                    batch['hidden_states'].to(device),
                    batch['mask'].to(device),
                    batch['T'].to(device),
                    batch['scalar_features'].to(device),
                )

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_probs), np.array(all_labels)


def get_baseline_confidences(dataset):
    """Extract baseline confidence signals from trajectory data."""
    raw_data = dataset.data if hasattr(dataset, 'data') else dataset.dataset.data

    signals = {
        "LENS": [],
        "mean_entropy_neg": [],
        "last_token_conf": [],
        "length_neg": [],
        "H_spec": [],
    }
    labels = []

    for item in raw_data:
        T = item["T"]
        ent = item["token_entropies"]
        top1 = item["top1_probs"]
        spec = item["spectral_features"]

        # LENS = H_spec / log(T)
        H_spec = spec.get("H_spec", 0.0)
        lens = H_spec / np.log(max(T, 2))
        signals["LENS"].append(lens)
        signals["H_spec"].append(H_spec)

        # Negative mean entropy (lower entropy = more confident)
        mean_ent = ent.mean().item() if len(ent) > 0 else 0.0
        signals["mean_entropy_neg"].append(-mean_ent)

        # Last token confidence
        last_conf = top1[-1].item() if len(top1) > 0 else 0.0
        signals["last_token_conf"].append(last_conf)

        # Negative length (shorter = more confident, for Latent-SFT this is reversed)
        signals["length_neg"].append(-float(T))

        labels.append(float(item["correct"]))

    for k in signals:
        signals[k] = np.array(signals[k])
    labels = np.array(labels)

    return signals, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--head_dir', type=str, default='./trained_heads')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--max_T', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load full dataset
    print(f"Loading trajectories from {args.trajectory_file}...")
    dataset = TrajectoryDataset(args.trajectory_file, max_T=args.max_T)
    print(f"Loaded {len(dataset)} samples.")

    sample = dataset[0]
    hidden_dim = sample['hidden_states'].shape[-1]

    # Get baseline signals
    baseline_signals, labels = get_baseline_confidences(dataset)
    overall_acc = labels.mean()
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Evaluate all methods
    all_results = {}
    coverage_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Baselines
    print("\n--- Baseline Confidence Signals ---")
    for name, confidences in baseline_signals.items():
        sel_acc = selective_accuracy(confidences, labels, coverage_levels)
        aurc = compute_aurc(confidences, labels)
        auccr = compute_auccr(confidences, labels)

        # AUC
        if len(np.unique(labels)) >= 2:
            auc = roc_auc_score(labels, confidences)
        else:
            auc = float('nan')

        all_results[name] = {
            "auc": float(auc),
            "aurc": aurc,
            "auccr": auccr,
            "selective_accuracy": sel_acc,
        }
        print(f"  {name:<20} AUC={auc:.4f} | AURC={aurc:.4f} | AUCCR={auccr:.4f} | Acc@50%={sel_acc[0.5]:.4f}")

    # Trained heads
    print("\n--- Trained Confidence Heads ---")
    head_files = [f for f in os.listdir(args.head_dir) if f.startswith("confidence_head_") and f.endswith(".pt")]

    for hf in sorted(head_files):
        ckpt = torch.load(os.path.join(args.head_dir, hf), map_location=device)
        mt = ckpt["model_type"]
        saved_args = ckpt.get("args", {})

        # Reconstruct model
        if mt == 'mlp':
            model = MLPConfidenceHead(input_dim=8, hidden_dim=64).to(device)
        elif mt == 'attention':
            proj_dim = saved_args.get('proj_dim', 128)
            model = AttentionConfidenceHead(
                hidden_dim=hidden_dim, proj_dim=proj_dim,
                n_heads=4, dropout=0.0
            ).to(device)
        else:
            proj_dim = saved_args.get('proj_dim', 128)
            model = HybridConfidenceHead(
                hidden_dim=hidden_dim, proj_dim=proj_dim,
                n_heads=4, n_scalar_features=8, dropout=0.0
            ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        confidences, head_labels = get_confidence_from_head(model, dataset, device, mt)

        sel_acc = selective_accuracy(confidences, head_labels, coverage_levels)
        aurc = compute_aurc(confidences, head_labels)
        auccr = compute_auccr(confidences, head_labels)
        ece = compute_ece(confidences, head_labels)

        if len(np.unique(head_labels)) >= 2:
            auc = roc_auc_score(head_labels, confidences)
        else:
            auc = float('nan')

        method_name = f"trained_{mt}"
        all_results[method_name] = {
            "auc": float(auc),
            "ece": float(ece),
            "aurc": aurc,
            "auccr": auccr,
            "selective_accuracy": sel_acc,
        }
        print(f"  {method_name:<20} AUC={auc:.4f} | ECE={ece:.4f} | AURC={aurc:.4f} | AUCCR={auccr:.4f} | Acc@50%={sel_acc[0.5]:.4f}")

    # Summary table
    print(f"\n{'='*80}")
    print("SELECTIVE PREDICTION RESULTS")
    print(f"{'='*80}")
    print(f"Overall accuracy (always answer): {overall_acc:.4f}")
    print(f"\n{'Method':<22} {'AUC':>7} {'AURC↓':>7} {'AUCCR↑':>7} | {'Acc@50%':>8} {'Acc@70%':>8} {'Acc@90%':>8}")
    print("-" * 80)
    for name, res in all_results.items():
        sel = res["selective_accuracy"]
        print(f"  {name:<20} {res['auc']:>7.4f} {res['aurc']:>7.4f} {res['auccr']:>7.4f} | {sel[0.5]:>8.4f} {sel[0.7]:>8.4f} {sel[0.9]:>8.4f}")

    # Save results
    output_file = os.path.join(args.output_dir, "selective_prediction_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "overall_accuracy": float(overall_acc),
            "n_samples": len(dataset),
            "methods": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
