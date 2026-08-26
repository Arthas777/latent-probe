"""
Risk C probe: Does the last-step hidden state encode the answer itself?

Two probes:
  1. Answer token probe: predict the first answer token from last-step h_T
  2. Answer value probe: predict the numeric answer value from last-step h_T

If probe accuracy is high → h_T encodes the answer (confidence-as-self-knowledge)
If probe accuracy is low → h_T encodes abstract confidence, not answer content

Additional analysis:
  3. Compare: does the confidence head's AUC correlate with answer-probe accuracy?
     If samples where the probe is correct also have high confidence → head IS doing answer reconstruction
     If samples where the probe is wrong still show calibrated confidence → head learns something beyond answer identity

Usage:
    python probe_answer_leakage.py \
        --trajectory_file ./trajectory_data/trajectories_GSM8k.pt \
        --latent_model_path ../../Latent-SFT/checkpoints/latent-4 \
        --output_dir ./ablation_results \
        --device cuda:0
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'Latent-SFT'))
from transformers import AutoTokenizer, AutoConfig

from train_confidence_head import (
    TrajectoryDataset, AttentionConfidenceHead, roc_auc_score, compute_ece
)


class AnswerProbe(nn.Module):
    """Linear probe: last-step hidden state → answer token logits."""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, vocab_size),
        )

    def forward(self, h):
        return self.probe(h)


class NumericProbe(nn.Module):
    """Regression probe: last-step hidden state → numeric answer value."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return self.probe(h).squeeze(-1)


class ProbeDataset(Dataset):
    def __init__(self, trajectory_file, tokenizer, test_data_path):
        self.data = torch.load(trajectory_file, map_location='cpu', weights_only=False)

        with open(test_data_path) as f:
            if test_data_path.endswith('.json'):
                test_data = json.load(f)
                self.answers = [str(d['answer']) for d in test_data]
            else:
                test_data = [json.loads(l) for l in f]
                self.answers = [str(d['answer']) for d in test_data]

        self.tokenizer = tokenizer

        # Pre-tokenize answers
        self.answer_token_ids = []
        self.answer_values = []
        for ans in self.answers:
            tokens = tokenizer.encode(ans.strip(), add_special_tokens=False)
            self.answer_token_ids.append(tokens[0] if tokens else 0)
            try:
                self.answer_values.append(float(ans.strip().replace(',', '')))
            except ValueError:
                self.answer_values.append(0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        T = item['T']
        hs = item['hidden_states']  # [T, hidden_dim]

        # Get last-step hidden state
        if T > 0 and hs.shape[0] > 0:
            last_h = hs[T-1] if T <= hs.shape[0] else hs[-1]
        else:
            last_h = torch.zeros(hs.shape[-1] if hs.numel() > 0 else 2048)

        return {
            'last_hidden': last_h.float(),
            'answer_token_id': torch.tensor(self.answer_token_ids[idx], dtype=torch.long),
            'answer_value': torch.tensor(self.answer_values[idx], dtype=torch.float32),
            'correct': torch.tensor(float(item['correct']), dtype=torch.float32),
            'T': T,
        }


def train_answer_probe(dataset, device, hidden_dim, vocab_size, epochs=50, batch_size=64, seed=42):
    """Train answer token probe with 5-fold CV."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_folds = 5
    fold_size = n // n_folds

    all_preds = np.zeros(n, dtype=int)
    all_targets = np.zeros(n, dtype=int)
    all_top5_correct = np.zeros(n, dtype=bool)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)

        model = AnswerProbe(hidden_dim, vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                h = batch['last_hidden'].to(device)
                target = batch['answer_token_id'].to(device)
                logits = model(h)
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                h = batch['last_hidden'].to(device)
                target = batch['answer_token_id'].numpy()
                logits = model(h)
                preds = logits.argmax(dim=-1).cpu().numpy()
                top5 = logits.topk(5, dim=-1).indices.cpu().numpy()

                batch_indices = test_idx[len(all_preds[all_preds != 0]):len(all_preds[all_preds != 0])+len(preds)]
                # This indexing is tricky in subset, just collect sequentially

        # Re-evaluate cleanly
        model.eval()
        test_preds = []
        test_targets = []
        test_top5 = []
        with torch.no_grad():
            for batch in test_loader:
                h = batch['last_hidden'].to(device)
                target = batch['answer_token_id']
                logits = model(h)
                preds = logits.argmax(dim=-1).cpu()
                top5 = logits.topk(5, dim=-1).indices.cpu()
                test_preds.append(preds)
                test_targets.append(target)
                test_top5.append(top5)

        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()
        test_top5 = torch.cat(test_top5).numpy()

        all_preds[test_idx] = test_preds
        all_targets[test_idx] = test_targets
        for i, ti in enumerate(test_idx):
            all_top5_correct[ti] = test_targets[i] in test_top5[i]

        fold_acc = (test_preds == test_targets).mean()
        fold_top5 = sum(test_targets[i] in test_top5[i] for i in range(len(test_targets))) / len(test_targets)
        print(f"  Fold {fold+1}: token_acc={fold_acc:.4f}, top5_acc={fold_top5:.4f}")

    return all_preds, all_targets, all_top5_correct


def train_numeric_probe(dataset, device, hidden_dim, epochs=50, batch_size=64, seed=42):
    """Train numeric value probe with 5-fold CV."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_folds = 5
    fold_size = n // n_folds

    all_preds = np.zeros(n)
    all_targets = np.zeros(n)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)

        model = NumericProbe(hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.MSELoss()

        # Normalize targets
        train_values = [dataset[i]['answer_value'].item() for i in train_idx]
        mean_val = np.mean(train_values)
        std_val = np.std(train_values) + 1e-8

        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                h = batch['last_hidden'].to(device)
                target = (batch['answer_value'].to(device) - mean_val) / std_val
                pred = model(h)
                loss = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        test_preds_list = []
        test_targets_list = []
        with torch.no_grad():
            for batch in test_loader:
                h = batch['last_hidden'].to(device)
                pred = model(h).cpu().numpy() * std_val + mean_val
                test_preds_list.append(pred)
                test_targets_list.append(batch['answer_value'].numpy())

        test_preds_arr = np.concatenate(test_preds_list)
        test_targets_arr = np.concatenate(test_targets_list)
        all_preds[test_idx] = test_preds_arr
        all_targets[test_idx] = test_targets_arr

        # Exact match (within 1% or ±0.5)
        exact = np.sum(np.abs(test_preds_arr - test_targets_arr) < np.maximum(np.abs(test_targets_arr) * 0.01, 0.5))
        fold_exact = exact / len(test_targets_arr)
        r2 = 1 - np.sum((test_preds_arr - test_targets_arr)**2) / (np.sum((test_targets_arr - test_targets_arr.mean())**2) + 1e-8)
        print(f"  Fold {fold+1}: exact_match={fold_exact:.4f}, R²={r2:.4f}")

    return all_preds, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--latent_model_path', type=str, default='../../Latent-SFT/checkpoints/latent-4')
    parser.add_argument('--test_data', type=str, default='../../Latent-SFT/data/GSM8k-Aug-test.jsonl')
    parser.add_argument('--output_dir', type=str, default='./ablation_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.latent_model_path)
    config = AutoConfig.from_pretrained(args.latent_model_path)
    hidden_dim = config.hidden_size
    vocab_size = config.vocab_size

    print(f"Hidden dim: {hidden_dim}, Vocab size: {vocab_size}")

    # Load probe dataset
    dataset = ProbeDataset(args.trajectory_file, tokenizer, args.test_data)
    print(f"Loaded {len(dataset)} samples")

    # Verify data
    sample = dataset[0]
    print(f"Sample: last_hidden shape={sample['last_hidden'].shape}, answer_token={sample['answer_token_id'].item()}, value={sample['answer_value'].item()}")

    # ============================================================
    # Probe 1: Answer token prediction
    # ============================================================
    print(f"\n{'='*60}")
    print("PROBE 1: Answer Token Prediction from last-step h_T")
    print(f"{'='*60}")

    token_preds, token_targets, top5_correct = train_answer_probe(
        dataset, device, hidden_dim, vocab_size, epochs=50
    )

    token_acc = (token_preds == token_targets).mean()
    top5_acc = top5_correct.mean()
    print(f"\n  Overall: token_acc={token_acc:.4f}, top5_acc={top5_acc:.4f}")

    # ============================================================
    # Probe 2: Numeric value prediction
    # ============================================================
    print(f"\n{'='*60}")
    print("PROBE 2: Numeric Answer Value Prediction from last-step h_T")
    print(f"{'='*60}")

    value_preds, value_targets = train_numeric_probe(dataset, device, hidden_dim, epochs=50)

    # Metrics
    exact_match = np.mean(np.abs(value_preds - value_targets) < np.maximum(np.abs(value_targets) * 0.01, 0.5))
    within_10pct = np.mean(np.abs(value_preds - value_targets) < np.maximum(np.abs(value_targets) * 0.1, 1.0))
    r2 = 1 - np.sum((value_preds - value_targets)**2) / (np.sum((value_targets - value_targets.mean())**2) + 1e-8)
    print(f"\n  Overall: exact_match={exact_match:.4f}, within_10%={within_10pct:.4f}, R²={r2:.4f}")

    # ============================================================
    # Analysis: Confidence head vs answer probe correlation
    # ============================================================
    print(f"\n{'='*60}")
    print("ANALYSIS: Confidence head behavior conditioned on probe accuracy")
    print(f"{'='*60}")

    # Load trained confidence head
    head_path = os.path.join(os.path.dirname(args.output_dir), 'trained_heads', 'confidence_head_attention.pt')
    if os.path.exists(head_path):
        ckpt = torch.load(head_path, map_location=device, weights_only=False)
        head = AttentionConfidenceHead(hidden_dim=hidden_dim, proj_dim=128, n_heads=4, dropout=0.0).to(device)
        head.load_state_dict(ckpt['model_state_dict'])
        head.eval()

        # Get confidence scores for all samples
        traj_dataset = TrajectoryDataset(args.trajectory_file, max_T=64)
        loader = DataLoader(traj_dataset, batch_size=64, shuffle=False, num_workers=0)
        all_conf = []
        with torch.no_grad():
            for batch in loader:
                logits = head(batch['hidden_states'].to(device), batch['mask'].to(device), batch['T'].to(device))
                all_conf.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_conf = np.array(all_conf)

        correctness = np.array([dataset[i]['correct'].item() for i in range(len(dataset))])

        # Split by whether answer probe got the token right
        probe_correct_mask = (token_preds == token_targets)
        probe_wrong_mask = ~probe_correct_mask

        if probe_correct_mask.sum() > 10 and probe_wrong_mask.sum() > 10:
            # AUC on samples where probe is correct (head knows answer)
            if len(np.unique(correctness[probe_correct_mask])) >= 2:
                auc_probe_correct = roc_auc_score(correctness[probe_correct_mask], all_conf[probe_correct_mask])
            else:
                auc_probe_correct = float('nan')

            # AUC on samples where probe is wrong (head doesn't know answer)
            if len(np.unique(correctness[probe_wrong_mask])) >= 2:
                auc_probe_wrong = roc_auc_score(correctness[probe_wrong_mask], all_conf[probe_wrong_mask])
            else:
                auc_probe_wrong = float('nan')

            print(f"  Samples where answer probe CORRECT: n={probe_correct_mask.sum()}")
            print(f"    Confidence head AUC on these: {auc_probe_correct:.4f}")
            print(f"  Samples where answer probe WRONG: n={probe_wrong_mask.sum()}")
            print(f"    Confidence head AUC on these: {auc_probe_wrong:.4f}")
            print()
            if auc_probe_wrong > 0.70:
                print("  ✓ Confidence head works EVEN when it can't predict the answer")
                print("    → Confidence signal is NOT purely answer reconstruction")
            else:
                print("  ⚠️ Confidence head relies heavily on answer-predictable samples")
        else:
            auc_probe_correct = float('nan')
            auc_probe_wrong = float('nan')
            print("  Insufficient samples in one group for meaningful split")
    else:
        auc_probe_correct = float('nan')
        auc_probe_wrong = float('nan')
        print(f"  No trained head found at {head_path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("RISK C SUMMARY")
    print(f"{'='*60}")
    print(f"  Answer token probe accuracy: {token_acc:.4f} (top-5: {top5_acc:.4f})")
    print(f"  Numeric value exact match: {exact_match:.4f} (within 10%: {within_10pct:.4f})")
    print(f"  Confidence AUC when probe correct: {auc_probe_correct:.4f}")
    print(f"  Confidence AUC when probe wrong:   {auc_probe_wrong:.4f}")
    print()

    if token_acc >= 0.80:
        print("  INTERPRETATION: h_T strongly encodes the answer.")
        print("  But this is expected — the model IS about to generate the answer.")
        print("  Key question: does confidence head work beyond answer identity?")
    elif token_acc >= 0.30:
        print("  INTERPRETATION: h_T partially encodes the answer.")
        print("  Confidence head may use answer identity as one of several signals.")
    else:
        print("  INTERPRETATION: h_T does NOT directly encode the answer.")
        print("  Confidence head learns abstract confidence, not answer reconstruction.")

    # Save results
    results = {
        "answer_token_probe": {
            "accuracy": float(token_acc),
            "top5_accuracy": float(top5_acc),
        },
        "numeric_value_probe": {
            "exact_match": float(exact_match),
            "within_10pct": float(within_10pct),
            "r2": float(r2),
        },
        "conditional_analysis": {
            "confidence_auc_when_probe_correct": float(auc_probe_correct),
            "confidence_auc_when_probe_wrong": float(auc_probe_wrong),
            "n_probe_correct": int(probe_correct_mask.sum()) if 'probe_correct_mask' in dir() else 0,
            "n_probe_wrong": int(probe_wrong_mask.sum()) if 'probe_wrong_mask' in dir() else 0,
        }
    }
    output_file = os.path.join(args.output_dir, "risk_c_probe_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
