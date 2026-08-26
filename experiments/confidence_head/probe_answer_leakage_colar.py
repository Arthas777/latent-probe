"""
Risk C probe for CoLaR: Does the last-step hidden state encode the answer itself?

Adapted from probe_answer_leakage.py for CoLaR trajectories.
Uses Llama-3.2-1B tokenizer and CoLaR trajectory data.

Usage:
    python probe_answer_leakage_colar.py \
        --trajectory_file ./trajectory_data_colar/trajectories_GSM8k_colar.pt \
        --output_dir ./ablation_results_colar \
        --device cuda:0
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(__file__))
from train_confidence_head import (
    TrajectoryDataset, AttentionConfidenceHead, roc_auc_score
)


BASE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '../../models/llms/LLM-Research/Llama-3___2-1B-Instruct'
)
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '../../Latent-SFT/data/GSM8k-Aug-test.jsonl'
)


class AnswerProbe(nn.Module):
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
            else:
                test_data = [json.loads(l) for l in f]
            self.answers = [str(d['answer']) for d in test_data]

        self.tokenizer = tokenizer

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
        hs = item['hidden_states']

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

        exact = np.sum(np.abs(test_preds_arr - test_targets_arr) < np.maximum(np.abs(test_targets_arr) * 0.01, 0.5))
        fold_exact = exact / len(test_targets_arr)
        r2 = 1 - np.sum((test_preds_arr - test_targets_arr)**2) / (np.sum((test_targets_arr - test_targets_arr.mean())**2) + 1e-8)
        print(f"  Fold {fold+1}: exact_match={fold_exact:.4f}, R²={r2:.4f}")

    return all_preds, all_targets


def train_confidence_head_nested_cv(trajectory_file, device, hidden_dim, seed=42, n_folds=5):
    """Train confidence head with nested CV and return per-sample predictions."""
    dataset = TrajectoryDataset(trajectory_file, max_T=64)
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
        print(f"  Confidence head fold {fold+1}: AUC={fold_auc:.4f}")

    overall_auc = roc_auc_score(all_labels, all_probs)
    print(f"  Overall confidence head AUC (nested CV): {overall_auc:.4f}")
    return all_probs, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=BASE_MODEL_PATH)
    parser.add_argument('--test_data', type=str, default=TEST_DATA_PATH)
    parser.add_argument('--output_dir', type=str, default='./ablation_results_colar')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    hidden_dim = config.hidden_size
    vocab_size = config.vocab_size

    print(f"Hidden dim: {hidden_dim}, Vocab size: {vocab_size}")

    dataset = ProbeDataset(args.trajectory_file, tokenizer, args.test_data)
    print(f"Loaded {len(dataset)} samples")

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

    exact_match = np.mean(np.abs(value_preds - value_targets) < np.maximum(np.abs(value_targets) * 0.01, 0.5))
    within_10pct = np.mean(np.abs(value_preds - value_targets) < np.maximum(np.abs(value_targets) * 0.1, 1.0))
    r2 = 1 - np.sum((value_preds - value_targets)**2) / (np.sum((value_targets - value_targets.mean())**2) + 1e-8)
    print(f"\n  Overall: exact_match={exact_match:.4f}, within_10%={within_10pct:.4f}, R²={r2:.4f}")

    # ============================================================
    # Conditional analysis: confidence head vs answer probe
    # ============================================================
    print(f"\n{'='*60}")
    print("ANALYSIS: Train confidence head (nested CV) for conditional analysis")
    print(f"{'='*60}")

    conf_probs, conf_labels = train_confidence_head_nested_cv(
        args.trajectory_file, device, hidden_dim, seed=42
    )

    correctness = conf_labels
    probe_correct_mask = (token_preds == token_targets)
    probe_wrong_mask = ~probe_correct_mask

    if probe_correct_mask.sum() > 10 and probe_wrong_mask.sum() > 10:
        if len(np.unique(correctness[probe_correct_mask])) >= 2:
            auc_probe_correct = roc_auc_score(correctness[probe_correct_mask], conf_probs[probe_correct_mask])
        else:
            auc_probe_correct = float('nan')

        if len(np.unique(correctness[probe_wrong_mask])) >= 2:
            auc_probe_wrong = roc_auc_score(correctness[probe_wrong_mask], conf_probs[probe_wrong_mask])
        else:
            auc_probe_wrong = float('nan')

        print(f"  Samples where answer probe CORRECT: n={probe_correct_mask.sum()}")
        print(f"    Confidence head AUC on these: {auc_probe_correct:.4f}")
        print(f"  Samples where answer probe WRONG: n={probe_wrong_mask.sum()}")
        print(f"    Confidence head AUC on these: {auc_probe_wrong:.4f}")
        print()
        if auc_probe_wrong > 0.70:
            print("  => Confidence head works EVEN when it can't predict the answer")
            print("     Confidence signal is NOT purely answer reconstruction")
        else:
            print("  => Confidence head relies heavily on answer-predictable samples")
    else:
        auc_probe_correct = float('nan')
        auc_probe_wrong = float('nan')
        print("  Insufficient samples in one group for meaningful split")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("RISK C SUMMARY (CoLaR)")
    print(f"{'='*60}")
    print(f"  Answer token probe accuracy: {token_acc:.4f} (top-5: {top5_acc:.4f})")
    print(f"  Numeric value exact match: {exact_match:.4f} (within 10%: {within_10pct:.4f}, R²: {r2:.4f})")
    print(f"  Confidence AUC when probe correct: {auc_probe_correct:.4f}")
    print(f"  Confidence AUC when probe wrong:   {auc_probe_wrong:.4f}")

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
            "n_probe_correct": int(probe_correct_mask.sum()),
            "n_probe_wrong": int(probe_wrong_mask.sum()),
        }
    }
    output_file = os.path.join(args.output_dir, "risk_c_probe_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
