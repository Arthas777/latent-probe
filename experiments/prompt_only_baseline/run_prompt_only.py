"""
Prompt-Only Baseline (Appendix, Table 8).
Extract hidden state at the <think> token (before latent reasoning begins)
and train a probe to predict correctness.

Compares:
  (A) Prompt-only: last hidden state at <think> token
  (B) Full trajectory: current main result (0.805 ± 0.011)

Protocol: Same nested CV splits as §5 main result, 6 runs.
"""
import os, sys, json, torch, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
lsft_root = project_root / 'Latent-SFT'
sys.path.insert(0, str(lsft_root))

LATENT4_PATH = str(lsft_root / 'checkpoints/latent-4')
DATA_PATH = str(lsft_root / 'data/GSM8k-Aug-test.jsonl')
TRAJ_PATH = str(project_root / 'experiments/confidence_head/trajectory_data/trajectories_GSM8k.pt')
OUTPUT_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda:0'
NUM_RUNS = 6
N_FOLDS = 5
HIDDEN_DIM = 2048
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-4
BATCH_SIZE = 64


class MLPProbe(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_prompt_hidden_states():
    """Extract hidden state at <think> token for all problems."""
    cache_path = OUTPUT_DIR / 'prompt_hidden_states.pt'
    if cache_path.exists():
        print(f"Loading cached prompt hidden states from {cache_path}")
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    print("Extracting prompt hidden states...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(LATENT4_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LATENT4_PATH, attn_implementation='sdpa',
        torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    with open(DATA_PATH) as f:
        problems = [json.loads(line) for line in f]

    # Load trajectory data for correctness labels
    traj_data = torch.load(TRAJ_PATH, map_location='cpu', weights_only=False)
    assert len(traj_data) == len(problems), f"Mismatch: {len(traj_data)} vs {len(problems)}"

    prompt_hiddens = []
    labels = []

    for i, (prob, traj) in enumerate(tqdm(zip(problems, traj_data), total=len(problems))):
        question = prob['question'] if 'question' in prob else prob.get('problem', '')
        # Use exact same prompt format as routing script
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{question}<|eot_id|>"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        input_ids = tokenizer.encode(input_prefix, return_tensors='pt', add_special_tokens=False).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            # Last layer, last token (the <think> token)
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()

        prompt_hiddens.append(last_hidden)
        labels.append(int(traj['correct']))

    result = {
        'hidden_states': torch.stack(prompt_hiddens),  # [N, 2048]
        'labels': torch.tensor(labels),  # [N]
    }
    torch.save(result, cache_path)
    print(f"Saved {len(labels)} prompt hidden states to {cache_path}")
    return result


def train_probe_nested_cv(hidden_states, labels, seed=42):
    """Train MLP probe with nested CV, return test AUCs."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    N = len(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_aucs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), labels.numpy())):
        # Inner split for early stopping
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + fold_idx)
        inner_train_idx, val_idx = next(inner_skf.split(train_idx, labels[train_idx].numpy()))
        inner_train_idx = train_idx[inner_train_idx]
        val_idx = train_idx[val_idx]

        X_train = hidden_states[inner_train_idx].to(DEVICE)
        y_train = labels[inner_train_idx].float().to(DEVICE)
        X_val = hidden_states[val_idx].to(DEVICE)
        y_val = labels[val_idx].float().to(DEVICE)
        X_test = hidden_states[test_idx].to(DEVICE)
        y_test = labels[test_idx].numpy()

        model = MLPProbe(HIDDEN_DIM, 256).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0
        patience_count = 0
        best_state = None

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(len(X_train))
            for start in range(0, len(X_train), BATCH_SIZE):
                batch_idx = perm[start:start+BATCH_SIZE]
                logits = model(X_train[batch_idx])
                loss = criterion(logits, y_train[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_auc = roc_auc_score(y_val.cpu().numpy(), val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    break

        # Test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_auc = roc_auc_score(y_test, test_probs)
        fold_aucs.append(test_auc)

    return np.mean(fold_aucs)


def main():
    print("=" * 60)
    print("Prompt-Only Baseline")
    print("=" * 60)

    # Step 1: Extract prompt hidden states
    data = extract_prompt_hidden_states()
    hidden_states = data['hidden_states']
    labels = data['labels']
    print(f"\nData: {len(labels)} problems, {labels.sum().item()} correct ({labels.float().mean()*100:.1f}%)")

    # Step 2: Run multiple nested CV runs
    print(f"\nRunning {NUM_RUNS} nested CV runs...")
    run_aucs = []
    for run in range(NUM_RUNS):
        seed = 42 + run * 7
        auc = train_probe_nested_cv(hidden_states, labels, seed=seed)
        run_aucs.append(auc)
        print(f"  Run {run+1}/{NUM_RUNS}: AUC = {auc:.4f}")

    mean_auc = np.mean(run_aucs)
    std_auc = np.std(run_aucs)
    print(f"\n{'='*60}")
    print(f"PROMPT-ONLY BASELINE (MLP on <think> token hidden state)")
    print(f"  AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  Full trajectory (§5 main result): 0.805 ± 0.011")
    print(f"  Gap (full - prompt): {0.805 - mean_auc:.3f}")
    print(f"{'='*60}")

    # Also compute: last latent step only (for comparison)
    traj_data = torch.load(TRAJ_PATH, map_location='cpu', weights_only=False)
    last_step_hiddens = torch.stack([d['hidden_states'][-1] for d in traj_data])  # [N, 2048]

    print(f"\nRunning last-latent-step probe for comparison...")
    last_step_aucs = []
    for run in range(NUM_RUNS):
        seed = 42 + run * 7
        auc = train_probe_nested_cv(last_step_hiddens, labels, seed=seed)
        last_step_aucs.append(auc)
        print(f"  Run {run+1}/{NUM_RUNS}: AUC = {auc:.4f}")

    last_mean = np.mean(last_step_aucs)
    last_std = np.std(last_step_aucs)

    results = {
        'prompt_only': {'mean': float(mean_auc), 'std': float(std_auc), 'runs': [float(x) for x in run_aucs]},
        'last_latent_step': {'mean': float(last_mean), 'std': float(last_std), 'runs': [float(x) for x in last_step_aucs]},
        'full_trajectory_reference': {'mean': 0.805, 'std': 0.011},
        'protocol': 'MLP probe, nested 5-fold CV, 6 runs, same problem-level splits'
    }

    out_path = OUTPUT_DIR / 'prompt_only_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
