"""
Prompt-Only Baseline for CoLaR.
Extract hidden state at the last prompt token (before latent reasoning begins)
and train a probe to predict correctness.

Compares:
  (A) Prompt-only: last hidden state at end of prompt (before latent z_1)
  (B) Full trajectory: current CoLaR main result (0.853 ± 0.004)

Protocol: Same nested CV splits as main result, 4 runs (different seeds).

CoLaR prompt format:
  "Question: {question} Let's think step by step:(Thinking speed: 5)###"
  → last token hidden state is the state right before the first latent forward step.

Model: Llama-3.2-1B-Instruct + LoRA (from CoLaR checkpoint)
"""
import os, sys, json, torch, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Paths
COLAR_CKPT = str(project_root / 'logs/colar/qsa-gsm/20260428-195821_350334_full-cotsft-baseline/checkpoints/last.ckpt')
LLM_PATH = str(project_root / 'models/llms/Llama-3.2-1B-Instruct')
DATA_PATH = str(project_root / 'datasets/text_reasoning/gsm/test.json')
TRAJ_PATH = str(project_root / 'experiments/confidence_head/trajectory_data_colar/trajectories_GSM8k_colar.pt')
OUTPUT_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda:0'
NUM_RUNS = 4
N_FOLDS = 5
HIDDEN_DIM = 2048
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5
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


def load_colar_model():
    """Load CoLaR backbone (LLM + LoRA) from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"Loading CoLaR checkpoint: {COLAR_CKPT}")
    ckpt = torch.load(COLAR_CKPT, map_location='cpu', weights_only=False)
    hparams = ckpt['hyper_parameters']
    model_kwargs = hparams['model_kwargs']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load base LLM
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    )
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    lora_cfg = model_kwargs['lora_config']
    lora_config = LoraConfig(r=lora_cfg['r'], lora_alpha=lora_cfg['lora_alpha'])
    model = get_peft_model(model, peft_config=lora_config)

    # Load LLM state dict from checkpoint
    state_dict = ckpt['state_dict']
    llm_state = {}
    for k, v in state_dict.items():
        if k.startswith('llm.'):
            llm_state[k.replace('llm.', '')] = v

    model.load_state_dict(llm_state, strict=False)
    model = model.to(DEVICE)
    model.eval()

    print(f"  Loaded model with LoRA (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})")
    return model, tokenizer


def extract_prompt_hidden_states():
    """Extract hidden state at end of CoLaR prompt for all problems."""
    cache_path = OUTPUT_DIR / 'prompt_hidden_states_colar.pt'
    if cache_path.exists():
        print(f"Loading cached prompt hidden states from {cache_path}")
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    print("Extracting prompt hidden states for CoLaR...")
    model, tokenizer = load_colar_model()

    # Load test data
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test samples")

    # Load trajectory data for correctness labels
    traj_data = torch.load(TRAJ_PATH, map_location='cpu', weights_only=False)
    assert len(traj_data) == len(data), f"Mismatch: {len(traj_data)} vs {len(data)}"

    # CoLaR prompt format (from collect_trajectories_colar.py)
    question_template = "Question: {} Let's think step by step:"
    speed_template = "(Thinking speed: {})"
    compression_factor = 5

    prompt_hiddens = []
    labels = []

    for i, (example, traj) in enumerate(tqdm(zip(data, traj_data), total=len(data), desc="Extracting")):
        question = example['question']
        # Build the full prompt that CoLaR sees before latent reasoning
        suffix = speed_template.format(compression_factor) + "###"
        text = question_template.format(question) + suffix

        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, padding=False)
        input_ids = inputs['input_ids'].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            # Last layer, last token (the ### separator - right before latent reasoning)
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()

        prompt_hiddens.append(last_hidden)
        labels.append(int(traj['correct']))

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(data)}] done")

    result = {
        'hidden_states': torch.stack(prompt_hiddens),  # [N, 2048]
        'labels': torch.tensor(labels),  # [N]
    }
    torch.save(result, cache_path)
    print(f"Saved {len(labels)} prompt hidden states to {cache_path}")
    print(f"  Correct: {sum(labels)}/{len(labels)} = {sum(labels)/len(labels):.4f}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


def train_probe_nested_cv(hidden_states, labels, seed=42):
    """Train MLP probe with nested CV, return test AUCs per fold."""
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

        probe = MLPProbe(HIDDEN_DIM, 256).to(DEVICE)
        optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0
        patience_count = 0
        best_state = None

        for epoch in range(MAX_EPOCHS):
            probe.train()
            perm = torch.randperm(len(X_train))
            for start in range(0, len(X_train), BATCH_SIZE):
                batch_idx = perm[start:start+BATCH_SIZE]
                logits = probe(X_train[batch_idx])
                loss = criterion(logits, y_train[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_auc = roc_auc_score(y_val.cpu().numpy(), val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_count = 0
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    break

        # Test
        probe.load_state_dict(best_state)
        probe.eval()
        with torch.no_grad():
            test_logits = probe(X_test)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_auc = roc_auc_score(y_test, test_probs)
        fold_aucs.append(test_auc)

    return np.mean(fold_aucs)


def main():
    print("=" * 60)
    print("Prompt-Only Baseline for CoLaR")
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
    print(f"PROMPT-ONLY BASELINE (CoLaR)")
    print(f"  Model: Llama-3.2-1B-Instruct + LoRA (CoLaR checkpoint)")
    print(f"  Prompt: 'Question: {{q}} Let's think step by step:(Thinking speed: 5)###'")
    print(f"  AUC = {mean_auc:.3f} +/- {std_auc:.3f}")
    print(f"  Full trajectory (CoLaR main result): 0.853 +/- 0.004")
    print(f"  Gap (full - prompt): {0.853 - mean_auc:.3f}")
    print(f"{'='*60}")

    # Comparison with LSFT
    print(f"\n  LSFT prompt-only:  0.769 +/- 0.007")
    print(f"  CoLaR prompt-only: {mean_auc:.3f} +/- {std_auc:.3f}")

    # Save results
    results = {
        'prompt_only': {
            'mean': float(mean_auc),
            'std': float(std_auc),
            'runs': [float(x) for x in run_aucs]
        },
        'full_trajectory_reference': {'mean': 0.853, 'std': 0.004},
        'gap': float(0.853 - mean_auc),
        'lsft_comparison': {
            'lsft_prompt_only': {'mean': 0.769, 'std': 0.007},
            'lsft_full_trajectory': {'mean': 0.805, 'std': 0.011},
        },
        'protocol': 'MLP probe on last prompt token hidden state (before latent reasoning), nested 5-fold CV, 4 runs',
        'model': 'Llama-3.2-1B-Instruct + LoRA (CoLaR checkpoint)',
        'prompt_format': "Question: {q} Let's think step by step:(Thinking speed: 5)###",
        'hyperparams': {
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
            'patience': PATIENCE,
            'hidden_dim': 256,
            'n_folds': N_FOLDS,
            'max_epochs': MAX_EPOCHS,
        }
    }

    out_path = OUTPUT_DIR / 'prompt_only_colar_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
