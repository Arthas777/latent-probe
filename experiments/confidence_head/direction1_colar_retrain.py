"""
Direction 1: Retrain confidence head on mixed data (50% deterministic + 50% σ=1000 stochastic)
then evaluate as selector on σ=1000 diverse rollouts.

Pipeline:
1. Collect deterministic trajectories (1319 problems × 1)
2. Collect σ=1000 stochastic trajectories (1319 problems × 4 rollouts → pick 1319 to balance)
3. Train attention head on mixed data (50/50 split)
4. Evaluate: run fresh σ=1000 N=8 rollouts, use retrained head as selector
"""
import os, sys, json, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.modules.projector import LatentPolicy
from src.utils.constants import MODEL_EMB_STD
from src.utils.utils import get_position_ids_from_attention_mask
from train_confidence_head import AttentionConfidenceHead, TrajectoryDataset, train, evaluate, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn


class CoLaRModel:
    def __init__(self, ckpt_path, device='cuda:1'):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        hparams = ckpt['hyper_parameters']
        model_kwargs = hparams['model_kwargs']

        model_id = model_kwargs['model_id']
        llm_path = os.path.join(project_root, 'models', 'llms', model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.llm = AutoModelForCausalLM.from_pretrained(llm_path).to(self.device)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.embedding = self.llm.get_input_embeddings()

        lp_config = model_kwargs['latent_policy_config']
        hidden_size = self.llm.config.hidden_size
        self.latent_policy = LatentPolicy(
            feature_size=hidden_size,
            intermediate_size=lp_config.get('lp_intermediate_size', hidden_size),
            deterministic=lp_config.get('lp_determinisitc', False),
        ).to(self.device)

        self.embeds_std = MODEL_EMB_STD[model_id]
        self.thinking_separator = "###"
        self.thinking_separator_id = self.tokenizer.convert_tokens_to_ids(self.thinking_separator)
        self.hidden_size = hidden_size

        self.latent_gen_config = model_kwargs['latent_generation_config']
        self.answer_gen_config = model_kwargs['answer_generation_config']
        self.question_template = "Question: {} Let's think step by step:"
        self.speed_template = "(Thinking speed: {})"

        state_dict = ckpt['state_dict']
        llm_state = {}
        lp_state = {}
        for k, v in state_dict.items():
            if k.startswith('latent_policy.'):
                lp_state[k.replace('latent_policy.', '')] = v
            elif k.startswith('llm.'):
                llm_state[k.replace('llm.', '')] = v

        if model_kwargs.get('do_lora', False):
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(**model_kwargs['lora_config'])
            self.llm = get_peft_model(self.llm, peft_config=lora_config)
            self.llm.load_state_dict(llm_state, strict=False)
            self.embedding = self.llm.get_input_embeddings()

        self.latent_policy.load_state_dict(lp_state)
        self.llm.eval()
        self.latent_policy.eval()
        print(f"CoLaR loaded. hidden_size={hidden_size}, embeds_std={self.embeds_std}")

    @torch.no_grad()
    def generate_rollout(self, question, sigma_scale=1.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        speed = self.latent_gen_config['compression_factor']
        max_latent_steps = self.latent_gen_config['max_n_latent_forward']
        suffix = self.speed_template.format(speed) + self.thinking_separator

        text = self.question_template.format(question) + suffix
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, padding=False)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        position_ids = get_position_ids_from_attention_mask(attention_mask)
        question_embeds = self.embedding(input_ids)

        outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        all_attention_mask = attention_mask
        current_position_ids = position_ids[:, -1:]
        past_key_values = outputs.past_key_values

        hidden_states_list = []
        latent_embeds_list = []

        latent_temperature = self.latent_gen_config.get('latent_temperature', 1.0)

        for step in range(max_latent_steps):
            h = outputs.hidden_states[-1][:, -1:, :]
            hidden_states_list.append(h.squeeze(0).squeeze(0).cpu())

            distribution = self.latent_policy.forward(h, temperature=latent_temperature)
            mean = distribution.mean
            std = distribution.stddev * sigma_scale
            noise = torch.randn_like(mean)
            z_t = (mean + std * noise) * self.embeds_std

            latent_embeds_list.append(z_t.squeeze(0).squeeze(0).cpu())

            all_attention_mask = torch.cat([
                all_attention_mask,
                torch.ones(1, 1, device=self.device, dtype=torch.long),
            ], dim=1)
            current_position_ids = current_position_ids + 1

            outputs = self.llm.forward(
                inputs_embeds=z_t,
                attention_mask=all_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values

            last_logits = outputs.logits[:, -1]
            eol_temperature = self.latent_gen_config.get('eol_temperature', 1.0)
            probs = torch.softmax(last_logits / eol_temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == self.thinking_separator_id:
                break

        T = len(hidden_states_list)

        # Answer generation
        end_of_thinking_ids = torch.tensor([[self.thinking_separator_id]], device=self.device)
        end_of_thinking_embeds = self.embedding(end_of_thinking_ids)

        all_embeds = torch.cat(
            [question_embeds] +
            [e.unsqueeze(0).unsqueeze(0).to(self.device) for e in latent_embeds_list] +
            [end_of_thinking_embeds], dim=1
        )
        full_attention_mask = torch.cat([
            all_attention_mask,
            torch.ones(1, 1, device=self.device, dtype=torch.long),
        ], dim=1)

        pred_ids = self.llm.generate(
            inputs_embeds=all_embeds,
            attention_mask=full_attention_mask,
            **self.answer_gen_config
        )
        decoded = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

        return {
            "text": decoded,
            "hidden_states": torch.stack(hidden_states_list) if hidden_states_list else torch.zeros(0, self.hidden_size),
            "T": T,
        }


def extract_answer_colar(text, gt_str):
    pred_str = text.strip('#\n ').split('Answer:')[-1] if 'Answer:' in text else text.strip('#\n ')
    pred_clean = pred_str.strip().rstrip('.').replace(',', '').lower()
    gt_clean = gt_str.strip().rstrip('.').replace(',', '').lower()
    try:
        return abs(float(pred_clean) - float(gt_clean)) < 1e-4
    except (ValueError, TypeError):
        return pred_clean == gt_clean


class MixedTrajectoryDataset(Dataset):
    """Dataset that holds mixed deterministic + stochastic trajectories."""
    def __init__(self, data_list, max_T=64):
        self.data = data_list
        self.max_T = max_T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        h = item['hidden_states']  # [T, hidden_dim]
        T = item['T']
        correct = item['correct']

        if h.shape[0] > self.max_T:
            h = h[-self.max_T:]
            T = self.max_T

        hidden_dim = h.shape[-1]
        padded = torch.zeros(self.max_T, hidden_dim)
        padded[:h.shape[0]] = h
        mask = torch.zeros(self.max_T, dtype=torch.bool)
        mask[:h.shape[0]] = True

        return {
            'hidden_states': padded,
            'mask': mask,
            'T': torch.tensor(float(T)),
            'label': torch.tensor(float(correct)),
        }


def head_predict(head, hidden_states, T, max_T=64, device='cuda:1'):
    h = hidden_states.to(device).float()
    if h.shape[0] > max_T:
        h = h[-max_T:]
    T_actual = h.shape[0]
    if T_actual < max_T:
        pad = torch.zeros(max_T - T_actual, h.shape[1], device=device)
        h = torch.cat([h, pad], dim=0)
    mask = torch.zeros(max_T, dtype=torch.bool, device=device)
    mask[:T_actual] = True
    h = h.unsqueeze(0)
    mask = mask.unsqueeze(0)
    T_tensor = torch.tensor([float(T_actual)], device=device)
    logit = head(h, mask, T_tensor)
    return torch.sigmoid(logit).item()


def train_head_on_dataset(dataset, hidden_dim, device, n_folds=5, epochs=40, seed=42):
    """Train attention head with 5-fold CV, return best model and CV AUC."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    all_probs = np.zeros(n)
    all_labels = np.zeros(n)
    best_model_state = None
    best_auc = 0

    class Args:
        batch_size = 64
        epochs = epochs
        lr = 1e-3
        weight_decay = 0.01

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False, num_workers=0)

        model = AttentionConfidenceHead(hidden_dim=hidden_dim, proj_dim=128, n_heads=4, dropout=0.1).to(device)
        _, fold_auc = train(model, train_loader, val_loader, torch.device(device), Args(), model_type='attention')

        metrics = evaluate(model, val_loader, torch.device(device), 'attention')
        all_probs[val_idx] = metrics['probs']
        all_labels[val_idx] = metrics['labels']

        if fold_auc > best_auc:
            best_auc = fold_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Fold {fold+1}/{n_folds}: val_auc={metrics['auc']:.4f}")

    cv_auc = roc_auc_score(all_labels, all_probs)
    print(f"  CV AUC: {cv_auc:.4f}")
    return best_model_state, cv_auc


def main():
    device = 'cuda:1'
    sigma_scale = 1000.0
    N = 8

    ckpt_path = os.path.join(project_root, 'logs/colar/qsa-gsm/20260428-195821_350334_full-cotsft-baseline/checkpoints/epoch44__step67815__monitor0.312.ckpt')
    print("Loading CoLaR model...")
    colar = CoLaRModel(ckpt_path, device=device)

    # Load test data
    test_path = os.path.join(project_root, 'datasets', 'text_reasoning', 'gsm', 'test.json')
    with open(test_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems")

    # =========================================================
    # Phase 1: Collect training data (deterministic + σ=1000)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Collecting training trajectories")
    print("=" * 70)

    train_data = []

    # 1a: Deterministic trajectories (σ=1, seed=0)
    print("\nCollecting deterministic trajectories...")
    for pi, example in enumerate(tqdm(data, desc="Deterministic")):
        out = colar.generate_rollout(example['question'], sigma_scale=1.0, seed=0)
        correct = extract_answer_colar(out['text'], str(example['answer']))
        train_data.append({
            'hidden_states': out['hidden_states'],
            'T': out['T'],
            'correct': bool(correct),
            'source': 'deterministic',
        })

    det_acc = np.mean([d['correct'] for d in train_data])
    print(f"  Deterministic acc: {det_acc:.4f}, n={len(train_data)}")

    # 1b: Stochastic trajectories (σ=1000, different seeds)
    print("\nCollecting σ=1000 stochastic trajectories...")
    stochastic_data = []
    for pi, example in enumerate(tqdm(data, desc="Stochastic")):
        seed = 99999 + pi  # different from eval seeds
        out = colar.generate_rollout(example['question'], sigma_scale=sigma_scale, seed=seed)
        correct = extract_answer_colar(out['text'], str(example['answer']))
        stochastic_data.append({
            'hidden_states': out['hidden_states'],
            'T': out['T'],
            'correct': bool(correct),
            'source': 'stochastic',
        })

    stoch_acc = np.mean([d['correct'] for d in stochastic_data])
    print(f"  Stochastic acc: {stoch_acc:.4f}, n={len(stochastic_data)}")

    # Combine 50/50
    train_data.extend(stochastic_data)
    print(f"\nTotal training data: {len(train_data)} (50% det + 50% stoch)")
    print(f"  Overall label balance: {np.mean([d['correct'] for d in train_data]):.4f}")

    # =========================================================
    # Phase 2: Train confidence head on mixed data
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Training confidence head on mixed data")
    print("=" * 70)

    dataset = MixedTrajectoryDataset(train_data, max_T=64)
    best_state, cv_auc = train_head_on_dataset(
        dataset, hidden_dim=colar.hidden_size, device=device, n_folds=5, epochs=40
    )

    # Load best model
    head = AttentionConfidenceHead(hidden_dim=colar.hidden_size, proj_dim=128, n_heads=4, dropout=0.1)
    head.load_state_dict(best_state)
    head = head.to(device).eval()
    print(f"\nRetrained head CV AUC: {cv_auc:.4f}")

    # Save retrained head
    os.makedirs('./direction1_results', exist_ok=True)
    torch.save({'model_state_dict': best_state, 'cv_auc': cv_auc},
               './direction1_results/colar_head_mixed_sigma1000.pt')

    # =========================================================
    # Phase 3: Evaluate on fresh σ=1000 rollouts
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Evaluating retrained head as selector (σ=1000, N=8)")
    print("=" * 70)

    results = []
    for pi, example in enumerate(tqdm(data, desc="Eval rollouts")):
        gt_answer = str(example['answer']).strip()

        # Deterministic baseline
        det_out = colar.generate_rollout(example['question'], sigma_scale=1.0, seed=0)
        det_correct = extract_answer_colar(det_out['text'], gt_answer)
        det_conf = head_predict(head, det_out['hidden_states'], det_out['T'], device=device)

        # N fresh rollouts (different seeds from training!)
        rollouts = []
        for n in range(N):
            seed = 200000 + n * 10000 + pi  # distinct from training seeds
            out = colar.generate_rollout(example['question'], sigma_scale=sigma_scale, seed=seed)
            correct = extract_answer_colar(out['text'], gt_answer)
            conf = head_predict(head, out['hidden_states'], out['T'], device=device)
            rollouts.append({
                "correct": bool(correct), "T": out['T'], "confidence": conf,
            })

        results.append({
            "idx": pi,
            "det_correct": bool(det_correct),
            "det_conf": det_conf,
            "rollouts": rollouts,
        })

        if (pi + 1) % 100 == 0:
            det_so_far = np.mean([r["det_correct"] for r in results])
            oracle_so_far = np.mean([any(ro["correct"] for ro in r["rollouts"]) for r in results])
            head_so_far = np.mean([r["rollouts"][max(range(N), key=lambda k: r["rollouts"][k]["confidence"])]["correct"] for r in results])
            print(f"  [{pi+1}] det={det_so_far:.4f}, oracle={oracle_so_far:.4f}, head={head_so_far:.4f}")

    # =========================================================
    # Phase 4: Analysis
    # =========================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS: CoLaR σ=1000, Retrained Head")
    print("=" * 70)

    det_acc = np.mean([r["det_correct"] for r in results])
    oracle_acc = np.mean([any(ro["correct"] for ro in r["rollouts"]) for r in results])
    mean_rollout_acc = np.mean([np.mean([ro["correct"] for ro in r["rollouts"]]) for r in results])

    # Selectors
    head_selected = []
    random_selected = []
    for r in results:
        rollouts = r["rollouts"]
        best_idx = max(range(N), key=lambda k: rollouts[k]["confidence"])
        head_selected.append(rollouts[best_idx]["correct"])
        rng = np.random.default_rng(r["idx"])
        random_selected.append(rollouts[rng.integers(N)]["correct"])

    head_acc = np.mean(head_selected)
    random_acc = np.mean(random_selected)

    # Within-problem AUC
    wp_aucs = []
    for r in results:
        confs = [ro["confidence"] for ro in r["rollouts"]]
        labels = [float(ro["correct"]) for ro in r["rollouts"]]
        if len(set(labels)) == 2:
            auc = roc_auc_score(labels, confs)
            if not np.isnan(auc):
                wp_aucs.append(auc)

    # Pooled AUC
    all_confs = [ro["confidence"] for r in results for ro in r["rollouts"]]
    all_labels = [float(ro["correct"]) for r in results for ro in r["rollouts"]]
    pooled_auc = roc_auc_score(all_labels, all_confs)

    print(f"\n  Deterministic acc:      {det_acc:.4f}")
    print(f"  Mean rollout acc:       {mean_rollout_acc:.4f} (Δ: {mean_rollout_acc-det_acc:+.4f})")
    print(f"  Oracle (any correct):   {oracle_acc:.4f} (Δ: {oracle_acc-det_acc:+.4f})")
    print(f"  Random selector:        {random_acc:.4f} (Δ: {random_acc-det_acc:+.4f})")
    print(f"  Head selector:          {head_acc:.4f} (Δ: {head_acc-det_acc:+.4f})")
    print(f"\n  Within-problem AUC:     {np.mean(wp_aucs):.4f} ({len(wp_aucs)} mixed problems)")
    print(f"  Pooled AUC:             {pooled_auc:.4f}")
    print(f"  Retrained head CV AUC:  {cv_auc:.4f}")

    # Go/No-Go
    head_delta = head_acc - det_acc
    print(f"\n  === GO/NO-GO ===")
    print(f"  Head Δ vs deterministic: {head_delta:+.4f} ({head_delta*100:+.1f}pp)")
    print(f"  Within-problem AUC:      {np.mean(wp_aucs):.4f} (target: ≥0.65)")
    if head_delta >= 0.03 and np.mean(wp_aucs) >= 0.65:
        print(f"  → GO: Retrained head works as selector")
    elif head_delta >= 0.02:
        print(f"  → CONDITIONAL GO: Marginal improvement")
    else:
        print(f"  → NO-GO: Head cannot select even after retraining")

    # Save
    output = {
        "config": {"sigma_scale": sigma_scale, "N": N, "n_problems": len(results),
                   "paradigm": "CoLaR-5", "head": "retrained_mixed_50_50"},
        "head_cv_auc": float(cv_auc),
        "det_acc": float(det_acc),
        "oracle_acc": float(oracle_acc),
        "oracle_delta": float(oracle_acc - det_acc),
        "mean_rollout_acc": float(mean_rollout_acc),
        "head_acc": float(head_acc),
        "head_delta": float(head_delta),
        "random_acc": float(random_acc),
        "within_problem_auc": float(np.mean(wp_aucs)) if wp_aucs else None,
        "pooled_auc": float(pooled_auc),
        "n_mixed_problems": len(wp_aucs),
    }
    with open('./direction1_results/colar_retrained_head_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    with open('./direction1_results/colar_retrained_per_problem.json', 'w') as f:
        json.dump(results, f)
    print(f"\nSaved to direction1_results/colar_retrained_head_results.json")


if __name__ == "__main__":
    main()
