"""
Direction 1 Experiment C: CoLaR diversity injection via σ-scaling.

CoLaR's latent policy outputs a Gaussian distribution N(μ, σ²).
Training collapses σ → ~2.7e-5, eliminating stochasticity.
We inject diversity by scaling σ up by a factor (σ_scale).

Tests multiple σ_scale values to find the goldilocks zone.
Computes oracle ceiling and head-as-selector accuracy.
"""
import os, sys, json, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf
from src.modules.projector import LatentPolicy
from src.utils.constants import MODEL_EMB_STD
from src.utils.utils import get_position_ids_from_attention_mask
from train_confidence_head import AttentionConfidenceHead


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    desc_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_idx]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return float(np.trapz(tpr, fpr))


def load_head(path, hidden_dim, device):
    model = AttentionConfidenceHead(hidden_dim=hidden_dim, proj_dim=128, n_heads=4, dropout=0.1)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def head_predict(head, hidden_states, T, max_T=64, device='cuda:0'):
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


class CoLaRDiversityTest:
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
        """Generate one rollout with optional sigma scaling for diversity."""
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

            # === DIVERSITY INJECTION ===
            # Instead of z = distribution.rsample(), we manually sample with scaled sigma
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
    """Extract and check answer from CoLaR output."""
    pred_str = text.strip('#\n ').split('Answer:')[-1] if 'Answer:' in text else text.strip('#\n ')
    pred_clean = pred_str.strip().rstrip('.').replace(',', '').lower()
    gt_clean = gt_str.strip().rstrip('.').replace(',', '').lower()
    try:
        return abs(float(pred_clean) - float(gt_clean)) < 1e-4
    except (ValueError, TypeError):
        return pred_clean == gt_clean


def main():
    device = 'cuda:1'
    N = 8
    ckpt_path = os.path.join(project_root, 'logs/colar/qsa-gsm/20260428-195821_350334_full-cotsft-baseline/checkpoints/epoch44__step67815__monitor0.312.ckpt')

    print("Loading CoLaR model...")
    colar = CoLaRDiversityTest(ckpt_path, device=device)

    # Train a quick CoLaR confidence head from existing trajectory data
    colar_traj_path = './trajectory_data_colar/trajectories_GSM8k_colar.pt'
    if os.path.exists(colar_traj_path):
        print("Training CoLaR confidence head from existing trajectories...")
        from train_confidence_head import TrajectoryDataset, train, evaluate
        from torch.utils.data import DataLoader
        dataset = TrajectoryDataset(colar_traj_path, max_T=64)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        head = AttentionConfidenceHead(hidden_dim=colar.hidden_size, proj_dim=128, n_heads=4, dropout=0.1).to(device)

        class QuickArgs:
            epochs = 30
            lr = 1e-3
            weight_decay = 0.01
            batch_size = 64
        _, best_auc = train(head, train_loader, val_loader, torch.device(device), QuickArgs(), model_type='attention')
        print(f"  CoLaR head trained, best AUC={best_auc:.4f}")
        head.eval()
    else:
        print("WARNING: No CoLaR trajectory data found. Using Latent-SFT head (OOD).")
        head = load_head('./trained_heads/confidence_head_attention.pt', hidden_dim=colar.hidden_size, device=device)

    # Load test data
    test_path = os.path.join(project_root, 'datasets', 'text_reasoning', 'gsm', 'test.json')
    with open(test_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems")

    # Test multiple sigma scales
    sigma_scales = [1.0, 100.0, 1000.0, 5000.0, 10000.0]

    for sigma_scale in sigma_scales:
        print(f"\n{'='*70}")
        print(f"SIGMA_SCALE = {sigma_scale}")
        print(f"{'='*70}")

        results = []
        for pi, example in enumerate(tqdm(data, desc=f"σ={sigma_scale}")):
            gt_answer = str(example['answer']).strip()

            # Deterministic (sigma_scale=1, seed=0 for reproducible EOL sampling)
            if sigma_scale == sigma_scales[0]:
                det_out = colar.generate_rollout(example['question'], sigma_scale=1.0, seed=0)
                det_correct = extract_answer_colar(det_out['text'], gt_answer)
                det_conf = head_predict(head, det_out['hidden_states'], det_out['T'], device=device)
            else:
                det_correct = results_baseline[pi]["det_correct"]
                det_conf = results_baseline[pi]["det_conf"]

            # N rollouts with diversity
            rollouts = []
            for n in range(N):
                seed = n * 10000 + pi + 1
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
                mean_acc = np.mean([np.mean([ro["correct"] for ro in r["rollouts"]]) for r in results])
                print(f"  [{pi+1}] det={det_so_far:.4f}, mean_rollout={mean_acc:.4f}, oracle={oracle_so_far:.4f}")

        # Store baseline for reuse
        if sigma_scale == sigma_scales[0]:
            results_baseline = results

        # Analysis
        det_acc = np.mean([r["det_correct"] for r in results])
        oracle_acc = np.mean([any(ro["correct"] for ro in r["rollouts"]) for r in results])
        mean_rollout_acc = np.mean([np.mean([ro["correct"] for ro in r["rollouts"]]) for r in results])

        # Head selector
        head_selected = []
        sc_answers_not_available = True  # CoLaR doesn't give us parsed answers easily
        for r in results:
            best_idx = max(range(N), key=lambda k: r["rollouts"][k]["confidence"])
            head_selected.append(r["rollouts"][best_idx]["correct"])
        head_acc = np.mean(head_selected)

        # Within-problem AUC
        wp_aucs = []
        for r in results:
            confs = [ro["confidence"] for ro in r["rollouts"]]
            labels = [float(ro["correct"]) for ro in r["rollouts"]]
            if len(set(labels)) == 2:
                auc = roc_auc_score(labels, confs)
                if not np.isnan(auc):
                    wp_aucs.append(auc)

        print(f"\n  σ_scale={sigma_scale}:")
        print(f"    Deterministic acc:    {det_acc:.4f}")
        print(f"    Mean rollout acc:     {mean_rollout_acc:.4f} (Δ: {mean_rollout_acc-det_acc:+.4f})")
        print(f"    Oracle (any correct): {oracle_acc:.4f} (Δ: {oracle_acc-det_acc:+.4f})")
        print(f"    Head selector:        {head_acc:.4f} (Δ: {head_acc-det_acc:+.4f})")
        print(f"    Within-problem AUC:   {np.mean(wp_aucs):.4f} ({len(wp_aucs)} mixed problems)")

        # Save
        os.makedirs('./direction1_results', exist_ok=True)
        output = {
            "config": {"sigma_scale": sigma_scale, "N": N, "n_problems": len(results), "paradigm": "CoLaR-5"},
            "det_acc": float(det_acc),
            "oracle_acc": float(oracle_acc),
            "oracle_delta": float(oracle_acc - det_acc),
            "mean_rollout_acc": float(mean_rollout_acc),
            "head_acc": float(head_acc),
            "within_problem_auc": float(np.mean(wp_aucs)) if wp_aucs else None,
            "n_mixed_problems": len(wp_aucs),
        }
        with open(f'./direction1_results/colar_sigma{sigma_scale}_results.json', 'w') as f:
            json.dump(output, f, indent=2)

    print("\n\nDone. Results saved to direction1_results/colar_sigma*_results.json")


if __name__ == "__main__":
    main()
