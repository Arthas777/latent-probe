"""
Collect latent trajectory features from CoLaR checkpoint.

CoLaR differs from Latent-SFT:
  - Latent tokens are continuous hidden states z_t ∈ R^2048 (sampled from Gaussian policy)
  - NOT vocab-space softmax distributions
  - Trajectory length T is dynamic (model decides when to emit ### separator)
  - Input features for confidence head: per-step hidden states from transformer layers

Usage:
    python collect_trajectories_colar.py \
        --ckpt_path ../../logs/colar/qsa-gsm/.../checkpoints/xxx.ckpt \
        --output_dir ./trajectory_data_colar
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.modules.projector import LatentPolicy
from src.utils.constants import MODEL_EMB_STD
from src.utils.utils import get_position_ids_from_attention_mask


class CoLaRInference:
    """Standalone CoLaR inference (no Lightning dependency)."""

    def __init__(self, ckpt_path, device='cuda:0'):
        self.device = torch.device(device)

        # Load checkpoint to get hparams
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        hparams = ckpt['hyper_parameters']
        model_kwargs = hparams['model_kwargs']

        model_id = model_kwargs['model_id']
        llm_path = os.path.join(project_root, 'models', 'llms', model_id)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path).to(self.device)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.embedding = self.llm.get_input_embeddings()

        # LatentPolicy
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

        # Generation config
        self.latent_gen_config = model_kwargs['latent_generation_config']
        self.answer_gen_config = model_kwargs['answer_generation_config']

        # Templates
        self.question_template = "Question: {} Let's think step by step:"
        self.speed_template = "(Thinking speed: {})"

        # Load state dict (LoRA weights + latent_policy weights)
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
        self.hidden_size = self.llm.config.hidden_size if not model_kwargs.get('do_lora') else hidden_size

        self.llm.eval()
        self.latent_policy.eval()

        print(f"Loaded CoLaR from {ckpt_path}")
        print(f"  Model: {model_id}, hidden_size={hidden_size}")
        print(f"  embeds_std={self.embeds_std}")
        print(f"  max_n_latent_forward={self.latent_gen_config['max_n_latent_forward']}")
        print(f"  compression_factor={self.latent_gen_config['compression_factor']}")

    @torch.no_grad()
    def generate_with_trajectory(self, question, max_latent_steps=None):
        """
        Run CoLaR inference and return trajectory features.
        Returns:
          - text: decoded answer
          - hidden_states: [T, hidden_dim] last-layer hidden states before each latent step
          - latent_embeds: [T, hidden_dim] sampled latent embeddings z_t
          - policy_means: [T, hidden_dim] mean of the Gaussian policy
          - policy_stds: [T, hidden_dim] std of the Gaussian policy
          - T: number of latent forward steps
          - correct: None (to be filled externally)
        """
        if max_latent_steps is None:
            max_latent_steps = self.latent_gen_config['max_n_latent_forward']

        speed = self.latent_gen_config['compression_factor']
        suffix = self.speed_template.format(speed) + self.thinking_separator

        # Prepare question input
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

        # Latent generation loop
        all_attention_mask = attention_mask
        current_position_ids = position_ids[:, -1:]
        past_key_values = outputs.past_key_values

        hidden_states_list = []
        latent_embeds_list = []
        policy_means_list = []
        policy_stds_list = []

        latent_temperature = self.latent_gen_config.get('latent_temperature', 1.0)

        for step in range(max_latent_steps):
            # Get hidden state before policy
            h = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden_dim]
            hidden_states_list.append(h.squeeze(0).squeeze(0).cpu())

            # Policy forward
            distribution = self.latent_policy.forward(h, temperature=latent_temperature)
            z_t = distribution.rsample() * self.embeds_std  # [1, 1, hidden_dim]

            latent_embeds_list.append(z_t.squeeze(0).squeeze(0).cpu())
            policy_means_list.append(distribution.mean.squeeze(0).squeeze(0).cpu())
            policy_stds_list.append(distribution.stddev.squeeze(0).squeeze(0).cpu())

            # Update attention mask
            all_attention_mask = torch.cat([
                all_attention_mask,
                torch.ones(1, 1, device=self.device, dtype=torch.long),
            ], dim=1)

            current_position_ids = current_position_ids + 1

            # Forward with latent embedding
            outputs = self.llm.forward(
                inputs_embeds=z_t,
                attention_mask=all_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values

            # Check EOL (end of latent thinking)
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

        # Build full input for answer generation
        all_embeds = torch.cat([question_embeds] + [e.unsqueeze(0).unsqueeze(0).to(self.device) for e in latent_embeds_list] + [end_of_thinking_embeds], dim=1)
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
            "hidden_states": torch.stack(hidden_states_list) if hidden_states_list else torch.zeros(0, self.llm.config.hidden_size),
            "latent_embeds": torch.stack(latent_embeds_list) if latent_embeds_list else torch.zeros(0, self.llm.config.hidden_size),
            "policy_means": torch.stack(policy_means_list) if policy_means_list else torch.zeros(0, self.llm.config.hidden_size),
            "policy_stds": torch.stack(policy_stds_list) if policy_stds_list else torch.zeros(0, self.llm.config.hidden_size),
            "T": T,
        }


def compute_colar_spectral_features(hidden_states, top_k=512):
    """Compute spectral features from CoLaR hidden state trajectory."""
    T = hidden_states.shape[0]
    if T < 2:
        return {"H_spec": 0.0, "top_sv_ratio": 0.0, "sv_entropy_norm": 0.0, "rank_90": 0}

    mat = hidden_states.float()
    if mat.shape[1] > top_k:
        # PCA-like: use SVD on the trajectory matrix directly
        pass  # hidden_dim > top_k is fine, SVD handles it

    try:
        S = torch.linalg.svdvals(mat)
    except Exception:
        return {"H_spec": 0.0, "top_sv_ratio": 0.0, "sv_entropy_norm": 0.0, "rank_90": 0}

    S = S[S > 1e-10]
    if len(S) == 0:
        return {"H_spec": 0.0, "top_sv_ratio": 0.0, "sv_entropy_norm": 0.0, "rank_90": 0}

    p = S / S.sum()
    H_spec = -(p * torch.log(p)).sum().item()
    top_sv_ratio = (S[0] / S.sum()).item()
    max_H = np.log(len(S)) if len(S) > 1 else 1.0
    sv_entropy_norm = H_spec / max_H if max_H > 0 else 0.0
    cumulative_energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    rank_90 = int((cumulative_energy < 0.9).sum().item()) + 1

    return {
        "H_spec": H_spec,
        "top_sv_ratio": top_sv_ratio,
        "sv_entropy_norm": sv_entropy_norm,
        "rank_90": rank_90,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./trajectory_data_colar')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--test_times', type=int, default=1,
                        help='Number of times to run each sample (for stochastic policy)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = CoLaRInference(args.ckpt_path, device=args.device)

    # Load test data
    test_path = os.path.join(project_root, 'datasets', 'text_reasoning', 'gsm', 'test.json')
    with open(test_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test samples.")

    all_features = []

    for run_idx in range(args.test_times):
        if args.test_times > 1:
            print(f"\n--- Run {run_idx+1}/{args.test_times} ---")
            torch.manual_seed(42 + run_idx)

        for i, example in enumerate(tqdm(data, desc=f"Collecting (run {run_idx+1})")):
            output = model.generate_with_trajectory(example['question'])

            # Check correctness
            pred_answer = model.tokenizer.decode([], skip_special_tokens=True) if not output['text'] else output['text']
            # Extract answer from CoLaR output format: "###Answer:XXX" or just the number
            pred_str = pred_answer.strip('#\n ').split('Answer:')[-1] if 'Answer:' in pred_answer else pred_answer.strip('#\n ')

            gt_answer = str(example['answer']).strip()
            pred_clean = pred_str.strip().rstrip('.').replace(',', '').lower()
            gt_clean = gt_answer.strip().rstrip('.').replace(',', '').lower()

            try:
                correct = abs(float(pred_clean) - float(gt_clean)) < 1e-4
            except (ValueError, TypeError):
                correct = pred_clean == gt_clean

            # Compute spectral features on hidden states
            spectral = compute_colar_spectral_features(output['hidden_states'])

            # Compute policy-specific features
            T = output['T']
            if T > 0:
                mean_std = output['policy_stds'].mean().item()
                std_of_std = output['policy_stds'].std().item() if T > 1 else 0.0
                mean_norm = output['latent_embeds'].norm(dim=-1).mean().item()
            else:
                mean_std = 0.0
                std_of_std = 0.0
                mean_norm = 0.0

            feature = {
                "idx": i,
                "run": run_idx,
                "correct": bool(correct),
                "T": T,
                "hidden_states": output["hidden_states"],  # [T, hidden_dim]
                "token_entropies": torch.zeros(T),  # placeholder for compatibility
                "top1_probs": torch.zeros(T),  # placeholder
                "spectral_features": {
                    **spectral,
                    "mean_policy_std": mean_std,
                    "std_of_policy_std": std_of_std,
                    "mean_latent_norm": mean_norm,
                },
            }
            all_features.append(feature)

            if (i + 1) % 100 == 0:
                acc = sum(f["correct"] for f in all_features) / len(all_features)
                print(f"  [{i+1}/{len(data)}] Running acc: {acc:.4f}, mean T: {np.mean([f['T'] for f in all_features[-100:]]):.1f}")

    # Save
    output_file = os.path.join(args.output_dir, "trajectories_GSM8k_colar.pt")
    torch.save(all_features, output_file)
    print(f"\nSaved {len(all_features)} trajectories to {output_file}")

    acc = sum(f["correct"] for f in all_features) / len(all_features)
    Ts = [f['T'] for f in all_features]
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Mean T: {np.mean(Ts):.1f} (std={np.std(Ts):.1f})")


if __name__ == "__main__":
    main()
