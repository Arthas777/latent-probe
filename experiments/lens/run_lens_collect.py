"""
LENS multi-rollout data collection (Section 4.2).
For each problem in GSM8k test set, run N latent generations and collect:
- LENS (H_spec / log T)
- Raw spectral entropy H
- Mean token entropy (baseline)
- Last-token confidence
- Correct/incorrect label

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_lens_collect.py --n_rollouts 8 --seed 42
  CUDA_VISIBLE_DEVICES=0 python run_lens_collect.py --n_rollouts 8 --seed 42 --n_samples 50  # quick test
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.modules.projector import LatentPolicy
from src.utils.constants import MODEL_EMB_STD


def load_model(llm_path, ckpt_path, device):
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    llm = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
    llm.resize_token_embeddings(len(tokenizer))
    llm.generation_config.pad_token_id = tokenizer.pad_token_id
    llm.generation_config.eos_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(r=128, lora_alpha=32)
    llm = get_peft_model(llm, peft_config=lora_config)

    latent_policy = LatentPolicy(
        feature_size=llm.config.hidden_size,
        intermediate_size=2048,
        deterministic=False,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    llm_state = {}
    lp_state = {}
    for k, v in state_dict.items():
        if k.startswith("llm."):
            llm_state[k[4:]] = v
        elif k.startswith("latent_policy."):
            lp_state[k[len("latent_policy."):]] = v

    llm.load_state_dict(llm_state, strict=False)
    latent_policy.load_state_dict(lp_state, strict=True)

    llm = llm.to(device).eval()
    latent_policy = latent_policy.to(device=device, dtype=torch.bfloat16).eval()
    embedding = llm.get_input_embeddings()

    return tokenizer, llm, latent_policy, embedding


@torch.no_grad()
def latent_generate_with_metrics(
    tokenizer, llm, latent_policy, embedding,
    question, embeds_std, device,
    compression_factor=5, max_n_latent_forward=64,
):
    """
    Single latent generation, returns:
    - trajectory: (T, d) tensor
    - pred_answer: str
    - token_entropies: list of per-step vocab entropy
    - last_token_conf: float (final step max prob)
    """
    thinking_separator = "###"
    thinking_separator_id = tokenizer.convert_tokens_to_ids(thinking_separator)
    speed_template = "(Thinking speed: {})"
    question_template = "Question: {} Let's think step by step:"

    suffix = speed_template.format(compression_factor) + thinking_separator
    text = question_template.format(question) + suffix
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    question_input_ids = inputs["input_ids"].to(device)
    question_attention_mask = inputs["attention_mask"].to(device)

    question_position_ids = torch.clamp_min(
        torch.cumsum(question_attention_mask, dim=1) - 1, 0
    )
    question_embeds = embedding(question_input_ids)

    outputs = llm.forward(
        inputs_embeds=question_embeds,
        attention_mask=question_attention_mask,
        position_ids=question_position_ids,
        output_hidden_states=True,
    )

    all_attention_mask = question_attention_mask
    current_position_ids = question_position_ids[:, -1:]
    past_key_values = outputs.past_key_values
    is_done = torch.tensor([[False]], device=device)

    latent_embeddings = []
    token_entropies = []
    last_token_conf = 0.0

    for _ in range(max_n_latent_forward):
        distributions = latent_policy.forward(outputs.hidden_states[-1][:, -1:, :])
        current_inputs_embeds = distributions.rsample() * embeds_std

        latent_embeddings.append(current_inputs_embeds.squeeze(0).squeeze(0).cpu().float())

        not_done = (~is_done).long()
        all_attention_mask = torch.cat([all_attention_mask, not_done], dim=1)
        current_position_ids = current_position_ids + not_done

        outputs = llm.forward(
            inputs_embeds=current_inputs_embeds,
            attention_mask=all_attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values

        last_logits = outputs.logits[:, -1]
        probs = torch.softmax(last_logits, dim=-1)

        # token-level entropy
        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum().item()
        token_entropies.append(entropy)

        # last-token confidence (max prob)
        last_token_conf = probs.max().item()

        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == thinking_separator_id:
            break

    # generate answer
    end_of_thinking_ids = torch.tensor([[thinking_separator_id]], device=device)
    end_of_thinking_embeds = embedding(end_of_thinking_ids)
    all_attention_mask = torch.cat(
        [all_attention_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1
    )

    all_embeds_list = [question_embeds]
    for le in latent_embeddings:
        all_embeds_list.append(le.unsqueeze(0).unsqueeze(0).to(device=device, dtype=question_embeds.dtype))
    all_embeds_list.append(end_of_thinking_embeds)
    all_inputs_embeds = torch.cat(all_embeds_list, dim=1)

    pred_ids = llm.generate(
        inputs_embeds=all_inputs_embeds,
        attention_mask=all_attention_mask,
        max_new_tokens=16,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )
    pred_str = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    if len(latent_embeddings) > 0:
        trajectory = torch.stack(latent_embeddings, dim=0)
    else:
        trajectory = None

    return trajectory, pred_str, token_entropies, last_token_conf


def compute_spectral_metrics(trajectory):
    if trajectory is None or trajectory.shape[0] < 2:
        return None
    U, S, Vh = torch.linalg.svd(trajectory, full_matrices=False)
    S = S.float()
    S_sq = S ** 2
    total = S_sq.sum()
    if total < 1e-12:
        return None
    p = S_sq / total

    log_p = torch.log(p + 1e-12)
    spectral_entropy = -(p * log_p).sum().item()
    effective_rank = np.exp(spectral_entropy)
    top_sv_ratio = (S_sq[0] / total).item()

    normed = trajectory / (trajectory.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim_matrix = normed @ normed.T
    n = cos_sim_matrix.shape[0]
    if n > 1:
        mask = ~torch.eye(n, dtype=torch.bool)
        mean_cos_sim = cos_sim_matrix[mask].mean().item()
    else:
        mean_cos_sim = 1.0

    T = trajectory.shape[0]
    lens = spectral_entropy / np.log(T) if T >= 2 else 0.0

    return {
        "spectral_entropy": spectral_entropy,
        "lens": lens,
        "effective_rank": effective_rank,
        "top_sv_ratio": top_sv_ratio,
        "mean_cos_sim": mean_cos_sim,
        "n_latent": T,
    }


def extract_answer(output_string):
    try:
        return output_string.strip('#').split("Answer:")[-1].strip()
    except (ValueError, IndexError):
        return output_string


def verify_answer(gt_answer, pred_answer):
    def clean(s):
        return s.strip("#\n ").rstrip(".").replace(",", "").lower()
    gt = clean(gt_answer)
    pred = clean(pred_answer)
    try:
        gt_f = float(gt)
        pred_f = float(pred)
        return float(gt_f == pred_f)
    except ValueError:
        return float(gt == pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=0, help="0=all 1319 samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression_factor", type=int, default=5)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent
    llm_path = project_root / "models" / "llms" / "Llama-3.2-1B-Instruct"
    ckpt_path = (
        project_root / "experiments" / "spectral_pretest" / "checkpoints" / "logs"
        / "colar" / "qsa-gsm" / "colar-final" / "checkpoints" / "colar_best.ckpt"
    )
    data_path = project_root / "experiments" / "spectral_pretest" / "data" / "gsm8k_test.json"

    suffix = f"_N{args.n_rollouts}_c{args.compression_factor}_seed{args.seed}"
    if args.output_suffix:
        suffix += f"_{args.output_suffix}"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    device = args.device
    embeds_std = MODEL_EMB_STD["Llama-3.2-1B-Instruct"]

    print(f"Loading model...")
    tokenizer, llm, latent_policy, embedding = load_model(
        str(llm_path), str(ckpt_path), device
    )

    with open(data_path) as f:
        data = json.load(f)
    if args.n_samples > 0:
        data = data[:args.n_samples]

    print(f"N_rollouts={args.n_rollouts}, N_samples={len(data)}, c={args.compression_factor}")
    print(f"Total inference calls: {args.n_rollouts * len(data)}")

    all_results = []

    for i, sample in enumerate(tqdm(data, desc="Questions")):
        question = sample["question"]
        gt_answer = sample["answer"]

        question_rollouts = []
        for r in range(args.n_rollouts):
            trajectory, pred_str, token_entropies, last_token_conf = \
                latent_generate_with_metrics(
                    tokenizer, llm, latent_policy, embedding,
                    question, embeds_std, device,
                    compression_factor=args.compression_factor,
                )

            pred_answer = extract_answer(pred_str)
            correct = verify_answer(gt_answer, pred_answer)
            spectral = compute_spectral_metrics(trajectory)

            if spectral is None:
                continue

            mean_token_entropy = np.mean(token_entropies) if token_entropies else 0.0

            rollout_data = {
                "rollout_idx": r,
                "correct": correct,
                "pred_answer": pred_answer,
                **spectral,
                "mean_token_entropy": mean_token_entropy,
                "last_token_conf": last_token_conf,
            }
            question_rollouts.append(rollout_data)

        all_results.append({
            "question_idx": i,
            "gt_answer": gt_answer,
            "n_valid_rollouts": len(question_rollouts),
            "rollouts": question_rollouts,
        })

        if (i + 1) % 50 == 0:
            n_total_rollouts = sum(len(q["rollouts"]) for q in all_results)
            n_correct = sum(
                sum(r["correct"] for r in q["rollouts"])
                for q in all_results
            )
            print(f"  [{i+1}/{len(data)}] total_rollouts={n_total_rollouts}, "
                  f"per-rollout acc={n_correct/max(n_total_rollouts,1):.3f}")

    # save
    output_path = output_dir / f"lens_rollouts{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # quick stats
    n_questions = len(all_results)
    all_rollouts_flat = [r for q in all_results for r in q["rollouts"]]
    per_rollout_acc = np.mean([r["correct"] for r in all_rollouts_flat])
    print(f"Questions: {n_questions}, Total rollouts: {len(all_rollouts_flat)}")
    print(f"Per-rollout accuracy: {per_rollout_acc:.4f}")


if __name__ == "__main__":
    main()
