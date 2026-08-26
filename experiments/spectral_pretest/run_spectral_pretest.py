"""
Spectral pretest: correlation between spectral features of CoLaR latent
trajectories and correctness (Section 4.1).

Steps:
1. Load CoLaR checkpoint (Llama-3.2-1B-Instruct + LoRA, GSM8k SFT)
2. Run latent generation on GSM8k test set, collect latent trajectories
3. Compute spectral features: H(E_c), effective rank, top-SV ratio, etc.
4. Compute Spearman rho(spectral features, correctness)
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.modules.projector import LatentPolicy
from src.utils.constants import MODEL_EMB_STD


def load_model(llm_path, ckpt_path, device):
    """Load LLM + LoRA + LatentPolicy from checkpoint."""
    print(f"Loading tokenizer: {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"Loading LLM: {llm_path}")
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

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    llm_state = {}
    lp_state = {}
    for k, v in state_dict.items():
        if k.startswith("llm."):
            llm_state[k[4:]] = v  # strip "llm." prefix
        elif k.startswith("latent_policy."):
            lp_state[k[len("latent_policy."):]] = v

    print(f"  LLM keys: {len(llm_state)}, LatentPolicy keys: {len(lp_state)}")
    llm.load_state_dict(llm_state, strict=False)
    latent_policy.load_state_dict(lp_state, strict=True)

    llm = llm.to(device).eval()
    latent_policy = latent_policy.to(device=device, dtype=torch.bfloat16).eval()
    embedding = llm.get_input_embeddings()

    return tokenizer, llm, latent_policy, embedding


def get_position_ids(attention_mask):
    return torch.clamp_min(torch.cumsum(attention_mask, dim=1) - 1, 0)


@torch.no_grad()
def latent_generate_single(
    tokenizer, llm, latent_policy, embedding, question,
    embeds_std, device,
    compression_factor=5, max_n_latent_forward=64,
):
    """Run latent generation for a single question, return trajectory and predicted answer."""
    thinking_separator = "###"
    thinking_separator_id = tokenizer.convert_tokens_to_ids(thinking_separator)
    speed_template = "(Thinking speed: {})"
    question_template = "Question: {} Let's think step by step:"
    answer_template = "Answer:{}"

    # Prepare question input
    suffix = speed_template.format(compression_factor) + thinking_separator
    text = question_template.format(question) + suffix
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    question_input_ids = inputs["input_ids"].to(device)
    question_attention_mask = inputs["attention_mask"].to(device)

    question_position_ids = get_position_ids(question_attention_mask)
    question_embeds = embedding(question_input_ids)

    outputs = llm.forward(
        inputs_embeds=question_embeds,
        attention_mask=question_attention_mask,
        position_ids=question_position_ids,
        output_hidden_states=True,
    )

    # latent generation loop
    all_attention_mask = question_attention_mask
    current_position_ids = question_position_ids[:, -1:]
    past_key_values = outputs.past_key_values
    is_done = torch.tensor([[False]], device=device)

    latent_embeddings = []

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
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == thinking_separator_id:
            break

    # Generate answer
    end_of_thinking_ids = torch.tensor([[thinking_separator_id]], device=device)
    end_of_thinking_embeds = embedding(end_of_thinking_ids)
    all_attention_mask = torch.cat(
        [all_attention_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1
    )

    # Collect all embeds for generation
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

    # stack latent trajectory: (L_c, d)
    if len(latent_embeddings) > 0:
        trajectory = torch.stack(latent_embeddings, dim=0)  # (L_c, d)
    else:
        trajectory = None

    return trajectory, pred_str


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
        gt = float(gt)
        pred = float(pred)
    except ValueError:
        pass
    return float(gt == pred)


def compute_spectral_metrics(trajectory):
    """
    Compute spectral features of a latent trajectory.
    trajectory: (L_c, d) tensor
    """
    if trajectory is None or trajectory.shape[0] < 2:
        return None

    # SVD
    U, S, Vh = torch.linalg.svd(trajectory, full_matrices=False)
    S = S.float()

    # Normalized spectrum
    S_sq = S ** 2
    total = S_sq.sum()
    if total < 1e-12:
        return None
    p = S_sq / total

    # Spectral entropy
    log_p = torch.log(p + 1e-12)
    spectral_entropy = -(p * log_p).sum().item()

    # effective rank
    effective_rank = np.exp(spectral_entropy)

    # σ1² / Σσi² (top singular value dominance)
    top_sv_ratio = (S_sq[0] / total).item()

    # Cumulative ratio of top-k singular values
    cumulative = torch.cumsum(S_sq, dim=0) / total
    top5_ratio = cumulative[min(4, len(cumulative) - 1)].item()

    # Mean pairwise cosine similarity
    normed = trajectory / (trajectory.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim_matrix = normed @ normed.T
    n = cos_sim_matrix.shape[0]
    if n > 1:
        mask = ~torch.eye(n, dtype=torch.bool)
        mean_cos_sim = cos_sim_matrix[mask].mean().item()
    else:
        mean_cos_sim = 1.0

    return {
        "spectral_entropy": spectral_entropy,
        "effective_rank": effective_rank,
        "top_sv_ratio": top_sv_ratio,
        "top5_sv_ratio": top5_ratio,
        "mean_cos_sim": mean_cos_sim,
        "n_latent": trajectory.shape[0],
        "n_singular_values": len(S),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=0, help="0=all samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeats", type=int, default=1, help="repeats per sample (majority vote)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent
    llm_path = project_root / "models" / "llms" / "Llama-3.2-1B-Instruct"
    ckpt_path = base_dir / "checkpoints" / "logs" / "colar" / "qsa-gsm" / "colar-final" / "checkpoints" / "colar_best.ckpt"
    data_path = base_dir / "data" / "gsm8k_test.json"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    device = args.device
    embeds_std = MODEL_EMB_STD["Llama-3.2-1B-Instruct"]

    tokenizer, llm, latent_policy, embedding = load_model(
        str(llm_path), str(ckpt_path), device
    )

    with open(data_path) as f:
        data = json.load(f)

    if args.n_samples > 0:
        data = data[:args.n_samples]

    print(f"\nStarting inference: {len(data)} samples, device={device}")

    results = []
    for i, sample in enumerate(tqdm(data, desc="Inference")):
        question = sample["question"]
        gt_answer = sample["answer"]

        trajectory, pred_str = latent_generate_single(
            tokenizer, llm, latent_policy, embedding, question,
            embeds_std, device,
        )

        pred_answer = extract_answer(pred_str)
        correct = verify_answer(gt_answer, pred_answer)

        spectral = compute_spectral_metrics(trajectory)
        if spectral is None:
            continue

        result = {
            "idx": i,
            "correct": correct,
            "pred_answer": pred_answer,
            "gt_answer": gt_answer,
            **spectral,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            acc_so_far = np.mean([r["correct"] for r in results])
            print(f"  [{i+1}/{len(data)}] acc={acc_so_far:.3f}, "
                  f"mean H={np.mean([r['spectral_entropy'] for r in results]):.3f}")

    # Save raw results
    with open(output_dir / "spectral_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Statistical analysis
    print("\n" + "=" * 60)
    print("Statistical analysis results")
    print("=" * 60)

    correctness = np.array([r["correct"] for r in results])
    n_total = len(correctness)
    acc = correctness.mean()
    print(f"Total samples: {n_total}, accuracy: {acc:.4f}")
    print(f"Correct: {int(correctness.sum())}, Incorrect: {int(n_total - correctness.sum())}")

    metrics = ["spectral_entropy", "effective_rank", "top_sv_ratio", "mean_cos_sim"]
    metric_names = {
        "spectral_entropy": "Spectral entropy H(E_c)",
        "effective_rank": "Effective Rank",
        "top_sv_ratio": "top-SV ratio (collapse indicator)",
        "mean_cos_sim": "Mean cosine similarity (collapse indicator)",
    }

    analysis = {}
    for metric in metrics:
        values = np.array([r[metric] for r in results])
        rho, pvalue = stats.spearmanr(values, correctness)

        correct_vals = values[correctness == 1]
        wrong_vals = values[correctness == 0]

        # Welch's t-test
        if len(correct_vals) > 1 and len(wrong_vals) > 1:
            t_stat, t_pvalue = stats.ttest_ind(correct_vals, wrong_vals, equal_var=False)
            # Mann-Whitney U test (more robust)
            u_stat, u_pvalue = stats.mannwhitneyu(correct_vals, wrong_vals, alternative='two-sided')
        else:
            t_stat, t_pvalue = float('nan'), float('nan')
            u_stat, u_pvalue = float('nan'), float('nan')

        analysis[metric] = {
            "spearman_rho": rho,
            "spearman_pvalue": pvalue,
            "correct_mean": float(correct_vals.mean()) if len(correct_vals) > 0 else None,
            "correct_std": float(correct_vals.std()) if len(correct_vals) > 0 else None,
            "wrong_mean": float(wrong_vals.mean()) if len(wrong_vals) > 0 else None,
            "wrong_std": float(wrong_vals.std()) if len(wrong_vals) > 0 else None,
            "welch_t": float(t_stat),
            "welch_pvalue": float(t_pvalue),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_pvalue": float(u_pvalue),
        }

        name = metric_names.get(metric, metric)
        print(f"\n--- {name} ---")
        print(f"  Spearman ρ = {rho:.4f}  (p = {pvalue:.2e})")
        if len(correct_vals) > 0 and len(wrong_vals) > 0:
            print(f"  Correct: {correct_vals.mean():.4f} +/- {correct_vals.std():.4f}")
            print(f"  Incorrect: {wrong_vals.mean():.4f} +/- {wrong_vals.std():.4f}")
            print(f"  Mann-Whitney U p = {u_pvalue:.2e}")

    # Latent length analysis
    n_latents = np.array([r["n_latent"] for r in results])
    rho_len, p_len = stats.spearmanr(n_latents, correctness)
    print(f"\n--- Latent sequence length ---")
    print(f"  Spearman ρ(length, correctness) = {rho_len:.4f}  (p = {p_len:.2e})")
    print(f"  Mean length: {n_latents.mean():.1f}, correct: {n_latents[correctness==1].mean():.1f}, "
          f"incorrect: {n_latents[correctness==0].mean():.1f}")

    analysis["n_latent"] = {
        "spearman_rho": rho_len,
        "spearman_pvalue": p_len,
        "overall_mean": float(n_latents.mean()),
    }

    # Save analysis results
    with open(output_dir / "spectral_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    # Conclusion
    rho_H = analysis["spectral_entropy"]["spearman_rho"]
    print("\n" + "=" * 60)
    print("Pretest conclusion")
    print("=" * 60)
    if abs(rho_H) >= 0.3:
        print(f"PASS: rho(H, correctness) = {rho_H:.4f} >= 0.3")
        print("  -> Spectral entropy is a strong signal")
    elif abs(rho_H) >= 0.1:
        print(f"WEAK: rho(H, correctness) = {rho_H:.4f} in [0.1, 0.3)")
        print("  -> Signal exists but weak, consider combined metrics")
    else:
        print(f"FAIL: rho(H, correctness) = {rho_H:.4f} < 0.1")
        print("  -> Consider alternative metrics (top_sv_ratio or mean_cos_sim)")

    # Find best metric
    best_metric = max(metrics, key=lambda m: abs(analysis[m]["spearman_rho"]))
    best_rho = analysis[best_metric]["spearman_rho"]
    print(f"\nBest correlated metric: {metric_names.get(best_metric, best_metric)}, rho = {best_rho:.4f}")


if __name__ == "__main__":
    main()
