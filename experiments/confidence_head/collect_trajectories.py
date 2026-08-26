"""
Step 1: Collect latent trajectory features from Latent-SFT checkpoint.
For each sample, extracts:
  - Per-step token entropies H(α_t)
  - Per-step top-1 probability (max of α_t)
  - Per-step hidden state (last layer, last token)
  - Trajectory length T
  - Correctness label

Saves features as .pt files for training the confidence head.

Usage:
    python collect_trajectories.py \
        --latent_model_path ../../Latent-SFT/checkpoints/latent-4 \
        --dataset GSM8k \
        --output_dir ./trajectory_data
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
latent_sft_dir = os.path.join(current_dir, '../../Latent-SFT')
sys.path.append(latent_sft_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def extract_answer(pred_str):
    pred_str = pred_str.replace("ки", "")
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            return a
        else:
            return ans.split("$")[0].strip()
    return ""


def check_is_correct(pred, gt):
    pred = pred.strip().replace(",", "").replace("$", "").replace("%", "")
    gt = str(gt).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except (ValueError, TypeError):
        pass
    return pred.strip() == gt.strip()


@torch.no_grad()
def generate_with_full_trajectory(model, inputs, max_new_tokens=128,
                                   temperature=0.6, top_p=0.95):
    """
    Run Latent-SFT inference and return rich trajectory features:
      - hidden_states: last-layer hidden at each latent step [T, hidden_dim]
      - token_entropies: H(α_t) per step [T]
      - top1_probs: max(α_t) per step [T]
      - spectral_features: SVD-based features of α_t matrix
      - T: number of latent steps
      - text: decoded answer
    """
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids = []
    past_key_values = None

    hidden_states_list = []
    token_entropies = []
    top1_probs = []
    latent_probs_list = []
    latent_embeds = []

    W = model.model.embed_tokens.weight.detach()

    for latent_step in range(max_new_tokens):
        if input_ids is not None:
            input_embeddings = model.model.embed_tokens(input_ids)

        outputs = model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]
        hidden_states_list.append(last_hidden.squeeze(0).cpu())

        next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        probs = torch.softmax(next_token_logits.float(), dim=-1)  # [1, vocab_size]
        p = probs[0]

        # Token entropy
        log_p = torch.log(p + 1e-12)
        h = -(p * log_p).sum().item()
        token_entropies.append(h)

        # Top-1 probability
        top1_probs.append(p.max().item())

        # Store probs for spectral analysis
        latent_probs_list.append(p.cpu())

        # Soft embedding for next step
        input_embeddings = (probs.to(W.dtype) @ W).unsqueeze(1)
        latent_embeds.append(input_embeddings)

        attention_mask = torch.cat([
            attention_mask,
            torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
        ], dim=1)

        if next_token[0, 0].item() == model.latent_token_ids[1][0]:
            input_ids = next_token
            break
        else:
            input_ids = None
            generated_ids.append(int(next_token.item()))

    T = len(hidden_states_list)

    # Answer phase
    if latent_embeds:
        latent_state = torch.cat(latent_embeds, dim=1)
    else:
        latent_state = torch.zeros(1, 0, W.shape[1], device=W.device, dtype=W.dtype)

    think_ids = torch.LongTensor(model.latent_token_ids[1]).unsqueeze(0).to(model.device)
    think_end_embeddings = model.model.embed_tokens(think_ids)
    input_ids_orig = inputs['input_ids']
    attn_len = input_ids_orig.size(1) + latent_state.size(1) + len(model.latent_token_ids[1])
    attention_mask_full = torch.ones(1, attn_len, dtype=torch.long, device=model.device)

    input_embeddings_full = model.model.embed_tokens(input_ids_orig)
    input_embeddings_full = torch.cat([input_embeddings_full, latent_state, think_end_embeddings], dim=1)

    generated_output = model.generate(
        inputs_embeds=input_embeddings_full,
        attention_mask=attention_mask_full,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    decoded_text = model.tokenizer.decode(generated_output[0], skip_special_tokens=False)

    # Compute spectral features from α_t matrix
    spectral_feats = compute_spectral_features(latent_probs_list)

    return {
        "text": decoded_text,
        "hidden_states": torch.stack(hidden_states_list) if hidden_states_list else torch.zeros(0, W.shape[1]),  # [T, hidden_dim]
        "token_entropies": torch.tensor(token_entropies, dtype=torch.float32),  # [T]
        "top1_probs": torch.tensor(top1_probs, dtype=torch.float32),  # [T]
        "spectral_features": spectral_feats,  # dict of scalar features
        "T": T,
    }


def compute_spectral_features(probs_list, top_k=512):
    """Compute spectral features from the α_t trajectory matrix."""
    T = len(probs_list)
    if T < 2:
        return {"H_spec": 0.0, "top_sv_ratio": 0.0, "sv_entropy_norm": 0.0, "rank_90": 0}

    mat = torch.stack(probs_list)  # [T, V]
    if mat.shape[1] > top_k:
        mean_probs = mat.mean(dim=0)
        _, topk_idx = mean_probs.topk(top_k)
        mat = mat[:, topk_idx]

    try:
        S = torch.linalg.svdvals(mat.float())
    except Exception:
        return {"H_spec": 0.0, "top_sv_ratio": 0.0, "sv_entropy_norm": 0.0, "rank_90": 0}

    S = S[S > 1e-10]
    if len(S) == 0:
        return {"H_spec": 0.0, "top_sv_ratio": 0.0, "sv_entropy_norm": 0.0, "rank_90": 0}

    p = S / S.sum()
    H_spec = -(p * torch.log(p)).sum().item()

    # Top singular value ratio
    top_sv_ratio = (S[0] / S.sum()).item()

    # Normalized spectral entropy
    max_H = np.log(len(S)) if len(S) > 1 else 1.0
    sv_entropy_norm = H_spec / max_H if max_H > 0 else 0.0

    # Effective rank (90% energy)
    cumulative_energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    rank_90 = int((cumulative_energy < 0.9).sum().item()) + 1

    return {
        "H_spec": H_spec,
        "top_sv_ratio": top_sv_ratio,
        "sv_entropy_norm": sv_entropy_norm,
        "rank_90": rank_90,
    }


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='GSM8k')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='./trajectory_data')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.latent_model_path}...")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.latent_model_path,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16,
        use_cache=False,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.latent_model_path)
    model.eval()
    model.tokenizer = tokenizer
    model.latent_token_ids = tokenizer(['<think>', '</think>'], add_special_tokens=False)['input_ids']
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Load data
    data_dir = os.path.join(latent_sft_dir, 'data')
    data_map = {
        'GSM8k': 'GSM8k-Aug-test.jsonl',
        'Math500': 'Math-500-test.jsonl',
        'AIME24': 'AIME-2024-test.jsonl',
    }
    data_path = os.path.join(data_dir, data_map.get(args.dataset, f'{args.dataset}-test.jsonl'))
    print(f"Loading data from {data_path}...")
    data = read_jsonl(data_path)
    print(f"Loaded {len(data)} samples.")

    config = AutoConfig.from_pretrained(args.latent_model_path)
    model_type = config.model_type

    all_features = []

    for i, example in enumerate(tqdm(data, desc="Collecting trajectories")):
        # Format input
        if model_type == 'qwen2':
            messages = [{"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + example["problem"]}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_prefix = input_text + "<｜Assistant｜>"
        else:  # llama
            input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
            input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        input_ids = tokenizer(input_prefix, truncation=False, padding=False,
                             add_special_tokens=False, return_attention_mask=False)['input_ids']

        text_input = {
            'input_ids': torch.tensor(input_ids + model.latent_token_ids[0], dtype=torch.long).to(device).unsqueeze(0),
            'attention_mask': torch.tensor([1] * (len(input_ids) + len(model.latent_token_ids[0])), dtype=torch.long).to(device).unsqueeze(0),
        }

        output = generate_with_full_trajectory(
            model, text_input,
            max_new_tokens=args.max_new_tokens,
            temperature=0.6, top_p=0.95,
        )

        # Check correctness
        pred = extract_answer(output["text"])
        correct = check_is_correct(pred, example["answer"])

        feature = {
            "idx": i,
            "correct": bool(correct),
            "T": output["T"],
            "hidden_states": output["hidden_states"],  # [T, hidden_dim]
            "token_entropies": output["token_entropies"],  # [T]
            "top1_probs": output["top1_probs"],  # [T]
            "spectral_features": output["spectral_features"],  # dict
        }
        all_features.append(feature)

        if (i + 1) % 100 == 0:
            acc = sum(f["correct"] for f in all_features) / len(all_features)
            print(f"  [{i+1}/{len(data)}] Running acc: {acc:.4f}")

    # Save
    output_file = os.path.join(args.output_dir, f"trajectories_{args.dataset}.pt")
    torch.save(all_features, output_file)
    print(f"\nSaved {len(all_features)} trajectories to {output_file}")

    acc = sum(f["correct"] for f in all_features) / len(all_features)
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Mean T: {np.mean([f['T'] for f in all_features]):.1f}")


if __name__ == "__main__":
    main()
