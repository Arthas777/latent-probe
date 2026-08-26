"""
Collect multiple rollouts from Latent-SFT(2) with noise injection (ε=0.001)
to measure within-problem AUC for the compression ratio sweep (Experiment D).

This parallels the existing direction1_lsft_full.py but for Latent-SFT(2).
"""

import os, sys, json, torch, numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments', 'confidence_head'))

from transformers import AutoModelForCausalLM, AutoTokenizer
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
    with torch.no_grad():
        logit = head(h, mask, T_tensor)
    return torch.sigmoid(logit).item()


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
def generate_with_noise(model, inputs, eps=0.001, max_new_tokens=128,
                        temperature=0.6, top_p=0.95, seed=None):
    """
    Latent-SFT inference with Gaussian noise injection on soft embeddings.
    Returns hidden states for confidence head scoring.
    """
    if seed is not None:
        torch.manual_seed(seed)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    device = input_ids.device

    W = model.model.embed_tokens.weight.detach()
    hidden_states_list = []
    past_key_values = None
    latent_embeds = []

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

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        hidden_states_list.append(last_hidden.squeeze(0).cpu())

        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        probs = torch.softmax(next_token_logits.float(), dim=-1)

        # Soft embedding + noise injection
        soft_emb = (probs.to(W.dtype) @ W)
        if eps > 0:
            noise = torch.randn_like(soft_emb) * eps
            soft_emb = soft_emb + noise

        input_embeddings = soft_emb.unsqueeze(1)
        latent_embeds.append(input_embeddings)

        attention_mask = torch.cat([
            attention_mask,
            torch.ones((attention_mask.size(0), 1), device=device)
        ], dim=1)

        if next_token[0, 0].item() == model.latent_token_ids[1][0]:
            input_ids = next_token
            break
        else:
            input_ids = None

    T = len(hidden_states_list)

    # Answer phase
    if latent_embeds:
        latent_state = torch.cat(latent_embeds, dim=1)
    else:
        latent_state = torch.zeros(1, 0, W.shape[1], device=device, dtype=W.dtype)

    think_ids = torch.LongTensor(model.latent_token_ids[1]).unsqueeze(0).to(device)
    think_end_embeddings = model.model.embed_tokens(think_ids)
    input_ids_orig = inputs['input_ids']
    input_embeddings_full = model.model.embed_tokens(input_ids_orig)
    input_embeddings_full = torch.cat([input_embeddings_full, latent_state, think_end_embeddings], dim=1)
    attn_len = input_embeddings_full.size(1)
    attention_mask_full = torch.ones(1, attn_len, dtype=torch.long, device=device)

    generated_output = model.generate(
        inputs_embeds=input_embeddings_full,
        attention_mask=attention_mask_full,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    decoded_text = model.tokenizer.decode(generated_output[0], skip_special_tokens=False)

    return {
        "text": decoded_text,
        "hidden_states": torch.stack(hidden_states_list) if hidden_states_list else torch.zeros(0, W.shape[1]),
        "T": T,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Latent-SFT(2) model
    model_path = os.path.join(project_root, 'Latent-SFT', 'checkpoints', 'latent-2')
    print(f"Loading Latent-SFT(2) from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation='sdpa',
        torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    model.tokenizer = tokenizer
    model.latent_token_ids = tokenizer(['<think>', '</think>'], add_special_tokens=False)['input_ids']
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Load confidence head (trained on latent-4 data, but architecturally compatible)
    head_path = os.path.join(project_root, 'experiments', 'confidence_head',
                            'trained_heads', 'confidence_head_attention.pt')
    print(f"Loading confidence head from {head_path}...")
    head = load_head(head_path, hidden_dim=2048, device=args.device)

    # Load test data
    data_path = os.path.join(project_root, 'Latent-SFT', 'data', 'GSM8k-Aug-test.jsonl')
    print(f"Loading data from {data_path}...")
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"  Loaded {len(data)} problems")

    # Collect rollouts
    per_problem_results = []
    all_correct = 0
    all_total = 0

    for pi, example in enumerate(tqdm(data, desc=f"LSFT(2) ε={args.eps} N={args.N}")):
        gt_answer = str(example['answer']).strip()

        # Format input for Llama
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids = tokenizer(input_prefix, truncation=False, padding=False,
                             add_special_tokens=False, return_attention_mask=False)['input_ids']

        text_input = {
            'input_ids': torch.tensor(input_ids + model.latent_token_ids[0], dtype=torch.long).to(device).unsqueeze(0),
            'attention_mask': torch.tensor([1] * (len(input_ids) + len(model.latent_token_ids[0])), dtype=torch.long).to(device).unsqueeze(0),
        }

        # Deterministic rollout (no noise)
        det_out = generate_with_noise(model, text_input, eps=0, seed=777)
        det_pred = extract_answer(det_out['text'])
        det_correct = check_is_correct(det_pred, gt_answer)
        det_conf = head_predict(head, det_out['hidden_states'], det_out['T'], device=args.device)

        # N rollouts with noise
        rollouts = []
        for n in range(args.N):
            seed = 42 * 10000 + pi * 100 + n
            out = generate_with_noise(model, text_input, eps=args.eps, seed=seed)
            pred = extract_answer(out['text'])
            correct = check_is_correct(pred, gt_answer)
            conf = head_predict(head, out['hidden_states'], out['T'], device=args.device)
            rollouts.append({
                'correct': correct,
                'answer': pred,
                'T': out['T'],
                'confidence': conf,
            })
            all_correct += int(correct)
            all_total += 1

        per_problem_results.append({
            'idx': pi,
            'gt': gt_answer,
            'det_correct': det_correct,
            'det_conf': det_conf,
            'det_T': det_out['T'],
            'rollouts': rollouts,
        })

        if (pi + 1) % 100 == 0:
            det_acc = sum(p['det_correct'] for p in per_problem_results) / len(per_problem_results)
            mean_acc = all_correct / all_total
            print(f"  [{pi+1}/{len(data)}] det_acc={det_acc:.4f}, rollout_acc={mean_acc:.4f}")

    # Compute statistics
    det_acc = sum(p['det_correct'] for p in per_problem_results) / len(per_problem_results)
    oracle_correct = sum(1 for p in per_problem_results if any(r['correct'] for r in p['rollouts']))
    oracle_acc = oracle_correct / len(per_problem_results)
    mean_rollout_acc = all_correct / all_total

    # Within-problem AUC
    within_aucs = []
    n_mixed = 0
    for p in per_problem_results:
        labels = [int(r['correct']) for r in p['rollouts']]
        scores = [r['confidence'] for r in p['rollouts']]
        if len(set(labels)) == 2:
            n_mixed += 1
            auc = roc_auc_score(labels, scores)
            if not np.isnan(auc):
                within_aucs.append(auc)

    within_problem_auc = np.mean(within_aucs) if within_aucs else float('nan')

    # Head as selector
    head_correct = 0
    for p in per_problem_results:
        best_idx = max(range(len(p['rollouts'])), key=lambda i: p['rollouts'][i]['confidence'])
        head_correct += p['rollouts'][best_idx]['correct']
    head_acc = head_correct / len(per_problem_results)

    # Pooled AUC
    all_labels = []
    all_scores = []
    for p in per_problem_results:
        for r in p['rollouts']:
            all_labels.append(int(r['correct']))
            all_scores.append(r['confidence'])
    pooled_auc = roc_auc_score(all_labels, all_scores)

    # Endpoint dominance: last-1 step AUC vs full trajectory AUC
    # (would need last-1 hidden states — skip for now, use full T info)

    results = {
        'config': {
            'model': 'Latent-SFT-2',
            'eps': args.eps,
            'N': args.N,
            'n_problems': len(data),
        },
        'det_acc': det_acc,
        'oracle_acc': oracle_acc,
        'oracle_delta': oracle_acc - det_acc,
        'mean_rollout_acc': mean_rollout_acc,
        'head_acc': head_acc,
        'head_delta': head_acc - det_acc,
        'within_problem_auc': within_problem_auc,
        'pooled_auc': pooled_auc,
        'n_mixed_problems': n_mixed,
        'mean_T': np.mean([p['det_T'] for p in per_problem_results]),
    }

    print(f"\n{'='*60}")
    print(f"Latent-SFT(2) ε={args.eps} Results")
    print(f"{'='*60}")
    print(f"  Det acc:            {det_acc:.4f}")
    print(f"  Oracle acc:         {oracle_acc:.4f} (Δ={oracle_acc-det_acc:.4f})")
    print(f"  Mean rollout acc:   {mean_rollout_acc:.4f}")
    print(f"  Head selector acc:  {head_acc:.4f} (Δ={head_acc-det_acc:.4f})")
    print(f"  Pooled AUC:         {pooled_auc:.4f}")
    print(f"  Within-problem AUC: {within_problem_auc:.4f}")
    print(f"  Mixed problems:     {n_mixed}")
    print(f"  Mean T:             {results['mean_T']:.1f}")

    # Save results
    out_path = os.path.join(args.output_dir, 'lsft2_rollouts_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save per-problem data
    per_problem_path = os.path.join(args.output_dir, 'lsft2_per_problem.json')
    with open(per_problem_path, 'w') as f:
        json.dump(per_problem_results, f)
    print(f"Saved: {per_problem_path}")


if __name__ == "__main__":
    main()
