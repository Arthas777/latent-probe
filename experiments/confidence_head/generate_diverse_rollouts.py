"""
Direction 1 Step 1-4: Generate diverse rollouts via Gumbel-Softmax injection
in the latent stage, then evaluate confidence head on the diverse trajectories.

For each problem, generates N rollouts with stochastic latent decoding,
dumps: hidden states (endpoint), confidence scores, correctness labels.

Usage:
    python generate_diverse_rollouts.py \
        --latent_model_path ../../Latent-SFT/checkpoints/latent-4 \
        --head_path ./trained_heads/confidence_head_attention.pt \
        --N 8 --tau 1.0 --seed 42 \
        --output_dir ./direction1_results
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
latent_sft_dir = os.path.join(current_dir, '../../Latent-SFT')
sys.path.append(latent_sft_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from train_confidence_head import AttentionConfidenceHead, TrajectoryDataset


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
def generate_stochastic_rollout(model, inputs, max_new_tokens=128,
                                 tau=1.0, answer_temperature=0.6,
                                 answer_top_p=0.95, seed=None):
    """
    Run Latent-SFT inference with Gumbel-Softmax noise in latent stage.

    Key difference from deterministic: instead of probs = softmax(logits),
    we use probs = gumbel_softmax(logits, tau) to inject diversity while
    staying close to the training distribution (Latent-SFT trains with
    soft mixing via softmax, so Gumbel at tau=1 is a natural extension).
    """
    if seed is not None:
        torch.manual_seed(seed)

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

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        hidden_states_list.append(last_hidden.squeeze(0).cpu())

        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # === DIVERSITY INJECTION ===
        # Gumbel-Softmax: adds Gumbel noise before softmax
        # At tau=1.0 this matches the scale of training distribution
        g = -torch.log(-torch.log(torch.rand_like(next_token_logits) + 1e-20) + 1e-20)
        probs = F.softmax((next_token_logits.float() + g) / tau, dim=-1)

        p = probs[0]
        log_p = torch.log(p + 1e-12)
        h = -(p * log_p).sum().item()
        token_entropies.append(h)
        top1_probs.append(p.max().item())
        latent_probs_list.append(p.cpu())

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

    # Answer phase (unchanged — uses model's own sampling)
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
        temperature=answer_temperature,
        top_p=answer_top_p,
    )

    decoded_text = model.tokenizer.decode(generated_output[0], skip_special_tokens=False)

    return {
        "text": decoded_text,
        "hidden_states": torch.stack(hidden_states_list) if hidden_states_list else torch.zeros(0, W.shape[1]),
        "token_entropies": torch.tensor(token_entropies, dtype=torch.float32),
        "top1_probs": torch.tensor(top1_probs, dtype=torch.float32),
        "T": T,
    }


@torch.no_grad()
def generate_deterministic_rollout(model, inputs, max_new_tokens=128,
                                    answer_temperature=0.6, answer_top_p=0.95):
    """Deterministic baseline: standard softmax, no Gumbel noise."""
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids = []
    past_key_values = None

    hidden_states_list = []
    token_entropies = []
    top1_probs = []
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

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        hidden_states_list.append(last_hidden.squeeze(0).cpu())

        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        probs = F.softmax(next_token_logits.float(), dim=-1)
        p = probs[0]
        log_p = torch.log(p + 1e-12)
        h = -(p * log_p).sum().item()
        token_entropies.append(h)
        top1_probs.append(p.max().item())

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
        temperature=answer_temperature,
        top_p=answer_top_p,
    )

    decoded_text = model.tokenizer.decode(generated_output[0], skip_special_tokens=False)

    return {
        "text": decoded_text,
        "hidden_states": torch.stack(hidden_states_list) if hidden_states_list else torch.zeros(0, W.shape[1]),
        "token_entropies": torch.tensor(token_entropies, dtype=torch.float32),
        "top1_probs": torch.tensor(top1_probs, dtype=torch.float32),
        "T": T,
    }


def compute_spectral_features(token_entropies, T):
    """Compute LENS score from token entropies."""
    if T < 2:
        return 0.0
    mean_ent = token_entropies.mean().item()
    return mean_ent


def load_confidence_head(head_path, hidden_dim=2048, proj_dim=128, device='cuda:0'):
    """Load trained attention confidence head."""
    model = AttentionConfidenceHead(
        hidden_dim=hidden_dim, proj_dim=proj_dim, n_heads=4, dropout=0.1
    )
    checkpoint = torch.load(head_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    return model


def head_predict(head, hidden_states, T, max_T=64, device='cuda:0'):
    """Get confidence prediction from the attention head."""
    h = hidden_states.to(device).float()  # [T, hidden_dim]
    if h.shape[0] > max_T:
        h = h[-max_T:]
    T_actual = h.shape[0]
    # Pad to max_T
    if T_actual < max_T:
        pad = torch.zeros(max_T - T_actual, h.shape[1], device=device)
        h = torch.cat([h, pad], dim=0)
    mask = torch.zeros(max_T, dtype=torch.bool, device=device)
    mask[:T_actual] = True

    h = h.unsqueeze(0)  # [1, max_T, hidden_dim]
    mask = mask.unsqueeze(0)  # [1, max_T]
    T_tensor = torch.tensor([T_actual], dtype=torch.float32, device=device)

    logit = head(h, mask, T_tensor)
    prob = torch.sigmoid(logit).item()
    return prob


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def roc_auc_score(y_true, y_score):
    """Manual AUC implementation."""
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
    auc = np.trapz(tpr, fpr)
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_model_path', type=str, default='../../Latent-SFT/checkpoints/latent-4')
    parser.add_argument('--head_path', type=str, default='./trained_heads/confidence_head_attention.pt')
    parser.add_argument('--dataset', type=str, default='GSM8k')
    parser.add_argument('--N', type=int, default=8, help='Number of diverse rollouts per problem')
    parser.add_argument('--tau', type=float, default=1.0, help='Gumbel-Softmax temperature')
    parser.add_argument('--seed', type=int, default=42, help='Base seed')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--max_problems', type=int, default=None, help='Limit problems for sanity check')
    parser.add_argument('--output_dir', type=str, default='./direction1_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_deterministic', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    print(f"Loading Latent-SFT model from {args.latent_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.latent_model_path,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
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

    # Load confidence head
    print(f"Loading confidence head from {args.head_path}...")
    config = AutoConfig.from_pretrained(args.latent_model_path)
    head = load_confidence_head(args.head_path, hidden_dim=config.hidden_size, device=args.device)

    # Load data
    data_dir = os.path.join(latent_sft_dir, 'data')
    data_map = {
        'GSM8k': 'GSM8k-Aug-test.jsonl',
        'SVAMP': 'SVAMP-test.jsonl',
    }
    data_path = os.path.join(data_dir, data_map.get(args.dataset, f'{args.dataset}-test.jsonl'))
    print(f"Loading data from {data_path}...")
    data = read_jsonl(data_path)

    if args.max_problems is not None:
        data = data[:args.max_problems]
    print(f"Running on {len(data)} problems, N={args.N} rollouts each, tau={args.tau}")

    all_results = []

    for i, example in enumerate(tqdm(data, desc="Generating rollouts")):
        # Format input
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        input_ids = tokenizer(input_prefix, truncation=False, padding=False,
                             add_special_tokens=False, return_attention_mask=False)['input_ids']

        text_input = {
            'input_ids': torch.tensor(input_ids + model.latent_token_ids[0], dtype=torch.long).to(device).unsqueeze(0),
            'attention_mask': torch.tensor([1] * (len(input_ids) + len(model.latent_token_ids[0])), dtype=torch.long).to(device).unsqueeze(0),
        }

        problem_result = {
            "idx": i,
            "gt_answer": str(example["answer"]),
            "rollouts": [],
            "deterministic": None,
        }

        # Deterministic baseline
        if not args.skip_deterministic:
            det_out = generate_deterministic_rollout(
                model, text_input, max_new_tokens=args.max_new_tokens
            )
            det_pred = extract_answer(det_out["text"])
            det_correct = check_is_correct(det_pred, example["answer"])
            det_conf = head_predict(head, det_out["hidden_states"], det_out["T"], device=args.device)
            problem_result["deterministic"] = {
                "correct": bool(det_correct),
                "answer": det_pred,
                "T": det_out["T"],
                "confidence": det_conf,
            }

        # Stochastic rollouts
        for n in range(args.N):
            rollout_seed = args.seed * 10000 + i * 100 + n
            out = generate_stochastic_rollout(
                model, text_input, max_new_tokens=args.max_new_tokens,
                tau=args.tau, seed=rollout_seed,
            )
            pred = extract_answer(out["text"])
            correct = check_is_correct(pred, example["answer"])
            conf = head_predict(head, out["hidden_states"], out["T"], device=args.device)

            problem_result["rollouts"].append({
                "seed": rollout_seed,
                "correct": bool(correct),
                "answer": pred,
                "T": out["T"],
                "confidence": conf,
                "mean_entropy": out["token_entropies"].mean().item() if len(out["token_entropies"]) > 0 else 0.0,
            })

        all_results.append(problem_result)

        # Progress report every 50 problems
        if (i + 1) % 50 == 0:
            n_done = i + 1
            det_acc = sum(1 for r in all_results if r["deterministic"] and r["deterministic"]["correct"]) / n_done if not args.skip_deterministic else 0
            stoch_accs = [sum(1 for ro in r["rollouts"] if ro["correct"]) / len(r["rollouts"]) for r in all_results]
            mean_stoch_acc = np.mean(stoch_accs)
            print(f"\n  [{n_done}/{len(data)}] Det acc: {det_acc:.4f}, Mean stochastic acc (per-problem): {mean_stoch_acc:.4f}")

    # Save raw results
    output_file = os.path.join(args.output_dir, f"rollouts_N{args.N}_tau{args.tau}_seed{args.seed}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} problems × {args.N} rollouts to {output_file}")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("DIRECTION 1 ANALYSIS")
    print("=" * 70)

    # Step 2: Diversity sanity check
    print("\n--- Step 2: Diversity Sanity Check ---")
    answer_agreement_rates = []
    T_stds = []
    conf_stds = []
    for r in all_results:
        answers = [ro["answer"] for ro in r["rollouts"]]
        most_common = max(set(answers), key=answers.count)
        agreement = answers.count(most_common) / len(answers)
        answer_agreement_rates.append(agreement)
        Ts = [ro["T"] for ro in r["rollouts"]]
        T_stds.append(np.std(Ts))
        confs = [ro["confidence"] for ro in r["rollouts"]]
        conf_stds.append(np.std(confs))

    print(f"  Top-1 answer agreement rate: {np.mean(answer_agreement_rates):.4f} (target: ≤ 0.85)")
    print(f"  Mean T std across rollouts: {np.mean(T_stds):.4f}")
    print(f"  Mean confidence std across rollouts: {np.mean(conf_stds):.4f} (target: ≥ 0.05)")

    # Step 3: Single-rollout accuracy
    print("\n--- Step 3: Single-Rollout Accuracy ---")
    single_rollout_accs = [r["rollouts"][0]["correct"] for r in all_results]
    single_acc = np.mean(single_rollout_accs)
    det_acc = np.mean([r["deterministic"]["correct"] for r in all_results]) if not args.skip_deterministic else 0
    print(f"  Deterministic accuracy: {det_acc:.4f}")
    print(f"  Single stochastic rollout accuracy: {single_acc:.4f}")
    print(f"  Delta: {single_acc - det_acc:+.4f} (target: ≥ -0.03)")

    # Step 4: Head OOD diagnostic
    print("\n--- Step 4: Head OOD Diagnostic ---")
    all_confs = []
    all_labels = []
    within_problem_aucs = []
    for r in all_results:
        problem_confs = []
        problem_labels = []
        for ro in r["rollouts"]:
            all_confs.append(ro["confidence"])
            all_labels.append(float(ro["correct"]))
            problem_confs.append(ro["confidence"])
            problem_labels.append(float(ro["correct"]))
        # Within-problem AUC (only if there's both correct and incorrect)
        if len(set(problem_labels)) == 2:
            wp_auc = roc_auc_score(problem_labels, problem_confs)
            if not np.isnan(wp_auc):
                within_problem_aucs.append(wp_auc)

    pooled_auc = roc_auc_score(all_labels, all_confs)
    mean_within_auc = np.mean(within_problem_aucs) if within_problem_aucs else float('nan')
    print(f"  Pooled AUC (all rollouts): {pooled_auc:.4f} (target: ≥ 0.75)")
    print(f"  Within-problem AUC (mean): {mean_within_auc:.4f} (target: ≥ 0.65)")
    print(f"  Problems with mixed outcomes: {len(within_problem_aucs)}/{len(all_results)}")

    # Step 5: Best-of-N selection
    print("\n--- Step 5: Best-of-N Selection ---")
    selectors = {
        "deterministic": [],
        "random_pick": [],
        "self_consistency": [],
        "last_token_conf": [],  # using mean_entropy as proxy
        "trained_head": [],
        "oracle": [],
    }

    for r in all_results:
        rollouts = r["rollouts"]
        N_actual = len(rollouts)

        # Deterministic
        if r["deterministic"]:
            selectors["deterministic"].append(r["deterministic"]["correct"])

        # Random pick
        rng = np.random.default_rng(r["idx"])
        random_idx = rng.integers(0, N_actual)
        selectors["random_pick"].append(rollouts[random_idx]["correct"])

        # Self-consistency (majority vote on answer)
        answers = [ro["answer"] for ro in rollouts]
        if answers:
            most_common_answer = max(set(answers), key=answers.count)
            sc_correct = check_is_correct(most_common_answer, r["gt_answer"])
            selectors["self_consistency"].append(bool(sc_correct))

        # Mean entropy selector (lower entropy = higher confidence, proxy for last-token)
        ent_idx = min(range(N_actual), key=lambda k: rollouts[k]["mean_entropy"])
        selectors["last_token_conf"].append(rollouts[ent_idx]["correct"])

        # Trained head selector (highest confidence)
        head_idx = max(range(N_actual), key=lambda k: rollouts[k]["confidence"])
        selectors["trained_head"].append(rollouts[head_idx]["correct"])

        # Oracle (any correct rollout exists)
        selectors["oracle"].append(any(ro["correct"] for ro in rollouts))

    print(f"\n  {'Selector':<25} {'Accuracy':>10} {'Δ vs det':>10}")
    print("  " + "-" * 47)
    det_baseline = np.mean(selectors["deterministic"]) if selectors["deterministic"] else 0
    for name, results in selectors.items():
        if results:
            acc = np.mean(results)
            delta = acc - det_baseline
            marker = " ✓" if delta >= 0.03 else ""
            print(f"  {name:<25} {acc:>10.4f} {delta:>+10.4f}{marker}")

    # Go/No-Go summary
    print("\n" + "=" * 70)
    print("GO / NO-GO DECISION")
    print("=" * 70)
    head_acc = np.mean(selectors["trained_head"])
    sc_acc = np.mean(selectors["self_consistency"]) if selectors["self_consistency"] else 0
    head_delta = head_acc - det_baseline
    head_vs_sc = head_acc - sc_acc

    print(f"  Trained head Δ vs deterministic: {head_delta:+.4f} ({head_delta*100:+.1f}pp)")
    print(f"  Trained head vs self-consistency: {head_vs_sc:+.4f} ({head_vs_sc*100:+.1f}pp)")

    if head_delta >= 0.04 and head_vs_sc > 0:
        print("  → Situation A: GO — Head is clearly the best selector")
    elif 0.02 <= head_delta < 0.04:
        print("  → Situation B: CONDITIONAL GO — Head works but not dominating SC")
    elif head_delta >= 0.04 and head_vs_sc <= 0:
        print("  → Situation C: PIVOT — Diversity helps but SC suffices as selector")
    else:
        print("  → Situation D: NO-GO — Insufficient improvement from diversity + head")

    # Save analysis
    analysis = {
        "config": {
            "N": args.N, "tau": args.tau, "seed": args.seed,
            "n_problems": len(all_results),
            "model": args.latent_model_path,
        },
        "step2_diversity": {
            "answer_agreement_rate": float(np.mean(answer_agreement_rates)),
            "mean_T_std": float(np.mean(T_stds)),
            "mean_conf_std": float(np.mean(conf_stds)),
            "pass": np.mean(answer_agreement_rates) <= 0.85,
        },
        "step3_single_rollout": {
            "deterministic_acc": float(det_acc),
            "single_stochastic_acc": float(single_acc),
            "delta": float(single_acc - det_acc),
            "pass": (single_acc - det_acc) >= -0.03,
        },
        "step4_head_ood": {
            "pooled_auc": float(pooled_auc),
            "within_problem_auc": float(mean_within_auc),
            "n_mixed_problems": len(within_problem_aucs),
            "pass_pooled": pooled_auc >= 0.75,
            "pass_within": mean_within_auc >= 0.65 if not np.isnan(mean_within_auc) else False,
        },
        "step5_selection": {k: float(np.mean(v)) for k, v in selectors.items() if v},
        "decision": {
            "head_delta_vs_det": float(head_delta),
            "head_vs_sc": float(head_vs_sc),
        },
    }
    analysis_file = os.path.join(args.output_dir, f"analysis_N{args.N}_tau{args.tau}_seed{args.seed}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {analysis_file}")


if __name__ == "__main__":
    main()
