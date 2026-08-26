"""
Direction 1 Experiment A: Full 1319-problem oracle ceiling + head-as-selector
on Latent-SFT with eps=0.001 embedding noise, N=8 rollouts.
"""
import os, sys, json, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Latent-SFT'))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from train_confidence_head import AttentionConfidenceHead


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


def check_correct(pred, gt):
    pred = pred.strip().replace(",", "").replace("$", "").replace("%", "")
    gt = str(gt).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except (ValueError, TypeError):
        pass
    return pred.strip() == gt.strip()


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
    model.load_state_dict(ckpt['model_state_dict'])
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


def main():
    device = torch.device('cuda:0')
    eps = 0.001
    N = 8

    # Load model
    model_path = '../../Latent-SFT/checkpoints/latent-4'
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation='sdpa', torch_dtype=torch.bfloat16,
        use_cache=False, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    model.tokenizer = tokenizer
    model.latent_token_ids = tokenizer(['<think>', '</think>'], add_special_tokens=False)['input_ids']
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    W = model.model.embed_tokens.weight.detach()

    # Load head
    head = load_head('./trained_heads/confidence_head_attention.pt',
                     hidden_dim=model.config.hidden_size, device=str(device))

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '../../Latent-SFT/data/GSM8k-Aug-test.jsonl')
    with open(data_path) as f:
        problems = [json.loads(line) for line in f]
    print(f"Running {len(problems)} problems, N={N}, eps={eps}")

    results = []
    for pi, example in enumerate(tqdm(problems, desc="LSFT full")):
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        iids_raw = tokenizer(input_prefix, truncation=False, padding=False,
                            add_special_tokens=False, return_attention_mask=False)['input_ids']
        text_input = {
            'input_ids': torch.tensor(iids_raw + model.latent_token_ids[0], dtype=torch.long).to(device).unsqueeze(0),
            'attention_mask': torch.tensor([1] * (len(iids_raw) + len(model.latent_token_ids[0])), dtype=torch.long).to(device).unsqueeze(0),
        }

        think_ids = torch.LongTensor(model.latent_token_ids[1]).unsqueeze(0).to(device)
        think_end_emb = model.model.embed_tokens(think_ids)
        input_ids_orig = text_input['input_ids']

        # --- Deterministic baseline ---
        with torch.no_grad():
            iids = text_input['input_ids'].clone()
            attn = text_input['attention_mask'].clone()
            past_kv = None
            latent_embeds_det = []
            hidden_det = []
            for step in range(50):
                if iids is not None:
                    emb = model.model.embed_tokens(iids)
                out = model(inputs_embeds=emb, attention_mask=attn, past_key_values=past_kv,
                           use_cache=True, output_hidden_states=True)
                hidden_det.append(out.hidden_states[-1][:, -1, :].squeeze(0).cpu())
                logits = out.logits[:, -1, :]
                past_kv = out.past_key_values
                nt = torch.argmax(logits, dim=-1, keepdim=True)
                probs = F.softmax(logits.float(), dim=-1)
                z = (probs.to(W.dtype) @ W)
                emb = z.unsqueeze(1)
                latent_embeds_det.append(emb)
                attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
                if nt[0, 0].item() == model.latent_token_ids[1][0]:
                    break
                else:
                    iids = None

            latent_state_det = torch.cat(latent_embeds_det, dim=1)
            T_det = latent_state_det.shape[1]
            hidden_det_tensor = torch.stack(hidden_det)

            attn_len = input_ids_orig.size(1) + latent_state_det.size(1) + len(model.latent_token_ids[1])
            attn_full = torch.ones(1, attn_len, dtype=torch.long, device=device)
            emb_full = torch.cat([model.model.embed_tokens(input_ids_orig), latent_state_det, think_end_emb], dim=1)

            torch.manual_seed(0)
            gen_out = model.generate(inputs_embeds=emb_full, attention_mask=attn_full,
                                    max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95)
            text = tokenizer.decode(gen_out[0], skip_special_tokens=False)
            det_ans = extract_answer(text)
            det_correct = check_correct(det_ans, example['answer'])
            det_conf = head_predict(head, hidden_det_tensor, T_det, device=str(device))

        # --- N stochastic rollouts ---
        rollouts = []
        for n in range(N):
            seed = n * 10000 + pi
            torch.manual_seed(seed)
            with torch.no_grad():
                iids = text_input['input_ids'].clone()
                attn = text_input['attention_mask'].clone()
                past_kv = None
                latent_embeds = []
                hidden_list = []
                for step in range(50):
                    if iids is not None:
                        emb = model.model.embed_tokens(iids)
                    out = model(inputs_embeds=emb, attention_mask=attn, past_key_values=past_kv,
                               use_cache=True, output_hidden_states=True)
                    hidden_list.append(out.hidden_states[-1][:, -1, :].squeeze(0).cpu())
                    logits = out.logits[:, -1, :]
                    past_kv = out.past_key_values
                    nt = torch.argmax(logits, dim=-1, keepdim=True)
                    probs = F.softmax(logits.float(), dim=-1)
                    z = (probs.to(W.dtype) @ W)
                    noise = eps * z.norm() * torch.randn_like(z)
                    z = z + noise
                    emb = z.unsqueeze(1)
                    latent_embeds.append(emb)
                    attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
                    if nt[0, 0].item() == model.latent_token_ids[1][0]:
                        break
                    else:
                        iids = None

                latent_state = torch.cat(latent_embeds, dim=1)
                T_roll = latent_state.shape[1]
                hidden_tensor = torch.stack(hidden_list)

                attn_len = input_ids_orig.size(1) + latent_state.size(1) + len(model.latent_token_ids[1])
                attn_full = torch.ones(1, attn_len, dtype=torch.long, device=device)
                emb_full = torch.cat([model.model.embed_tokens(input_ids_orig), latent_state, think_end_emb], dim=1)

                gen_out = model.generate(inputs_embeds=emb_full, attention_mask=attn_full,
                                        max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95)
                text = tokenizer.decode(gen_out[0], skip_special_tokens=False)
                ans = extract_answer(text)
                correct = check_correct(ans, example['answer'])
                conf = head_predict(head, hidden_tensor, T_roll, device=str(device))

            rollouts.append({
                "correct": bool(correct), "answer": ans,
                "T": T_roll, "confidence": conf,
            })

        results.append({
            "idx": pi,
            "gt": str(example["answer"]),
            "det_correct": bool(det_correct),
            "det_conf": det_conf,
            "rollouts": rollouts,
        })

        if (pi + 1) % 100 == 0:
            det_so_far = np.mean([r["det_correct"] for r in results])
            oracle_so_far = np.mean([any(ro["correct"] for ro in r["rollouts"]) for r in results])
            print(f"  [{pi+1}] det={det_so_far:.4f}, oracle={oracle_so_far:.4f}, Δ={oracle_so_far-det_so_far:+.4f}")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("LATENT-SFT FULL (1319 problems, eps=0.001, N=8)")
    print("=" * 70)

    det_acc = np.mean([r["det_correct"] for r in results])
    oracle_acc = np.mean([any(ro["correct"] for ro in r["rollouts"]) for r in results])
    mean_rollout_acc = np.mean([np.mean([ro["correct"] for ro in r["rollouts"]]) for r in results])

    # Selectors
    head_selected = []
    sc_selected = []
    random_selected = []
    for r in results:
        rollouts = r["rollouts"]
        # Head selector
        best_idx = max(range(N), key=lambda k: rollouts[k]["confidence"])
        head_selected.append(rollouts[best_idx]["correct"])
        # Self-consistency
        answers = [ro["answer"] for ro in rollouts]
        most_common = max(set(answers), key=answers.count) if answers else ""
        sc_correct = check_correct(most_common, r["gt"])
        sc_selected.append(bool(sc_correct))
        # Random
        rng = np.random.default_rng(r["idx"])
        random_selected.append(rollouts[rng.integers(N)]["correct"])

    head_acc = np.mean(head_selected)
    sc_acc = np.mean(sc_selected)
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

    print(f"\n  Deterministic acc:      {det_acc:.4f}")
    print(f"  Mean rollout acc:       {mean_rollout_acc:.4f}")
    print(f"  Oracle (any correct):   {oracle_acc:.4f}  (Δ vs det: {oracle_acc-det_acc:+.4f})")
    print(f"  Random selector:        {random_acc:.4f}  (Δ vs det: {random_acc-det_acc:+.4f})")
    print(f"  Self-consistency:       {sc_acc:.4f}  (Δ vs det: {sc_acc-det_acc:+.4f})")
    print(f"  Head selector:          {head_acc:.4f}  (Δ vs det: {head_acc-det_acc:+.4f})")
    print(f"\n  Within-problem AUC:     {np.mean(wp_aucs):.4f} ({len(wp_aucs)} problems with mixed outcomes)")
    print(f"  Pooled AUC:             {roc_auc_score([float(ro['correct']) for r in results for ro in r['rollouts']], [ro['confidence'] for r in results for ro in r['rollouts']]):.4f}")

    # Save
    os.makedirs('./direction1_results', exist_ok=True)
    output = {
        "config": {"eps": eps, "N": N, "n_problems": len(results), "paradigm": "Latent-SFT-4"},
        "det_acc": float(det_acc),
        "oracle_acc": float(oracle_acc),
        "oracle_delta": float(oracle_acc - det_acc),
        "mean_rollout_acc": float(mean_rollout_acc),
        "head_acc": float(head_acc),
        "sc_acc": float(sc_acc),
        "random_acc": float(random_acc),
        "within_problem_auc": float(np.mean(wp_aucs)) if wp_aucs else None,
        "n_mixed_problems": len(wp_aucs),
    }
    with open('./direction1_results/lsft_full_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    # Also save per-problem data
    with open('./direction1_results/lsft_full_per_problem.json', 'w') as f:
        json.dump(results, f)
    print(f"\nResults saved to direction1_results/lsft_full_results.json")


if __name__ == "__main__":
    main()
