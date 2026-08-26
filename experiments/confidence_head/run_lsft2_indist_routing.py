"""
In-distribution routing experiment: train probe on LSFT(2) trajectories, then route on LSFT(2).

Steps:
  1. Collect LSFT(2) trajectories with hidden states
  2. Train AttentionConfidenceHead on LSFT(2) data (nested CV)
  3. Run routing using in-distribution probe
  4. Compare with cross-r probe (trained on r=4, applied to r=2)

Usage: python run_lsft2_indist_routing.py [--device cuda:0]
"""
import os, sys, json, torch, argparse, numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
lsft_root = os.path.join(project_root, 'Latent-SFT')
sys.path.insert(0, lsft_root)
sys.path.insert(0, os.path.join(lsft_root, 'eval'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer

sys.path.insert(0, os.path.dirname(__file__))
from train_confidence_head import AttentionConfidenceHead

LATENT2_PATH = os.path.join(lsft_root, 'checkpoints/latent-2')
COT_SFT_PATH = os.path.join(lsft_root, 'checkpoints/cot-sft')
DATA_PATH = os.path.join(lsft_root, 'data/GSM8k-Aug-test.jsonl')
RESULTS_DIR = './routing_results_lsft2_indist'


def collect_lsft2_trajectories(device):
    """Collect LSFT(2) trajectories with hidden states using single-pass inference."""
    cache_path = os.path.join(RESULTS_DIR, 'lsft2_trajectories.pt')
    if os.path.exists(cache_path):
        print(f"Loading cached LSFT(2) trajectories from {cache_path}")
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    print(f"Collecting LSFT(2) trajectories from {LATENT2_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        LATENT2_PATH, attn_implementation='sdpa',
        torch_dtype=torch.bfloat16, use_cache=True, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(LATENT2_PATH)
    model.eval()
    model.latent_token_ids = tokenizer(['<think>', '</think>'], add_special_tokens=False)['input_ids']
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    W = model.model.embed_tokens.weight.detach()

    with open(DATA_PATH) as f:
        problems = [json.loads(l) for l in f]

    results = []
    for pi, example in enumerate(tqdm(problems, desc="Collecting LSFT(2) trajectories")):
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids_raw = tokenizer(input_prefix, truncation=False, padding=False,
                                  add_special_tokens=False, return_attention_mask=False)['input_ids']
        text_input = {
            'input_ids': torch.tensor(input_ids_raw + model.latent_token_ids[0], dtype=torch.long).to(device).unsqueeze(0),
            'attention_mask': torch.tensor([1] * (len(input_ids_raw) + len(model.latent_token_ids[0])), dtype=torch.long).to(device).unsqueeze(0),
        }

        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
        past_key_values = None
        hidden_list = []
        generated_ids = []
        latent_states = []

        with torch.no_grad():
            for latent_step in range(128):
                if input_ids is not None:
                    input_embeddings = model.model.embed_tokens(input_ids)

                outputs = model(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                hidden_list.append(outputs.hidden_states[-1][:, -1, :].squeeze(0).cpu())
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                probs = torch.softmax(next_token_logits, dim=-1).to(W.dtype)
                input_embeddings = (probs @ W).unsqueeze(1)
                latent_states.append(input_embeddings)

                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                ], dim=1)

                if next_token[0, 0].item() == model.latent_token_ids[1][0]:
                    input_ids = next_token
                    break
                else:
                    input_ids = None
                    generated_ids.append(int(next_token.item()))

            T = len(generated_ids)

            # Generate answer using accumulated latent states
            latent_state = torch.cat(latent_states, dim=1)
            think_ids = torch.LongTensor(model.latent_token_ids[1]).unsqueeze(0).to(device)
            think_end_embeddings = model.model.embed_tokens(think_ids)
            input_ids_orig = text_input['input_ids']
            attn_len = input_ids_orig.size(1) + latent_state.size(1) + len(model.latent_token_ids[1])
            attn_full = torch.ones(1, attn_len, dtype=torch.long, device=device)
            emb_full = torch.cat([model.model.embed_tokens(input_ids_orig), latent_state, think_end_embeddings], dim=1)

            gen_out = model.generate(
                inputs_embeds=emb_full, attention_mask=attn_full,
                max_new_tokens=128, do_sample=False,
            )
            text = tokenizer.decode(gen_out[0], skip_special_tokens=False)

        pred = extract_answer(text)
        correct = check_is_correct(pred, example['answer'])
        hidden_tensor = torch.stack(hidden_list[:T]) if T > 0 else torch.zeros(1, 2048)

        results.append({
            'idx': pi,
            'correct': bool(correct),
            'T': T,
            'hidden_states': hidden_tensor.half(),
        })

        if (pi + 1) % 100 == 0:
            acc = np.mean([r['correct'] for r in results])
            avg_T = np.mean([r['T'] for r in results])
            print(f"  [{pi+1}/{len(problems)}] acc={acc:.4f}, avg_T={avg_T:.1f}")

    del model
    torch.cuda.empty_cache()

    torch.save(results, cache_path)
    print(f"Saved {len(results)} trajectories to {cache_path}")
    return results


def train_probe_nested_cv(trajectories, device, n_outer=5, n_seeds=3, max_T=64):
    """Train AttentionConfidenceHead with nested CV on LSFT(2) data."""
    N = len(trajectories)
    hidden_dim = trajectories[0]['hidden_states'].shape[-1]

    # Prepare data
    labels = torch.tensor([t['correct'] for t in trajectories], dtype=torch.float32)
    Ts = torch.tensor([t['T'] for t in trajectories], dtype=torch.float32)

    # Pad hidden states
    H_padded = torch.zeros(N, max_T, hidden_dim)
    masks = torch.zeros(N, max_T, dtype=torch.bool)
    for i, t in enumerate(trajectories):
        hs = t['hidden_states'].float()
        T_i = min(hs.shape[0], max_T)
        H_padded[i, :T_i] = hs[:T_i]
        masks[i, :T_i] = True

    all_aucs = []
    all_confs = np.zeros(N)
    conf_counts = np.zeros(N)

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42 + seed)
        fold_aucs = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), labels.numpy())):
            # Inner split for early stopping
            inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed * 100 + fold_idx)
            inner_train_idx, val_idx = next(inner_skf.split(train_idx, labels[train_idx].numpy()))
            inner_train_idx = train_idx[inner_train_idx]
            val_idx = train_idx[val_idx]

            # Build model
            head = AttentionConfidenceHead(hidden_dim=hidden_dim, proj_dim=128).to(device)
            optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0.01)
            criterion = torch.nn.BCEWithLogitsLoss()

            # Training
            best_val_auc = 0
            best_state = None
            patience_count = 0
            batch_size = 64

            for epoch in range(100):
                head.train()
                perm = np.random.permutation(len(inner_train_idx))
                epoch_loss = 0
                n_batches = 0

                for start in range(0, len(inner_train_idx), batch_size):
                    batch_idx = inner_train_idx[perm[start:start + batch_size]]
                    h_batch = H_padded[batch_idx].to(device)
                    m_batch = masks[batch_idx].to(device)
                    t_batch = Ts[batch_idx].to(device)
                    y_batch = labels[batch_idx].to(device)

                    logits = head(h_batch, m_batch, t_batch).squeeze(-1)
                    loss = criterion(logits, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                # Validation
                head.eval()
                with torch.no_grad():
                    h_val = H_padded[val_idx].to(device)
                    m_val = masks[val_idx].to(device)
                    t_val = Ts[val_idx].to(device)
                    val_logits = head(h_val, m_val, t_val).squeeze(-1)
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()
                    val_labels = labels[val_idx].numpy()
                    val_auc = roc_auc_score(val_labels, val_probs)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= 10:
                        break

            # Evaluate on test fold
            head.load_state_dict(best_state)
            head.eval()
            with torch.no_grad():
                h_test = H_padded[test_idx].to(device)
                m_test = masks[test_idx].to(device)
                t_test = Ts[test_idx].to(device)
                test_logits = head(h_test, m_test, t_test).squeeze(-1)
                test_probs = torch.sigmoid(test_logits).cpu().numpy()
                test_labels = labels[test_idx].numpy()
                test_auc = roc_auc_score(test_labels, test_probs)

            fold_aucs.append(test_auc)

            # Store confidence scores
            for i, idx in enumerate(test_idx):
                all_confs[idx] += test_probs[i]
                conf_counts[idx] += 1

            print(f"  Seed {seed+1}, Fold {fold_idx+1}: AUC={test_auc:.4f} (val={best_val_auc:.4f})")

        seed_auc = np.mean(fold_aucs)
        all_aucs.append(seed_auc)
        print(f"  Seed {seed+1} mean AUC: {seed_auc:.4f}")

    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    print(f"\n  LSFT(2) in-distribution probe AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    # Average confidences
    avg_confs = all_confs / np.maximum(conf_counts, 1)
    return mean_auc, std_auc, avg_confs


def run_routing(trajectories, cot_results, confs):
    """Run routing using in-distribution probe confidences."""
    N = len(trajectories)
    assert len(cot_results) == N

    # Pareto sweep
    taus = np.arange(0.0, 1.01, 0.05).tolist()
    results = []
    for tau in taus:
        n_latent = 0
        n_correct = 0
        total_tokens = 0
        for i in range(N):
            if confs[i] > tau:
                n_latent += 1
                n_correct += int(trajectories[i]['correct'])
                total_tokens += trajectories[i]['T']
            else:
                n_correct += int(cot_results[i]['correct'])
                total_tokens += cot_results[i]['total_tokens']
        results.append({
            'tau': tau,
            'latent_frac': n_latent / N,
            'accuracy': n_correct / N,
            'avg_tokens': total_tokens / N,
        })
    return results


def cv_routing(trajectories, cot_results, confs, n_folds=5):
    """5-fold CV routing with argmax-accuracy threshold selection."""
    N = len(trajectories)
    indices = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    cot_avg_tokens = np.mean([c['total_tokens'] for c in cot_results])
    fold_size = N // n_folds
    taus_to_try = np.arange(0.01, 0.99, 0.01).tolist()

    fold_results = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else N
        val_idx = set(indices[val_start:val_end].tolist())
        train_idx = [i for i in range(N) if i not in val_idx]

        # Select best tau on train (argmax accuracy)
        best_tau = 0.5
        best_acc = 0
        for tau in taus_to_try:
            n_correct = 0
            for i in train_idx:
                if confs[i] > tau:
                    n_correct += int(trajectories[i]['correct'])
                else:
                    n_correct += int(cot_results[i]['correct'])
            acc = n_correct / len(train_idx)
            if acc > best_acc:
                best_acc = acc
                best_tau = tau

        # Evaluate on val fold
        val_list = sorted(val_idx)
        n_correct = 0
        total_tokens = 0
        n_latent = 0
        for i in val_list:
            if confs[i] > best_tau:
                n_latent += 1
                n_correct += int(trajectories[i]['correct'])
                total_tokens += trajectories[i]['T']
            else:
                n_correct += int(cot_results[i]['correct'])
                total_tokens += cot_results[i]['total_tokens']

        val_acc = n_correct / len(val_list)
        val_tokens = total_tokens / len(val_list)
        val_latent_frac = n_latent / len(val_list)

        fold_results.append({
            'fold': fold,
            'best_tau': best_tau,
            'val_acc': val_acc,
            'val_avg_tokens': val_tokens,
            'val_latent_frac': val_latent_frac,
        })

    accs = [r['val_acc'] for r in fold_results]
    tokens = [r['val_avg_tokens'] for r in fold_results]
    latent_fracs = [r['val_latent_frac'] for r in fold_results]
    token_saving = 1 - np.mean(tokens) / cot_avg_tokens

    return {
        'cv_acc_mean': float(np.mean(accs)),
        'cv_acc_std': float(np.std(accs)),
        'cv_tokens_mean': float(np.mean(tokens)),
        'cv_latent_frac_mean': float(np.mean(latent_fracs)),
        'cv_token_saving': float(token_saving),
        'folds': fold_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(args.device)

    # Step 1: Collect LSFT(2) trajectories
    print("=" * 70)
    print("STEP 1: Collect LSFT(2) trajectories with hidden states")
    print("=" * 70)
    trajectories = collect_lsft2_trajectories(device)
    N = len(trajectories)
    latent_acc = np.mean([t['correct'] for t in trajectories])
    avg_T = np.mean([t['T'] for t in trajectories])
    print(f"  LSFT(2): N={N}, acc={latent_acc:.4f}, avg_T={avg_T:.1f}")

    # Step 2: Train in-distribution probe
    print("\n" + "=" * 70)
    print("STEP 2: Train AttentionConfidenceHead on LSFT(2) (nested CV)")
    print("=" * 70)
    mean_auc, std_auc, confs = train_probe_nested_cv(trajectories, device)

    # Step 3: Load CoT results
    print("\n" + "=" * 70)
    print("STEP 3: Load CoT-SFT results and run routing")
    print("=" * 70)

    # Try to load existing CoT results
    cot_paths = [
        os.path.join(os.path.dirname(__file__), 'routing_results_official/cot_results.pt'),
        os.path.join(os.path.dirname(__file__), 'routing_results_v2/cot_results.pt'),
        os.path.join(os.path.dirname(__file__), 'routing_results/cot_sft_inference.json'),
    ]
    cot_results = None
    for cp in cot_paths:
        if os.path.exists(cp):
            print(f"  Loading CoT results from {cp}")
            if cp.endswith('.json'):
                with open(cp) as f:
                    cot_data = json.load(f)
                cot_results = cot_data['per_problem'] if 'per_problem' in cot_data else cot_data
            else:
                cot_results = torch.load(cp, map_location='cpu', weights_only=False)
            break

    if cot_results is None:
        print("  ERROR: No CoT results found. Running CoT-SFT inference...")
        # Run CoT inference
        cot_results = run_cot_inference(device)

    cot_acc = np.mean([c['correct'] for c in cot_results])
    cot_avg_tokens = np.mean([c['total_tokens'] for c in cot_results])
    print(f"  CoT-SFT: N={len(cot_results)}, acc={cot_acc:.4f}, avg_tokens={cot_avg_tokens:.1f}")

    # Step 4: Routing
    print("\n--- Pareto Sweep ---")
    sweep = run_routing(trajectories, cot_results, confs)
    print(f"{'tau':>6} | {'Latent%':>8} | {'Acc':>7} | {'Tokens':>7} | {'Saving':>7}")
    print("-" * 50)
    for r in sweep:
        saving = 1 - r['avg_tokens'] / cot_avg_tokens
        print(f"{r['tau']:6.2f} | {r['latent_frac']*100:7.1f}% | {r['accuracy']*100:6.2f}% | {r['avg_tokens']:7.1f} | {saving*100:6.1f}%")

    print("\n--- 5-Fold CV Routing ---")
    cv = cv_routing(trajectories, cot_results, confs)
    print(f"  CV Acc: {cv['cv_acc_mean']*100:.1f} ± {cv['cv_acc_std']*100:.1f}%")
    print(f"  CV Tokens: {cv['cv_tokens_mean']:.1f}")
    print(f"  CV Latent Frac: {cv['cv_latent_frac_mean']*100:.1f}%")
    print(f"  Token Saving: {cv['cv_token_saving']*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: LSFT(2) In-Distribution Routing")
    print("=" * 70)
    print(f"  Probe AUC (in-dist): {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  Probe AUC (cross-r, from paper): 0.805 ± 0.011")
    print(f"  Pure latent acc: {latent_acc*100:.1f}%")
    print(f"  Pure CoT acc: {cot_acc*100:.1f}%")
    print(f"  Routing CV acc (in-dist): {cv['cv_acc_mean']*100:.1f} ± {cv['cv_acc_std']*100:.1f}%")
    print(f"  Routing CV acc (cross-r, from paper): 52.5 ± 1.5%")
    print(f"  Token saving (in-dist): {cv['cv_token_saving']*100:.1f}%")
    print(f"  Token saving (cross-r, from paper): 43%")

    # Save results
    output = {
        'probe_auc_mean': mean_auc,
        'probe_auc_std': std_auc,
        'latent_acc': float(latent_acc),
        'cot_acc': float(cot_acc),
        'cot_avg_tokens': float(cot_avg_tokens),
        'avg_T': float(avg_T),
        'routing_sweep': sweep,
        'routing_cv': cv,
        'comparison': {
            'cross_r_auc': 0.805,
            'cross_r_routing_acc': 0.525,
            'cross_r_saving': 0.43,
        }
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR}/results.json")


def run_cot_inference(device):
    """Fallback: run CoT-SFT inference if no cached results."""
    print(f"  Loading CoT-SFT from {COT_SFT_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        COT_SFT_PATH, attn_implementation='sdpa',
        torch_dtype=torch.bfloat16, use_cache=True, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(COT_SFT_PATH)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    with open(DATA_PATH) as f:
        problems = [json.loads(l) for l in f]

    results = []
    for pi, example in enumerate(tqdm(problems, desc="CoT-SFT inference")):
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}"}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            gen_out = model.generate(input_ids, max_new_tokens=512, do_sample=False)

        gen_tokens = gen_out[0][input_ids.shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_answer(text)
        correct = check_is_correct(pred, example['answer'])

        results.append({
            'idx': pi,
            'correct': bool(correct),
            'total_tokens': len(gen_tokens),
        })

    del model
    torch.cuda.empty_cache()

    # Save
    save_path = os.path.join(RESULTS_DIR, 'cot_results.json')
    with open(save_path, 'w') as f:
        json.dump({'per_problem': results}, f)
    print(f"  Saved CoT results to {save_path}")
    return results


if __name__ == "__main__":
    main()
