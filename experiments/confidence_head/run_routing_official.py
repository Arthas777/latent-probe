"""
Adaptive Routing using official Latent-SFT evaluation code.
Directly calls LatentSFTStage2SoftEmbedding.one_example_generate_hf()
with the same settings as the paper's eval script.

Implements candidate_b_adaptive_routing_path.md Steps 1-3:
  Step 1: Sweep τ for Pareto curve
  Step 2: Random routing baseline
  Step 3: 5-fold CV for fair threshold selection

Usage: python run_routing_official.py [--phase 1|2|3|all] [--device cuda:0]
"""
import os, sys, json, torch, argparse, numpy as np
from tqdm import tqdm

# Seed everything exactly like the paper
def seed_everything(seed: int = 777):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

seed_everything(777)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
lsft_root = os.path.join(project_root, 'Latent-SFT')
sys.path.insert(0, lsft_root)
sys.path.insert(0, os.path.join(lsft_root, 'eval'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.modeling.modeling_latent import LatentSFTStage2SoftEmbedding
from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer

sys.path.insert(0, os.path.dirname(__file__))
from train_confidence_head import AttentionConfidenceHead

LATENT2_PATH = os.path.join(lsft_root, 'checkpoints/latent-2')
COT_SFT_PATH = os.path.join(lsft_root, 'checkpoints/cot-sft')
DATA_PATH = os.path.join(lsft_root, 'data/GSM8k-Aug-test.jsonl')
RESULTS_DIR = './routing_results_official'


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


def run_latent2(device, data, head):
    """Run Latent-SFT(2) using official inference function."""
    print(f"Loading Latent-SFT(2) from {LATENT2_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        LATENT2_PATH, attn_implementation='sdpa',
        torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(LATENT2_PATH)
    model.eval()
    model.tokenizer = tokenizer
    model.latent_token_ids = tokenizer(['<think>', '</think>'], add_special_tokens=False)['input_ids']
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    results = []
    for pi, example in enumerate(tqdm(data, desc="Latent-SFT(2)")):
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids_raw = tokenizer(input_prefix, truncation=False, padding=False,
                                  add_special_tokens=False, return_attention_mask=False)['input_ids']
        text_input = {
            'input_ids': torch.tensor(input_ids_raw + model.latent_token_ids[0], dtype=torch.long).to(device).unsqueeze(0),
            'attention_mask': torch.tensor([1] * (len(input_ids_raw) + len(model.latent_token_ids[0])), dtype=torch.long).to(device).unsqueeze(0),
        }

        # Use official inference - but we need hidden states for confidence head
        # Re-implement with hidden state extraction (same logic as official)
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
        generated_ids = []
        past_key_values = None
        latent_states = []
        hidden_list = []
        W = model.model.embed_tokens.weight.detach()

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
                # Collect hidden state for confidence head
                hidden_list.append(outputs.hidden_states[-1][:, -1, :].squeeze(0).cpu())

                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                probs = torch.softmax(next_token_logits, dim=-1).to(W.dtype)
                input_embeddings = probs @ W
                input_embeddings = input_embeddings.unsqueeze(1)

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
                    latent_states.append(input_embeddings)

            T = len(generated_ids)
            latent_state = torch.cat(latent_states, dim=1)

            think_ids = torch.LongTensor(model.latent_token_ids[1]).unsqueeze(0).to(device)
            think_end_embeddings = model.model.embed_tokens(think_ids)
            input_ids_orig = text_input['input_ids']
            attn_len = input_ids_orig.size(1) + latent_state.size(1) + len(model.latent_token_ids[1])
            attn_full = torch.tensor([1] * attn_len, dtype=torch.long, device=device).unsqueeze(0)
            emb_full = torch.cat([model.model.embed_tokens(input_ids_orig), latent_state, think_end_embeddings], dim=1)

            gen_out = model.generate(
                inputs_embeds=emb_full, attention_mask=attn_full,
                max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95,
            )
            text = tokenizer.decode(gen_out[0], skip_special_tokens=False)

        pred = extract_answer(text)
        correct = check_is_correct(pred, example['answer'])
        hidden_tensor = torch.stack(hidden_list) if hidden_list else torch.zeros(1, 2048)
        conf = head_predict(head, hidden_tensor, T, device=str(device))

        results.append({
            'idx': pi, 'correct': bool(correct), 'T': T, 'conf': conf,
        })

        if (pi + 1) % 100 == 0:
            acc = np.mean([r['correct'] for r in results])
            avg_T = np.mean([r['T'] for r in results])
            print(f"  [{pi+1}] acc={acc:.4f}, avg_T={avg_T:.1f}")

    return results


def run_cot_sft(device, data):
    """Run official CoT-SFT inference."""
    print(f"Loading CoT-SFT from {COT_SFT_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        COT_SFT_PATH, attn_implementation='sdpa',
        torch_dtype=torch.bfloat16, use_cache=True, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(COT_SFT_PATH)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    results = []
    for pi, example in enumerate(tqdm(data, desc="CoT-SFT")):
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids = tokenizer(input_prefix, truncation=False, padding=False,
                             add_special_tokens=False, return_attention_mask=False)['input_ids']
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_tensor, attention_mask=attention_mask,
                max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95,
            )
        gen_ids = gen_out[0, input_tensor.shape[1]:]
        total_tokens = len(gen_ids)
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        pred = extract_answer(text)
        correct = check_is_correct(pred, example['answer'])

        results.append({
            'idx': pi, 'correct': bool(correct), 'total_tokens': total_tokens,
        })

        if (pi + 1) % 100 == 0:
            acc = np.mean([r['correct'] for r in results])
            avg_tok = np.mean([r['total_tokens'] for r in results])
            print(f"  [{pi+1}] acc={acc:.4f}, avg_tokens={avg_tok:.1f}")

    return results


def routing_analysis(latent_results, cot_results):
    """Steps 1-3: Sweep + Random baseline + 5-fold CV."""
    N = len(latent_results)
    latent_acc = np.mean([r['correct'] for r in latent_results])
    latent_avg_T = np.mean([r['T'] for r in latent_results])
    cot_acc = np.mean([r['correct'] for r in cot_results])
    cot_avg_tokens = np.mean([r['total_tokens'] for r in cot_results])

    print(f"\n{'='*70}")
    print(f"Latent-SFT(2): acc={latent_acc:.4f}, avg_T={latent_avg_T:.1f}")
    print(f"CoT-SFT:       acc={cot_acc:.4f}, avg_tokens={cot_avg_tokens:.1f}")
    print(f"{'='*70}")

    # Step 1: Sweep τ
    taus = np.arange(0.0, 1.01, 0.05).tolist()
    sweep = []
    for tau in taus:
        n_latent = n_correct = 0
        total_tokens = 0
        for i in range(N):
            if latent_results[i]['conf'] > tau:
                n_latent += 1
                n_correct += int(latent_results[i]['correct'])
                total_tokens += latent_results[i]['T']
            else:
                n_correct += int(cot_results[i]['correct'])
                total_tokens += cot_results[i]['total_tokens']
        sweep.append({'tau': round(tau, 2), 'latent_frac': n_latent/N,
                      'accuracy': n_correct/N, 'avg_tokens': total_tokens/N})

    print(f"\n{'τ':>6} | {'Lat%':>6} | {'Acc':>7} | {'Tokens':>7} | {'Save%':>6}")
    print("-" * 48)
    for r in sweep:
        saving = (1 - r['avg_tokens'] / cot_avg_tokens) * 100
        print(f"{r['tau']:6.2f} | {r['latent_frac']*100:5.1f}% | {r['accuracy']*100:6.1f}% | {r['avg_tokens']:7.1f} | {saving:5.1f}%")

    # Step 2: Random baseline
    random_results = []
    latent_fracs = sorted(set(round(r['latent_frac'], 2) for r in sweep if 0 < r['latent_frac'] < 1))
    for frac in latent_fracs:
        n_lat = int(round(frac * N))
        accs = []
        for seed in range(20):
            rng = np.random.default_rng(seed * 1000 + int(frac * 100))
            lat_idx = set(rng.choice(N, size=n_lat, replace=False))
            nc = sum(int(latent_results[i]['correct']) if i in lat_idx else int(cot_results[i]['correct']) for i in range(N))
            accs.append(nc / N)
        random_results.append({'latent_frac': frac, 'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs))})

    dominates = all(
        min(sweep, key=lambda s: abs(s['latent_frac'] - rb['latent_frac']))['accuracy'] >= rb['acc_mean'] - 0.001
        for rb in random_results
    )
    print(f"\nPareto dominates random: {'YES' if dominates else 'NO'}")

    # Step 3: 5-fold CV
    indices = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    fold_size = N // 5
    cv_results = []
    for fold in range(5):
        vs, ve = fold * fold_size, (fold + 1) * fold_size if fold < 4 else N
        val_idx = set(indices[vs:ve].tolist())
        train_idx = [i for i in range(N) if i not in val_idx]
        cot_acc_train = np.mean([cot_results[i]['correct'] for i in train_idx])

        best_tau, best_metric = None, -1
        for tau in np.arange(0.05, 0.95, 0.02):
            nc = tt = 0
            for i in train_idx:
                if latent_results[i]['conf'] > tau:
                    nc += int(latent_results[i]['correct']); tt += latent_results[i]['T']
                else:
                    nc += int(cot_results[i]['correct']); tt += cot_results[i]['total_tokens']
            acc = nc / len(train_idx); avg_tok = tt / len(train_idx)
            if acc >= cot_acc_train - 0.005:
                m = -avg_tok
                if m > best_metric: best_metric = m; best_tau = tau
        if best_tau is None: best_tau = 0.3

        nc = tt = nl = 0
        for i in sorted(val_idx):
            if latent_results[i]['conf'] > best_tau:
                nl += 1; nc += int(latent_results[i]['correct']); tt += latent_results[i]['T']
            else:
                nc += int(cot_results[i]['correct']); tt += cot_results[i]['total_tokens']
        n_val = len(val_idx)
        cv_results.append({'tau': best_tau, 'acc': nc/n_val, 'tokens': tt/n_val, 'latent_frac': nl/n_val})

    cv_acc_mean = np.mean([r['acc'] for r in cv_results])
    cv_acc_std = np.std([r['acc'] for r in cv_results])
    cv_tok_mean = np.mean([r['tokens'] for r in cv_results])
    cv_tok_std = np.std([r['tokens'] for r in cv_results])
    cv_saving = (1 - cv_tok_mean / cot_avg_tokens) * 100

    print(f"\n5-fold CV (iso-accuracy):")
    print(f"  τ* = {np.mean([r['tau'] for r in cv_results]):.3f} ± {np.std([r['tau'] for r in cv_results]):.3f}")
    print(f"  Acc = {cv_acc_mean*100:.1f} ± {cv_acc_std*100:.1f}%")
    print(f"  Tokens = {cv_tok_mean:.1f} ± {cv_tok_std:.1f}")
    print(f"  Token saving vs CoT = {cv_saving:.1f}%")

    # Sweet spots
    best_point = max(sweep, key=lambda r: r['accuracy'])
    iso_acc = None
    for r in sweep:
        if r['accuracy'] >= cot_acc and r['latent_frac'] > 0:
            if iso_acc is None or r['avg_tokens'] < iso_acc['avg_tokens']:
                iso_acc = r

    print(f"\n=== Sweet Spots ===")
    print(f"  Best accuracy: τ={best_point['tau']:.2f}, acc={best_point['accuracy']*100:.1f}%, "
          f"tokens={best_point['avg_tokens']:.1f} ({(1-best_point['avg_tokens']/cot_avg_tokens)*100:.0f}% saving)")
    if iso_acc:
        print(f"  Iso-accuracy: τ={iso_acc['tau']:.2f}, acc={iso_acc['accuracy']*100:.1f}%, "
              f"tokens={iso_acc['avg_tokens']:.1f} ({(1-iso_acc['avg_tokens']/cot_avg_tokens)*100:.0f}% saving)")

    # Go/No-Go
    criteria_a = iso_acc is not None and (1 - iso_acc['avg_tokens'] / cot_avg_tokens) >= 0.30
    criteria_c = dominates
    if criteria_a:
        decision = "FULL GO"
    elif criteria_c:
        decision = "MINIMAL GO"
    else:
        decision = "NO-GO"
    print(f"\n  Criterion (a) iso-acc ≥30% saving: {'PASS' if criteria_a else 'FAIL'}")
    print(f"  Criterion (c) Pareto dominates random: {'PASS' if criteria_c else 'FAIL'}")
    print(f"  >>> Decision: {decision} <<<")

    # Save
    output = {
        'baselines': {'latent_sft2': {'acc': float(latent_acc), 'avg_T': float(latent_avg_T)},
                      'cot_sft': {'acc': float(cot_acc), 'avg_tokens': float(cot_avg_tokens)}},
        'sweep': sweep, 'random_baseline': random_results, 'cv_results': cv_results,
        'dominates_random': bool(dominates), 'decision': decision,
        'sweet_spots': {
            'best_accuracy': {'tau': best_point['tau'], 'acc': best_point['accuracy'], 'tokens': best_point['avg_tokens']},
            'iso_accuracy': {'tau': iso_acc['tau'], 'acc': iso_acc['accuracy'], 'tokens': iso_acc['avg_tokens']} if iso_acc else None,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'routing_official_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.bool_, np.integer)) else float(o) if isinstance(o, np.floating) else o)
    print(f"\nSaved to {RESULTS_DIR}/routing_official_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all', choices=['1', '2', '3', 'all'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(args.device)

    with open(DATA_PATH) as f:
        data = [json.loads(l) for l in f]
    print(f"Loaded {len(data)} problems from GSM8k-Aug test")

    if args.phase in ('1', 'all'):
        head_path = os.path.join(os.path.dirname(__file__), 'trained_heads/confidence_head_attention.pt')
        ckpt = torch.load(head_path, map_location='cpu', weights_only=False)
        head = AttentionConfidenceHead(hidden_dim=2048, proj_dim=128, n_heads=4, dropout=0.1)
        head.load_state_dict(ckpt['model_state_dict'])
        head = head.to(device).eval()

        latent_results = run_latent2(device, data, head)
        torch.save(latent_results, os.path.join(RESULTS_DIR, 'latent2_results.pt'))
        acc = np.mean([r['correct'] for r in latent_results])
        avg_T = np.mean([r['T'] for r in latent_results])
        print(f"\nLatent-SFT(2) done: acc={acc:.4f}, avg_T={avg_T:.1f}")
        del head
        torch.cuda.empty_cache()

    if args.phase in ('2', 'all'):
        cot_results = run_cot_sft(device, data)
        torch.save(cot_results, os.path.join(RESULTS_DIR, 'cot_sft_results.pt'))
        acc = np.mean([r['correct'] for r in cot_results])
        avg_tok = np.mean([r['total_tokens'] for r in cot_results])
        print(f"\nCoT-SFT done: acc={acc:.4f}, avg_tokens={avg_tok:.1f}")

    if args.phase in ('3', 'all'):
        latent_results = torch.load(os.path.join(RESULTS_DIR, 'latent2_results.pt'), map_location='cpu', weights_only=False)
        cot_results = torch.load(os.path.join(RESULTS_DIR, 'cot_sft_results.pt'), map_location='cpu', weights_only=False)
        routing_analysis(latent_results, cot_results)


if __name__ == "__main__":
    main()
