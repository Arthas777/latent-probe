"""
Risk 2 mitigation: Cross-task (OOD) evaluation of trained confidence heads.
Tests the GSM8k-trained calibrator on GSM-Hard, SVAMP, MultiArith, Math500.
This is the acid test — if cross-task AUC >= 0.78, the signal is transferable.
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
latent_sft_dir = os.path.join(current_dir, '../../Latent-SFT')
sys.path.append(latent_sft_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collect_trajectories import (
    generate_with_full_trajectory, extract_answer, check_is_correct,
    compute_spectral_features
)
from train_confidence_head import (
    AttentionConfidenceHead, HybridConfidenceHead, MLPConfidenceHead,
    TrajectoryDataset, roc_auc_score, compute_ece
)
from evaluate_selective import selective_accuracy, compute_aurc, compute_auccr


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def collect_and_evaluate(model, tokenizer, model_type, data, device,
                         confidence_heads, max_new_tokens=128):
    """Collect trajectories and evaluate confidence heads in one pass."""
    all_features = []

    for i, example in enumerate(tqdm(data, desc="Inference")):
        if model_type == 'qwen2':
            messages = [{"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + example["problem"]}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_prefix = input_text + "<｜Assistant｜>"
        else:
            input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{example['problem']}<|eot_id|>"
            input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        input_ids = tokenizer(input_prefix, truncation=False, padding=False,
                             add_special_tokens=False, return_attention_mask=False)['input_ids']

        text_input = {
            'input_ids': torch.tensor(input_ids + model.latent_token_ids[0], dtype=torch.long).to(device).unsqueeze(0),
            'attention_mask': torch.tensor([1] * (len(input_ids) + len(model.latent_token_ids[0])), dtype=torch.long).to(device).unsqueeze(0),
        }

        output = generate_with_full_trajectory(model, text_input, max_new_tokens=max_new_tokens)

        pred = extract_answer(output["text"])
        correct = check_is_correct(pred, example["answer"])

        all_features.append({
            "idx": i,
            "correct": bool(correct),
            "T": output["T"],
            "hidden_states": output["hidden_states"],
            "token_entropies": output["token_entropies"],
            "top1_probs": output["top1_probs"],
            "spectral_features": output["spectral_features"],
        })

    return all_features


def evaluate_heads_on_features(features, confidence_heads, device, max_T=64):
    """Evaluate all confidence heads on collected features."""
    from torch.utils.data import DataLoader

    # Create a temporary .pt to reuse TrajectoryDataset
    tmp_path = '/tmp/_ood_traj_tmp.pt'
    torch.save(features, tmp_path)
    dataset = TrajectoryDataset(tmp_path, max_T=max_T)

    labels = np.array([f['correct'] for f in features], dtype=float)
    overall_acc = labels.mean()

    results = {}

    # Baselines
    lens_scores = []
    ent_scores = []
    for f in features:
        T = f['T']
        H_spec = f['spectral_features'].get('H_spec', 0.0)
        lens = H_spec / np.log(max(T, 2))
        lens_scores.append(lens)
        mean_ent = f['token_entropies'].mean().item() if len(f['token_entropies']) > 0 else 0.0
        ent_scores.append(-mean_ent)

    lens_scores = np.array(lens_scores)
    ent_scores = np.array(ent_scores)

    if len(np.unique(labels)) >= 2:
        results['LENS'] = {
            'auc': float(roc_auc_score(labels, lens_scores)),
            'sel_acc_50': selective_accuracy(lens_scores, labels)[0.5],
        }
        results['mean_entropy_neg'] = {
            'auc': float(roc_auc_score(labels, ent_scores)),
            'sel_acc_50': selective_accuracy(ent_scores, labels)[0.5],
        }
    else:
        results['LENS'] = {'auc': float('nan'), 'sel_acc_50': float('nan')}
        results['mean_entropy_neg'] = {'auc': float('nan'), 'sel_acc_50': float('nan')}

    # Trained heads
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    for head_name, (head_model, head_type) in confidence_heads.items():
        head_model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch_labels = batch['label']
                if head_type == 'mlp':
                    logits = head_model(batch['scalar_features'].to(device))
                elif head_type == 'attention':
                    logits = head_model(
                        batch['hidden_states'].to(device),
                        batch['mask'].to(device),
                        batch['T'].to(device),
                    )
                else:
                    logits = head_model(
                        batch['hidden_states'].to(device),
                        batch['mask'].to(device),
                        batch['T'].to(device),
                        batch['scalar_features'].to(device),
                    )
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(batch_labels.numpy().tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        if len(np.unique(all_labels)) >= 2:
            auc = roc_auc_score(all_labels, all_probs)
            sel_acc = selective_accuracy(all_probs, all_labels)
            ece = compute_ece(all_probs, all_labels)
            results[head_name] = {
                'auc': float(auc),
                'ece': float(ece),
                'sel_acc_50': sel_acc[0.5],
            }
        else:
            results[head_name] = {'auc': float('nan'), 'ece': float('nan'), 'sel_acc_50': float('nan')}

    os.remove(tmp_path)
    return overall_acc, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_model_path', type=str, default='../../Latent-SFT/checkpoints/latent-4')
    parser.add_argument('--head_dir', type=str, default='./trained_heads')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['GSM8k-Hard', 'Svamp', 'Multiarith'])
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load LLM
    print(f"Loading model from {args.latent_model_path}...")
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

    config = AutoConfig.from_pretrained(args.latent_model_path)
    model_type_str = config.model_type
    hidden_dim = config.hidden_size

    # Load confidence heads
    print("Loading confidence heads...")
    confidence_heads = {}

    for hf in os.listdir(args.head_dir):
        if not hf.startswith("confidence_head_") or not hf.endswith(".pt"):
            continue
        ckpt = torch.load(os.path.join(args.head_dir, hf), map_location=device)
        mt = ckpt["model_type"]
        saved_args = ckpt.get("args", {})

        if mt == 'mlp':
            head = MLPConfidenceHead(input_dim=8, hidden_dim=64).to(device)
        elif mt == 'attention':
            head = AttentionConfidenceHead(
                hidden_dim=hidden_dim, proj_dim=saved_args.get('proj_dim', 128),
                n_heads=4, dropout=0.0
            ).to(device)
        else:
            head = HybridConfidenceHead(
                hidden_dim=hidden_dim, proj_dim=saved_args.get('proj_dim', 128),
                n_heads=4, n_scalar_features=8, dropout=0.0
            ).to(device)

        head.load_state_dict(ckpt["model_state_dict"])
        confidence_heads[f"trained_{mt}"] = (head, mt)
        print(f"  Loaded {mt} head")

    # Dataset file mapping
    data_map = {
        'GSM8k-Hard': 'GSM8k-Hard-test.jsonl',
        'Svamp': 'Svamp-test.jsonl',
        'Multiarith': 'Multiarith-test.jsonl',
        'Math500': 'Math-500-test.jsonl',
    }

    all_dataset_results = {}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name}")
        print(f"{'='*60}")

        data_path = os.path.join(latent_sft_dir, 'data', data_map[ds_name])
        data = read_jsonl(data_path)
        print(f"Loaded {len(data)} samples.")

        # Collect trajectories
        features = collect_and_evaluate(
            model, tokenizer, model_type_str, data, device,
            confidence_heads, max_new_tokens=args.max_new_tokens
        )

        # Save trajectories for later use
        traj_path = os.path.join(args.output_dir, f"trajectories_{ds_name}.pt")
        torch.save(features, traj_path)

        # Evaluate
        overall_acc, results = evaluate_heads_on_features(features, confidence_heads, device)

        all_dataset_results[ds_name] = {
            "n_samples": len(data),
            "overall_accuracy": float(overall_acc),
            "n_correct": int(sum(f['correct'] for f in features)),
            "methods": results,
        }

        print(f"\n  Overall accuracy: {overall_acc:.4f} ({int(overall_acc*len(data))}/{len(data)})")
        print(f"  {'Method':<22} {'AUC':>7} {'Acc@50%':>8}")
        print(f"  {'-'*40}")
        for name, res in results.items():
            print(f"  {name:<22} {res['auc']:>7.4f} {res['sel_acc_50']:>8.4f}")

    # Cross-dataset summary
    print(f"\n{'='*70}")
    print("CROSS-TASK GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Acc':>6} {'N':>5} | {'LENS':>7} {'Attn':>7} {'Hybrid':>7} {'MLP':>7}")
    print("-" * 70)

    for ds_name, ds_res in all_dataset_results.items():
        methods = ds_res['methods']
        lens_auc = methods.get('LENS', {}).get('auc', float('nan'))
        attn_auc = methods.get('trained_attention', {}).get('auc', float('nan'))
        hybrid_auc = methods.get('trained_hybrid', {}).get('auc', float('nan'))
        mlp_auc = methods.get('trained_mlp', {}).get('auc', float('nan'))
        print(f"  {ds_name:<13} {ds_res['overall_accuracy']:>6.3f} {ds_res['n_samples']:>5} | {lens_auc:>7.4f} {attn_auc:>7.4f} {hybrid_auc:>7.4f} {mlp_auc:>7.4f}")

    # Add GSM8k reference
    print(f"  {'GSM8k (ref)':<13} {'0.450':>6} {'1319':>5} | {'0.7132':>7} {'0.8352':>7} {'0.8306':>7} {'0.7203':>7}")

    # Save
    output_file = os.path.join(args.output_dir, "ood_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_dataset_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
