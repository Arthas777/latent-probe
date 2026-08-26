"""
Collect CoLaR latent trajectory features on SVAMP dataset (OOD evaluation).
Reuses CoLaRInference from collect_trajectories_colar.py.

Usage:
    python collect_trajectories_colar_svamp.py \
        --ckpt_path ../../logs/colar/qsa-gsm/20260428-195821_350334_full-cotsft-baseline/checkpoints/epoch44__step67815__monitor0.312.ckpt \
        --output_dir ./trajectory_data_colar \
        --device cuda:1
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_trajectories_colar import CoLaRInference, compute_colar_spectral_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--svamp_path', type=str,
                        default='../../Latent-SFT/data/Svamp-test.jsonl')
    parser.add_argument('--output_dir', type=str, default='./trajectory_data_colar')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    svamp_path = os.path.join(base_dir, args.svamp_path) if not os.path.isabs(args.svamp_path) else args.svamp_path

    model = CoLaRInference(args.ckpt_path, device=args.device)

    with open(svamp_path) as f:
        data = [json.loads(l) for l in f]
    print(f"Loaded {len(data)} SVAMP samples.")

    all_features = []

    for i, example in enumerate(tqdm(data, desc="Collecting SVAMP trajectories")):
        question = example['problem']
        output = model.generate_with_trajectory(question)

        # Extract answer
        pred_answer = output['text']
        pred_str = pred_answer.strip('#\n ').split('Answer:')[-1] if 'Answer:' in pred_answer else pred_answer.strip('#\n ')

        gt_answer = str(example['answer']).strip()
        pred_clean = pred_str.strip().rstrip('.').replace(',', '').lower()
        gt_clean = gt_answer.strip().rstrip('.').replace(',', '').lower()

        try:
            correct = abs(float(pred_clean) - float(gt_clean)) < 1e-4
        except (ValueError, TypeError):
            correct = pred_clean == gt_clean

        spectral = compute_colar_spectral_features(output['hidden_states'])

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
            "run": 0,
            "correct": bool(correct),
            "T": T,
            "hidden_states": output["hidden_states"],
            "token_entropies": torch.zeros(T),
            "top1_probs": torch.zeros(T),
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

    output_file = os.path.join(args.output_dir, "trajectories_Svamp_colar.pt")
    torch.save(all_features, output_file)
    print(f"\nSaved {len(all_features)} trajectories to {output_file}")

    acc = sum(f["correct"] for f in all_features) / len(all_features)
    Ts = [f['T'] for f in all_features]
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Mean T: {np.mean(Ts):.1f} (std={np.std(Ts):.1f})")


if __name__ == "__main__":
    main()
