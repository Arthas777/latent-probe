"""
Compute confidence scores for CoLaR deterministic trajectories using the trained head.
This is pure post-processing (no model inference needed).
"""
import os, sys, json, torch, numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from train_confidence_head import AttentionConfidenceHead


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


def main():
    device = 'cuda:0'
    base = os.path.dirname(__file__)

    # Load CoLaR head (trained on mixed det+stoch data, cv_auc=0.94)
    head_path = os.path.join(base, 'direction1_results/colar_head_mixed.pt')
    ckpt = torch.load(head_path, map_location='cpu', weights_only=False)
    head = AttentionConfidenceHead(hidden_dim=2048, proj_dim=128, n_heads=4, dropout=0.1)
    head.load_state_dict(ckpt['model_state_dict'])
    head = head.to(device).eval()
    print(f"Loaded CoLaR head (cv_auc={ckpt['cv_auc']:.4f})")

    # Load trajectories
    traj_path = os.path.join(base, 'trajectory_data_colar/trajectories_GSM8k_colar.pt')
    trajectories = torch.load(traj_path, map_location='cpu', weights_only=False)
    print(f"Loaded {len(trajectories)} CoLaR trajectories")

    # Compute confidence for each
    results = []
    for i, traj in enumerate(trajectories):
        conf = head_predict(head, traj['hidden_states'], traj['T'], device=device)
        results.append({
            'idx': i,
            'correct': bool(traj['correct']),
            'T': traj['T'],
            'conf': conf,
        })

    acc = np.mean([r['correct'] for r in results])
    avg_T = np.mean([r['T'] for r in results])
    confs = [r['conf'] for r in results]
    labels = [float(r['correct']) for r in results]

    # Compute AUC
    y_true = np.array(labels)
    y_score = np.array(confs)
    desc_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_idx]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    pooled_auc = float(np.trapz(tpr, fpr))

    print(f"\nCoLaR deterministic: acc={acc:.4f}, avg_T={avg_T:.1f}")
    print(f"Head pooled AUC: {pooled_auc:.4f}")
    print(f"Conf stats: mean={np.mean(confs):.3f}, median={np.median(confs):.3f}, "
          f"min={min(confs):.3f}, max={max(confs):.3f}")

    os.makedirs('./routing_results', exist_ok=True)
    with open('./routing_results/colar_deterministic_confs.json', 'w') as f:
        json.dump({'per_problem': results, 'pooled_auc': pooled_auc,
                   'accuracy': float(acc), 'avg_T': float(avg_T)}, f, indent=2)
    print(f"Saved to routing_results/colar_deterministic_confs.json")


if __name__ == "__main__":
    main()
