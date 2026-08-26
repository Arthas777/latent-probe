"""Re-run in-dist probe routing with official CoT results."""
import torch, numpy as np, os, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from train_confidence_head import AttentionConfidenceHead

device = torch.device('cuda:0')

# Load LSFT(2) trajectories
traj = torch.load('./routing_results_lsft2_indist/lsft2_trajectories.pt', map_location='cpu', weights_only=False)
N = len(traj)
latent_acc = np.mean([t['correct'] for t in traj])
avg_T = np.mean([t['T'] for t in traj])
print(f'LSFT(2): N={N}, acc={latent_acc:.4f}, avg_T={avg_T:.1f}')

# Load official CoT results
cot = torch.load('./routing_results_official/cot_sft_results.pt', map_location='cpu', weights_only=False)
cot_acc = np.mean([c['correct'] for c in cot])
cot_avg_tokens = np.mean([c['total_tokens'] for c in cot])
print(f'CoT-SFT (official): N={len(cot)}, acc={cot_acc:.4f}, avg_tokens={cot_avg_tokens:.1f}')

# Prepare data
hidden_dim = traj[0]['hidden_states'].shape[-1]
max_T = 64
labels = torch.tensor([t['correct'] for t in traj], dtype=torch.float32)
Ts = torch.tensor([t['T'] for t in traj], dtype=torch.float32)

H_padded = torch.zeros(N, max_T, hidden_dim)
masks = torch.zeros(N, max_T, dtype=torch.bool)
for i, t in enumerate(traj):
    hs = t['hidden_states'].float()
    T_i = min(hs.shape[0], max_T)
    H_padded[i, :T_i] = hs[:T_i]
    masks[i, :T_i] = True

# Train probe (nested CV, 3 seeds)
all_confs = np.zeros(N)
conf_counts = np.zeros(N)
all_aucs = []

for seed in range(3):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + seed)
    fold_aucs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), labels.numpy())):
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed*100+fold_idx)
        inner_train_idx, val_idx = next(inner_skf.split(train_idx, labels[train_idx].numpy()))
        inner_train_idx = train_idx[inner_train_idx]
        val_idx = train_idx[val_idx]

        head = AttentionConfidenceHead(hidden_dim=hidden_dim, proj_dim=128).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_val_auc = 0
        best_state = None
        patience_count = 0

        for epoch in range(100):
            head.train()
            perm = np.random.permutation(len(inner_train_idx))
            for start in range(0, len(inner_train_idx), 64):
                batch_idx = inner_train_idx[perm[start:start+64]]
                h_b = H_padded[batch_idx].to(device)
                m_b = masks[batch_idx].to(device)
                t_b = Ts[batch_idx].to(device)
                y_b = labels[batch_idx].to(device)
                logits = head(h_b, m_b, t_b).squeeze(-1)
                loss = criterion(logits, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            head.eval()
            with torch.no_grad():
                val_logits = head(H_padded[val_idx].to(device), masks[val_idx].to(device), Ts[val_idx].to(device)).squeeze(-1)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_auc = roc_auc_score(labels[val_idx].numpy(), val_probs)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 10:
                    break

        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            test_logits = head(H_padded[test_idx].to(device), masks[test_idx].to(device), Ts[test_idx].to(device)).squeeze(-1)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_auc = roc_auc_score(labels[test_idx].numpy(), test_probs)
        fold_aucs.append(test_auc)
        for i, idx in enumerate(test_idx):
            all_confs[idx] += test_probs[i]
            conf_counts[idx] += 1
    all_aucs.append(np.mean(fold_aucs))
    print(f'  Seed {seed+1} AUC: {np.mean(fold_aucs):.4f}')

mean_auc = np.mean(all_aucs)
std_auc = np.std(all_aucs)
print(f'\nIn-dist probe AUC: {mean_auc:.3f} +/- {std_auc:.3f}')

confs = all_confs / np.maximum(conf_counts, 1)

# === Pareto sweep with official CoT ===
print(f'\n=== Pareto Sweep (Official CoT) ===')
print(f'{"tau":>6} | {"Latent%":>8} | {"Acc":>7} | {"Tokens":>7} | {"Saving":>7}')
print('-' * 50)
for tau in np.arange(0.0, 1.01, 0.05):
    n_latent = n_correct = 0
    total_tokens = 0
    for i in range(N):
        if confs[i] > tau:
            n_latent += 1
            n_correct += int(traj[i]['correct'])
            total_tokens += traj[i]['T']
        else:
            n_correct += int(cot[i]['correct'])
            total_tokens += cot[i]['total_tokens']
    acc = n_correct / N
    avg_tok = total_tokens / N
    saving = 1 - avg_tok / cot_avg_tokens
    print(f'{tau:6.2f} | {n_latent/N*100:7.1f}% | {acc*100:6.2f}% | {avg_tok:7.1f} | {saving*100:6.1f}%')

# === 5-fold CV routing ===
indices = np.arange(N)
rng = np.random.default_rng(42)
rng.shuffle(indices)
fold_size = N // 5
taus_to_try = np.arange(0.01, 0.99, 0.01)

fold_accs = []
fold_tokens = []
fold_latent_fracs = []
for fold in range(5):
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < 4 else N
    val_idx_set = set(indices[val_start:val_end].tolist())
    train_idx_list = [i for i in range(N) if i not in val_idx_set]

    best_tau = 0.5
    best_train_acc = 0
    for tau in taus_to_try:
        nc = sum(int(traj[i]['correct']) if confs[i] > tau else int(cot[i]['correct']) for i in train_idx_list)
        acc = nc / len(train_idx_list)
        if acc > best_train_acc:
            best_train_acc = acc
            best_tau = tau

    val_list = sorted(val_idx_set)
    nc = tt = nl = 0
    for i in val_list:
        if confs[i] > best_tau:
            nl += 1
            nc += int(traj[i]['correct'])
            tt += traj[i]['T']
        else:
            nc += int(cot[i]['correct'])
            tt += cot[i]['total_tokens']
    fold_accs.append(nc / len(val_list))
    fold_tokens.append(tt / len(val_list))
    fold_latent_fracs.append(nl / len(val_list))

cv_acc = np.mean(fold_accs)
cv_std = np.std(fold_accs)
cv_tokens = np.mean(fold_tokens)
cv_saving = 1 - cv_tokens / cot_avg_tokens
cv_latent = np.mean(fold_latent_fracs)

print(f'\n=== 5-Fold CV Routing (Official CoT) ===')
print(f'  CV Acc: {cv_acc*100:.1f} +/- {cv_std*100:.1f}%')
print(f'  CV Tokens: {cv_tokens:.1f}')
print(f'  CV Latent Frac: {cv_latent*100:.1f}%')
print(f'  Token Saving: {cv_saving*100:.1f}%')
print(f'  Delta over CoT: {(cv_acc - cot_acc)*100:+.1f}pp')

print(f'\n{"="*70}')
print(f'FINAL COMPARISON')
print(f'{"="*70}')
print(f'  Pure LSFT(2):  {latent_acc*100:.1f}%')
print(f'  Pure CoT:      {cot_acc*100:.1f}%')
print(f'  ---')
print(f'  In-dist probe AUC:  {mean_auc:.3f} +/- {std_auc:.3f}')
print(f'  Cross-r probe AUC:  0.805 +/- 0.011 (paper)')
print(f'  ---')
print(f'  Routing (in-dist):  {cv_acc*100:.1f} +/- {cv_std*100:.1f}% | saving {cv_saving*100:.1f}% | delta {(cv_acc-cot_acc)*100:+.1f}pp')
print(f'  Routing (cross-r):  52.5 +/- 1.5%           | saving 43%    | delta +2.0pp (paper)')
