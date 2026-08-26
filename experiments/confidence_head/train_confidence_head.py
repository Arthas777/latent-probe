"""
Step 2: Train a Calibrated Confidence Head on latent trajectory features.

Architecture options:
  A) MLP on aggregated features (T, H_spec, mean_entropy, etc.) — lightweight baseline
  B) Attention-pooling over per-step hidden states + T — the main proposed method

Both are trained with BCE loss to predict P(correct | trajectory, T).

Usage:
    python train_confidence_head.py \
        --trajectory_file ./trajectory_data/trajectories_GSM8k.pt \
        --output_dir ./trained_heads \
        --model_type attention  # or 'mlp'
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm


def roc_auc_score(y_true, y_score):
    """Manual AUC implementation (no sklearn dependency)."""
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


# ============================================================
# Model Architectures
# ============================================================

class MLPConfidenceHead(nn.Module):
    """Lightweight MLP on hand-crafted trajectory features."""

    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttentionConfidenceHead(nn.Module):
    """
    Attention-pooling over per-step hidden states, conditioned on T.

    g_phi({h_t}_{t=1}^T, T) → P(correct)

    Attention-pooling over per-step hidden states, conditioned on T.
    """

    def __init__(self, hidden_dim=2048, proj_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.T_embed = nn.Sequential(
            nn.Linear(1, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        # Self-attention to summarize trajectory
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(proj_dim)
        # Final prediction
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, hidden_states, mask, T):
        """
        hidden_states: [B, max_T, hidden_dim]
        mask: [B, max_T] — 1 for valid, 0 for pad
        T: [B] — trajectory lengths (float)
        """
        x = self.proj(hidden_states)  # [B, max_T, proj_dim]

        # Self-attention with mask
        key_padding_mask = ~mask.bool()  # True = ignore
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)

        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1).float()  # [B, max_T, 1]
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # [B, proj_dim]

        # T embedding
        T_feat = self.T_embed(T.unsqueeze(-1))  # [B, proj_dim]

        # Concatenate and predict
        combined = torch.cat([pooled, T_feat], dim=-1)  # [B, proj_dim*2]
        logit = self.head(combined).squeeze(-1)  # [B]
        return logit


class HybridConfidenceHead(nn.Module):
    """
    Combines per-step hidden state attention with hand-crafted spectral features.
    Best of both worlds: learned representation + domain knowledge.
    """

    def __init__(self, hidden_dim=2048, proj_dim=128, n_heads=4,
                 n_scalar_features=8, dropout=0.1):
        super().__init__()
        self.attn_head = AttentionConfidenceHead(
            hidden_dim=hidden_dim, proj_dim=proj_dim,
            n_heads=n_heads, dropout=dropout
        )
        # Scalar feature branch
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalar_features, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2 + proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )
        # Override the attention head's final layer
        self.proj = self.attn_head.proj
        self.T_embed = self.attn_head.T_embed
        self.attn = self.attn_head.attn
        self.norm = self.attn_head.norm

    def forward(self, hidden_states, mask, T, scalar_features):
        """
        scalar_features: [B, n_scalar_features] — hand-crafted features
        """
        x = self.proj(hidden_states)
        key_padding_mask = ~mask.bool()
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        T_feat = self.T_embed(T.unsqueeze(-1))
        scalar_feat = self.scalar_net(scalar_features)

        combined = torch.cat([pooled, T_feat, scalar_feat], dim=-1)
        logit = self.fusion(combined).squeeze(-1)
        return logit


# ============================================================
# Dataset
# ============================================================

class TrajectoryDataset(Dataset):
    """Dataset of collected trajectory features."""

    def __init__(self, trajectory_file, max_T=64):
        self.data = torch.load(trajectory_file, map_location='cpu')
        self.max_T = max_T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        T = item["T"]
        hidden_dim = item["hidden_states"].shape[-1] if item["hidden_states"].numel() > 0 else 2048

        # Pad/truncate hidden states to max_T
        hs = item["hidden_states"]  # [T, hidden_dim]
        if T == 0:
            hs_padded = torch.zeros(self.max_T, hidden_dim)
            mask = torch.zeros(self.max_T)
        elif T > self.max_T:
            hs_padded = hs[:self.max_T]
            mask = torch.ones(self.max_T)
            T = self.max_T
        else:
            hs_padded = torch.zeros(self.max_T, hidden_dim)
            hs_padded[:T] = hs
            mask = torch.zeros(self.max_T)
            mask[:T] = 1.0

        # Scalar features
        ent = item["token_entropies"]
        top1 = item["top1_probs"]
        spec = item["spectral_features"]

        mean_ent = ent.mean().item() if len(ent) > 0 else 0.0
        std_ent = ent.std().item() if len(ent) > 1 else 0.0
        mean_top1 = top1.mean().item() if len(top1) > 0 else 0.0
        last_top1 = top1[-1].item() if len(top1) > 0 else 0.0

        scalar_features = torch.tensor([
            float(item["T"]),
            np.log(max(item["T"], 1)),
            mean_ent,
            std_ent,
            mean_top1,
            last_top1,
            spec.get("H_spec", 0.0),
            spec.get("sv_entropy_norm", 0.0),
        ], dtype=torch.float32)

        return {
            "hidden_states": hs_padded.float(),  # [max_T, hidden_dim]
            "mask": mask.float(),  # [max_T]
            "T": torch.tensor(float(item["T"]), dtype=torch.float32),
            "scalar_features": scalar_features,  # [8]
            "label": torch.tensor(float(item["correct"]), dtype=torch.float32),
        }


# ============================================================
# Training
# ============================================================

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    probs = np.array(probs)
    labels = np.array(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
    return ece


def evaluate(model, dataloader, device, model_type='attention'):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)

            if model_type == 'mlp':
                logits = model(batch['scalar_features'].to(device))
            elif model_type == 'attention':
                logits = model(
                    batch['hidden_states'].to(device),
                    batch['mask'].to(device),
                    batch['T'].to(device),
                )
            else:  # hybrid
                logits = model(
                    batch['hidden_states'].to(device),
                    batch['mask'].to(device),
                    batch['T'].to(device),
                    batch['scalar_features'].to(device),
                )

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # AUC
    if len(np.unique(all_labels)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels, all_probs)

    # ECE
    ece = compute_ece(all_probs, all_labels)

    # Brier score
    brier = ((all_probs - all_labels) ** 2).mean()

    return {"auc": auc, "ece": ece, "brier": brier, "probs": all_probs, "labels": all_labels}


def train(model, train_loader, val_loader, device, args, model_type='attention'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            labels = batch['label'].to(device)

            if model_type == 'mlp':
                logits = model(batch['scalar_features'].to(device))
            elif model_type == 'attention':
                logits = model(
                    batch['hidden_states'].to(device),
                    batch['mask'].to(device),
                    batch['T'].to(device),
                )
            else:  # hybrid
                logits = model(
                    batch['hidden_states'].to(device),
                    batch['mask'].to(device),
                    batch['T'].to(device),
                    batch['scalar_features'].to(device),
                )

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, model_type)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_auc": val_metrics["auc"],
            "val_ece": val_metrics["ece"],
            "val_brier": val_metrics["brier"],
        })

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} | val_auc={val_metrics['auc']:.4f} | val_ece={val_metrics['ece']:.4f} | val_brier={val_metrics['brier']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./trained_heads')
    parser.add_argument('--model_type', type=str, default='all',
                        choices=['mlp', 'attention', 'hybrid', 'all'])
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--max_T', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading trajectories from {args.trajectory_file}...")
    dataset = TrajectoryDataset(args.trajectory_file, max_T=args.max_T)
    print(f"Loaded {len(dataset)} samples.")

    # Infer hidden_dim from data
    sample = dataset[0]
    hidden_dim = sample['hidden_states'].shape[-1]
    print(f"Hidden dim: {hidden_dim}")
    args.hidden_dim = hidden_dim

    # Train/val split
    n = len(dataset)
    indices = np.random.permutation(n)
    n_val = int(n * args.val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Determine which models to train
    model_types = [args.model_type] if args.model_type != 'all' else ['mlp', 'attention', 'hybrid']

    all_results = {}

    for mt in model_types:
        print(f"\n{'='*60}")
        print(f"Training {mt.upper()} confidence head")
        print(f"{'='*60}")

        if mt == 'mlp':
            model = MLPConfidenceHead(input_dim=8, hidden_dim=64).to(device)
        elif mt == 'attention':
            model = AttentionConfidenceHead(
                hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                n_heads=4, dropout=0.1
            ).to(device)
        else:  # hybrid
            model = HybridConfidenceHead(
                hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                n_heads=4, n_scalar_features=8, dropout=0.1
            ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        history, best_auc = train(model, train_loader, val_loader, device, args, model_type=mt)

        # Final evaluation
        final_metrics = evaluate(model, val_loader, device, mt)
        print(f"\n  Final: AUC={final_metrics['auc']:.4f} | ECE={final_metrics['ece']:.4f} | Brier={final_metrics['brier']:.4f}")

        # Save model
        model_path = os.path.join(args.output_dir, f"confidence_head_{mt}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": mt,
            "args": vars(args),
            "best_auc": best_auc,
            "final_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                             for k, v in final_metrics.items() if k not in ('probs', 'labels')},
        }, model_path)
        print(f"  Saved to {model_path}")

        all_results[mt] = {
            "n_params": n_params,
            "best_auc": best_auc,
            "final_auc": final_metrics["auc"],
            "final_ece": final_metrics["ece"],
            "final_brier": final_metrics["brier"],
            "history": history,
            "val_probs": final_metrics["probs"].tolist(),
            "val_labels": final_metrics["labels"].tolist(),
        }

    # Save comprehensive results
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON: Confidence Head vs LENS Baselines")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'AUC':>8} {'ECE':>8} {'Brier':>8} {'Params':>10}")
    print("-" * 60)
    for mt, res in all_results.items():
        print(f"  {mt:<18} {res['final_auc']:>8.4f} {res['final_ece']:>8.4f} {res['final_brier']:>8.4f} {res['n_params']:>10,}")

    # Compare with LENS baseline (from existing results if available)
    print("\n  --- Baselines (from lens_signal_validation) ---")
    print(f"  {'LENS (hand-crafted)':<18} {'0.7136':>8} {'  N/A':>8} {'  N/A':>8} {'0':>10}")
    print(f"  {'Mean entropy':<18} {'0.6953':>8} {'  N/A':>8} {'  N/A':>8} {'0':>10}")


if __name__ == "__main__":
    main()
