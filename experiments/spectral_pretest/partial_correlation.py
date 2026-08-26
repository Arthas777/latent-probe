"""
Partial correlation analysis: does the correlation between spectral features
and correctness persist after controlling for latent sequence length L?

Method:
1. Bucket by L (L=1,2,...,max), compute Spearman rho within each bucket
2. Weighted average by bucket size -> conditional rho
3. Residual-based continuous partial correlation as cross-validation
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

results_dir = Path(__file__).resolve().parent / "results"
with open(results_dir / "d2_spectral_results.json") as f:
    results = json.load(f)

correctness = np.array([r["correct"] for r in results])
n_latent = np.array([r["n_latent"] for r in results])
spectral_entropy = np.array([r["spectral_entropy"] for r in results])
mean_cos_sim = np.array([r["mean_cos_sim"] for r in results])
effective_rank = np.array([r["effective_rank"] for r in results])
top_sv_ratio = np.array([r["top_sv_ratio"] for r in results])

# ============================================================
# 1. Bucketed partial correlation
# ============================================================
unique_L = sorted(set(n_latent))
print(f"Latent length distribution: {dict(zip(*np.unique(n_latent, return_counts=True)))}")
print(f"Total samples: {len(results)}, accuracy: {correctness.mean():.4f}\n")

metrics = {
    "spectral_entropy": spectral_entropy,
    "mean_cos_sim": mean_cos_sim,
    "effective_rank": effective_rank,
    "top_sv_ratio": top_sv_ratio,
}

print("=" * 80)
print("Bucketed partial correlation: rho(metric, correct | L)")
print("=" * 80)

bucket_results = {}

for mname, mvals in metrics.items():
    print(f"\n--- {mname} ---")
    print(f"{'L':>4} | {'N':>5} | {'N_correct':>9} | {'Acc':>6} | {'ρ':>8} | {'p':>10} | {'mean_correct':>13} | {'mean_wrong':>11}")
    print("-" * 90)

    rho_list = []
    weight_list = []
    all_bucket_data = []

    for L in unique_L:
        mask = n_latent == L
        n_in_bucket = mask.sum()
        if n_in_bucket < 10:
            continue

        c = correctness[mask]
        v = mvals[mask]
        n_correct = int(c.sum())
        acc = c.mean()

        if c.std() == 0 or v.std() == 0:
            rho, p = 0.0, 1.0
        else:
            rho, p = stats.spearmanr(v, c)

        correct_vals = v[c == 1]
        wrong_vals = v[c == 0]
        mc = correct_vals.mean() if len(correct_vals) > 0 else float('nan')
        mw = wrong_vals.mean() if len(wrong_vals) > 0 else float('nan')

        print(f"{L:>4} | {n_in_bucket:>5} | {n_correct:>9} | {acc:>6.3f} | {rho:>8.4f} | {p:>10.2e} | {mc:>13.4f} | {mw:>11.4f}")

        rho_list.append(rho)
        weight_list.append(n_in_bucket)
        all_bucket_data.append({
            "L": int(L), "N": int(n_in_bucket), "rho": rho, "p": p,
            "acc": float(acc), "mean_correct": float(mc), "mean_wrong": float(mw),
        })

    weights = np.array(weight_list, dtype=float)
    rhos = np.array(rho_list)
    weighted_rho = np.average(rhos, weights=weights)
    print(f"\n  Weighted avg rho|L = {weighted_rho:.4f}  (vs unconditional rho = {stats.spearmanr(mvals, correctness)[0]:.4f})")

    bucket_results[mname] = {
        "weighted_rho": weighted_rho,
        "unconditional_rho": float(stats.spearmanr(mvals, correctness)[0]),
        "buckets": all_bucket_data,
    }

# ============================================================
# 2. Continuous partial correlation (manual, no extra dependencies)
# ============================================================
print("\n\n" + "=" * 80)
print("Continuous partial correlation (residual method)")
print("=" * 80)
print("Spearman rho between residuals after regressing out L\n")

def partial_spearman(x, y, z):
    """ρ(x, y | z) via residual method"""
    # Rank-based regression
    from numpy.polynomial.polynomial import polyfit
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # Residuals
    coef_x = np.polyfit(rz, rx, 1)
    res_x = rx - np.polyval(coef_x, rz)
    coef_y = np.polyfit(rz, ry, 1)
    res_y = ry - np.polyval(coef_y, rz)
    return stats.spearmanr(res_x, res_y)

for mname, mvals in metrics.items():
    rho_raw, p_raw = stats.spearmanr(mvals, correctness)
    rho_partial, p_partial = partial_spearman(mvals, correctness, n_latent)
    print(f"{mname:>20s}:  raw ρ = {rho_raw:+.4f} (p={p_raw:.1e})  |  partial ρ|L = {rho_partial:+.4f} (p={p_partial:.1e})")
    bucket_results[mname]["partial_rho_residual"] = float(rho_partial)
    bucket_results[mname]["partial_p_residual"] = float(p_partial)

# Latent length itself
rho_len, p_len = stats.spearmanr(n_latent, correctness)
print(f"\n{'n_latent':>20s}:  ρ = {rho_len:+.4f} (p={p_len:.1e})")

# ============================================================
# 3. Visualization
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Partial Correlation Analysis: Controlling for Latent Length L",
             fontsize=14, fontweight='bold')

# (a) Bucketed rho
ax = axes[0]
for mname, color, marker in [
    ("spectral_entropy", "#e74c3c", "o"),
    ("mean_cos_sim", "#3498db", "s"),
    ("effective_rank", "#2ecc71", "^"),
    ("top_sv_ratio", "#9b59b6", "D"),
]:
    bdata = bucket_results[mname]["buckets"]
    Ls = [b["L"] for b in bdata]
    rhos = [b["rho"] for b in bdata]
    Ns = [b["N"] for b in bdata]
    ax.plot(Ls, rhos, marker=marker, label=mname, color=color, linewidth=1.5, markersize=6)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel("Latent Sequence Length L")
ax.set_ylabel("Spearman rho (within bucket)")
ax.set_title("(a) Per-bucket rho(metric, correct)")
ax.legend(fontsize=8)
ax.set_ylim(-0.5, 0.5)

# (b) Bar chart: raw rho vs partial rho
ax = axes[1]
mnames = ["spectral_entropy", "mean_cos_sim", "effective_rank", "top_sv_ratio"]
short_names = ["H(E_c)", "cos_sim", "erank", "sv_ratio"]
x_pos = np.arange(len(mnames))
raw_rhos = [bucket_results[m]["unconditional_rho"] for m in mnames]
partial_rhos = [bucket_results[m]["partial_rho_residual"] for m in mnames]
weighted_rhos = [bucket_results[m]["weighted_rho"] for m in mnames]

w = 0.25
ax.bar(x_pos - w, raw_rhos, w, label="Raw rho", color="#e74c3c", alpha=0.8)
ax.bar(x_pos, weighted_rhos, w, label="Weighted bucket rho", color="#f39c12", alpha=0.8)
ax.bar(x_pos + w, partial_rhos, w, label="Partial rho (residual)", color="#3498db", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel("Spearman rho")
ax.set_title("(b) Raw vs Partial rho (controlling L)")
ax.legend(fontsize=9)

for i, (r1, r2, r3) in enumerate(zip(raw_rhos, weighted_rhos, partial_rhos)):
    ax.text(i - w, r1 - 0.02, f"{r1:.3f}", ha='center', va='top', fontsize=8)
    ax.text(i, r2 - 0.02, f"{r2:.3f}", ha='center', va='top', fontsize=8)
    ax.text(i + w, r3 - 0.02, f"{r3:.3f}", ha='center', va='top', fontsize=8)

# (c) Bucket sample count distribution
ax = axes[2]
all_Ls = sorted(set(n_latent))
counts = [int((n_latent == L).sum()) for L in all_Ls]
accs = [float(correctness[n_latent == L].mean()) if (n_latent == L).sum() > 0 else 0 for L in all_Ls]

ax2 = ax.twinx()
bars = ax.bar(all_Ls, counts, color='#bdc3c7', edgecolor='gray', label='N samples')
line = ax2.plot(all_Ls, accs, 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Accuracy')
ax.set_xlabel("Latent Sequence Length L")
ax.set_ylabel("Number of Samples")
ax2.set_ylabel("Accuracy", color='#e74c3c')
ax.set_title("(c) Sample Distribution & Accuracy by L")
ax.set_xlim(0, max(all_Ls) + 1)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(results_dir / "d2_partial_correlation.png", dpi=150, bbox_inches='tight')
plt.savefig(results_dir / "d2_partial_correlation.pdf", bbox_inches='tight')
print(f"\nFigure saved to {results_dir / 'd2_partial_correlation.pdf'}")

# Save data
with open(results_dir / "d2_partial_correlation.json", "w") as f:
    json.dump(bucket_results, f, indent=2)

# ============================================================
# Conclusion
# ============================================================
print("\n" + "=" * 80)
print("Conclusion")
print("=" * 80)
for mname in mnames:
    raw = bucket_results[mname]["unconditional_rho"]
    partial = bucket_results[mname]["partial_rho_residual"]
    reduction = 1 - abs(partial) / abs(raw) if abs(raw) > 0.001 else float('nan')
    print(f"{mname:>20s}:  raw={raw:+.4f}  partial|L={partial:+.4f}  reduction={reduction:+.1%}")
