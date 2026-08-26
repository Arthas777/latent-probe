"""
Statistical significance for routing (52.5% vs 50.5%).
Paired bootstrap + McNemar's test.
"""
import torch, json, numpy as np
from pathlib import Path
from scipy import stats

results_dir = Path(__file__).parent.parent / 'confidence_head' / 'routing_results_v2'
output_dir = Path(__file__).parent / 'results'
output_dir.mkdir(exist_ok=True)

# Load per-problem results
latent_data = torch.load(results_dir / 'latent2_results.pt', map_location='cpu', weights_only=False)
cot_data = torch.load(results_dir / 'cot_sft_results.pt', map_location='cpu', weights_only=False)

N = len(latent_data)
confs = np.array([r['conf'] for r in latent_data])
latent_correct = np.array([int(r['correct']) for r in latent_data])
cot_correct = np.array([int(r['correct']) for r in cot_data])

# CV-deployable threshold (from results: τ ≈ 0.02-0.03 for argmax-acc)
# Use τ=0.03 as the CV-selected threshold (corresponds to ~59% latent fraction)
tau_cv = 0.03
latent_mask = confs > tau_cv
routing_correct = np.where(latent_mask, latent_correct, cot_correct)

routing_acc = routing_correct.mean()
cot_acc = cot_correct.mean()
print(f"N = {N}")
print(f"Routing acc: {routing_acc*100:.2f}%")
print(f"Pure CoT acc: {cot_acc*100:.2f}%")
print(f"Δ: {(routing_acc - cot_acc)*100:.2f}pp")
print(f"Latent fraction: {latent_mask.mean()*100:.1f}%")

# McNemar's Test
n10 = ((routing_correct == 1) & (cot_correct == 0)).sum()  # routing wins
n01 = ((routing_correct == 0) & (cot_correct == 1)).sum()  # CoT wins
print(f"\nMcNemar's test:")
print(f"  Routing correct, CoT wrong (N10): {n10}")
print(f"  Routing wrong, CoT correct (N01): {n01}")
print(f"  Net gain: {n10 - n01}")

if n10 + n01 > 0:
    chi2 = (n10 - n01) ** 2 / (n10 + n01)
    p_mcnemar = 1 - stats.chi2.cdf(chi2, df=1)
    print(f"  χ² = {chi2:.2f}, p = {p_mcnemar:.4f}")
else:
    chi2, p_mcnemar = 0, 1.0
    print("  No discordant pairs!")

# Paired Bootstrap
np.random.seed(42)
n_bootstrap = 10000
boot_diffs = []
for _ in range(n_bootstrap):
    idx = np.random.choice(N, size=N, replace=True)
    boot_routing = routing_correct[idx].mean()
    boot_cot = cot_correct[idx].mean()
    boot_diffs.append(boot_routing - boot_cot)

boot_diffs = np.array(boot_diffs)
ci_lower = np.percentile(boot_diffs, 2.5) * 100
ci_upper = np.percentile(boot_diffs, 97.5) * 100
p_bootstrap = (boot_diffs <= 0).mean()

print(f"\nPaired Bootstrap (10000 resamples):")
print(f"  Mean Δ: {boot_diffs.mean()*100:.2f}pp")
print(f"  95% CI: [{ci_lower:.2f}pp, {ci_upper:.2f}pp]")
print(f"  P(routing ≤ CoT): {p_bootstrap:.4f}")

# Also test peak τ=0.05
latent_mask_peak = confs > 0.05
routing_peak = np.where(latent_mask_peak, latent_correct, cot_correct)
peak_acc = routing_peak.mean()
n10_peak = ((routing_peak == 1) & (cot_correct == 0)).sum()
n01_peak = ((routing_peak == 0) & (cot_correct == 1)).sum()
chi2_peak = (n10_peak - n01_peak) ** 2 / (n10_peak + n01_peak) if (n10_peak + n01_peak) > 0 else 0
p_peak = 1 - stats.chi2.cdf(chi2_peak, df=1)

print(f"\nPeak τ=0.05:")
print(f"  Acc: {peak_acc*100:.2f}%, Δ: {(peak_acc-cot_acc)*100:.2f}pp")
print(f"  McNemar N10={n10_peak}, N01={n01_peak}, χ²={chi2_peak:.2f}, p={p_peak:.4f}")

results = {
    'cv_tau': float(tau_cv),
    'routing_acc': float(routing_acc),
    'cot_acc': float(cot_acc),
    'delta_pp': float((routing_acc - cot_acc) * 100),
    'mcnemar': {'n10': int(n10), 'n01': int(n01), 'chi2': float(chi2), 'p': float(p_mcnemar)},
    'bootstrap': {'mean_delta_pp': float(boot_diffs.mean()*100), 'ci_lower_pp': float(ci_lower), 'ci_upper_pp': float(ci_upper), 'p_leq_0': float(p_bootstrap)},
    'peak_tau005': {'acc': float(peak_acc), 'delta_pp': float((peak_acc-cot_acc)*100), 'mcnemar_p': float(p_peak)}
}

with open(output_dir / 'routing_significance.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output_dir / 'routing_significance.json'}")
