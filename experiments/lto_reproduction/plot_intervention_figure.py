"""
Experiment B: Reframe stochasticity intervention data as mechanistic evidence.

Generates Figure for §10: Pooled AUC (stable) vs Within-problem AUC (at 0.5)
across multiple stochasticity injection scales.

Key message: stochasticity injection creates rollout diversity (oracle rises),
but the trained head's within-problem AUC stays at ~0.5 — proving the signal
is problem-level, not trajectory-level. Retraining on stochastic data doesn't help.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_colar_results():
    """Load CoLaR stochasticity results across sigma scales."""
    base_dir = os.path.join(PROJ_ROOT, 'experiments', 'confidence_head', 'direction1_results')
    sigmas = [1.0, 100.0, 1000.0, 5000.0, 10000.0]
    results = []
    for sigma in sigmas:
        path = os.path.join(base_dir, f'colar_sigma{sigma}_results.json')
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            results.append({
                'sigma': sigma,
                'oracle_delta': d['oracle_delta'],
                'within_problem_auc': d['within_problem_auc'],
                'n_mixed': d['n_mixed_problems'],
                'head_acc': d.get('head_acc', None),
                'det_acc': d.get('det_acc', None),
            })
    return results


def load_lsft_results():
    """Load LSFT stochasticity results."""
    base_dir = os.path.join(PROJ_ROOT, 'experiments', 'confidence_head', 'direction1_results')
    path = os.path.join(base_dir, 'lsft_full_results.json')
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return {
            'eps': d['config']['eps'],
            'oracle_delta': d['oracle_delta'],
            'within_problem_auc': d['within_problem_auc'],
            'n_mixed': d['n_mixed_problems'],
            'head_acc': d.get('head_acc', None),
            'det_acc': d.get('det_acc', None),
        }
    return None


def load_retrained_results():
    """Load results from retraining head on stochastic CoLaR data."""
    base_dir = os.path.join(PROJ_ROOT, 'experiments', 'confidence_head', 'direction1_results')
    path = os.path.join(base_dir, 'colar_retrained_quick_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_lto_results():
    """Load LTO reproduction results."""
    path = os.path.join(PROJ_ROOT, 'experiments', 'lto_reproduction', 'results',
                        'lto_reproduction_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_main_figure(colar_results, lsft_result, retrained_result, output_dir):
    """
    Main figure: Two-panel plot.
    Left: Oracle ceiling vs stochasticity scale (diversity exists)
    Right: Within-problem AUC vs stochasticity scale (head can't use it)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Data
    sigmas = [r['sigma'] for r in colar_results]
    oracle_deltas = [r['oracle_delta'] * 100 for r in colar_results]
    within_aucs = [r['within_problem_auc'] for r in colar_results]

    # Left panel: Oracle ceiling shows diversity exists
    ax = axes[0]
    ax.plot(range(len(sigmas)), oracle_deltas, 'o-', color='#2196F3', linewidth=2,
            markersize=8, label='CoLaR oracle Δ')

    # Add LSFT point
    if lsft_result:
        ax.axhline(lsft_result['oracle_delta'] * 100, color='#FF9800', linestyle='--',
                   linewidth=1.5, label=f'LSFT ε=0.001 ({lsft_result["oracle_delta"]*100:.1f}pp)')

    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'σ={int(s)}' if s >= 1 else f'σ={s}' for s in sigmas], fontsize=9)
    ax.set_ylabel('Oracle ceiling Δ (pp)', fontsize=11)
    ax.set_xlabel('Stochasticity injection magnitude', fontsize=11)
    ax.set_title('(a) Rollout diversity exists', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(oracle_deltas) * 1.3)

    # Right panel: Within-problem AUC stays at 0.5
    ax = axes[1]
    ax.plot(range(len(sigmas)), within_aucs, 's-', color='#F44336', linewidth=2,
            markersize=8, label='CoLaR within-problem AUC')

    # Add LSFT point
    if lsft_result:
        ax.axhline(lsft_result['within_problem_auc'], color='#FF9800', linestyle='--',
                   linewidth=1.5, label=f'LSFT ε=0.001 ({lsft_result["within_problem_auc"]:.3f})')

    # Add retrained point
    if retrained_result and 'within_problem_auc' in retrained_result:
        retrained_auc = retrained_result['within_problem_auc']
        ax.axhline(retrained_auc, color='#9C27B0', linestyle=':',
                   linewidth=2, label=f'Retrained head ({retrained_auc:.3f})')

    # Random baseline
    ax.axhline(0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Random (0.5)')

    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'σ={int(s)}' if s >= 1 else f'σ={s}' for s in sigmas], fontsize=9)
    ax.set_ylabel('Within-problem AUC', fontsize=11)
    ax.set_xlabel('Stochasticity injection magnitude', fontsize=11)
    ax.set_title('(b) Head cannot discriminate within-problem', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'intervention_figure.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_lto_comparison(lto_results, output_dir):
    """
    Figure: LTO Algorithm 1 performance across β values.
    Shows that LTO is ineffective on compression-based paradigms.
    """
    if lto_results is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    colors = ['#2196F3', '#FF9800', '#F44336']
    markers = ['o', 's', '^']

    for i, setting in enumerate(lto_results):
        betas = sorted(setting['lto_results'].keys(), key=float)
        accs = [setting['lto_results'][b]['acc_mean'] for b in betas]
        stds = [setting['lto_results'][b]['acc_std'] for b in betas]

        label = setting['setting'].split(':')[1].strip() if ':' in setting['setting'] else setting['setting']
        ax.errorbar(range(len(betas)), accs, yerr=stds,
                    fmt=f'{markers[i]}-', color=colors[i], linewidth=1.5,
                    markersize=6, capsize=3, label=label)
        # Det baseline
        ax.axhline(setting['det_acc'], color=colors[i], linestyle=':', alpha=0.5)

    # Huginn reference
    ax.axhline(0.385, color='#4CAF50', linestyle='--', linewidth=2,
               alpha=0.7, label='LTO on Huginn (+5.9pp)')
    ax.axhline(0.326, color='#4CAF50', linestyle=':', alpha=0.5)

    betas = sorted(lto_results[0]['lto_results'].keys(), key=float)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{float(b):.3f}' if float(b) < 0.01 else f'{float(b):.2f}' if float(b) < 1 else f'{float(b):.1f}' for b in betas],
                       fontsize=8, rotation=45)
    ax.set_xlabel('β (KL regularization weight)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('LTO Algorithm 1: Paradigm-Conditional Effectiveness', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'lto_beta_comparison.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def generate_summary_table(colar_results, lsft_result, retrained_result, lto_results, output_dir):
    """Generate a LaTeX-ready summary table."""
    lines = []
    lines.append("% Auto-generated from plot_intervention_figure.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{LTO Algorithm~1 fails on compression-based paradigms. "
                 "Huginn-3.5B gains +5.9pp; our paradigms gain $\\leq$+0.3pp despite "
                 "oracle ceilings of +3.8--5.3pp.}")
    lines.append("\\label{tab:lto-comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Method & Paradigm & Stoch. & $\\beta$ & Acc & $\\Delta$ \\\\")
    lines.append("\\midrule")

    if lto_results:
        for res in lto_results:
            paradigm = "LSFT(2)" if "LSFT(2)" in res['setting'] else \
                       "LSFT(4)" if "LSFT" in res['setting'] else "CoLaR"
            stoch = "none" if "deterministic" in res['setting'] else \
                    "$\\epsilon$=0.001" if "0.001" in res['setting'] else "$\\sigma$=1000"
            det_acc = res['det_acc']
            best_beta = max(res['lto_results'].items(), key=lambda x: x[1]['acc_mean'])
            beta_str, best_r = best_beta
            delta = best_r['delta_vs_det']
            delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"

            lines.append(f"Base model & {paradigm} & {stoch} & -- & {det_acc:.3f} & -- \\\\")
            lines.append(f"LTO Alg.~1 & {paradigm} & {stoch} & {beta_str} & {best_r['acc_mean']:.3f} & {delta_str} \\\\")

    lines.append("\\midrule")
    lines.append("LTO (reported) & Huginn-3.5B & native & best & 0.385 & +0.059 \\\\")
    lines.append("Our routing & LSFT(2) & none & -- & 0.525 & +0.025 \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path = os.path.join(output_dir, 'table_lto_comparison.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {out_path}")


def main():
    output_dir = os.path.join(PROJ_ROOT, 'experiments', 'lto_reproduction', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    colar_results = load_colar_results()
    lsft_result = load_lsft_results()
    retrained_result = load_retrained_results()
    lto_results = load_lto_results()

    print(f"  CoLaR results: {len(colar_results)} sigma values")
    print(f"  LSFT result: {'found' if lsft_result else 'missing'}")
    print(f"  Retrained: {'found' if retrained_result else 'missing'}")
    print(f"  LTO results: {'found' if lto_results else 'missing'}")

    if retrained_result:
        print(f"  Retrained head details: {json.dumps(retrained_result, indent=2)[:300]}")

    print("\nGenerating figures...")
    plot_main_figure(colar_results, lsft_result, retrained_result, output_dir)
    plot_lto_comparison(lto_results, output_dir)
    generate_summary_table(colar_results, lsft_result, retrained_result, lto_results, output_dir)

    # Print intervention data summary for paper writing
    print("\n" + "="*70)
    print("INTERVENTION DATA SUMMARY (for §10 writing)")
    print("="*70)
    print("\nCoLaR σ-scaling results:")
    print(f"{'σ':<10} {'Oracle Δ (pp)':<15} {'Within-prob AUC':<18} {'Mixed problems':<15}")
    print("-"*58)
    for r in colar_results:
        print(f"{r['sigma']:<10.0f} {r['oracle_delta']*100:<15.1f} {r['within_problem_auc']:<18.4f} {r['n_mixed']:<15}")

    if lsft_result:
        print(f"\nLSFT ε=0.001: oracle Δ={lsft_result['oracle_delta']*100:.1f}pp, "
              f"within-prob AUC={lsft_result['within_problem_auc']:.4f}, "
              f"n_mixed={lsft_result['n_mixed']}")

    if retrained_result:
        print(f"\nRetrained head on σ=1000 data: {json.dumps(retrained_result, indent=2)[:500]}")


if __name__ == "__main__":
    main()
