# Decoding Self-Knowledge from Latent Reasoning Models for Selective Prediction and Adaptive Routing

Code for the paper *"Decoding Self-Knowledge from Latent Reasoning Models for Selective Prediction and Adaptive Routing"*, accepted at EMNLP 2026 Findings.

## Overview

We study whether compression-based latent reasoning models retain decodable self-knowledge about reasoning correctness. We find a sharp dichotomy:

- **Problem-level signal exists**: A lightweight attention probe (~0.4M params, 378,369 on the 1B backbone) achieves AUC 0.805/0.853 on Latent-SFT/CoLaR, transfers zero-shot across paradigms, datasets, and compression ratios.
- **Trajectory-level signal does not**: The probe cannot rank rollouts of the same problem (within-problem AUC ~ 0.5), causing LTO-style best-of-N to fail (<=+0.3pp).
- **Practical applications**: Selective prediction (+20-25pp at 50% coverage); adaptive routing (52.5% acc at 43% token saving, exceeding both pure solvers).

## Setup

### Dependencies

```bash
conda create -n latent-probe python=3.10
conda activate latent-probe
pip install -r requirements.txt
```

### External Model Checkpoints

This project analyzes two latent reasoning paradigms. You need to obtain their checkpoints from their original authors:

| Paradigm | Source | Expected Path |
|----------|--------|---------------|
| **Latent-SFT** | Released by the original authors (cited in Section 3) | `./checkpoints/latent-sft-r4/` |
| **CoLaR** | Released by the original authors (cited in Section 3) | `./checkpoints/colar/` |
| **CoT-SFT** (base) | Released by CoLaR authors | `./checkpoints/cot-sft/` |
| **Llama-3.2-1B-Instruct** | Publicly available on HuggingFace | (HuggingFace cache or local) |

### Data

- **GSM8k-Aug**: Included in `experiments/spectral_pretest/data/gsm8k_test.json` (test split).
- **SVAMP**: Downloaded automatically via the collection scripts.

## Repository Structure

```
latent-probe/
├── src/                         # Minimal utilities for CoLaR inference
│   ├── modules/projector.py     # LatentPolicy (CoLaR's latent head)
│   ├── utils/constants.py       # Model embedding std constants
│   └── utils/utils.py           # Position ID utilities
├── experiments/
│   ├── spectral_pretest/        # §4.1: Spectral features & length confound analysis
│   ├── lens/                    # §4.2: LENS closed-form baseline
│   ├── confidence_head/         # §5-6: Probe training, selective prediction, routing
│   ├── lto_reproduction/        # §7: LTO best-of-N reproduction (failure case)
│   └── prompt_only_baseline/    # Appendix: Prompt-only & routing baselines
├── requirements.txt
└── README.md
```

## Quick Verification (GPU Required)

Pre-collected trajectories are included, so you can directly reproduce the main probe results (Table 3) without re-running model inference:

```bash
cd experiments/confidence_head

# Latent-SFT probe AUC (paper: 0.805 +/- 0.011)
python eval_crossval.py \
    --trajectory_file ./trajectory_data/trajectories_GSM8k.pt \
    --output_dir ./eval_results_verify \
    --device cuda:0

# CoLaR probe AUC (paper: 0.853 +/- 0.004)
python eval_crossval.py \
    --trajectory_file ./trajectory_data_colar/trajectories_GSM8k_colar.pt \
    --output_dir ./eval_results_verify_colar \
    --device cuda:0
```

Each run takes ~20 minutes on a single GPU (trains MLP, Attention, and Hybrid probes with 5-fold CV). This reproduces:
- **§5 (Table 3)**: Probe AUC with 5-fold nested CV + LENS baseline + all baselines
- **§6.1**: Selective prediction accuracy at multiple coverage levels

## Reproducing Experiments

### Step 1: Collect Trajectories

**Latent-SFT trajectories** (requires Latent-SFT checkpoint):
```bash
cd experiments/confidence_head
python collect_trajectories.py \
    --latent_model_path /path/to/checkpoints/latent-sft-r4 \
    --dataset GSM8k \
    --output_dir ./trajectory_data
```

**CoLaR trajectories** (requires CoLaR checkpoint):
```bash
python collect_trajectories_colar.py \
    --ckpt_path /path/to/checkpoints/colar_best.ckpt \
    --output_dir ./trajectory_data_colar
```

### Step 2: Train Confidence Probe

```bash
python train_confidence_head.py \
    --trajectory_file ./trajectory_data/trajectories_GSM8k.pt \
    --output_dir ./trained_heads \
    --model_type attention
```

### Step 3: Evaluate

**Nested cross-validation (§5, main probe result)**:
```bash
python eval_crossval.py
```

**Selective prediction (§6.1)**:
```bash
python eval_selective_complete.py
```

**Adaptive routing (§6.2)**:
```bash
python adaptive_routing.py
```

**LTO reproduction (§7)**:
```bash
cd ../lto_reproduction
python lto_algorithm1.py
```

### Spectral Pretest & LENS (§4)

```bash
cd experiments/spectral_pretest
python run_spectral_pretest.py --ckpt_path /path/to/checkpoints/colar_best.ckpt

cd ../lens
python run_lens_collect.py --n_rollouts 8 --seed 42
python analyze_lens.py
```

## Pre-computed Results

Trajectory data and trained probe weights are included for convenience (no need to re-run Step 1-2 for evaluation):

- `experiments/confidence_head/trajectory_data/` — Latent-SFT hidden states
- `experiments/confidence_head/trajectory_data_colar/` — CoLaR hidden states
- `experiments/confidence_head/trained_heads/` — Trained probe checkpoints (~3MB)
- `experiments/confidence_head/routing_results_official/` — Routing evaluation results

## Citation

```bibtex
@inproceedings{latent-probe-2026,
  title     = {Decoding Self-Knowledge from Latent Reasoning Models for Selective Prediction and Adaptive Routing},
  author    = {Author Names},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

This code is released under the MIT License. See [LICENSE](LICENSE).
