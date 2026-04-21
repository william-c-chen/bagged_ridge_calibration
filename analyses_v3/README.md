# Option B reproduction pipeline (Tables 7 + 8 of the v3 manuscript)

This directory holds the **fully reproducible** implementation of the v3
manuscript's Option B comparison: an independent reproduction of the published
500-bag glmnet ridge ensemble on the FFPE-paired training data, contrasted
with single-model alternatives (ridge, LASSO, elastic net, OLS) — each
re-fit at 100 independent random seeds to characterize procedural variance.

## How to reproduce

```bash
bash run_all.sh
```

End-to-end runtime ≈ 25 min, dominated by step 04 (100-seed glmnet bagging).
Output is deterministic — fixed seed `RANDOM_SEED = 42` in `scripts/common.py`
governs the CV-fold shuffles and bootstrap resamples; the multi-seed scripts
sweep `42 + 100m` for `m = 1..100`.

## Pipeline (canonical scripts)

| Script | Stage | Output |
|---|---|---|
| `00_load_deployed_coefs.py`     | (Optional) Load deployed 500-bag β coefficient matrix from internal CSVs. **Not in public release.** Skipped silently if files unavailable. | `data/deployed_ensemble_true.npz` |
| `01_load_training_frozen.py`    | Extract 173-sample UCSF *frozen* discovery cohort from `dat.integ.master`; HK-normalize. (Diagnostic context — not used for ridge training.) | `data/training_173.npz` |
| `02_build_ffpe_training.py`     | Extract 64-sample paired **FFPE** training set (X = log2 HK-normalized counts, y = Chen score) from `Previous_calibration_run/original_ffpe_frozen_paired/Results Study 20220706 UCSF UDR3097.xlsx`. | `data/ffpe_paired_training_pc.{npz,csv}` |
| `03_parse_rcc_bridging.py`      | Parse 2023 + 2026 bridging RCC files; HK-normalize. | `data/bridging_normalized.npz`, `bridging_counts.json` |
| `04_glmnet_bagging_multiseed.R` | 100 independent reproductions of the published 500-bootstrap glmnet ridge bagging (`cv.glmnet`, `λ_{1se}`). | `data/glmnet_ffpe_multiseed_betas.csv`, `meta.csv` |
| `05_score_bagging_multiseed.py` | Score the 100 reproduced β̄'s on bridging samples; compute LOO-CV K=6/K=3, cross-experiment \|Δslope\|/\|Δintercept\|, probe-failure. | `data/glmnet_ffpe_multiseed_scoring.json` |
| `06_single_models_multiseed.py` | Fit single ridge, LASSO, elastic net (each at 100 CV-fold seeds) and deterministic OLS on the FFPE training set; score and metric on bridging. | `data/single_models_multiseed.json` |
| `07_make_tables.py`             | Aggregate medians (Table 7) and SDs (Table 8) from the multi-seed outputs. | `data/table7_headline_metrics.csv`, `table8_seed_variance.csv` |
| `08_per_gene_lot_drift.py`      | §3.4 per-gene lot-drift magnitude across all 9 lot pairs; identifies most/least lot-responsive genes and score-impact concentration. | `data/per_gene_lot_drift.json` |
| `09_common_cause_mechanism.py`  | §3.4 mechanism test: within-training expression SD as common cause of lot drift and ridge shrinkage ($r = +0.24$ and $-0.62$). | `data/common_cause_mechanism.json` |
| `10_misclassification_sim.py`   | Seed-variability misclassification scenario (superseded by script 12 for the paper's primary framing). | `data/misclassification_sim.json` |
| `11_k_sweep.py`                 | Bridging-panel size sweep: LOO-CV RMSE vs K for each architecture across 100 seeds. | `data/k_sweep.json` |
| `12_calibration_error_misclass.py` | §3.6 primary clinical-translation: closed-form expected misclassification rate at thresholds, per architecture and K. | `data/calibration_error_misclass.json` |
| `13_deployment_ammo.py`         | §3.6 stressor sensitivity (3$\times$ drift) and analyst range (P10-P90) supporting tables. | `data/deployment_ammo.json` |

## Headline numbers

Median across 100 random seeds, all models trained on the same 64-sample
FFPE-paired training set:

| Model | ‖β̄‖₂ | LOO K=6 | LOO K=3 | \|Δslope\| | Probe |
|---|---|---|---|---|---|
| Deployed bagged ridge | 0.101 | 0.027 | 0.042 | 0.018 | 0.62 |
| Reproduced bagged ridge | 0.090 | 0.032 | 0.036 | 0.003 | 0.85 |
| Single ridge | 0.065 | 0.039 | 0.089 | 0.046 | 0.41 |
| Single LASSO | 0.105 | 0.051 | 0.102 | 0.062 | 0.74 |
| Single elastic net | 0.103 | 0.051 | 0.101 | 0.063 | 0.73 |
| Single OLS | 0.145 | 0.100 | 0.112 | 0.133 | 1.09 |

SD across 100 random seeds (lower = less reliance on lucky CV-fold draws):

| Metric | Bagged | Ridge | LASSO | EN | Ridge/Bagged |
|---|---|---|---|---|---|
| LOO K=6 | 0.0006 | 0.0105 | 0.0104 | 0.0102 | 17× |
| LOO K=3 | 0.0007 | 0.1248 | 0.0424 | 0.0429 | **188×** |
| \|Δslope\| | 0.0025 | 0.0175 | 0.0185 | 0.0171 | 7× |

## Caveats

- **Reproduction limit**: Our 100 bagged-ridge reproductions cluster tightly
  around K=6 LOO-CV ≈ 0.032 vs.\ deployed 0.027 — close but not identical.
  Cosine similarity between reproduced and deployed β̄ is 0.82–0.83 across all
  100 seeds. The reproduction matches the deployed on cross-experiment
  reproducibility (in fact slightly better) but not on K=6 LOO-CV.
- **Deployed coefficients are NOT in this directory or in the public repo.**
  Step `00` reads them from `analysis_scripts/internal_data/raleigh75_*.csv`
  if available. Without those, downstream scripts skip deployed-comparison
  columns (cos similarity); all other metrics are computed from bridging
  scores in the public repo.

## Directory layout

```
analyses_v3/
  scripts/                      Canonical reproducible pipeline (00-07 main; 08-13 extensions for §3.4 and §3.6)
    common.py                   Shared constants, paths, normalization helpers
  data/                         Intermediate outputs (NPZ, JSON, CSV)
                                The six JSON files that back paper numbers are
                                committed (whitelisted in .gitignore); coefficient
                                matrices are not redistributed.
  figures/                      Reserved for plot outputs
  logs/                         stdout capture per script
  run_all.sh                    Single-command full pipeline
  README.md                     This file
```
