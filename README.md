# Cross-Lot Calibration of a Bagged Ridge Meningioma Biomarker — Reproduction Code and Data

This repository contains de-identified data and Python analysis code sufficient to reproduce the main tables and figures of:

> Chen WC, Braman B, Chanoutsi N, Mirchia K, Raleigh DR. Cross-lot calibration of a bagged ridge meningioma biomarker: ensemble generalization, cross-experiment reproducibility, and comparison with alternative regularizers. 2026.

The paper characterizes how a 34-gene meningioma gene-expression risk score ([Chen et al., *Nature Medicine* 2023](https://www.nature.com/articles/s41591-023-02586-z)) deployed as a 500-model bagged ridge ensemble behaves across codeset manufacturing lots of the NanoString nCounter platform. See the paper for full methodology.

## Quick start

```bash
git clone https://github.com/william-c-chen/bagged_ridge_calibration.git
cd bagged_ridge_calibration
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_cross_experiment_reproducibility.py
python scripts/02_leave_one_out_cv.py
python scripts/03_coefficient_correlations.py
python scripts/04_ridge_vs_sparsified.py
python scripts/05_hk_and_pos.py
python scripts/06_master_comparison.py
```

Each script prints a self-contained block of results corresponding to specific tables/sections of the paper.

## Repository layout

```
data/
  biomarker_gene_list.csv         The 34 biomarker genes
  scores_2023.csv                 2023 cohort ensemble scores (5 samples x 3 lots)
  scores_2026.csv                 2026 cohort ensemble scores (10 samples x 4 lots)
  normalized_expression_2026.csv  Housekeeping-normalized log2 expression (34 genes x 40 lot-sample combinations)
  raw_counts_2026.csv             Raw probe counts, 2026 cohort (all probe classes: biomarker, HK, Other)
  positive_controls_2026.csv      NanoString POS_A to POS_F spike-in counts (per sample per lot)
  historical_scores.csv           Historical ChenRobust scores for shared samples (original codeset)
scripts/
  utils.py                                  Shared data-loading and scoring helpers
  01_cross_experiment_reproducibility.py    Tables 2 and 3 (lot A -> B, 2023 vs 2026)
  02_leave_one_out_cv.py                    Table 1 (LOO-CV, K=6)
  03_coefficient_correlations.py            Table 5 (||beta||_2 and directional quantity correlations)
  04_ridge_vs_sparsified.py                 Table 4 (ridge vs sparsified alternatives)
  05_hk_and_pos.py                          Housekeeping variance absorption + POS spike-in ratios
  06_master_comparison.py                   Cross-time reproducibility vs historical database
figures/                          Output directory (created at runtime)
requirements.txt                  Python dependencies
```

## Model coefficients and the Shiny scoring app

The 500-model ridge coefficient matrix and per-model intercepts that define the deployed bagged-ensemble biomarker are *not* redistributed in this repository. To compute risk scores on new gene-expression data, use the biomarker's Shiny scoring app described in the original publication (Chen et al., *Nature Medicine* 2023). Internal reproduction of the single-model distributional statistics in Tables 3, 4, and 5 — and the single-model rows of Table 1 — requires the coefficients; contact the corresponding authors to obtain them for verification purposes.

Analyses that operate on the ensemble-level risk scores (Table 2, ensemble row of Table 1, the cross-time reproducibility result, housekeeping variance absorption, and positive-control spike-in ratios) remain fully reproducible from the public CSVs in this repository. The scripts print a clear notice when coefficient-dependent analyses are skipped because the coefficients are not locally available.

## Data scope and de-identification

- **Sample IDs** are anonymized: 2023 cohort (S01–S05), 2026 cohort (S06–S15). A small number of samples were excluded from cross-lot analyses for platform QC failures (see paper §2.1); the paper's N = 7 QC-pass set for 2026 is `S06, S07, S09, S10, S12, S13, S14` (encoded as `QC_PASS_N7` in `utils.py`).
- **Lot IDs** use the paper's anonymized labels A, B, C, D, E (physical manufacturing lots; exact identifiers are available to readers only through direct correspondence with the corresponding authors).
- **Gene names** are the HGNC symbols of the 34 published biomarker genes from Chen et al. 2023 Nature Medicine.
- **Both the 2023 and 2026 cohort raw counts and housekeeping-normalized expression are distributed here** (`raw_counts_{2023,2026}.csv` and `normalized_expression_{2023,2026}.csv`), which is sufficient to recompute all per-sample ensemble and single-model scores from the coefficient matrix. The `historical_scores.csv` file contains the ChenRobust (bagged-ridge ensemble) scores from the deployed scoring pipeline for 2026 cohort samples that were also present in the original validation cohort.

## Reproducing the paper's core claims

Scripts marked "Public" reproduce fully from the public CSVs in this repository. Scripts marked "Coefficient" additionally require the 500-model coefficient matrix (not distributed; see above).

| Claim | Script | Runs on public data? |
|---|---|---|
| 2023/2026 A→B linear corrections agree to 0.018 on slope and intercept (Table 2) | `01_cross_experiment_reproducibility.py` | Public |
| Ensemble LOO-CV linear RMSE = 0.017 (ensemble row of Table 1) | `02_leave_one_out_cv.py` | Public |
| HK normalization absorbs ~54% of cross-lot variance; POS log₂(B/A) ≈ +0.3 at high concentrations (§3.1) | `05_hk_and_pos.py` | Public |
| Historical scores reproduce at R² = 0.97–0.98 for lots B, D, E | `06_master_comparison.py` | Public |
| Ensemble outperforms 78% of single models on cross-experiment |Δslope| (Table 3) | `01_cross_experiment_reproducibility.py` | Coefficient |
| P50/P90 single-model LOO-CV RMSE rows (Table 1) | `02_leave_one_out_cv.py` | Coefficient |
| `r = 0.65` for ‖β‖₂ vs LOO-CV RMSE; `r = 0.33` for directional quantity (Table 5) | `03_coefficient_correlations.py` | Coefficient |
| 2.6× probe-failure protection vs 5-gene sparsified alternative (Table 4) | `04_ridge_vs_sparsified.py` | Coefficient |

## Citing

If you use this code or data, please cite the paper (see above) and the underlying biomarker publication:

> Chen WC, Choudhury A, Youngblood MW, *et al*. Targeted gene expression profiling predicts meningioma outcomes and radiotherapy responses. *Nat Med* 2023, 29:3067–3076.

## License

Code is released under the MIT License (see `LICENSE`). Data are released for academic and educational use only; any clinical or commercial use requires written permission from the corresponding authors.

## Contact

William C. Chen (william.chen@ucsf.edu) and David R. Raleigh — Department of Radiation Oncology, University of California San Francisco.
