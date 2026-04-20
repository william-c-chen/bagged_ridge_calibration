# Data Dictionary

All data are de-identified. Sample IDs map to arbitrary internal labels; original lab identifiers are not preserved in this distribution.

## Files

### Model coefficients (not distributed)
The 500-model ridge coefficient matrix and per-model intercepts are not included in this public distribution. To compute risk scores on new gene-expression data, use the biomarker's Shiny scoring app described in Chen et al. *Nature Medicine* 2023; for internal reproduction of single-model distributional statistics, contact the corresponding authors.

### `biomarker_gene_list.csv`
The ordered list of 34 biomarker genes (HGNC symbols). Row order matches the coefficient matrix.

### `scores_2023.csv`
Ensemble bagged-ridge risk scores for the five 2023 bridging samples on the three 2023 codeset lots (A, B, C). Long format: `cohort`, `sample_id`, `lot`, `ensemble_score`. Sample IDs S01–S05.

### `raw_counts_2023.csv`
Raw probe counts for the 2023 cohort, same schema as `raw_counts_2026.csv`. Covers 5 samples × 3 lots = 15 (lot, sample) combinations.

### `normalized_expression_2023.csv`
Housekeeping-normalized log₂ expression for the 2023 cohort, same schema as `normalized_expression_2026.csv`. Recomputed from `raw_counts_2023.csv` using the formula `log2((r_g + 1) / HK_geometric_mean)`.

### `scores_2026.csv`
Ensemble bagged-ridge risk scores for the ten 2026 bridging samples on the four 2026 codeset lots (A, B, D, E). Long format as above. Sample IDs S06–S15. Not every (sample, lot) pair is present: three samples were excluded from cross-lot analyses in the paper for lot-specific QC failures — S08 failed on lots B, D, and E; S11 failed on lot D; S15 showed anomalous counts on lot E. The remaining 7 samples used in most paper analyses are available via `utils.QC_PASS_N7`.

### `normalized_expression_2026.csv`
Housekeeping-normalized log₂ expression of the 34 biomarker genes for the 2026 cohort across four lots. Column naming: `<lot>_<sample_id>` (e.g. `A_S06`). Normalization formula: `x_g = log2((r_g + 1) / HK_geometric_mean)` with HK_geometric_mean taken over the 7 housekeeping probes per (sample, lot) run.

### `raw_counts_2026.csv`
Raw probe counts for the full codeset in the 2026 cohort. Columns: `gene`, `class` (`BM` = biomarker gene, `HK` = housekeeping gene, `Other` = additional codeset probes from the Raleigh_75 panel), then one column per (lot, sample) combination (same naming as `normalized_expression_2026.csv`). Use this file to re-derive the HK normalization from scratch if desired.

### `positive_controls_2026.csv`
NanoString POS_A–POS_F synthetic spike-in probe counts, extracted from per-lane RCC files. Long format: `sample_id`, `lot`, `pos_probe`, `concentration_fM`, `count`. The synthetic-oligonucleotide POS controls are loaded at known concentrations (POS_A = 128 fM, POS_B = 32 fM, POS_C = 8 fM, POS_D = 2 fM, POS_E = 0.5 fM, POS_F = 0.125 fM) on every cartridge and provide a global hybridization-efficiency reference.

### `historical_scores.csv`
Ensemble bagged-ridge risk scores for the 2026 cohort as originally scored in the project validation-cohort database, on a codeset lot no longer available. Used to establish cross-time reproducibility of the biomarker score across three distinct measurement occasions (original assay; 2023 calibration; 2026 calibration).

## Lot labeling

| Label | Physical codeset family |
|:---:|:---|
| A | Lot family shared across 2023 and 2026 experiments (original-era manufacturing family) |
| B | Second reagent lot, also shared across 2023 and 2026 |
| C | 2023-only lot (not included in 2026) |
| D | 2026-only lot |
| E | 2026-only lot |

Exact vendor lot identifiers are retained in internal lab records and are available from the corresponding authors for verification.

## Not included

- Original validation-cohort raw counts. `historical_scores.csv` contains only the ensemble-level ChenRobust scores for samples shared between the 2026 calibration cohort and the original validation cohort (sufficient to reproduce the "Additional cross-time reproducibility" results).
- Protected health information. No patient identifiers, MRNs, dates, clinical annotations, or survival outcomes are included.
