#!/usr/bin/env bash
# Reproducibly rerun the full Option B / Tables 7-8 analysis pipeline.
# Fixed seed (42) hard-coded in scripts/common.py and the R scripts.
# Outputs go to analyses/data/ (intermediate NPZ/JSON/CSV) and analyses/logs/.
#
# Total wall time: ~25 minutes (dominated by the 100-seed glmnet bagging in step 04).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data figures logs

run() {
  local script="$1"; local log="logs/$(basename "$script" | sed 's/\.[^.]*$//').log"
  echo ">>> Running $script  (log -> $log)"
  if [[ "$script" == *.R ]]; then
    Rscript "$script" 2>&1 | tee "$log"
  else
    python3 "$script" 2>&1 | tee "$log"
  fi
  echo
}

# 00 is OPTIONAL — loads the deployed coefficient files (NOT publicly distributed).
# Skips silently if internal_data/ is unavailable; downstream skips deployed comparisons.
run scripts/00_load_deployed_coefs.py

# Build training inputs (Python)
run scripts/01_load_training_frozen.py     # 173 UCSF frozen samples (diagnostic context)
run scripts/02_build_ffpe_training.py       # 64 FFPE-paired samples (the actual ridge training set)
run scripts/03_parse_rcc_bridging.py        # 2023 + 2026 bridging RCCs → normalized X

# Run the bagged-ridge reproduction at 100 random seeds (R glmnet, ~20 min)
run scripts/04_glmnet_bagging_multiseed.R

# Score the 100 bagged β̄'s on bridging samples (Python)
run scripts/05_score_bagging_multiseed.py

# Fit single ridge / LASSO / elastic net at 100 CV-fold seeds + OLS (Python)
run scripts/06_single_models_multiseed.py

# Build Tables 7 and 8 from the multi-seed outputs
run scripts/07_make_tables.py

echo "Done."
echo "  Headline metrics → data/table7_headline_metrics.csv"
echo "  Seed-variance SDs → data/table8_seed_variance.csv"
echo "  Per-seed scoring  → data/glmnet_ffpe_multiseed_scoring.json, single_models_multiseed.json"
