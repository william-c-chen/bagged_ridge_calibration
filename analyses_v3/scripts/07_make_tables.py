"""Build Tables 7 and 8 from the multi-seed scoring outputs.

Uses the paper-protocol metrics: LOO-CV target = mean(B,D,E) from A; probe
failure = max_{g,s} |β_g × X_norm(s, lot A)[g]|. See common.py.

Reads:
  data/glmnet_ffpe_multiseed_scoring.json   (from 05_score_bagging_multiseed.py)
  data/single_models_multiseed.json         (from 06_single_models_multiseed.py)
  data/deployed_ensemble_true.npz           (optional; from 00_load_deployed_coefs.py)

Writes:
  data/table7_headline_metrics.csv
  data/table8_seed_variance.csv
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import DATA_DIR

# ---- Load multi-seed outputs ----
with open(os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json')) as f:
    bag_data = json.load(f)
bagged = bag_data['per_seed']
deployed_metrics = bag_data.get('deployed', None)
have_deployed = deployed_metrics is not None

with open(os.path.join(DATA_DIR, 'single_models_multiseed.json')) as f:
    single = json.load(f)

# ---- Table 7: headline medians ----
def median_metrics(runs):
    return {k: float(np.median([r[k] for r in runs])) for k in
            ['L2','cos_dep','LOO_K6','LOO_K3','dslope','dintc','probe_failure']}

models_t7 = []
if deployed_metrics is not None:
    models_t7.append(('Deployed bagged ridge', deployed_metrics))
models_t7.append(('Reproduced bagged ridge', median_metrics(bagged)))
models_t7.append(('Single ridge',           median_metrics(single['single_ridge'])))
models_t7.append(('Single LASSO',           median_metrics(single['single_lasso'])))
models_t7.append(('Single elastic net',     median_metrics(single['single_elasticnet'])))
ols = single['single_ols'][0]
models_t7.append(('Single OLS',             {k: ols[k] for k in
                  ['L2','cos_dep','LOO_K6','LOO_K3','dslope','dintc','probe_failure']}))

print("\n=== Table 7: Headline metrics (median across 100 seeds; OLS deterministic) ===")
hdr_t7 = ['Model','||beta||_2','cos(dep)','LOO_K6','LOO_K3','|Δslope|','|Δintc|','probe_failure']
print('\t'.join(hdr_t7))
for name, m in models_t7:
    print(f"{name}\t{m['L2']:.4f}\t{m['cos_dep']:.3f}\t{m['LOO_K6']:.4f}\t{m['LOO_K3']:.4f}\t"
          f"{m['dslope']:.4f}\t{m['dintc']:.4f}\t{m['probe_failure']:.4f}")
with open(os.path.join(DATA_DIR, 'table7_headline_metrics.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(hdr_t7)
    for name, m in models_t7:
        w.writerow([name, f"{m['L2']:.4f}", f"{m['cos_dep']:.3f}", f"{m['LOO_K6']:.4f}",
                    f"{m['LOO_K3']:.4f}", f"{m['dslope']:.4f}", f"{m['dintc']:.4f}",
                    f"{m['probe_failure']:.4f}"])

# ---- Table 8: SDs and ratios ----
def sd_dict(runs):
    return {k: float(np.std([r[k] for r in runs], ddof=1)) for k in
            ['L2','LOO_K6','LOO_K3','dslope','dintc','probe_failure']}
sd_bag = sd_dict(bagged)
sd_ridge = sd_dict(single['single_ridge'])
sd_lasso = sd_dict(single['single_lasso'])
sd_en    = sd_dict(single['single_elasticnet'])

print("\n=== Table 8: Standard deviation across 100 random seeds ===")
hdr_t8 = ['Metric','Bagged','Ridge','LASSO','ElasticNet','Ridge/Bagged']
print('\t'.join(hdr_t8))
metric_labels = [('L2','||beta||_2'),('LOO_K6','LOO-CV K=6'),('LOO_K3','LOO-CV K=3'),
                 ('dslope','|Δslope|'),('dintc','|Δintercept|'),('probe_failure','probe-failure')]
table8_rows = []
for k, label in metric_labels:
    row = [label, sd_bag[k], sd_ridge[k], sd_lasso[k], sd_en[k], sd_ridge[k]/sd_bag[k]]
    table8_rows.append(row)
    print(f"{label}\t{sd_bag[k]:.4f}\t{sd_ridge[k]:.4f}\t{sd_lasso[k]:.4f}\t"
          f"{sd_en[k]:.4f}\t{sd_ridge[k]/sd_bag[k]:.1f}x")

with open(os.path.join(DATA_DIR, 'table8_seed_variance.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(hdr_t8)
    for row in table8_rows:
        w.writerow([row[0]] + [f"{v:.4f}" if isinstance(v, float) else v for v in row[1:5]] +
                   [f"{row[5]:.1f}x"])

print(f"\nWrote {os.path.join(DATA_DIR, 'table7_headline_metrics.csv')}")
print(f"Wrote {os.path.join(DATA_DIR, 'table8_seed_variance.csv')}")
