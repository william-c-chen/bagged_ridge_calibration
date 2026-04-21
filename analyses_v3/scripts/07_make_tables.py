"""Build Tables 7 and 8 from the multi-seed scoring outputs.

Reads:
  data/glmnet_ffpe_multiseed_scoring.json   (from 05_score_bagging_multiseed.py)
  data/single_models_multiseed.json         (from 06_single_models_multiseed.py)
  data/deployed_ensemble_true.npz           (optional; from 00_load_deployed_coefs.py)
  data/bridging_normalized.npz               (from 03_parse_rcc_bridging.py)

Writes:
  data/table7_headline_metrics.csv
  data/table8_seed_variance.csv

Both tables are also printed to stdout (latex-ready rows).
"""
import os, sys, json, csv, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import DATA_DIR, BRIDGING_SAMPLES_2026, BRIDGING_SAMPLES_2023_ALL

BRIDGING_SAMPLES_2023_PAPER = [s for s in BRIDGING_SAMPLES_2023_ALL if s != 'MEN-315']

# ---- Load multi-seed outputs ----
with open(os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json')) as f:
    bagged = json.load(f)['per_seed']
with open(os.path.join(DATA_DIR, 'single_models_multiseed.json')) as f:
    single = json.load(f)

# Deployed metrics
deployed_path = os.path.join(DATA_DIR, 'deployed_ensemble_true.npz')
have_deployed = os.path.exists(deployed_path)
if have_deployed:
    dep = np.load(deployed_path, allow_pickle=True)
    dep_beta, dep_intc = dep['beta'], float(dep['intercept'])
    br = np.load(os.path.join(DATA_DIR, 'bridging_normalized.npz'), allow_pickle=True)
    keys_26 = [tuple(k) for k in br['keys_2026']]
    keys_23 = [tuple(k) for k in br['keys_2023']]

    def fit_panel(store, samples, la, lb):
        xa = np.array([store[(s, la)] for s in samples])
        yb = np.array([store[(s, lb)] for s in samples])
        c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
        return float(c[0]), float(c[1])

    def loocv_k6(store, samples, la, lb):
        errs = []
        for hold in samples:
            trs = [s for s in samples if s != hold]
            xa = np.array([store[(s, la)] for s in trs]); yb = np.array([store[(s, lb)] for s in trs])
            c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
            errs.append(c[0]*store[(hold, la)] + c[1] - store[(hold, lb)])
        return float(np.sqrt(np.mean(np.array(errs)**2)))

    def loocv_k3(store, samples, la, lb):
        all_errs = []
        for trs in itertools.combinations(samples, 3):
            test = [s for s in samples if s not in trs]
            xa = np.array([store[(s, la)] for s in trs]); yb = np.array([store[(s, lb)] for s in trs])
            c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
            for s in test:
                all_errs.append(c[0]*store[(s, la)] + c[1] - store[(s, lb)])
        return float(np.sqrt(np.mean(np.array(all_errs)**2)))

    def probe_failure(coef, X_raw):
        return float(np.max(np.abs(-coef[None, :] * np.log2(X_raw + 1))))

    s26 = dict(zip(keys_26, dep_intc + br['X_norm_2026'] @ dep_beta))
    s23 = dict(zip(keys_23, dep_intc + br['X_norm_2023'] @ dep_beta))
    a23, b23 = fit_panel(s23, BRIDGING_SAMPLES_2023_PAPER, 'A', 'B')
    a26, b26 = fit_panel(s26, BRIDGING_SAMPLES_2026, 'A', 'B')
    deployed_metrics = {
        'L2': float(np.linalg.norm(dep_beta)), 'cos_dep': 1.0,
        'LOO_K6': loocv_k6(s26, BRIDGING_SAMPLES_2026, 'A', 'B'),
        'LOO_K3': loocv_k3(s26, BRIDGING_SAMPLES_2026, 'A', 'B'),
        'dslope': abs(a23 - a26), 'dintc': abs(b23 - b26),
        'probe_failure': probe_failure(dep_beta, br['X_raw_2026']),
    }
else:
    deployed_metrics = None

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
