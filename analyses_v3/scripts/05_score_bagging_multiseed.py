"""Score all 100 FFPE-bagged seed-run β̄'s against bridging data.

Per-seed metrics use the paper's exact LOO-CV and probe-failure conventions
(see common.py: loocv_amean_k6, loocv_amean_k3, probe_failure_paper):
  - LOO-CV: predict mean(B,D,E) from A; linear correction; iterate over hold-outs
  - Probe failure: max_{g,s} |β_g × X_norm(s, lot A)[g]|
  - Cross-experiment: A→B linear correction in 2023 vs 2026 (paper Table 3)
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import (DATA_DIR, BRIDGING_SAMPLES_2026, BRIDGING_SAMPLES_2023_ALL,
                    loocv_amean_k6, loocv_amean_k3, probe_failure_paper)

BRIDGING_SAMPLES_2023_PAPER = [s for s in BRIDGING_SAMPLES_2023_ALL if s != 'MEN-315']

betas_csv = os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_betas.csv')
meta_csv  = os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_meta.csv')

with open(betas_csv) as f:
    reader = csv.reader(f); hdr = next(reader)
    rows = [r for r in reader]
gene_names = [r[0] for r in rows]
betas = np.array([[float(v) for v in r[1:]] for r in rows])  # 34 × M
seeds = [int(h.replace('seed_', '')) for h in hdr[1:]]
M = betas.shape[1]
print(f"Loaded {M} seed-runs × {betas.shape[0]} genes")

with open(meta_csv) as f:
    reader = csv.DictReader(f)
    meta = {int(r['seed']): r for r in reader}

# Optional: deployed for cosine reference
dep_path = os.path.join(DATA_DIR, 'deployed_ensemble_true.npz')
have_dep = os.path.exists(dep_path)
if have_dep:
    dep = np.load(dep_path, allow_pickle=True)
    dep_beta, dep_intc = dep['beta'], float(dep['intercept'])

br = np.load(os.path.join(DATA_DIR, 'bridging_normalized.npz'), allow_pickle=True)
keys_26 = [tuple(k) for k in br['keys_2026']]
keys_23 = [tuple(k) for k in br['keys_2023']]
Xn_26, Xn_23 = br['X_norm_2026'], br['X_norm_2023']

# Per-sample lot-A normalized expression (used by probe_failure_paper)
sample_to_X_lotA_26 = {}
for (s, l), x in zip(keys_26, Xn_26):
    if l == 'A':
        sample_to_X_lotA_26[s] = x
X_lotA_2026 = [sample_to_X_lotA_26[s] for s in BRIDGING_SAMPLES_2026]


def fit_panel(store, samples, la, lb):
    xa = np.array([store[(s, la)] for s in samples]); yb = np.array([store[(s, lb)] for s in samples])
    c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
    return float(c[0]), float(c[1])


def metrics(beta, intc):
    s26 = dict(zip(keys_26, intc + Xn_26 @ beta))
    s23 = dict(zip(keys_23, intc + Xn_23 @ beta))
    a23, b23 = fit_panel(s23, BRIDGING_SAMPLES_2023_PAPER, 'A', 'B')
    a26, b26 = fit_panel(s26, BRIDGING_SAMPLES_2026, 'A', 'B')
    out = {
        'L2': float(np.linalg.norm(beta)),
        'LOO_K6': loocv_amean_k6(s26, BRIDGING_SAMPLES_2026),
        'LOO_K3': loocv_amean_k3(s26, BRIDGING_SAMPLES_2026),
        'dslope': abs(a23 - a26),
        'dintc':  abs(b23 - b26),
        'probe_failure': probe_failure_paper(beta, X_lotA_2026),
    }
    if have_dep:
        out['cos_dep'] = float(np.dot(beta, dep_beta) / (np.linalg.norm(beta)*np.linalg.norm(dep_beta)))
    return out


per_seed = []
for i, s in enumerate(seeds):
    intc = float(meta[s]['intercept'])
    per_seed.append(metrics(betas[:, i], intc))

if have_dep:
    dep_m = metrics(dep_beta, dep_intc)
    dep_m['cos_dep'] = 1.0
    print(f"\nDeployed (paper protocol): "
          f"L2={dep_m['L2']:.4f}  K6={dep_m['LOO_K6']:.4f}  K3={dep_m['LOO_K3']:.4f}  "
          f"|Δa|={dep_m['dslope']:.4f}  probe={dep_m['probe_failure']:.4f}")

print(f"\n=== Distribution across {M} reproductions (paper protocol) ===")
for k in ['L2','LOO_K6','LOO_K3','dslope','dintc','probe_failure'] + (['cos_dep'] if have_dep else []):
    arr = np.array([m[k] for m in per_seed])
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"  {k:<14} median={p50:.4f}  IQR [{p25:.4f}, {p75:.4f}]  90% CI [{p5:.4f}, {p95:.4f}]")

out = {'per_seed': per_seed, 'M': M}
if have_dep: out['deployed'] = dep_m
with open(os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json')}")
