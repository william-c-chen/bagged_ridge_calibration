"""Score all 100 FFPE-bagged seed-run β̄'s against bridging data.

Computes per-seed: ||β̄||_2, cos(deployed), LOO-CV K=6 / K=3 A→B,
cross-experiment |Δslope| / |Δintercept|, probe-failure.
Reports distribution and percentiles, comparing deployed against the distribution.
"""
import os, sys, json, csv, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import DATA_DIR, BRIDGING_SAMPLES_2026, BRIDGING_SAMPLES_2023_ALL

BRIDGING_SAMPLES_2023_PAPER = [s for s in BRIDGING_SAMPLES_2023_ALL if s != 'MEN-315']

# Load multi-seed β̄'s
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

# Deployed
dep = np.load(os.path.join(DATA_DIR, 'deployed_ensemble_true.npz'), allow_pickle=True)
dep_beta, dep_intc = dep['beta'], float(dep['intercept'])

br = np.load(os.path.join(DATA_DIR, 'bridging_normalized.npz'), allow_pickle=True)
keys_26 = [tuple(k) for k in br['keys_2026']]
keys_23 = [tuple(k) for k in br['keys_2023']]
Xn_26, Xn_23, Xraw_26 = br['X_norm_2026'], br['X_norm_2023'], br['X_raw_2026']

def loocv_k6(store, samples, la, lb):
    errs = []
    for hold in samples:
        trs = [s for s in samples if s != hold]
        xa = np.array([store[(s, la)] for s in trs]); yb = np.array([store[(s, lb)] for s in trs])
        c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
        errs.append(c[0]*store[(hold, la)] + c[1] - store[(hold, lb)])
    return float(np.sqrt(np.mean(np.array(errs)**2)))
def loocv_k3(store, samples, la, lb):
    errs = []
    for trs in itertools.combinations(samples, 3):
        test = [s for s in samples if s not in trs]
        xa = np.array([store[(s, la)] for s in trs]); yb = np.array([store[(s, lb)] for s in trs])
        c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
        for s in test:
            errs.append(c[0]*store[(s, la)] + c[1] - store[(s, lb)])
    return float(np.sqrt(np.mean(np.array(errs)**2)))
def fit_panel(store, samples, la, lb):
    xa = np.array([store[(s, la)] for s in samples]); yb = np.array([store[(s, lb)] for s in samples])
    c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
    return float(c[0]), float(c[1])
def probe_failure(coef, X_raw):
    return float(np.max(np.abs(-coef[None, :] * np.log2(X_raw + 1))))

# Score deployed and each seed
def metrics(beta, intc):
    s26 = dict(zip(keys_26, intc + Xn_26 @ beta))
    s23 = dict(zip(keys_23, intc + Xn_23 @ beta))
    a23, b23 = fit_panel(s23, BRIDGING_SAMPLES_2023_PAPER, 'A', 'B')
    a26, b26 = fit_panel(s26, BRIDGING_SAMPLES_2026, 'A', 'B')
    return {
        'L2': float(np.linalg.norm(beta)),
        'cos_dep': float(np.dot(beta, dep_beta)/(np.linalg.norm(beta)*np.linalg.norm(dep_beta))),
        'LOO_K6': loocv_k6(s26, BRIDGING_SAMPLES_2026, 'A', 'B'),
        'LOO_K3': loocv_k3(s26, BRIDGING_SAMPLES_2026, 'A', 'B'),
        'dslope': abs(a23 - a26),
        'dintc':  abs(b23 - b26),
        'probe_failure': probe_failure(beta, Xraw_26),
    }

dep_m = metrics(dep_beta, dep_intc)
print(f"\nDeployed: ||β̄||={dep_m['L2']:.4f}  LOO K6={dep_m['LOO_K6']:.4f}  "
      f"|Δslope|={dep_m['dslope']:.4f}  probe={dep_m['probe_failure']:.4f}")

per_seed = []
for i, seed in enumerate(seeds):
    intc = float(meta[seed]['intercept'])
    per_seed.append(metrics(betas[:, i], intc))

def pct(arr, key):
    v = np.array([m[key] for m in arr])
    return v.min(), np.percentile(v, 25), np.median(v), np.percentile(v, 75), v.max()

print(f"\n=== Distribution across {M} FFPE-bagged seed-runs ===")
print(f"{'Metric':<18} {'min':>10} {'P25':>10} {'median':>10} {'P75':>10} {'max':>10}  {'deployed':>10}  {'percentile':>10}")
for k in ['L2','cos_dep','LOO_K6','LOO_K3','dslope','dintc','probe_failure']:
    mn, p25, p50, p75, mx = pct(per_seed, k)
    arr = np.array([m[k] for m in per_seed])
    # what percentile does deployed sit at? lower-better metrics: lower percentile = deployed better
    if k == 'cos_dep':
        # deployed cos with itself is 1.0; we want lower-is-bad
        pctile = (arr < dep_m[k]).mean() * 100
    else:
        pctile = (arr < dep_m[k]).mean() * 100
    print(f"{k:<18} {mn:>10.4f} {p25:>10.4f} {p50:>10.4f} {p75:>10.4f} {mx:>10.4f}  {dep_m[k]:>10.4f}  {pctile:>9.0f}%")

with open(os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json'), 'w') as f:
    json.dump({'per_seed': per_seed, 'deployed': dep_m, 'M': M}, f, indent=2)
print(f"\nWrote {os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json')}")
