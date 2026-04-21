"""100-seed runs of single ridge / LASSO / elastic net on FFPE training,
plus deterministic OLS. Paper-protocol metrics (loocv_amean_*, probe_failure_paper).
Same seed range as 04 R script (42 + 100*m for m=1..100)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold
from common import (DATA_DIR, BRIDGING_SAMPLES_2026, BRIDGING_SAMPLES_2023_ALL,
                    loocv_amean_k6, loocv_amean_k3, probe_failure_paper)

BRIDGING_SAMPLES_2023_PAPER = [s for s in BRIDGING_SAMPLES_2023_ALL if s != 'MEN-315']

tr = np.load(os.path.join(DATA_DIR, 'ffpe_paired_training_pc.npz'), allow_pickle=True)
X_tr, y_tr = tr['X_norm'], tr['y']
br = np.load(os.path.join(DATA_DIR, 'bridging_normalized.npz'), allow_pickle=True)
keys_26 = [tuple(k) for k in br['keys_2026']]
keys_23 = [tuple(k) for k in br['keys_2023']]
Xn_26, Xn_23 = br['X_norm_2026'], br['X_norm_2023']

dep_path = os.path.join(DATA_DIR, 'deployed_ensemble_true.npz')
have_dep = os.path.exists(dep_path)
if have_dep:
    dep = np.load(dep_path, allow_pickle=True)
    dep_beta = dep['beta']

sample_to_X_lotA_26 = {s: x for (s, l), x in zip(keys_26, Xn_26) if l == 'A'}
X_lotA_2026 = [sample_to_X_lotA_26[s] for s in BRIDGING_SAMPLES_2026]


def fit_panel(store, samples, la, lb):
    xa = np.array([store[(s, la)] for s in samples]); yb = np.array([store[(s, lb)] for s in samples])
    c, *_ = np.linalg.lstsq(np.column_stack([xa, np.ones_like(xa)]), yb, rcond=None)
    return float(c[0]), float(c[1])


def evaluate(beta, intc):
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


M = 100
results = {'single_ridge': [], 'single_lasso': [], 'single_elasticnet': []}
ridge_alphas = np.logspace(-2, 3, 41)
lasso_alphas = np.logspace(-4, 0, 41)
en_alphas    = np.logspace(-4, 0, 41)

for m in range(M):
    seed = 42 + 100 * (m + 1)
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    r = RidgeCV(alphas=ridge_alphas, cv=cv).fit(X_tr, y_tr)
    results['single_ridge'].append(evaluate(r.coef_, float(r.intercept_)))
    l = LassoCV(alphas=lasso_alphas, cv=cv, random_state=seed, max_iter=10000).fit(X_tr, y_tr)
    results['single_lasso'].append(evaluate(l.coef_, float(l.intercept_)))
    e = ElasticNetCV(l1_ratio=0.5, alphas=en_alphas, cv=cv, random_state=seed, max_iter=20000).fit(X_tr, y_tr)
    results['single_elasticnet'].append(evaluate(e.coef_, float(e.intercept_)))
    if (m + 1) % 25 == 0: print(f"  seed {m+1}/{M} done")

ols = LinearRegression().fit(X_tr, y_tr)
results['single_ols'] = [evaluate(ols.coef_, float(ols.intercept_))]

print("\n=== Distribution across 100 CV-fold seeds (paper protocol) ===")
print(f"{'Model':<22} {'metric':<14} {'min':>9} {'median':>9} {'max':>9}")
for model, runs in results.items():
    for k in ['L2','LOO_K6','LOO_K3','dslope','dintc','probe_failure']:
        arr = np.array([r[k] for r in runs])
        if model == 'single_ols':
            print(f"{model:<22} {k:<14} {arr[0]:>9.4f} {arr[0]:>9.4f} {arr[0]:>9.4f}")
        else:
            print(f"{model:<22} {k:<14} {arr.min():>9.4f} {np.median(arr):>9.4f} {arr.max():>9.4f}")

with open(os.path.join(DATA_DIR, 'single_models_multiseed.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {os.path.join(DATA_DIR, 'single_models_multiseed.json')}")
