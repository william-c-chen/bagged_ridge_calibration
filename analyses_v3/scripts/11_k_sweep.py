"""K sweep: bridging panel size vs calibration RMSE by architecture.

For each architecture (bagged ridge, single ridge, LASSO, elastic net, OLS)
and each K in {3,4,5,6,7}, enumerate all C(7,K) training subsets of the 2026
bridging panel, fit a linear correction (lot A -> mean of lots B,D,E), and
evaluate RMSE on the (7-K) held-out samples. Report mean RMSE across subsets.

K=7 uses in-sample residual RMSE (no held-out).
"""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from itertools import combinations
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold
from scipy import stats
from common import (DATA_DIR, BRIDGING_SAMPLES_2026, OTHER_LOTS_2026)

# Training & bridging data
tr = np.load(os.path.join(DATA_DIR, 'ffpe_paired_training_pc.npz'), allow_pickle=True)
X_tr, y_tr = tr['X_norm'], tr['y']
br = np.load(os.path.join(DATA_DIR, 'bridging_normalized.npz'), allow_pickle=True)
keys_26 = [tuple(k) for k in br['keys_2026']]
Xn_26 = br['X_norm_2026']

# Deployed ensemble
dep = np.load(os.path.join(DATA_DIR, 'deployed_ensemble_true.npz'), allow_pickle=True)
bagged_beta, bagged_intc = dep['beta'], float(dep['intercept'])


def fit_one(model_name, seed=None):
    """Return (beta, intercept) for one model fit. Deterministic for OLS."""
    if model_name == 'bagged':
        return bagged_beta, bagged_intc
    if model_name == 'ols':
        m = LinearRegression().fit(X_tr, y_tr); return m.coef_, float(m.intercept_)
    cv = KFold(n_splits=10, shuffle=True, random_state=seed or 42)
    if model_name == 'ridge':
        m = RidgeCV(alphas=np.logspace(-2, 3, 41), cv=cv).fit(X_tr, y_tr)
    elif model_name == 'lasso':
        m = LassoCV(alphas=np.logspace(-4, 0, 41), cv=cv, random_state=seed, max_iter=10000).fit(X_tr, y_tr)
    elif model_name == 'en':
        m = ElasticNetCV(l1_ratio=0.5, alphas=np.logspace(-4, 0, 41), cv=cv,
                         random_state=seed, max_iter=20000).fit(X_tr, y_tr)
    return m.coef_, float(m.intercept_)


def score_bridging(beta, intc):
    """Return dict (sample, lot) -> score on 2026 bridging."""
    preds = intc + Xn_26 @ beta
    return dict(zip(keys_26, preds))


def k_rmse(scores, K):
    """Mean RMSE across C(7,K) training subsets: fit lin correction on K,
    evaluate on the (7-K) held-out samples."""
    ns = len(BRIDGING_SAMPLES_2026)
    yA = np.array([scores[(s, 'A')] for s in BRIDGING_SAMPLES_2026])
    yT = np.array([np.mean([scores[(s, L)] for L in OTHER_LOTS_2026])
                   for s in BRIDGING_SAMPLES_2026])
    if K == ns:
        sl, ic, *_ = stats.linregress(yA, yT)
        resid = yT - (sl * yA + ic)
        return float(np.sqrt(np.mean(resid ** 2)))
    rmses = []
    for train in combinations(range(ns), K):
        ti = list(train); hi = [i for i in range(ns) if i not in ti]
        if np.var(yA[ti]) < 1e-12: continue
        sl, ic, *_ = stats.linregress(yA[ti], yT[ti])
        err = yT[hi] - (sl * yA[hi] + ic)
        rmses.append(np.sqrt(np.mean(err ** 2)))
    return float(np.mean(rmses))


# Bagged is deterministic (single fit); single-model architectures vary with CV seed
# so we run them across 100 seeds and report the median RMSE per K.
Ks = [3, 4, 5, 6, 7]
M = 100
models_det = ['bagged', 'ols']
models_seeded = ['ridge', 'lasso', 'en']

# Deterministic models
det_scores = {m: score_bridging(*fit_one(m)) for m in models_det}
det_rmses = {m: {K: k_rmse(det_scores[m], K) for K in Ks} for m in models_det}

# Seeded models
print(f"Running {len(models_seeded)} seeded architectures × {M} seeds ...")
seeded_rmses = {m: {K: [] for K in Ks} for m in models_seeded}
for mm in range(M):
    sd = 42 + 100 * (mm + 1)
    for m in models_seeded:
        beta, intc = fit_one(m, seed=sd)
        sc = score_bridging(beta, intc)
        for K in Ks:
            seeded_rmses[m][K].append(k_rmse(sc, K))
    if (mm + 1) % 25 == 0: print(f"  {mm+1}/{M}")

# Median per K per architecture
print(f"\nMedian RMSE across 100 seeds (single-model architectures); single run for bagged/OLS.")
print(f"{'K':<4}" + ''.join(f'{m:>10}' for m in ['bagged', 'ridge', 'lasso', 'en', 'ols']))
results = {}
for K in Ks:
    row = {
        'bagged': det_rmses['bagged'][K],
        'ridge':  float(np.median(seeded_rmses['ridge'][K])),
        'lasso':  float(np.median(seeded_rmses['lasso'][K])),
        'en':     float(np.median(seeded_rmses['en'][K])),
        'ols':    det_rmses['ols'][K],
    }
    results[K] = row
    print(f"{K:<4}" + ''.join(f"{row[m]:>10.4f}" for m in ['bagged', 'ridge', 'lasso', 'en', 'ols']))

print(f"\nRatio: single-ridge median / bagged RMSE at each K:")
for K in Ks:
    print(f"  K={K}: {results[K]['ridge']/results[K]['bagged']:.2f}x")

# Also report P90 of single ridge (analyst-unluck)
print(f"\nP90 of single ridge across seeds (what the unlucky analyst sees):")
for K in Ks:
    p90 = float(np.percentile(seeded_rmses['ridge'][K], 90))
    print(f"  K={K}: bagged={results[K]['bagged']:.4f}, single ridge P90={p90:.4f}  ({p90/results[K]['bagged']:.2f}x)")

with open(os.path.join(DATA_DIR, 'k_sweep.json'), 'w') as f:
    out = {
        'median_by_K': {str(K): v for K, v in results.items()},
        'ridge_p90_by_K': {str(K): float(np.percentile(seeded_rmses['ridge'][K], 90)) for K in Ks},
    }
    json.dump(out, f, indent=2)
print(f"\nSaved {os.path.join(DATA_DIR, 'k_sweep.json')}")
