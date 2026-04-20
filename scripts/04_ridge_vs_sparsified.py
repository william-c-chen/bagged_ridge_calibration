"""Reproduces Table 4 (ridge vs sparsified alternatives).

Computes LOO-CV RMSE at K=6 and K=3 (all C(7,3)=35 subsets) and worst-gene probe
failure sensitivity for the bagged ensemble, sparsified versions of its mean
coefficients, a weakly regularized scaled version, and selected single constituent
ridge models.
"""
from itertools import combinations
import numpy as np
from scipy import stats

from utils import (
    load_coefficients, load_normalized_2026,
    ensemble_beta, ensemble_intercept, coefficients_available,
    QC_PASS_N7,
)

OTHER_LOTS = ['B', 'D', 'E']


def score_with(beta_vec, intercept, x):
    return intercept + beta_vec @ x


def loocv_rmse(beta_vec, intercept, X, samples=QC_PASS_N7, k=6, kind='linear'):
    yA = np.array([score_with(beta_vec, intercept, X[('A', s)]) for s in samples])
    yT = np.array([np.mean([score_with(beta_vec, intercept, X[(L, s)]) for L in OTHER_LOTS])
                   for s in samples])
    errs = []
    n = len(samples)
    if k == n - 1:  # classic LOO
        for i in range(n):
            m = np.ones(n, dtype=bool); m[i] = False
            if kind == 'additive':
                add_c = np.mean(yT[m] - yA[m])
                errs.append(yT[i] - (yA[i] + add_c))
            else:
                sl, ic, _, _, _ = stats.linregress(yA[m], yT[m])
                errs.append(yT[i] - (sl * yA[i] + ic))
    else:
        # Enumerate all size-k training subsets
        all_errs = []
        for train in combinations(range(n), k):
            train_idx = list(train)
            test_idx = [i for i in range(n) if i not in train_idx]
            yA_tr = yA[train_idx]; yT_tr = yT[train_idx]
            if np.var(yA_tr) < 1e-12:
                continue
            sl, ic, _, _, _ = stats.linregress(yA_tr, yT_tr)
            for i in test_idx:
                all_errs.append(yT[i] - (sl * yA[i] + ic))
        errs = all_errs
    return np.sqrt(np.mean(np.array(errs) ** 2))


def probe_failure(beta_vec, X, samples=QC_PASS_N7):
    worst = 0.0
    for g in range(len(beta_vec)):
        shifts = [abs(beta_vec[g] * X[('A', s)][g]) for s in samples]
        worst = max(worst, max(shifts))
    return worst


def sparsify(beta_vec, frac):
    """Zero out coefficients outside the top |frac| fraction; rescale remaining to preserve sum of |beta|."""
    k = max(1, int(round(frac * len(beta_vec))))
    order = np.argsort(np.abs(beta_vec))[::-1]
    keep = order[:k]
    sparse = np.zeros_like(beta_vec)
    sparse[keep] = beta_vec[keep]
    # Rescale to preserve total L1 norm
    if np.sum(np.abs(sparse)) > 0:
        sparse *= np.sum(np.abs(beta_vec)) / np.sum(np.abs(sparse))
    return sparse


def main():
    if not coefficients_available():
        print('This script requires the 500-model ridge coefficient matrix and')
        print('the ensemble mean coefficient vector to construct sparsified and')
        print('weakly regularized alternatives. The coefficients are not distributed')
        print('with this repository; see README.md for details on obtaining them.')
        return
    genes, beta, intercepts = load_coefficients()
    _, _, _, X = load_normalized_2026()
    bbar = ensemble_beta(beta)
    int_bar = ensemble_intercept(intercepts)

    print('Table 4: LOO-CV RMSE and worst-gene probe failure sensitivity')
    print(f'{"Model":<52} {"K=6":>6} {"K=3":>7} {"Probe fail":>10}')

    def row(label, bvec, ic):
        r6 = loocv_rmse(bvec, ic, X, k=6)
        r3 = loocv_rmse(bvec, ic, X, k=3)
        pf = probe_failure(bvec, X)
        print(f'{label:<52} {r6:.3f}  {r3:.3f}   {pf:.3f}')

    row('Bagged ridge (500-model ensemble)', bbar, int_bar)
    for frac, label in [(0.70, '70% sparsified (24 genes)'),
                        (0.50, '50% sparsified (17 genes)'),
                        (0.30, '30% sparsified (10 genes)'),
                        (0.15, '15% sparsified (5 genes)')]:
        row(label, sparsify(bbar, frac), int_bar)
    row('Weakly reg. (beta x 2.5)', 2.5 * bbar, int_bar)

    # Median-L2-norm single ridge constituent
    L2 = np.linalg.norm(beta, axis=0)
    med_idx = int(np.argsort(L2)[len(L2) // 2])
    row('Median-||b||_2 single ridge', beta[:, med_idx], intercepts[med_idx])
    row('  sparsified to 15%', sparsify(beta[:, med_idx], 0.15), intercepts[med_idx])
    p90_idx = int(np.argsort(L2)[int(0.9 * len(L2))])
    row('P90-||b||_2 single ridge', beta[:, p90_idx], intercepts[p90_idx])


if __name__ == '__main__':
    main()
