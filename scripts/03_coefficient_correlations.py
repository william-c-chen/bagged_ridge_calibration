"""Reproduces Table 5 correlations and the r = 0.65 / r = 0.33 figures from the paper.

Across the 500 bootstrap ridge constituents, computes:
  - ||beta||_2 per model
  - beta^T Sigma_delta beta per model (directional quantity)
  - LOO-CV RMSE per model (linear correction, lot A -> mean of others)
  - Worst-gene probe failure sensitivity per model
  - Correlations between coefficient-structure summaries and calibration outcomes.
"""
import numpy as np
from scipy import stats

from utils import (
    load_coefficients, load_normalized_2026,
    single_model_scores, coefficients_available,
    QC_PASS_N7,
)

OTHER_LOTS = ['B', 'D', 'E']


def loocv_lin_rmse(score_A, score_T):
    errs = []
    for i in range(len(score_A)):
        m = np.ones(len(score_A), dtype=bool); m[i] = False
        if np.var(score_A[m]) < 1e-12:
            continue
        sl, ic, _, _, _ = stats.linregress(score_A[m], score_T[m])
        errs.append(score_T[i] - (sl * score_A[i] + ic))
    errs = np.array(errs)
    return np.sqrt(np.mean(errs**2)) if len(errs) else np.nan


def main():
    if not coefficients_available():
        print('This script requires the 500-model ridge coefficient matrix, which')
        print('is not distributed with this repository. See README.md for details')
        print('on obtaining the coefficients for internal reproduction.')
        return
    genes, beta, intercepts = load_coefficients()
    _, _, _, X = load_normalized_2026()

    # Sigma_delta: covariance of delta^{AB}(s) across the 7 QC-pass samples
    deltas = np.array([X[('B', s)] - X[('A', s)] for s in QC_PASS_N7])  # 7 x 34
    Sigma_delta = np.cov(deltas, rowvar=False)  # 34 x 34

    N = beta.shape[1]
    L2 = np.linalg.norm(beta, axis=0)
    directional = np.array([beta[:, i] @ Sigma_delta @ beta[:, i] for i in range(N)])

    # Per-model LOO-CV RMSE (A -> mean of others)
    loocv_rmses = []
    for i in range(N):
        yA = np.array([single_model_scores(X[('A', s)], beta, intercepts)[i] for s in QC_PASS_N7])
        yT = np.array([np.mean([single_model_scores(X[(L, s)], beta, intercepts)[i] for L in OTHER_LOTS])
                       for s in QC_PASS_N7])
        loocv_rmses.append(loocv_lin_rmse(yA, yT))
    loocv_rmses = np.array(loocv_rmses)

    # Probe failure sensitivity: max over genes of absolute score change when one gene's
    # normalized expression is zeroed on lot A (approximated by setting that gene's count
    # effect to -mean across samples).
    probe_fail = []
    for i in range(N):
        gene_effects = []
        for g in range(34):
            # Score impact of gene g on lot A for each sample:
            # Change = coef * (0 - x_g) when probe fails completely.
            shifts = []
            for s in QC_PASS_N7:
                shifts.append(abs(beta[g, i] * X[('A', s)][g]))
            gene_effects.append(max(shifts))
        probe_fail.append(max(gene_effects))
    probe_fail = np.array(probe_fail)

    print('Per-model coefficient-structure summaries (across 500 models):')
    print(f'  ||beta||_2: median={np.median(L2):.3f}, range [{L2.min():.3f}, {L2.max():.3f}]')
    print(f'  beta^T Sigma_delta beta: median={np.median(directional):.4f}')
    print(f'  LOO-CV lin RMSE: median={np.median(loocv_rmses):.4f}')
    print(f'  Probe failure sensitivity: median={np.median(probe_fail):.3f}')
    print()

    print('Table 5 correlations:')
    print(f'{"Metric":<40} {"r vs ||b||_2":<14} {"r vs b^T Sd b":<14}')
    for lab, outcome in [('LOO-CV RMSE', loocv_rmses),
                         ('Worst-gene probe failure', probe_fail)]:
        r1 = stats.pearsonr(L2, outcome)[0]
        r2 = stats.pearsonr(directional, outcome)[0]
        print(f'{lab:<40} {r1:>8.3f}        {r2:>8.3f}')


if __name__ == '__main__':
    main()
