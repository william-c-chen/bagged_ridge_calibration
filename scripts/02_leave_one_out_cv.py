"""Reproduces Table 1 (LOO-CV of the calibration correction, 2026 data).

Target: for each sample s, predict the mean score on the three non-outlier lots
(B, D, E) from the lot-A score, using a correction fit on the other 6 samples.
Report RMSE and max |err| for ensemble (using ensemble scores) and for the 500
single-model distribution (requires coefficients, not distributed).
"""
import numpy as np
from scipy import stats

from utils import (
    load_scores_2026, load_coefficients, load_normalized_2026,
    single_model_scores, coefficients_available, QC_PASS_N7,
)

OTHER_LOTS = ['B', 'D', 'E']


def loocv(score_A, score_T):
    add_errs, lin_errs = [], []
    for i in range(len(score_A)):
        m = np.ones(len(score_A), dtype=bool); m[i] = False
        add_c = np.mean(score_T[m] - score_A[m])
        add_errs.append(score_T[i] - (score_A[i] + add_c))
        if np.var(score_A[m]) > 1e-12:
            sl, ic, _, _, _ = stats.linregress(score_A[m], score_T[m])
            lin_errs.append(score_T[i] - (sl * score_A[i] + ic))
    add_errs = np.array(add_errs); lin_errs = np.array(lin_errs)
    return (np.sqrt(np.mean(add_errs**2)), np.max(np.abs(add_errs)),
            np.sqrt(np.mean(lin_errs**2)), np.max(np.abs(lin_errs)))


def main():
    # ---- Ensemble row (from published ensemble scores) ----
    s26 = load_scores_2026()
    s26 = s26[s26['sample_id'].isin(QC_PASS_N7)].pivot(index='sample_id', columns='lot', values='ensemble_score')
    yA_ens = np.array([s26.loc[s, 'A'] for s in QC_PASS_N7])
    yT_ens = np.array([np.mean([s26.loc[s, L] for L in OTHER_LOTS]) for s in QC_PASS_N7])
    add_r, add_m, lin_r, lin_m = loocv(yA_ens, yT_ens)

    print('Table 1: LOO-CV of calibration correction (2026 data, N=7 held out positions)')
    print(f'{"":<24} {"Additive":>10} {"":>12} {"Linear":>10} {"":>12}')
    print(f'{"":<24} {"RMSE":>6} {"Max|err|":>10} {"RMSE":>6} {"Max|err|":>10}')
    print(f'{"Ensemble (500 models)":<24} {add_r:.3f}   {add_m:.3f}      {lin_r:.3f}   {lin_m:.3f}')

    if not coefficients_available():
        print(f'{"P50 single model":<24} [skipped: coefficients not distributed]')
        print(f'{"P90 single model":<24} [skipped: coefficients not distributed]')
        print('\nTo obtain single-model rows, request the 500-model coefficient matrix')
        print('from the corresponding authors.')
        return

    _, _, _, X = load_normalized_2026()
    _, beta, intercepts = load_coefficients()
    single_add_r, single_lin_r, single_add_m, single_lin_m = [], [], [], []
    for i in range(beta.shape[1]):
        yA_s = np.array([single_model_scores(X[('A', s)], beta, intercepts)[i] for s in QC_PASS_N7])
        yT_s = np.array([np.mean([single_model_scores(X[(L, s)], beta, intercepts)[i] for L in OTHER_LOTS])
                         for s in QC_PASS_N7])
        ar, am, lr, lm = loocv(yA_s, yT_s)
        single_add_r.append(ar); single_lin_r.append(lr)
        single_add_m.append(am); single_lin_m.append(lm)
    for pct, label in [(50, 'P50 single model'), (90, 'P90 single model')]:
        ar = np.percentile(single_add_r, pct); am = np.percentile(single_add_m, pct)
        lr = np.percentile(single_lin_r, pct); lm = np.percentile(single_lin_m, pct)
        print(f'{label:<24} {ar:.3f}   {am:.3f}      {lr:.3f}   {lm:.3f}')
    improvement = (np.median(single_lin_r) - lin_r) / np.median(single_lin_r)
    print(f'\nEnsemble linear-RMSE improvement over median single: {improvement:.1%}')


if __name__ == '__main__':
    main()
