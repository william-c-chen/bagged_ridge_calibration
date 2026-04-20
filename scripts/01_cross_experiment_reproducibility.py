"""Reproduces Tables 2 and 3 of the paper.

Table 2 (ensemble-level): Linear correction parameters for lot A -> lot B in
2023 (N=5) and 2026 (N=7). Uses published ensemble scores only.

Table 3 (single-model distribution): |Delta slope| and |Delta intercept| for
the ensemble vs 500 single bootstrap ridge models. Requires the model
coefficient matrix, which is not distributed in this repository (see README,
"Model coefficients and the Shiny scoring app").
"""
import numpy as np
from scipy import stats

from utils import (
    load_scores_2023, load_scores_2026, load_coefficients, load_normalized_2026,
    single_model_scores, coefficients_available, QC_PASS_N7,
)


def fit_ab(yA, yB):
    slope, intercept, r, _, _ = stats.linregress(yA, yB)
    return slope, intercept, r ** 2


def main():
    # ---- Table 2: ensemble A -> B, 2023 and 2026 (ensemble scores only) ----
    s23 = load_scores_2023().pivot(index='sample_id', columns='lot', values='ensemble_score')
    yA_23, yB_23 = s23['A'].values, s23['B'].values
    s23_slope, s23_ic, s23_r2 = fit_ab(yA_23, yB_23)

    s26 = load_scores_2026()
    s26 = s26[s26['sample_id'].isin(QC_PASS_N7)].pivot(index='sample_id', columns='lot', values='ensemble_score')
    yA_26, yB_26 = s26['A'].values, s26['B'].values
    s26_slope, s26_ic, s26_r2 = fit_ab(yA_26, yB_26)

    print('Table 2: Linear correction parameters for lot A -> B')
    print(f'{"Experiment":<10} {"N":<3} {"Slope":<8} {"Intercept":<10} {"R^2":<6}')
    print(f'{"2023":<10} {len(yA_23):<3} {s23_slope:.3f}    {s23_ic:+.3f}      {s23_r2:.3f}')
    print(f'{"2026":<10} {len(yA_26):<3} {s26_slope:.3f}    {s26_ic:+.3f}      {s26_r2:.3f}')
    print(f'{"Diff":<10}     {abs(s26_slope - s23_slope):.3f}    {abs(s26_ic - s23_ic):.3f}')

    # ---- Table 3: single-model distribution ----
    print()
    if not coefficients_available():
        print('Table 3 (single-model distribution): skipped.')
        print('  Requires the 34x500 ridge coefficient matrix and 500 intercepts,')
        print('  which are not distributed with this repository. Contact the')
        print('  corresponding authors to obtain these for internal reproduction.')
        return

    import pandas as pd
    from pathlib import Path
    ne23 = pd.read_csv(Path(__file__).resolve().parent.parent / 'data' / 'normalized_expression_2023.csv')
    X23 = {(c.split('_')[0], c.split('_')[1]): ne23[c].values for c in ne23.columns[1:]}
    _, _, _, X26 = load_normalized_2026()
    _, beta, intercepts = load_coefficients()

    SAMPLES_2023 = ['S01', 'S02', 'S03', 'S04', 'S05']
    yA_23_all = np.array([single_model_scores(X23[('A', s)], beta, intercepts) for s in SAMPLES_2023])
    yB_23_all = np.array([single_model_scores(X23[('B', s)], beta, intercepts) for s in SAMPLES_2023])
    yA_26_all = np.array([single_model_scores(X26[('A', s)], beta, intercepts) for s in QC_PASS_N7])
    yB_26_all = np.array([single_model_scores(X26[('B', s)], beta, intercepts) for s in QC_PASS_N7])

    single_dslope, single_dic = [], []
    for i in range(beta.shape[1]):
        s23, ic23, _ = fit_ab(yA_23_all[:, i], yB_23_all[:, i])
        s26, ic26, _ = fit_ab(yA_26_all[:, i], yB_26_all[:, i])
        single_dslope.append(abs(s26 - s23))
        single_dic.append(abs(ic26 - ic23))
    single_dslope = np.array(single_dslope); single_dic = np.array(single_dic)

    ens_dslope = abs(s26_slope - s23_slope); ens_dic = abs(s26_ic - s23_ic)
    print('Table 3: Cross-experiment |Delta| for lot A -> B correction parameters')
    print(f'{"":<20}  {"|Delta slope|":<14} {"|Delta intercept|":<16}')
    print(f'{"Ensemble":<20}  {ens_dslope:.3f}          {ens_dic:.3f}')
    print(f'{"Single P10":<20}  {np.percentile(single_dslope,10):.3f}          {np.percentile(single_dic,10):.3f}')
    print(f'{"Single P50 (median)":<20}  {np.percentile(single_dslope,50):.3f}          {np.percentile(single_dic,50):.3f}')
    print(f'{"Single P90":<20}  {np.percentile(single_dslope,90):.3f}          {np.percentile(single_dic,90):.3f}')
    print(f'\nEnsemble outperforms {(single_dslope > ens_dslope).mean():.0%} of single models on |Delta slope|')


if __name__ == '__main__':
    main()
