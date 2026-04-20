"""Reproduces the 'Additional cross-time reproducibility' section of the paper.

Compares 2026 ensemble scores on each lot against historical (database) scores
from the original validation cohort. QM313 (sample S11) is excluded a priori for
QC reasons; historical scores are reported for the subset present in the database.
"""
import numpy as np
import pandas as pd
from scipy import stats

from utils import load_scores_2026, load_historical_scores


# Samples to exclude from cross-time comparison (failed QC)
EXCLUDE = {'S11'}  # original ID QM313; failed hybridization on lot D


def main():
    s26 = load_scores_2026()
    hist = load_historical_scores().set_index('sample_id')

    # 2026 A -> B linear correction (from the paper): slope 0.9048, intercept 0.0909
    a_AB, b_AB = 0.9048, 0.0909

    s26_wide = s26[~s26['sample_id'].isin(EXCLUDE)].pivot(
        index='sample_id', columns='lot', values='ensemble_score'
    )

    def compare(col_name, get_score):
        xs, ys = [], []
        for sid in s26_wide.index:
            if sid not in hist.index:
                continue
            v = get_score(sid)
            if v is None or np.isnan(v):
                continue
            xs.append(v)
            ys.append(hist.loc[sid, 'historical_score'])
        xs = np.array(xs); ys = np.array(ys)
        sl, ic, r, _, _ = stats.linregress(xs, ys)
        direct_rmse = np.sqrt(np.mean((ys - xs) ** 2))
        print(f'{col_name:<40} N={len(xs):<2} slope={sl:.3f}  intercept={ic:+.3f}  R^2={r**2:.3f}  '
              f'direct RMSE={direct_rmse:.4f}')

    print('Current-codeset scores vs historical database scores (S11 excluded):')
    print()
    for lot in ['A', 'B', 'D', 'E']:
        compare(f'Lot {lot}', lambda sid, lot=lot: s26_wide.loc[sid, lot] if lot in s26_wide.columns else None)
    compare('Lot A, corrected (0.9048x + 0.0909)',
            lambda sid: a_AB * s26_wide.loc[sid, 'A'] + b_AB)


if __name__ == '__main__':
    main()
