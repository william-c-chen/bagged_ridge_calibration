"""Reproduces the HK variance absorption (Table 1) and POS spike-in ratios.

HK variance: for the 4 lots in 2026, compute cross-lot variance of
  - raw log2(r+1) for each (gene, sample)
  - housekeeping-normalized x_g
averaged over the 7 QC-pass samples and 34 biomarker genes.

POS spike-ins: report log2(lot B / lot A) count ratio by concentration, across samples.
"""
import numpy as np
import pandas as pd

from utils import (
    load_raw_counts_2026, load_positive_controls, load_normalized_2026,
    QC_PASS_N7,
)


def main():
    # ---- HK variance ----
    rc = load_raw_counts_2026()
    lots = ['A', 'B', 'D', 'E']

    bm = rc[rc['class'] == 'BM'].set_index('gene').drop(columns='class')
    hk = rc[rc['class'] == 'HK'].set_index('gene').drop(columns='class')

    def col(lot, s): return f'{lot}_{s}'

    raw_vars, norm_vars = [], []
    for s in QC_PASS_N7:
        bm_vals = {lot: bm[col(lot, s)].values for lot in lots}
        hk_vals = {lot: hk[col(lot, s)].values for lot in lots}
        for gi in range(bm.shape[0]):
            raw = [np.log2(bm_vals[lot][gi] + 1) for lot in lots]
            hk_gmean = [np.exp(np.mean(np.log(hk_vals[lot] + 1))) for lot in lots]
            norm = [np.log2((bm_vals[lot][gi] + 1) / hk_gmean[li]) for li, lot in enumerate(lots)]
            raw_vars.append(np.var(raw, ddof=0))
            norm_vars.append(np.var(norm, ddof=0))
    raw_mean, norm_mean = np.mean(raw_vars), np.mean(norm_vars)
    raw_sd, norm_sd = np.sqrt(raw_mean), np.sqrt(norm_mean)
    print('Cross-lot variance (averaged over 34 genes x 7 samples):')
    print(f'  Raw log2(r+1):       variance={raw_mean:.3f}   SD={raw_sd:.3f}')
    print(f'  HK-normalized x_g:   variance={norm_mean:.3f}   SD={norm_sd:.3f}')
    print(f'  Absorbed fraction:   {(1 - norm_mean/raw_mean):.0%}')
    print()

    # ---- POS spike-ins ----
    pos = load_positive_controls()
    print('Positive-control spike-ins: log2(lot B / lot A) ratio by probe')
    for probe, sub in pos.groupby('pos_probe'):
        wide = sub.pivot(index='sample_id', columns='lot', values='count')
        ok = wide.dropna(subset=['A', 'B'])
        if len(ok) == 0: continue
        ratios = np.log2(ok['B'] / ok['A']).values
        conc = sub['concentration_fM'].iloc[0]
        print(f'  {probe} ({conc:>5g} fM): mean log2(B/A)={np.mean(ratios):+.3f}  SD={np.std(ratios):.3f}  N={len(ratios)}')


if __name__ == '__main__':
    main()
