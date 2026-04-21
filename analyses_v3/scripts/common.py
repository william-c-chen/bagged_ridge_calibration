"""Shared constants and utilities for Option B analyses.

Fixed-seed, fully reproducible. All analysis scripts source this module.
"""
from __future__ import annotations
import os
import re
import numpy as np

# ---------- Reproducibility ----------
RANDOM_SEED = 42  # fixed; change here to rerun under a different seed

# ---------- Paths (relative to repo root) ----------
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR     = os.path.join(os.path.dirname(__file__), '..', 'figures')
LOG_DIR     = os.path.join(os.path.dirname(__file__), '..', 'logs')

DAT_INTEG            = os.path.join(REPO_ROOT, 'dat.integ.master.5.6.2025.xlsx')
CALIB_DEEPDIVE       = os.path.join(REPO_ROOT, 'Calibration_DeepDive_Results.xlsx')
SAMPLES_CALIB        = os.path.join(REPO_ROOT, 'Samples_for_calibration.xlsx')
RCC_2023_FIRST       = os.path.join(REPO_ROOT, 'Previous_calibration_run', 'first_batch')
RCC_2023_FIRST_IDX   = os.path.join(RCC_2023_FIRST, 'NanoString RCC index-Chen 20231007.xlsx')

# ---------- 34 biomarker genes (from Chen 2023 Nat Med Supp Table 1/3) ----------
GENE_NAMES = [
    'ARID1B','CCL21','CCN1','CCND2','CD3E','CDC20','CDK6','CDKN2A','CDKN2C','CHEK1',
    'CKS2','COL1A1','ESR1','EZH2','FBLIM1','FGFR4','GAS1','IFNGR1','IGF2','KDR',
    'KIF20A','KRT14','LINC02593','MDM4','MMP9','MUTYH','MYBL1','PGK1','PGR','PIM1',
    'SPOP','TAGLN','TMEM30B','USF1',
]

# ---------- 7 housekeeping genes ----------
HK_NAMES = ['ACTB','CASC3','GUSB','KIAA1715','MRPL19','POP4','TTC21B']

# ---------- 2026 experiment: 4 codeset lots (cartridge-identified) ----------
LOTS_2026 = {
    'A': 'C10904X1',
    'B': 'C9543',
    'D': 'C11199',
    'E': 'C11695X1',
}

# ---------- 2026 bridging samples (all 10 lanes of the PCR strip) ----------
BRIDGING_SAMPLES_2026_ALL = ['AM42','AM57','AM32','AM35','AM53','QM313','QM318','QM322','QM325','QM326']

# ---------- 2026 bridging samples with complete 4-lot coverage (QC-passing) ----------
BRIDGING_SAMPLES_2026 = ['AM57','AM32','AM35','AM53','QM318','QM322','QM325']  # N=7

# ---------- 2023 experiment: 3 codeset lots ----------
LOTS_2023 = {
    'A': 'C10904',
    'B': 'C9543',
    'C': 'C10132',
}

# ---------- 2023 bridging samples (6 ran; paper uses 5 after QC per Table 3) ----------
BRIDGING_SAMPLES_2023_ALL = ['MEN-322','MEN-315','MEN-268','MEN-303','MEN-308','MEN-256']


# ==================== RCC parsing ====================

_SAMPLE_RE   = re.compile(r'<Sample_Attributes>.*?\nID,([^\n]*)', re.S)
_CODESET_RE  = re.compile(r'GeneRLF,([^\n]*)')
_LANE_RE     = re.compile(r'<Lane_Attributes>.*?\nID,([^\n]*)', re.S)
_CART_RE     = re.compile(r'CartridgeID,([^\n]*)')

def parse_rcc_file(path: str) -> dict:
    """Parse a NanoString RCC file, return dict with raw counts + metadata.

    Output keys: sample_id, codeset, lane, cartridge, counts (dict probe_name -> int).
    """
    with open(path, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    sample_id = _SAMPLE_RE.search(text).group(1).strip()
    codeset   = _CODESET_RE.search(text).group(1).strip()
    lane      = _LANE_RE.search(text).group(1).strip()
    cart      = _CART_RE.search(text).group(1).strip()

    # Code_Summary section: CodeClass,Name,Accession,Count
    cs_match = re.search(r'<Code_Summary>(.*?)</Code_Summary>', text, re.S)
    counts = {}
    for line in cs_match.group(1).splitlines():
        parts = line.strip().split(',')
        if len(parts) == 4 and parts[0] != 'CodeClass':
            probe_class, probe_name, _acc, cnt = parts
            try:
                counts[probe_name] = int(cnt)
            except ValueError:
                pass
    return {
        'sample_id': sample_id, 'codeset': codeset, 'lane': lane,
        'cartridge': cart, 'counts': counts, 'path': path,
    }


def hk_normalize(counts_row: np.ndarray, hk_row: np.ndarray) -> np.ndarray:
    """Apply the Chen 2023 normalization: x = log2((r + 1) / geom_mean(hk + 1)).

    Vectorized. `counts_row` and `hk_row` can be 1D (one sample) or 2D (samples × genes).
    """
    if counts_row.ndim == 1:
        h = np.exp(np.mean(np.log(hk_row + 1)))
        return np.log2((counts_row + 1) / h)
    else:
        h = np.exp(np.mean(np.log(hk_row + 1), axis=1))  # per-sample h
        return np.log2((counts_row + 1) / h[:, None])


def ensure_dirs():
    for d in (DATA_DIR, FIG_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


# ==================== Calibration metrics (matching paper conventions) ====================
# These match the protocols in scripts/02_leave_one_out_cv.py and 04_ridge_vs_sparsified.py
# of the bagged_ridge_calibration repo (Tables 1, 2, 6 of the paper).

OTHER_LOTS_2026 = ['B', 'D', 'E']  # for "predict mean(B,D,E) from A" LOO-CV protocol


def loocv_amean_k6(scores, samples, lots_target=OTHER_LOTS_2026):
    """K=6 LOO-CV linear-correction RMSE, paper protocol:
    target = mean of scores on `lots_target`; predictor = score on lot A.
    Fit linear correction on K-1 samples, predict K-th, iterate, return RMSE."""
    import numpy as np
    from scipy import stats
    yA = np.array([scores[(s, 'A')] for s in samples])
    yT = np.array([np.mean([scores[(s, L)] for L in lots_target]) for s in samples])
    errs = []
    for i in range(len(samples)):
        m = np.ones(len(samples), dtype=bool); m[i] = False
        sl, ic, *_ = stats.linregress(yA[m], yT[m])
        errs.append(yT[i] - (sl * yA[i] + ic))
    return float(np.sqrt(np.mean(np.array(errs)**2)))


def loocv_amean_k3(scores, samples, lots_target=OTHER_LOTS_2026):
    """K=3 enumeration: all C(N,3) 3-sample training subsets, paper protocol."""
    import numpy as np
    from scipy import stats
    from itertools import combinations
    yA = np.array([scores[(s, 'A')] for s in samples])
    yT = np.array([np.mean([scores[(s, L)] for L in lots_target]) for s in samples])
    all_errs = []
    for train in combinations(range(len(samples)), 3):
        train_idx = list(train)
        test_idx = [i for i in range(len(samples)) if i not in train_idx]
        yA_tr = yA[train_idx]; yT_tr = yT[train_idx]
        if np.var(yA_tr) < 1e-12: continue
        sl, ic, *_ = stats.linregress(yA_tr, yT_tr)
        for i in test_idx:
            all_errs.append(yT[i] - (sl * yA[i] + ic))
    return float(np.sqrt(np.mean(np.array(all_errs)**2)))


def probe_failure_paper(beta, X_norm_lotA_per_sample):
    """Worst-gene probe-failure sensitivity, paper protocol:
    For each gene g and sample s: shift = |β_g × X_norm_lotA(s)[g]|
    Returns max over all (g, s)."""
    import numpy as np
    worst = 0.0
    for g in range(len(beta)):
        for s_vec in X_norm_lotA_per_sample:
            val = abs(beta[g] * s_vec[g])
            if val > worst: worst = val
    return float(worst)
