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
