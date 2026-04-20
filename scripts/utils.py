"""Shared data-loading and scoring helpers.

The 500-model ridge coefficient matrix and per-model intercepts are not
redistributed with this repository (see README.md, "Model coefficients and
the Shiny scoring app"). Scripts that require single-model resolution
(Tables 3, 4, 5 and the single-model rows of Table 1) will report that the
coefficient data is unavailable and skip those analyses. Analyses that
operate on the ensemble-level scores in scores_2023.csv, scores_2026.csv,
and historical_scores.csv remain fully reproducible from the public data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / 'data'


def coefficients_available() -> bool:
    """Return True if the model coefficient files are present locally."""
    return (DATA / 'ensemble_coefficients.csv').exists() and \
           (DATA / 'ensemble_intercepts.csv').exists()


def load_coefficients():
    """Return (genes, beta_matrix_34x500, intercept_500).

    Raises FileNotFoundError with a helpful message if the coefficient
    files are not present locally. See README for how to obtain them.
    """
    coef_path = DATA / 'ensemble_coefficients.csv'
    inter_path = DATA / 'ensemble_intercepts.csv'
    if not (coef_path.exists() and inter_path.exists()):
        raise FileNotFoundError(
            'Model coefficients are not distributed in this repository. '
            'To compute risk scores on new data, use the Shiny scoring app '
            'referenced in the Nature Medicine (2023) publication. '
            'For internal reproduction of single-model analyses, obtain the '
            'coefficient matrix from the corresponding authors.'
        )
    coefs = pd.read_csv(coef_path)
    genes = coefs['gene'].tolist()
    beta = coefs.drop(columns='gene').values  # 34 x 500
    intercepts = pd.read_csv(inter_path)['intercept'].values  # 500
    return genes, beta, intercepts


def load_normalized_2026():
    """Return (genes, sample_ids, lots, X[lot,sample] dict of 34-vec)."""
    ne = pd.read_csv(DATA / 'normalized_expression_2026.csv')
    genes = ne['gene'].tolist()
    X = {}
    for col in ne.columns[1:]:
        lot, sample_id = col.split('_', 1)
        X[(lot, sample_id)] = ne[col].values
    samples = sorted(set(k[1] for k in X))
    lots = sorted(set(k[0] for k in X))
    return genes, samples, lots, X


def load_raw_counts_2026():
    return pd.read_csv(DATA / 'raw_counts_2026.csv')


def load_scores_2026():
    return pd.read_csv(DATA / 'scores_2026.csv')


def load_scores_2023():
    return pd.read_csv(DATA / 'scores_2023.csv')


def load_positive_controls():
    return pd.read_csv(DATA / 'positive_controls_2026.csv')


def load_historical_scores():
    return pd.read_csv(DATA / 'historical_scores.csv')


def ensemble_score(x_vec, beta, intercepts):
    """Ensemble score = mean over 500 models of (intercept + beta_i . x)."""
    per_model = intercepts + x_vec @ beta  # 500
    return per_model.mean()


def single_model_scores(x_vec, beta, intercepts):
    """Return 500-vector of per-model scores."""
    return intercepts + x_vec @ beta


def ensemble_beta(beta):
    return beta.mean(axis=1)


def ensemble_intercept(intercepts):
    return intercepts.mean()


# Paper's QC-passing 2026 bridging sample set (N=7 used for cross-lot analyses)
QC_PASS_N7 = ['S06', 'S07', 'S09', 'S10', 'S12', 'S13', 'S14']
