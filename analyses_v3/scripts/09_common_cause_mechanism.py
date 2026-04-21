"""Test the common-cause mechanism for ridge's implicit cross-lot robustness.

Hypothesis: ridge was trained on a single codeset lot, yet its coefficients
negatively correlate with cross-lot drift (§3.4, r = -0.25). This can happen
without cross-lot supervision if within-training measurement variance and
cross-lot drift share a probe-level common cause.

Tests the causal chain:
  1. within-training expression SD ~ cross-lot drift magnitude (positive r)
  2. within-training expression SD ~ ridge coefficient shrinkage (negative r for |β̄|)
  3. Therefore emerges: lot drift ~ |β̄| is negative (matches §3.4)

Also reports bootstrap CV of coefficients across the 500 constituents, which
captures the same common-cause mechanism from the coefficient side.
"""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import DATA_DIR, GENE_NAMES

# Training expression and deployed β
tr = np.load(os.path.join(DATA_DIR, 'ffpe_paired_training_pc.npz'), allow_pickle=True)
X_tr, y_tr = tr['X_norm'], tr['y']
dep = np.load(os.path.join(DATA_DIR, 'deployed_ensemble_true.npz'), allow_pickle=True)
beta = dep['beta']

# 500 bootstrap constituents from the FFPE-paired reproduction
with open(os.path.join(DATA_DIR, 'glmnet_ffpe_pc_coefs_500.csv')) as f:
    rdr = csv.reader(f); next(rdr); rows = [r for r in rdr]
names_f = [r[0] for r in rows]
B = np.array([[float(v) for v in r[1:]] for r in rows])
order = np.argsort([GENE_NAMES.index(g) for g in names_f])
B = B[order]
assert [names_f[i] for i in order] == GENE_NAMES

boot_mean = B.mean(axis=1)
boot_sd = B.std(axis=1, ddof=1)
boot_cv = boot_sd / (np.abs(boot_mean) + 1e-10)

train_sd = X_tr.std(axis=0, ddof=1)
train_cor_y = np.array([np.corrcoef(X_tr[:, g], y_tr)[0, 1] for g in range(34)])

with open(os.path.join(DATA_DIR, 'per_gene_lot_drift.json')) as f:
    pg = json.load(f)
drift = np.array([next(x['avg_mean_delta'] for x in pg['per_gene'] if x['gene'] == g)
                  for g in GENE_NAMES])


def r(a, b): return float(np.corrcoef(a, b)[0, 1])


print("Chain of mediation:")
print(f"  [1] within-training SD   ~  cross-lot drift       r = {r(train_sd, drift):+.3f}")
print(f"  [2] within-training SD   ~  |β̄_g|                 r = {r(train_sd, np.abs(beta)):+.3f}")
print(f"  [3] cross-lot drift      ~  |β̄_g|                 r = {r(drift, np.abs(beta)):+.3f}")
print(f"\nAuxiliary:")
print(f"  |corr(X_g, y)|           ~  |β̄_g|                 r = {r(np.abs(train_cor_y), np.abs(beta)):+.3f}")
print(f"  bootstrap CV |SD/mean β| ~  cross-lot drift       r = {r(boot_cv, drift):+.3f}")
print(f"  bootstrap CV             ~  within-training SD    r = {r(boot_cv, train_sd):+.3f}")

out = {
    'r_train_sd_vs_drift':   r(train_sd, drift),
    'r_train_sd_vs_beta':    r(train_sd, np.abs(beta)),
    'r_drift_vs_beta':       r(drift, np.abs(beta)),
    'r_corXY_vs_beta':       r(np.abs(train_cor_y), np.abs(beta)),
    'r_bootCV_vs_drift':     r(boot_cv, drift),
    'r_bootCV_vs_train_sd':  r(boot_cv, train_sd),
}
with open(os.path.join(DATA_DIR, 'common_cause_mechanism.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {os.path.join(DATA_DIR, 'common_cause_mechanism.json')}")
