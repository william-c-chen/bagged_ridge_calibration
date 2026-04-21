"""Load the deployed 500-bag β coefficient matrix from internal data files.

NOT INCLUDED IN PUBLIC RELEASE: The deployed model's coefficients are sensitive
intellectual property and are not redistributed. This script reads them from
analysis_scripts/internal_data/raleigh75_*.csv (generated upstream from the
locked Chen 2023 model) and saves a re-ordered NPZ for downstream scripts.

If you do not have access to the deployed coefficient files, you can still run
the rest of the pipeline; downstream scripts that compare reproductions to the
deployed coefficient direction (e.g., cos similarity) will skip those columns.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import REPO_ROOT, DATA_DIR, GENE_NAMES, ensure_dirs

ensure_dirs()
COEFS = os.path.join(REPO_ROOT, 'analysis_scripts', 'internal_data', 'raleigh75_coef_matrix.csv')
INTCS = os.path.join(REPO_ROOT, 'analysis_scripts', 'internal_data', 'raleigh75_intercepts.csv')

if not (os.path.exists(COEFS) and os.path.exists(INTCS)):
    print(f"Deployed coefficient files not found at {COEFS}.\n"
          "Skipping. Downstream scripts will run without deployed comparison.")
    sys.exit(0)

with open(COEFS) as f:
    reader = csv.reader(f); next(reader)
    rows = [r for r in reader]
file_genes = [r[0].replace('_FFPE', '').replace('_Frozen', '') for r in rows]
all_coefs = np.array([[float(v) for v in r[1:]] for r in rows])  # genes × 500
with open(INTCS) as f:
    reader = csv.reader(f); next(reader)
    intercepts = np.array([float(r[0]) for r in reader])

assert set(file_genes) == set(GENE_NAMES), "Gene set mismatch"
reorder = np.array([file_genes.index(g) for g in GENE_NAMES])
coefs_ordered = all_coefs[reorder]
beta_bar = coefs_ordered.mean(axis=1)
intc_bar = float(intercepts.mean())

print(f"Loaded deployed 500-bag coefficients:")
print(f"  ||β̄||_2 = {np.linalg.norm(beta_bar):.4f}")
print(f"  median ||β_i|| = {float(np.median([np.linalg.norm(coefs_ordered[:, i]) for i in range(coefs_ordered.shape[1])])):.4f}")
print(f"  intercept mean = {intc_bar:.4f}")

np.savez(os.path.join(DATA_DIR, 'deployed_ensemble_true.npz'),
         beta=beta_bar, intercept=intc_bar,
         coefs_full=coefs_ordered, intercepts_full=intercepts,
         gene_names=np.array(GENE_NAMES))
print(f"Wrote {os.path.join(DATA_DIR, 'deployed_ensemble_true.npz')}")
