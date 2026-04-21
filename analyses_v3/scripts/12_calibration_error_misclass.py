"""Misclassification rate due to calibration-error residuals.

At deployment, a lab fits a linear correction on K bridging samples and applies
it to every patient. The correction carries residual error with scale σ, which
we take as the LOO-CV RMSE at that K (i.e. how well the correction generalizes
to a held-out sample, measured empirically in the K sweep).

For each patient with true score y, the corrected score = y + ε with ε ~ N(0, σ).
Closed-form misclassification probability per patient:

  Low group (y < T_LOW):     P(y+ε ≥ T_LOW)  = 1 - Φ((T_LOW - y)/σ)
  Int group (T_LOW ≤ y < T_HIGH):
     P(y+ε ≥ T_HIGH) + P(y+ε < T_LOW)
   = (1 - Φ((T_HIGH - y)/σ)) + Φ((T_LOW - y)/σ)
  High group (y ≥ T_HIGH):   P(y+ε < T_HIGH) = Φ((T_HIGH - y)/σ)

Averaged across the 499-patient validation-cohort distribution, this gives the
expected misclassification rate per architecture per K.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import norm
import openpyxl
from common import DATA_DIR

T_LOW, T_HIGH = 0.376, 0.565

# Patient score distribution from master DB.
#   column 24 = ChenRobust (deployed risk score used in practice)
#   column 2  = Site
# Validation cohort only: exclude site == "UCSF" (training).
wb = openpyxl.load_workbook(
    os.path.join(os.path.dirname(DATA_DIR), '..', 'dat.integ.master.5.6.2025.xlsx'),
    data_only=True, read_only=True)
scores = []
sites = []
for r in wb['Sheet1'].iter_rows(values_only=True):
    if r[0] is None or r[0] == 'ID': continue
    site = str(r[2]) if r[2] is not None else 'None'
    if site == 'UCSF': continue  # exclude training cohort
    if r[24] is None: continue
    try:
        scores.append(float(r[24])); sites.append(site)
    except: pass
scores = np.array(scores)
from collections import Counter
print(f"Validation patient population: N={len(scores)} (sites: {dict(Counter(sites))})")
print(f"  Low={np.mean(scores<T_LOW)*100:.1f}%, "
      f"Int={np.mean((scores>=T_LOW)&(scores<T_HIGH))*100:.1f}%, "
      f"High={np.mean(scores>=T_HIGH)*100:.1f}%")


def misclass_rate(sigma):
    """Expected misclassification per patient, averaged across population."""
    if sigma <= 0: return 0.0
    p = np.zeros_like(scores)
    low = scores < T_LOW
    int_ = (scores >= T_LOW) & (scores < T_HIGH)
    high = scores >= T_HIGH
    p[low]  = 1 - norm.cdf((T_LOW  - scores[low])  / sigma)
    p[int_] = (1 - norm.cdf((T_HIGH - scores[int_]) / sigma)) + \
              norm.cdf((T_LOW  - scores[int_]) / sigma)
    p[high] = norm.cdf((T_HIGH - scores[high]) / sigma)
    return float(np.mean(p))


# Read K-sweep RMSEs as σ estimates at each K
with open(os.path.join(DATA_DIR, 'k_sweep.json')) as f:
    ksw = json.load(f)

models = ['bagged', 'ridge', 'lasso', 'en', 'ols']
print(f"\nExpected misclassification % due to calibration-error residual at each K,"
      f"\nusing LOO-CV RMSE as σ (median across 100 seeds for single-model).\n")
print(f"{'K':<4}" + ''.join(f'{m:>10}' for m in models))
results = {}
for Kstr, row in ksw['median_by_K'].items():
    K = int(Kstr)
    rates = {m: misclass_rate(row[m]) for m in models}
    results[K] = {'sigma': row, 'misclass': rates}
    print(f"{K:<4}" + ''.join(f"{rates[m]*100:>9.2f}%" for m in models))

# Also report under the P90 single ridge (unlucky-analyst scenario)
print(f"\nUnlucky-analyst scenario (single ridge at its P90 RMSE across seeds):")
print(f"{'K':<4}{'bagged':>10}{'ridge_P90':>12}")
for Kstr, p90 in ksw['ridge_p90_by_K'].items():
    K = int(Kstr)
    bm = misclass_rate(ksw['median_by_K'][Kstr]['bagged'])
    rm_p90 = misclass_rate(p90)
    print(f"{K:<4}{bm*100:>9.2f}%{rm_p90*100:>11.2f}%")

# Absolute vs relative scale
print(f"\nClinical scale: decision-threshold gap = {T_HIGH - T_LOW:.3f}.")
print(f"At K=6, calibration σ ranges from {ksw['median_by_K']['6']['bagged']:.4f} (bagged) "
      f"to {ksw['median_by_K']['6']['ols']:.4f} (OLS), or "
      f"{ksw['median_by_K']['6']['bagged']/(T_HIGH-T_LOW)*100:.1f}%--"
      f"{ksw['median_by_K']['6']['ols']/(T_HIGH-T_LOW)*100:.1f}% of threshold gap.")

with open(os.path.join(DATA_DIR, 'calibration_error_misclass.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {os.path.join(DATA_DIR, 'calibration_error_misclass.json')}")
