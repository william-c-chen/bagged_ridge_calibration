"""Additional deployment-scenario simulations to strengthen the clinical argument.

Three framings:
  (1) Stressor sensitivity: scale observed (|Δslope|, |Δintercept|) by k = 1,
      1.5, 2, 3 to simulate "worse than observed" cross-lot drift and compare
      architecture resilience.
  (2) Per-1000-patients translation for clinical readers.
  (3) Analyst-range: P10 to P90 range of misclassification, per architecture.

Uses cross-experiment (|Δslope|, |Δintercept|) from the 100-seed scoring,
applied via closed-form threshold-crossing probability on the N=872 validation
cohort (ChenRobust scores, all non-UCSF sites).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import norm
import openpyxl
from common import DATA_DIR

T_LOW, T_HIGH = 0.376, 0.565

# Validation scores (site != "UCSF", column 24 ChenRobust)
wb = openpyxl.load_workbook(
    os.path.join(os.path.dirname(DATA_DIR), '..', 'dat.integ.master.5.6.2025.xlsx'),
    data_only=True, read_only=True)
scores = []
for r in wb['Sheet1'].iter_rows(values_only=True):
    if r[0] is None or r[0] == 'ID': continue
    site = str(r[2]) if r[2] is not None else 'None'
    if site == 'UCSF' or r[24] is None: continue
    try: scores.append(float(r[24]))
    except: pass
scores = np.array(scores)
print(f"Validation N = {len(scores)}")

with open(os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json')) as f:
    bag = json.load(f)
with open(os.path.join(DATA_DIR, 'single_models_multiseed.json')) as f:
    sng = json.load(f)

models = {
    'Bagged ridge':      bag['per_seed'],
    'Single ridge':      sng['single_ridge'],
    'Single LASSO':      sng['single_lasso'],
    'Single elastic net': sng['single_elasticnet'],
    'Single OLS':        sng['single_ols'],
}


def expected_misclass(dslope, dintc):
    """Closed-form expected misclass rate: ε has magnitude dslope*y + dintc,
    random sign, patient population as given."""
    sigma = dslope * scores + dintc  # per-patient scale
    d_up = np.where(scores < T_LOW, T_LOW - scores,
            np.where(scores < T_HIGH, T_HIGH - scores, np.inf))
    d_dn = np.where(scores >= T_HIGH, scores - T_HIGH,
            np.where(scores >= T_LOW, scores - T_LOW, np.inf))
    # With uniform-sign fixed-magnitude ε:
    # P(cross up)   = 0.5 * I(|ε| > d_up)
    # P(cross down) = 0.5 * I(|ε| > d_dn)
    p = 0.5 * ((sigma > d_up).astype(float) + (sigma > d_dn).astype(float))
    return float(np.mean(p))


# (1) Stressor sensitivity: misclass vs k * (observed drift), k in {1,1.5,2,3}
print("\n=== (1) Stressor sensitivity: misclass % at k × observed cross-experiment drift ===")
print(f"{'Model':<22}{'k=1':>8}{'k=1.5':>8}{'k=2':>8}{'k=3':>8}")
stress = {}
for name, runs in models.items():
    row = {}
    for k in [1.0, 1.5, 2.0, 3.0]:
        rates = [expected_misclass(k*r['dslope'], k*r['dintc']) for r in runs]
        row[k] = float(np.median(rates))
    stress[name] = row
    print(f"{name:<22}" + ''.join(f"{row[k]*100:>7.1f}%" for k in [1.0, 1.5, 2.0, 3.0]))

print(f"\nResilience (misclass at k=3 / misclass at k=1) — lower = more resilient:")
for name, row in stress.items():
    if row[1.0] > 0:
        print(f"  {name:<22}{row[3.0]/row[1.0]:>6.2f}x")

# (2) Per-1000-patients
print(f"\n=== (2) Per-1000-patients misclassification count at observed drift ===")
print(f"{'Model':<22}{'median':>10}{'P90':>10}{'P99':>10}")
per_1000 = {}
for name, runs in models.items():
    rates = [expected_misclass(r['dslope'], r['dintc']) for r in runs]
    med = np.median(rates) * 1000
    p90 = np.percentile(rates, 90) * 1000
    p99 = np.percentile(rates, 99) * 1000
    per_1000[name] = {'median': float(med), 'p90': float(p90), 'p99': float(p99)}
    print(f"{name:<22}{med:>8.0f}  {p90:>8.0f}  {p99:>8.0f}")

print(f"\nExcess misclassifications per 1000 vs bagged (median):")
bag_med = per_1000['Bagged ridge']['median']
for name, v in per_1000.items():
    print(f"  {name:<22}{v['median'] - bag_med:>+6.0f} per 1000")

# (3) Analyst-range: P10 to P90 per architecture
print(f"\n=== (3) Analyst range (P10 — P90 misclass %) ===")
print(f"{'Model':<22}{'P10':>8}{'P50':>8}{'P90':>8}{'range':>10}")
analyst = {}
for name, runs in models.items():
    rates = [expected_misclass(r['dslope'], r['dintc']) for r in runs]
    p10 = np.percentile(rates, 10); p50 = np.percentile(rates, 50); p90 = np.percentile(rates, 90)
    analyst[name] = {'p10': float(p10), 'p50': float(p50), 'p90': float(p90),
                     'range_pp': float((p90 - p10) * 100)}
    print(f"{name:<22}{p10*100:>7.2f}%{p50*100:>7.2f}%{p90*100:>7.2f}%{(p90-p10)*100:>9.1f}pp")

out = {'stressor': {k: v for k, v in stress.items()},
       'per_1000': per_1000, 'analyst_range': analyst}
with open(os.path.join(DATA_DIR, 'deployment_ammo.json'), 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved {os.path.join(DATA_DIR, 'deployment_ammo.json')}")
