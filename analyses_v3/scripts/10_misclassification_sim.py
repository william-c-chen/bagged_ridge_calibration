"""Translate cross-experiment calibration variability into clinical misclassification.

For each architecture and each of its 100 seeds, we have the magnitude of
correction variability (dslope, dintc) estimated across the 2023/2026
cross-experiment comparison. We treat these as the scale of correction-error
a lab would face under procedural-seed variation, apply the error to a
realistic patient score distribution, and count threshold crossings at the
clinical decision boundaries (0.376 and 0.565).

Outputs:
  - Median and P90 misclassification rate per architecture
  - Directional decomposition (upward vs downward crossings)

The misclassification rate is an upper bound because we treat error as always
pushing toward the nearest threshold; real errors are half-and-half, so actual
clinical misclassification is ~50% of reported values. Ranking across
architectures is preserved.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import openpyxl
from common import DATA_DIR

T_LOW, T_HIGH = 0.376, 0.565

# Validation-cohort patient score distribution (exclude site == "UCSF" training cohort).
wb = openpyxl.load_workbook(
    os.path.join(os.path.dirname(DATA_DIR), '..', 'dat.integ.master.5.6.2025.xlsx'),
    data_only=True, read_only=True)
ws = wb['Sheet1']
scores = []
for r in ws.iter_rows(values_only=True):
    if r[0] is None or r[0] == 'ID': continue
    site = str(r[2]) if r[2] is not None else 'None'
    if site == 'UCSF': continue
    if r[24] is not None:  # ChenRobust = deployed risk score
        try: scores.append(float(r[24]))
        except: pass
scores = np.array(scores)
print(f"Validation patient population: N={len(scores)}, "
      f"Low={np.mean(scores<T_LOW)*100:.1f}%, "
      f"Int={np.mean((scores>=T_LOW)&(scores<T_HIGH))*100:.1f}%, "
      f"High={np.mean(scores>=T_HIGH)*100:.1f}%")

# Multi-seed data
with open(os.path.join(DATA_DIR, 'glmnet_ffpe_multiseed_scoring.json')) as f:
    bag = json.load(f)
with open(os.path.join(DATA_DIR, 'single_models_multiseed.json')) as f:
    sng = json.load(f)

models = {
    'Bagged ridge':      [r for r in bag['per_seed']],
    'Single ridge':      sng['single_ridge'],
    'Single LASSO':      sng['single_lasso'],
    'Single elastic net': sng['single_elasticnet'],
    'Single OLS':        sng['single_ols'],
}


def group(y):
    return 0 if y < T_LOW else (1 if y < T_HIGH else 2)


def misclass_rate(dslope, dintc):
    """Expected fraction of patients misclassified given random error sign.

    Error at score y: ε(y) = ±dslope*y ± dintc with random sign per seed.
    Corrected score = y + ε. Misclassification = corrected in different
    risk group than y.

    Expected rate = 0.5 × [P(upward crossing) + P(downward crossing)].
    """
    err_mag = dslope * scores + dintc
    # Distance to next threshold above (if any)
    d_up = np.where(scores < T_LOW, T_LOW - scores,
            np.where(scores < T_HIGH, T_HIGH - scores, np.inf))
    d_dn = np.where(scores >= T_HIGH, scores - T_HIGH,
            np.where(scores >= T_LOW, scores - T_LOW, np.inf))
    p_up = (err_mag > d_up).astype(float)
    p_dn = (err_mag > d_dn).astype(float)
    return float(0.5 * np.mean(p_up + p_dn))


def directional_misclass(dslope, dintc):
    """Decompose into expected up-crossings and down-crossings (each *0.5)."""
    err_mag = dslope * scores + dintc
    d_up = np.where(scores < T_LOW, T_LOW - scores,
            np.where(scores < T_HIGH, T_HIGH - scores, np.inf))
    d_dn = np.where(scores >= T_HIGH, scores - T_HIGH,
            np.where(scores >= T_LOW, scores - T_LOW, np.inf))
    up = 0.5 * np.mean(err_mag > d_up)
    dn = 0.5 * np.mean(err_mag > d_dn)
    return float(up), float(dn)


print(f"\n{'Model':<22}{'median':>10}{'P90':>10}{'P99':>10}{'worst':>10}")
print('-' * 62)
results = {}
for name, runs in models.items():
    rates = [misclass_rate(r['dslope'], r['dintc']) for r in runs]
    median_r = float(np.median(rates))
    p90 = float(np.percentile(rates, 90))
    p99 = float(np.percentile(rates, 99))
    worst = float(max(rates))
    print(f"{name:<22}{median_r*100:>9.2f}%{p90*100:>9.2f}%{p99*100:>9.2f}%{worst*100:>9.2f}%")
    results[name] = {'median': median_r, 'p90': p90, 'p99': p99, 'worst': worst, 'rates': rates}

# Directional decomposition at median seed
print(f"\nDirectional misclassification at median seed:")
print(f"{'Model':<22}{'up-cross':>12}{'down-cross':>14}")
for name, runs in models.items():
    # sort by rate to pick median-seed
    rates = [misclass_rate(r['dslope'], r['dintc']) for r in runs]
    med_idx = int(np.argsort(rates)[len(rates)//2])
    up, dn = directional_misclass(runs[med_idx]['dslope'], runs[med_idx]['dintc'])
    print(f"{name:<22}{up*100:>11.2f}%{dn*100:>13.2f}%")

# Ratio of P90 to median — the "unlucky analyst" premium
print(f"\nAnalyst-unluck ratio (P90/median) — lower is safer for locked deployment:")
for name in models:
    r = results[name]
    ratio = r['p90'] / r['median'] if r['median'] > 0 else float('inf')
    print(f"  {name:<22}{ratio:>8.2f}x")

with open(os.path.join(DATA_DIR, 'misclassification_sim.json'), 'w') as f:
    out = {k: {kk: vv for kk, vv in v.items() if kk != 'rates'}
           for k, v in results.items()}
    json.dump(out, f, indent=2)
print(f"\nSaved {os.path.join(DATA_DIR, 'misclassification_sim.json')}")
