"""Per-gene lot-drift analysis across all 9 lot pairs.

For each gene g and each lot pair (l1, l2), compute
    δ_g(s) = x_g(s, l2) - x_g(s, l1)
then summarize per gene by averaging |mean δ_g| across the 9 lot pairs. Also
computes aggregate score impact (|β̄_g| × avg|mean δ_g|) and concentration
statistics.

Supports the §3.4 empirical claim that ridge's distributed weights offload
score sensitivity from the probes that actually drift between lots.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from itertools import combinations
from common import DATA_DIR, GENE_NAMES

br = np.load(os.path.join(DATA_DIR, 'bridging_normalized.npz'), allow_pickle=True)
keys_26 = [tuple(k) for k in br['keys_2026']]
keys_23 = [tuple(k) for k in br['keys_2023']]
Xn_26, Xn_23 = br['X_norm_2026'], br['X_norm_2023']

dep = np.load(os.path.join(DATA_DIR, 'deployed_ensemble_true.npz'), allow_pickle=True)
beta = dep['beta']


def build_table(keys, Xn):
    samples = sorted(set(s for s, _ in keys))
    lots = sorted(set(l for _, l in keys))
    tbl = {(s, l): x for (s, l), x in zip(keys, Xn)}
    return samples, lots, tbl


samples_26, lots_26, tbl_26 = build_table(keys_26, Xn_26)
samples_23, lots_23, tbl_23 = build_table(keys_23, Xn_23)

pair_stats = []
for label, samples, lots, tbl in [('2026', samples_26, lots_26, tbl_26),
                                   ('2023', samples_23, lots_23, tbl_23)]:
    for l1, l2 in combinations(lots, 2):
        valid = [s for s in samples if (s, l1) in tbl and (s, l2) in tbl]
        if len(valid) < 3: continue
        D = np.array([tbl[(s, l2)] - tbl[(s, l1)] for s in valid])
        pair_stats.append({'exp': label, 'l1': l1, 'l2': l2, 'N': len(valid),
                           'mean_delta': D.mean(axis=0), 'sd_delta': D.std(axis=0, ddof=1)})

mean_D = np.array([p['mean_delta'] for p in pair_stats])
per_gene_syst = np.mean(np.abs(mean_D), axis=0)

order = np.argsort(-per_gene_syst)
print(f"Per-gene average |mean δ| across {len(pair_stats)} lot pairs:")
print(f"{'Rank':<5}{'Gene':<12}{'|β̄|':>8}{'avg|mean δ|':>14}{'max|mean δ|':>14}")
for rk, gi in enumerate(order):
    print(f"{rk+1:<5}{GENE_NAMES[gi]:<12}{abs(beta[gi]):>8.4f}{per_gene_syst[gi]:>14.3f}{np.max(np.abs(mean_D[:,gi])):>14.3f}")

r_drift_beta = float(np.corrcoef(per_gene_syst, np.abs(beta))[0, 1])
impact = np.abs(beta) * per_gene_syst
imp_order = np.argsort(-impact)
total_impact = float(impact.sum())
top1 = float(impact[imp_order[0]]) / total_impact
top5 = float(impact[imp_order[:5]].sum()) / total_impact

print(f"\nPer-gene avg|mean δ| range: {per_gene_syst.min():.3f} to {per_gene_syst.max():.3f} "
      f"(ratio {per_gene_syst.max()/per_gene_syst.min():.2f}×)")
print(f"Correlation between per-gene drift magnitude and |β̄_g|: r = {r_drift_beta:+.3f}")
print(f"Top 1 gene score-impact share: {top1*100:.1f}% (uniform 2.9%)")
print(f"Top 5 gene score-impact share: {top5*100:.1f}% (uniform 14.7%)")

out = {
    'n_pairs': len(pair_stats),
    'per_gene': [
        {'gene': GENE_NAMES[i], 'avg_mean_delta': float(per_gene_syst[i]),
         'beta_abs': float(abs(beta[i])),
         'max_mean_delta': float(np.max(np.abs(mean_D[:, i]))),
         'impact': float(impact[i])}
        for i in range(34)
    ],
    'r_drift_vs_beta': r_drift_beta,
    'top1_impact_share': top1, 'top5_impact_share': top5,
    'range_min': float(per_gene_syst.min()), 'range_max': float(per_gene_syst.max()),
    'range_ratio': float(per_gene_syst.max() / per_gene_syst.min()),
}
with open(os.path.join(DATA_DIR, 'per_gene_lot_drift.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {os.path.join(DATA_DIR, 'per_gene_lot_drift.json')}")
