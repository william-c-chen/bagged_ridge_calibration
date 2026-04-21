"""Parse all relevant RCC files for 2023 and 2026 bridging-sample analyses.

Produces:
  analyses/data/bridging_counts.json  -- {experiment, sample, lot} -> {gene: raw_count, ...}
  analyses/data/bridging_normalized.npz -- normalized gene expression matrices

2023 experiment (Previous_calibration_run/first_batch):
  - 6 bridging samples (MEN-322, MEN-315, MEN-268, MEN-303, MEN-308, MEN-256)
  - 3 lots (C10904 = A, C9543 = B, C10132 = C)
  - Sample-to-lane mapping comes from 'NanoString RCC index-Chen 20231007.xlsx'

2026 experiment (RCC_files/):
  - 10 bridging samples × 4 lots × 12 lanes/cartridge
  - Sample-to-lane from Samples_for_calibration.xlsx Sheet2
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
import openpyxl
import numpy as np
from common import (REPO_ROOT, DATA_DIR, RCC_2023_FIRST, RCC_2023_FIRST_IDX,
                    SAMPLES_CALIB, GENE_NAMES, HK_NAMES, LOTS_2023, LOTS_2026,
                    BRIDGING_SAMPLES_2023_ALL, BRIDGING_SAMPLES_2026_ALL,
                    parse_rcc_file, hk_normalize, ensure_dirs)

ensure_dirs()

# ============== 2023 bridging RCCs ==============
# Build cartridge→lot lookup and sample-by-lane from the index file.
wb = openpyxl.load_workbook(RCC_2023_FIRST_IDX, read_only=True, data_only=True)
ws = wb['Sheet1']
rows = [r for r in ws.iter_rows(values_only=True) if any(v is not None for v in r)]
data_rows = [r for r in rows[5:] if isinstance(r[0], int)]
# Each row: (sample#, sample_id, cartridge_id, rcc_file_name)
# File-name contains suffix "_Raleigh75_NN" or "_C9543_NN" or "_C10132_NN";
# the prefix indicates the CARTRIDGE TYPE (== codeset)
# But cartridge 210512101124 carries BOTH C9543 and C10132 lanes (see index).
# So identifying the lot needs file-name parsing not cartridge ID.
lot_2023 = {}   # (sample_id, lot_letter) -> rcc path
for r in data_rows:
    sample_num, sample_id, cart_id, rcc_fname = r[0], r[1], r[2], r[3]
    if sample_id not in BRIDGING_SAMPLES_2023_ALL:
        continue
    # Determine lot from file name
    if '_C9543_' in rcc_fname:
        lot = 'B'
    elif '_C10132_' in rcc_fname:
        lot = 'C'
    elif '_Raleigh75_' in rcc_fname:
        # Raleigh75 cartridges hold the 1-192 runs on lot A (C10904)
        lot = 'A'
    else:
        continue
    # Locate the actual RCC file on disk (search recursively — some cartridges
    # have RCCs nested in a second-level directory inside the unzipped folder)
    cart_str = str(cart_id)
    candidates = glob.glob(os.path.join(RCC_2023_FIRST, f'unzipped_*{cart_str}_RCC',
                                         '**', rcc_fname), recursive=True)
    # Filter out __MACOSX resource-fork paths
    candidates = [c for c in candidates if '__MACOSX' not in c]
    if not candidates:
        print(f"  WARN: RCC file not found: {rcc_fname}")
        continue
    lot_2023[(sample_id, lot)] = candidates[0]

print(f"2023 bridging: {len(lot_2023)} (sample, lot) pairs found")
for (s, L), p in sorted(lot_2023.items()):
    print(f"  {s} | lot {L} | {os.path.basename(p)}")

# Parse and store counts
counts_2023 = {}
for (s, L), path in lot_2023.items():
    rcc = parse_rcc_file(path)
    counts_2023[(s, L)] = rcc['counts']

# ============== 2026 bridging RCCs ==============
# 2026 cartridge IDs (from Samples_for_calibration Sheet2):
#   211273141227 -> C10904X1 (lot A)
#   211272731227 -> C9543    (lot B)
#   211272721227 -> C11199   (lot D)
#   211272871227 -> C11695X1 (lot E)
cart_to_lot = {
    '211273141227': 'A',
    '211272731227': 'B',
    '211272721227': 'D',
    '211272871227': 'E',
}
# Lane → sample for 2026 (all 4 lots share the same 12-lane layout)
lane_to_sample_2026 = {
    1:'AM42', 2:'AM57', 3:'AM32', 4:'AM35', 5:'AM53',
    6:'QM313', 7:'QM318', 8:'QM322', 9:'QM325', 10:'QM326',
    11:'AM42_repeat_40ngul', 12:'50fM_ssDNA',
}

RCC_2026_DIR = os.path.join(REPO_ROOT, 'RCC_files')
cart_2026_dirs = glob.glob(os.path.join(RCC_2026_DIR, '20260414_*_RCC'))
cart_2026_dirs = [d for d in cart_2026_dirs if os.path.isdir(d)]

counts_2026 = {}
for d in cart_2026_dirs:
    cart_id = os.path.basename(d).split('_')[1]
    if cart_id not in cart_to_lot:
        continue
    lot = cart_to_lot[cart_id]
    for f in sorted(os.listdir(d)):
        if not f.endswith('.RCC'): continue
        # Lane number is the last "_NN.RCC" in the filename
        lane_match = f.rsplit('_', 1)[-1].replace('.RCC', '')
        try:
            lane = int(lane_match)
        except ValueError:
            continue
        sample = lane_to_sample_2026.get(lane)
        if sample is None or sample not in BRIDGING_SAMPLES_2026_ALL:
            continue
        rcc = parse_rcc_file(os.path.join(d, f))
        counts_2026[(sample, lot)] = rcc['counts']
print(f"\n2026 bridging: {len(counts_2026)} (sample, lot) pairs found")

# ============== Normalize and save ==============
def build_norm_matrix(counts_dict, gene_names, hk_names):
    keys = sorted(counts_dict.keys())
    X_raw = np.array([[counts_dict[k].get(g, 0) for g in gene_names] for k in keys], dtype=float)
    HK    = np.array([[counts_dict[k].get(h, 0) for h in hk_names] for k in keys], dtype=float)
    X_norm = hk_normalize(X_raw, HK)
    return keys, X_raw, HK, X_norm

keys_23, Xraw_23, HK_23, Xnorm_23 = build_norm_matrix(counts_2023, GENE_NAMES, HK_NAMES)
keys_26, Xraw_26, HK_26, Xnorm_26 = build_norm_matrix(counts_2026, GENE_NAMES, HK_NAMES)

print(f"\n2023 matrix: {Xnorm_23.shape} (sample-lot pairs × genes)")
print(f"2026 matrix: {Xnorm_26.shape}")

np.savez(os.path.join(DATA_DIR, 'bridging_normalized.npz'),
         keys_2023 = np.array([[s, l] for s,l in keys_23]),
         X_raw_2023=Xraw_23, HK_2023=HK_23, X_norm_2023=Xnorm_23,
         keys_2026 = np.array([[s, l] for s,l in keys_26]),
         X_raw_2026=Xraw_26, HK_2026=HK_26, X_norm_2026=Xnorm_26,
         gene_names=np.array(GENE_NAMES), hk_names=np.array(HK_NAMES))

# Also save raw counts as JSON for transparency/auditing
with open(os.path.join(DATA_DIR, 'bridging_counts.json'), 'w') as f:
    json.dump({
        '2023': {f'{s}|{l}': c for (s,l), c in counts_2023.items()},
        '2026': {f'{s}|{l}': c for (s,l), c in counts_2026.items()},
    }, f, indent=2)

print(f"\nWrote {os.path.join(DATA_DIR, 'bridging_normalized.npz')}")
print(f"Wrote {os.path.join(DATA_DIR, 'bridging_counts.json')}")
