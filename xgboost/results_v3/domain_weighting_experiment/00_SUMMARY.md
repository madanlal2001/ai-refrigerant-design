# Domain-Weighted Tc Estimated-Data Ablation — Summary (Frozen Protocol v3)

Read-only-in-spirit controlled experiment: frozen split, frozen features, frozen predicted-Tb methodology all reused unchanged. Only Tc estimated-data sample weights differ across D0-D4. No file under artifacts/ or the existing results_v3/ (outside this new directory) was modified.

**Reproduction gate:** D0 vs saved XGB-C max|diff|=0.000030 K (tolerance 0.01 K) — PASSED. D1 vs saved XGB-D max|diff|=0.000030 K — PASSED.

**Selection discipline:** D2/D3/D4 winners chosen by global validation RMSE only (refrigerant-like validation subgroup: 16 MEDIUM, 0 HIGH of 209 — reported, not used for selection).

## Winning configurations

- D2: 1.0 (global val RMSE=30.7495 K)
- D3: 2.5 (global val RMSE=31.2571 K)
- D4: 2.0 (global val RMSE=33.0609 K)

## Global Tc test metrics (n=209)

variant   n     R2    RMSE     MAE
     D0 209 0.9457 26.2513 17.0181
     D1 209 0.9505 25.0769 16.2905
     D2 209 0.9486 25.5555 16.2346
     D3 209 0.9461 26.1581 16.2866
     D4 209 0.9436 26.7685 17.5589

## Refrigerant-likeness bin sizes on the frozen test set (same bins as Script 05)

- LOW: n=202
- MEDIUM: n=6
- HIGH: n=1
- MEDIUM+HIGH: n=7

## RMSE by variant x refrigerant-likeness bin

refrig_likeness_bin     LOW  MEDIUM    HIGH  MEDIUM+HIGH     ALL
variant                                                         
D0                   26.629  10.593  10.620       10.597  26.251
D1                   25.373  15.183   1.689       14.071  25.077
D2                   25.925  10.988   1.732       10.194  25.556
D3                   26.522  12.286   2.534       11.415  26.158
D4                   27.139  12.776   0.786       11.832  26.768

## The 7 refrigerant-like frozen-test molecules — improved/worsened vs D0 and D1 (descriptive counts only, no significance claims)

- D2 vs D0: {'improved': 4, 'worsened': 3}  |  D2 vs D1: {'worsened': 5, 'improved': 2}
- D3 vs D0: {'improved': 4, 'worsened': 3}  |  D3 vs D1: {'worsened': 4, 'improved': 3}
- D4 vs D0: {'worsened': 4, 'improved': 3}  |  D4 vs D1: {'improved': 5, 'worsened': 2}

Full molecule-level table: `08_seven_refrigerant_like_molecules.csv`

_See `plots/weighting_functions.png` and `plots/seven_molecule_errors.png`._