# XGB A/B/C/D Ablation — Analysis Summary (Frozen Protocol v3)

**Single frozen split.** No error bars or statistical-significance claims. All deltas are *observed on this frozen test split* only.

**Residual sign:** residual = actual - predicted (positive => under-prediction).

**Ablation improvement convention:** delta = metric(baseline) - metric(variant); positive => variant reduces error.

**Negligibility rule:** for each contrast+target, resolution = 0.5 x median(pooled |error| of the two variants). |dRMSE| and |dMAE| both below resolution => NEGLIGIBLE on this split.

## Ablation effects
- **XGB-A->XGB-B** (estimated supervision), Tc (K): dRMSE=+3.1842 (+10.21%), dMAE=+1.3485 (+6.8%); resolution=5.5712 -> **NEGLIGIBLE**
- **XGB-A->XGB-B** (estimated supervision), Pc (bar): dRMSE=-0.0925 (-2.59%), dMAE=+0.0001 (+0.01%); resolution=0.4473 -> **NEGLIGIBLE**
- **XGB-A->XGB-B** (estimated supervision), omega (-): dRMSE=-0.0031 (-3.68%), dMAE=+0.0035 (+7.08%); resolution=0.0084 -> **NEGLIGIBLE**
- **XGB-A->XGB-C** (predicted Tb), Tc (K): dRMSE=+4.9209 (+15.79%), dMAE=+2.8201 (+14.22%); resolution=5.494 -> **NEGLIGIBLE**
- **XGB-A->XGB-C** (predicted Tb), Pc (bar): dRMSE=-0.0508 (-1.42%), dMAE=-0.0123 (-0.65%); resolution=0.4455 -> **NEGLIGIBLE**
- **XGB-A->XGB-C** (predicted Tb), omega (-): dRMSE=+0.0015 (+1.84%), dMAE=+0.0009 (+1.9%); resolution=0.01 -> **NEGLIGIBLE**
- **XGB-C->XGB-D** (estimated | Tb present), Tc (K): dRMSE=+1.1744 (+4.47%), dMAE=+0.7276 (+4.28%); resolution=5.4748 -> **NEGLIGIBLE**
- **XGB-C->XGB-D** (estimated | Tb present), Pc (bar): dRMSE=+0.0591 (+1.63%), dMAE=+0.0297 (+1.57%); resolution=0.444 -> **NEGLIGIBLE**
- **XGB-C->XGB-D** (estimated | Tb present), omega (-): dRMSE=-0.0033 (-4.01%), dMAE=+0.0026 (+5.34%); resolution=0.0081 -> **NEGLIGIBLE**
- **XGB-B->XGB-D** (Tb | estimated present), Tc (K): dRMSE=+2.9111 (+10.4%), dMAE=+2.1993 (+11.89%); resolution=5.5343 -> **NEGLIGIBLE**
- **XGB-B->XGB-D** (Tb | estimated present), Pc (bar): dRMSE=+0.1007 (+2.75%), dMAE=+0.0173 (+0.92%); resolution=0.4451 -> **NEGLIGIBLE**
- **XGB-B->XGB-D** (Tb | estimated present), omega (-): dRMSE=+0.0013 (+1.53%), dMAE=+0.0 (+0.06%); resolution=0.0073 -> **NEGLIGIBLE**

## Target-specific selection (RMSE primary, negligibility-aware)
- **Tc (K)**: XGB-D lowest RMSE; ['XGB-D', 'XGB-C', 'XGB-B'] effectively similar on this split (resolution=5.5045).
- **Pc (bar)**: XGB-D lowest RMSE; ['XGB-D', 'XGB-A', 'XGB-C', 'XGB-B'] effectively similar on this split (resolution=0.4455).
- **omega (-)**: XGB-C lowest RMSE; ['XGB-C', 'XGB-A', 'XGB-D', 'XGB-B'] effectively similar on this split (resolution=0.0084).

_Interpretation restricted to what a single frozen split supports. 'Effectively similar' variants should not be ranked as winners._