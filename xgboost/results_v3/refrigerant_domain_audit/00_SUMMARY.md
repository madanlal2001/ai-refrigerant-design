# Refrigerant-Domain Audit — Summary (Frozen Protocol v3)

Read-only diagnostic stage. No retraining, resplitting, retuning, or prediction modification. Fingerprints and descriptors recomputed from saved canonical SMILES (same Morgan r2/2048 protocol as Scripts 01/03/04) purely for similarity diagnostics.

**Single frozen split.** Descriptive only — no causal or significance claims.

## Part A — the 142 refrigerants vs the Tc-labeled experimental train pool (n=1668)
- LOW similarity (<0.4): 104 / 142
- MEDIUM similarity (0.4-0.6): 33 / 142
- HIGH similarity (>=0.6): 5 / 142
- Recurring refrigerant self-clusters (>=3 members, Butina cutoff 0.6): 16 clusters covering 77/142 refrigerants — see `03_refrigerant_self_clusters.csv`.

Largest recurring refrigerant families (by member count):
- cluster 0: n=16, mean sim to Tc-train=0.3377, LOW/MED/HIGH=13/3/0, mean halogen count=7.06
- cluster 1: n=10, mean sim to Tc-train=0.296, LOW/MED/HIGH=9/1/0, mean halogen count=3.3
- cluster 2: n=6, mean sim to Tc-train=0.2616, LOW/MED/HIGH=6/0/0, mean halogen count=7.0
- cluster 8: n=4, mean sim to Tc-train=0.2555, LOW/MED/HIGH=4/0/0, mean halogen count=6.0
- cluster 7: n=4, mean sim to Tc-train=0.286, LOW/MED/HIGH=4/0/0, mean halogen count=4.25
- cluster 3: n=4, mean sim to Tc-train=0.3296, LOW/MED/HIGH=4/0/0, mean halogen count=7.5

## Descriptor contrasts (142 refrigerants vs Tc-train pool, |standardized mean diff| >= 0.5)

- **halogen_fraction**: refrigerants mean=0.5047 vs train mean=0.1106 (standardized diff=+1.9276) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **F_to_C_ratio**: refrigerants mean=1.5264 vs train mean=0.1594 (standardized diff=+1.8413) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **MolMR**: refrigerants mean=16.7125 vs train mean=42.6508 (standardized diff=-1.6532) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **F_count**: refrigerants mean=4.2887 vs train mean=0.6181 (standardized diff=+1.5211) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **C_count**: refrigerants mean=2.838 vs train mean=7.9766 (standardized diff=-1.4439) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **halogen_count**: refrigerants mean=4.3451 vs train mean=0.9526 (standardized diff=+1.3744) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **num_rotatable_bonds**: refrigerants mean=0.507 vs train mean=3.7206 (standardized diff=-0.9678) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **fraction_Csp3**: refrigerants mean=0.5037 vs train mean=0.7527 (standardized diff=-0.7165) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **num_aromatic_rings**: refrigerants mean=0.0 vs train mean=0.2344 (standardized diff=-0.5962) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **TPSA**: refrigerants mean=5.882 vs train mean=14.1622 (standardized diff=-0.5551) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`
- **logP**: refrigerants mean=1.9415 vs train mean=2.7103 (standardized diff=-0.515) — full table in `02_refrigerant_vs_train_descriptor_comparison.csv`

## Part B — Tc TEST molecules mapped onto the refrigerant domain (n=209)
- LOW refrigerant-likeness (<0.4): 202 / 209
- MEDIUM refrigerant-likeness (0.4-0.6): 6 / 209
- HIGH refrigerant-likeness (>=0.6): 1 / 209

## XGB-D Tc RMSE: global vs refrigerant-like subsets

- Global (n=209): RMSE=25.0769 K
- HIGH refrigerant-likeness (n=1): RMSE=1.6886 K
- MEDIUM+HIGH refrigerant-likeness (n=7): RMSE=14.0713 K

Full per-variant, per-bin RMSE/MAE/n table: `05_variant_metrics_by_refrigerant_likeness.csv`

## A->C / C->D by refrigerant-likeness bin

Full table: `06_AtoC_CtoD_by_refrigerant_likeness.csv` (positive delta = later variant reduces |error|)

- LOW / A->C: n=202, median=+0.8892, mean=+2.9501, improved=126, worsened=76
- LOW / C->D: n=202, median=+0.1935, mean=+0.818, improved=105, worsened=97
- MEDIUM / A->C: n=6, median=-0.815, mean=-0.3406, improved=2, worsened=4
- MEDIUM / C->D: n=6, median=-4.8474, mean=-3.6815, improved=2, worsened=4
- HIGH / A->C: n=1, median=-4.4731, mean=-4.4731, improved=0, worsened=1
- HIGH / C->D: n=1, median=+8.9317, mean=+8.9317, improved=1, worsened=0
- MEDIUM+HIGH / A->C: n=7, median=-1.4417, mean=-0.9309, improved=2, worsened=5
- MEDIUM+HIGH / C->D: n=7, median=-4.0422, mean=-1.8796, improved=3, worsened=4
- ALL / A->C: n=209, median=+0.7527, mean=+2.8201, improved=128, worsened=81
- ALL / C->D: n=209, median=+0.1933, mean=+0.7276, improved=108, worsened=101

_See `plots/refrigerant_domain_diagnostic.png` for XGB-D |error| vs refrigerant-likeness._