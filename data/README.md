# Data

## What's committed here

| File | Rows | What it is |
|---|---|---|
| `final_master_experimental_finetuning_6644.csv` | 6,644 | Curated experimental molecule properties (Tc, Pc, omega, Tb, Tm, dHvap, dHfus — sparse, not every column filled per row). Primary fine-tuning/training-label source for both `xgboost/` and `gnn/`. |
| `final_pretraining_estimated_critprop.csv` | ~6,147 | Lower-confidence estimated/group-contribution critical-property data, used as weakly-weighted supplementary training signal (XGBoost variants B/D; GNN pretrain stage). |
| `Yaws Collection.tsv` | ~7,549 | Yaws reference database (CAS, name, Tc, Pc, Vc, omega — Pc column is in Pa despite its header). No per-row experimental/estimated quality flag. |
| `molecules_144_refrigerant.xlsx` | 142 | The locked deployment/screening target set. **Excluded from all training data everywhere in this repo** — every training script re-verifies zero structural overlap and fails loudly (or logs+drops, see each stage's docs) if leakage is found. |
| `molecules_smiles_BEST1.xlsx` | — | Alternate/working molecule-name-to-SMILES resolution table used during dataset assembly. |
| `unique_molecules_Tc_Pc_omega.csv` | 7,829 | Union of the finetune + Yaws molecule sets, deduplicated by InChIKey, **finetune value preferred per-property, Yaws fills only genuinely missing cells**. Names backfilled from PubChem where absent. Built for the `unique_molecules_xgboost` diagnostic experiment (see `xgboost/results/`). |
| `unique_molecules_Tc_Pc_omega_YAWSPRIORITY.csv` | 7,829 | Same union/molecule set as above, but **Yaws value preferred whenever present**, finetune only fills gaps. Built to test source-priority sensitivity (see `xgboost/results/unique_molecules_yawspriority_xgboost/`). |
| `mclinden_predicted_Tc_Pc_omega.csv` | — | GNN/XGBoost-predicted critical properties for the McLinden reference refrigerant set, used in `cop/mixtures/` mixture-screening scripts. |

## Provenance

`final_master_experimental_finetuning_6644.csv` and `final_pretraining_estimated_critprop.csv` are the two foundational label sources for this entire project — every downstream property model (XGBoost, GNN) is trained from them. `unique_molecules_*.csv` are derived files built during this project's later diagnostic work (see `xgboost/results/unique_molecules_xgboost/audit/` for a full read-only audit of label-source quality and known issues).

**Known data-quality caveats** (see `docs/validation.md` for detail):
- Yaws Collection has no experimental-vs-estimated flag per row — "yaws-sourced" in `value_sources` columns means "came from this file," not "verified experimental."
- The GNN pretraining/fine-tuning split has significant molecule overlap (~90% of fine-tuning molecules also appear in the pretraining file) — see `docs/validation.md`. This does **not** affect the XGBoost frozen-v3 protocol, which enforces val/test/refrigerant exclusion from the estimated pool explicitly (`xgboost/XGBoost.ipynb`, cell 1).

## What's NOT committed here

- Raw Yaws-with-SMILES resolution intermediates (`yaws_with_smiles*.csv`) — these were one-time structure-resolution working files, not needed to reproduce anything downstream of `unique_molecules_*.csv`, which already carries the resolved SMILES.
- Any file derived purely mechanically from the above (feature matrices, model binaries) — see `docs/reproducibility.md` for what regenerates what.
