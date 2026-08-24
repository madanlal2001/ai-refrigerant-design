# gnn/ — D-MPNN critical-property prediction (Tc, Pc, omega)

A hybrid message-passing GNN (PyTorch Geometric), pretrained on the larger estimated-property pool then fine-tuned on the curated experimental set — the same two-stage idea as XGBoost variants B/D, but with a shared learned representation instead of a weighted sample mix.

## Files here

- `gnn_pipeline.py` — the full training pipeline (data loading, featurization, hybrid pretrain/fine-tune loop, per-target specialist heads). Originally developed on Kaggle (`!pip install RDKit pubchempy torch_geometric openpyxl` in cell 1).
- `GNN_model.ipynb` — the notebook version of the same pipeline, used to produce the runs below.
- `predict_61_candidates.py` — inference script that screened a 61-molecule candidate set through the trained model.
- `dmpnn_pretrain_frozen.py` — a frozen-backbone pretraining variant.
- `results/` — small prediction/diagnostic CSVs and figures from the canonical run (see below).

## Which run is canonical

The training code was run multiple times as separate iterations (`GNN1`–`GNN4` in the original working tree). All four use **identical pipeline code** — they are re-runs, not methodologically different versions. **GNN3's run is treated as canonical here**: it is the most recently modified, has complete trained checkpoints for all four sub-models (Tc/pc-omega x pretrained/fine-tuned + a merged bundle), and is the only run that produced the actual downstream deliverable — `predicted_Tc_Pc_omega_61_candidates.csv` / `refrigerant_candidates_all_61.csv`.

GNN4 (an earlier, incomplete run — missing trained checkpoints and the pipeline script) and GNN1/GNN2 are **not** included; they remain on local disk at `2_Refrigrant/Gnn/GNN/GNN/GNN{1,2,4}/` if needed for provenance.

## Trained checkpoints — excluded from git

The five `.pt` files from the canonical run (`best_hybrid_tc_pretrained.pt`, `best_hybrid_tc_finetuned.pt`, `best_hybrid_pc_omega_pretrained.pt`, `best_hybrid_pc_omega_finetuned.pt`, `critical_property_merged_pipeline.pt` — ~63MB total) are **not committed**. To regenerate:

```bash
python gnn_pipeline.py   # trains on data/final_master_experimental_finetuning_6644.csv
                          # + data/final_pretraining_estimated_critprop.csv
```

Expected checkpoint location for downstream use (e.g. `cop/refrig/predictors/gnn.py`): `gnn/checkpoints/critical_property_merged_pipeline.pt`, or set `REFRIG_GNN_CHECKPOINT` (see `cop/.env.example`).

## Known caveat — read before trusting reported GNN test metrics

**~90% of the fine-tuning (experimental) molecules also appear in the pretraining (estimated) file.** This means the pretrain → fine-tune split has substantial train/test-adjacent overlap, so reported fine-tuning-stage test MAE is **not a clean generalization estimate** for this GNN pipeline. This is a different (and more serious) leakage concern than anything in the XGBoost frozen-v3 protocol, which explicitly filters the estimated pool against val/test/refrigerants (see `xgboost/README.md`). The 142 locked refrigerant molecules themselves are confirmed clean of this overlap — see `docs/validation.md`.
