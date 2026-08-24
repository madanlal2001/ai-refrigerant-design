# Reproducibility

## What's excluded from git, and how to get it back

Nothing excluded from git is deleted from the original research machine — this repo only controls what's *versioned*. Everything below is either regenerable by running a committed script, or documented as an external dependency.

| Excluded artifact | Size | Regenerate with | Notes |
|---|---|---|---|
| `xgboost/artifacts/X_benchmark.npy`, `X_est.npy`, `X_142.npy` | 19MB / 51MB / 1.3MB | `XGBoost.ipynb` cell 1 (`01_prepare_data.py`) | Deterministic given the same input CSVs and `seed=42` — RDKit-2D + Morgan(2048) feature matrices. |
| `xgboost/results/*/models/*.json` | 1-11MB each | `xgboost/results/<experiment>/train.py` | Diagnostic experiment models (not deployment models — see `docs/validation.md`). Deterministic given `seed=42`. |
| `gnn/*.pt` checkpoints | ~63MB total | `gnn/gnn_pipeline.py` | Not fully deterministic (GPU nondeterminism in PyTorch); expect small run-to-run variation. See `gnn/README.md` for expected filenames/paths. |
| `cop/checkpoints/`, `cop/merged_pipeline/` | ~82MB | `gnn/gnn_pipeline.py`, then point `cop/.env.example`'s `REFRIG_GNN_CHECKPOINT` at the output | Only needed to enable the GNN prediction option in the Streamlit app — the core thermodynamics (`cop/refrig/`) never imports torch and works without this. |
| `cop/mixtures/*_9comp.csv`, `*_81comp.csv` | 690MB / 95MB | `cop/mixtures/cop_calculation_for_ternary.py`, `cop_calculation_for_binary_mixtures.py` | Full parameter-sweep outputs. Smaller summary CSVs (`binary_mixture_cop_sweep.csv` etc.) are kept in git. |
| `docs/literature/` (not populated in this initial commit) | — | — | Reserved for reference papers if you choose to add them locally — third-party copyrighted PDFs should not be committed regardless (see `.gitignore`). |

## Environment

```bash
pip install -r requirements.txt          # core stack: numpy/pandas/scipy/sklearn/rdkit/xgboost/streamlit
# optional, for gnn/:
pip install torch torch_geometric        # install torch FIRST, then match torch_geometric's wheel to your CUDA version
# optional, for pr_eos/ validation scripts:
pip install CoolProp
```

## Determinism notes

- **XGBoost stages**: fully deterministic given the same input CSVs — `seed=42` fixes both the Butina/deficit split and every `XGBRegressor`. The `manifest.json` / `split_manifest.json` in each experiment records SHA-256 hashes of the exact input files used, so you can verify you're reproducing against the same data.
- **GNN stage**: PyTorch training is not bitwise-deterministic across hardware/driver versions even with a fixed seed; expect small (not qualitative) differences in reported metrics on re-training.
- **Split-cap parameter** (`unique_molecules_splitfix_xgboost/train.py`): `CLUSTER_SIZE_CAP=200`, `CLUSTER_STEP=0.15`, `CLUSTER_MIN_CUTOFF=0.10` are recorded in that experiment's `split_manifest.json` — these were chosen based on the specific 1,512-molecule super-cluster found in this dataset (see `docs/validation.md`); revisit them if you add substantially different data.

## Order of operations if reproducing from scratch

1. `xgboost/XGBoost.ipynb` (cells 1-4, in order) → produces `artifacts/` and `results_v3/`.
2. `xgboost/04_tc_outlier_audit.py` through `07_export_refrigerant_predictions.py`, in numeric order.
3. (Optional, diagnostic) `xgboost/results/<experiment>/train.py` then `audit.py`, for any of the three `unique_molecules_*` experiments.
4. `gnn/gnn_pipeline.py` to train, then `gnn/predict_61_candidates.py` for the candidate screening.
5. `cop/app.py` (`streamlit run cop/app.py`) for the interactive tool, or the scripts in `cop/pure_refrigerant/` and `cop/mixtures/` directly for batch COP calculations.
