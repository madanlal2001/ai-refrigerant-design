#!/usr/bin/env python
"""
Standalone inference script: scores refrigerant_candidates_all_61.csv (61
molecules, column 'SMILES') with the already-trained GNN3 merged bundle
(critical_property_merged_pipeline.pt), reusing the model/feature code from
../gnn_pipeline.py. Does NOT retrain anything.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

GNN3_DIR = Path(__file__).resolve().parent
PARENT_DIR = GNN3_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

import gnn_pipeline as gp  # noqa: E402

INPUT_CSV = GNN3_DIR / "refrigerant_candidates_all_61.csv"
OUTPUT_CSV = GNN3_DIR / "predicted_Tc_Pc_omega_61_candidates.csv"
MERGED_BUNDLE = GNN3_DIR / "critical_property_merged_pipeline.pt"


def main():
    device = torch.device("cpu")

    print(f"Loading merged bundle: {MERGED_BUNDLE}")
    model_A, model_B, scaler_A, scaler_B, loaded_mol_scaler = gp.load_merged_bundle(MERGED_BUNDLE, device)
    # gnn_pipeline module-level singleton used inside smiles_to_graph_multitask
    gp.mol_scaler.means = loaded_mol_scaler.means
    gp.mol_scaler.stds = loaded_mol_scaler.stds

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV.name}")

    if "smiles" not in df.columns:
        if "SMILES" in df.columns:
            df["smiles"] = df["SMILES"]
        elif "Smiles" in df.columns:
            df["smiles"] = df["Smiles"]
        else:
            raise ValueError("Input CSV must contain a SMILES/smiles column.")

    df = df.reset_index(drop=True).copy()
    df["row_id"] = np.arange(len(df))

    for col in gp.TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan

    print("Predicting Tc with Model A...")
    pred_a = gp.predict_multitask(model_A, df, gp.TARGETS_A, scaler_A, device, batch_size=50)
    pred_a = pred_a[["row_id", "Tc (K)"]].rename(columns={"Tc (K)": "Tc_predicted (K)"})

    print("Predicting Pc/omega with Model B...")
    pred_b = gp.predict_multitask(model_B, df, gp.TARGETS_B, scaler_B, device, batch_size=50)
    pred_b = pred_b[["row_id", "Pc (bar)", "omega (-)"]].rename(
        columns={"Pc (bar)": "Pc_predicted (bar)", "omega (-)": "omega_predicted (-)"}
    )

    out = df.drop(columns=["Tc (K)", "Pc (bar)", "omega (-)"])
    out = out.merge(pred_a, on="row_id", how="left")
    out = out.merge(pred_b, on="row_id", how="left")
    out = out.drop(columns=["row_id"])

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions for {len(out)} molecules to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
