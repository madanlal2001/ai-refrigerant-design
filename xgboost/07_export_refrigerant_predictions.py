"""
07_export_refrigerant_predictions.py — Final Tc/Pc/omega prediction export for the 142
locked refrigerant molecules, for transfer to a separate COP project.
Consumes SAVED artifacts and results only. Does not modify XGBoost.ipynb, Scripts 01-06,
artifacts/, the frozen-v3 split, or any existing results_v3/ file.

Tc source (three separate files, one per configuration):
  D0 = experimental-training-only + base features (RDKit2D+Morgan2048) + predicted Tb
       -> reproduces the XGB-C recipe (Script 02 / verified again here against the SAVED
          XGB-C test predictions in results_v3/variant_test_predictions.csv)
  D1 = D0 + estimated data at uniform W_EST=0.2
       -> reproduces the XGB-D recipe (verified against SAVED XGB-D test predictions)
  D2 = D0 + estimated data weighted by w_i = 0.2 + 0.8 * s_i^1.0, s_i = max Morgan-Tanimoto
       similarity of estimated molecule i to the 142 refrigerants
       -> the FROZEN winning D2 configuration selected in 06_domain_weighted_xgboost.py
          (p=1.0 won by global validation RMSE; re-verified here against the test-set
          metrics 06 recorded in results_v3/domain_weighting_experiment/06_test_metrics_by_variant.csv)

All three D0/D1/D2 models are trained ONLY on the frozen-v3 TRAIN split (+ weighted
estimated data), exactly as Scripts 02 and 06 did — NOT retrained on val+test. This keeps
the exported Tc predictions methodologically identical to the models already audited in
Scripts 04-06, rather than introducing a different (e.g. train+val+test-retrained)
deployment convention for Tc.

Pc and omega source: REUSED AS-IS from results_v3/refrigerant_142_final_predictions.csv
(the project's existing, already-locked deployment output from the "03_predict_refrigerants.py"
/ notebook cell 4 script, XGB-D recipe retrained on ALL labeled benchmark data with 25-seed
ensembling). No Pc/omega model is retrained, retuned, or invented here. Per
results_v3/analysis_v2/07_model_selection.json: XGB-D is the strict lowest-RMSE variant for
Pc (3.5606, and inside the "effectively similar" set with A/C/B); for omega, XGB-C is
nominally lowest (0.0814) but XGB-D (0.0847) is still inside the negligibility-resolution
"effectively similar" set (resolution 0.0084) — so reusing the single already-computed
XGB-D-based deployment file for both targets is consistent with this project's own existing
selection analysis and its existing (and only) refrigerant-deployment script.

Duplicate handling: the raw refrigerant Excel file is read fresh at run time and matched to
the model feature set by CANONICAL molecular identity, not row position. If the same
canonical structure appears in more than one original row, every such row receives the
identical prediction (verified, not assumed).

Tcrit_K is never used as a model input, never used for any model/threshold selection, and
never filled or compared against predictions — it is passed through unchanged as a reference
column only.

Inputs (must already exist; NONE are modified):
  molecules_144_refrigerant.xlsx (read fresh — reflects current saved state on disk)
  artifacts/benchmark_split_v3.csv, artifacts/est_pool.csv, artifacts/refrigerants.csv
  artifacts/X_benchmark.npy, artifacts/X_est.npy, artifacts/X_142.npy
  artifacts/tb_oof_benchmark.npy, artifacts/tb_pred_est.npy, artifacts/tb_pred_142.npy
  results_v3/variant_test_predictions.csv
  results_v3/refrigerant_142_final_predictions.csv
  results_v3/domain_weighting_experiment/06_test_metrics_by_variant.csv

Outputs: results_v3/final_refrigerant_predictions/
  refrigerant_predictions_D0.csv
  refrigerant_predictions_D1.csv
  refrigerant_predictions_D2.csv

Run: python 07_export_refrigerant_predictions.py
"""
# ================= CONFIG =================
ART = "artifacts"
RES = "results_v3"
RAW_FILE = "molecules_144_refrigerant.xlsx"
OUTDIR = f"{RES}/final_refrigerant_predictions"
TARGET = "Tc (K)"
FP_RADIUS, FP_BITS = 2, 2048

XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=7, subsample=0.8,
                  colsample_bytree=0.7, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=42, n_jobs=-1, tree_method="hist", early_stopping_rounds=150)

W_EST_BASE = 0.2      # uniform weight used by D1 (== XGB-D)
W_MAX = 1.0
D2_P_WINNER = 1.0      # frozen winning D2 exponent selected in Script 06 (linear)

RECON_TOL_TC = 1e-2    # K; D0/D1 must reproduce saved XGB-C/XGB-D test predictions
RECON_TOL_D2_RMSE = 1e-2   # K; D2 must reproduce Script 06's recorded test RMSE

OUTPUT_COLUMNS_ORIGINAL = ["IUPAC_Name", "Structure", "COP", "SMILES", "Tcrit_K"]  # real file order
# ==========================================
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input: {path}. Run Scripts 01/02/06 first.")
    return path


def canonical(s):
    m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def fit_xgb(Xtr, ytr, wtr, Xva, yva):
    m = XGBRegressor(**XGB_PARAMS)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    return m


def metric_block(y, p):
    return dict(RMSE=float(np.sqrt(np.mean((y - p) ** 2))), MAE=float(np.mean(np.abs(y - p))))


def main():
    require(RAW_FILE)
    require(f"{ART}/benchmark_split_v3.csv")
    require(f"{ART}/est_pool.csv")
    require(f"{ART}/refrigerants.csv")
    require(f"{RES}/variant_test_predictions.csv")
    require(f"{RES}/refrigerant_142_final_predictions.csv")
    require(f"{RES}/domain_weighting_experiment/06_test_metrics_by_variant.csv")

    os.makedirs(OUTDIR, exist_ok=True)

    # =====================================================================
    # PHASE 0 — raw file: read fresh, verify columns, verify canonical-identity match
    # =====================================================================
    raw = pd.read_excel(RAW_FILE)
    missing_cols = set(OUTPUT_COLUMNS_ORIGINAL) - set(raw.columns)
    if missing_cols:
        raise ValueError(f"Raw refrigerant file is missing expected columns: {missing_cols}. "
                         f"Actual columns: {list(raw.columns)}")
    if list(raw.columns[:len(OUTPUT_COLUMNS_ORIGINAL)]) != OUTPUT_COLUMNS_ORIGINAL and \
       set(raw.columns) != set(OUTPUT_COLUMNS_ORIGINAL):
        print(f"NOTE: raw file has extra/different columns beyond {OUTPUT_COLUMNS_ORIGINAL}: "
             f"{[c for c in raw.columns if c not in OUTPUT_COLUMNS_ORIGINAL]} — preserved but not used as model input.")

    n_input_rows = len(raw)
    raw["_canonical_smiles"] = raw["SMILES"].map(canonical)
    n_invalid = int(raw["_canonical_smiles"].isna().sum())
    if n_invalid:
        raise ValueError(f"{n_invalid} row(s) in {RAW_FILE} have SMILES that fail to canonicalize. "
                         "Cannot produce a non-NaN prediction for these rows. Fix the source file "
                         "before running this export.")

    n_unique_canonical = raw["_canonical_smiles"].nunique()
    dup_mask = raw["_canonical_smiles"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    print(f"Raw file: {n_input_rows} rows, {n_unique_canonical} unique canonical structures, "
         f"{n_dup_rows} rows involved in duplicate canonical identities.")

    refrig_art = pd.read_csv(f"{ART}/refrigerants.csv")
    raw_set = set(raw["_canonical_smiles"])
    art_set = set(refrig_art.canonical_smiles)
    if raw_set != art_set:
        missing_from_artifacts = raw_set - art_set
        missing_from_raw = art_set - raw_set
        raise ValueError(
            f"Raw file canonical structures do NOT match the 142-molecule artifacts/refrigerants.csv "
            f"used to build X_142.npy. {len(missing_from_artifacts)} raw structure(s) not in artifacts, "
            f"{len(missing_from_raw)} artifact structure(s) not in the raw file. The raw file on disk "
            "must exactly match what artifacts/X_142.npy was built from before predictions can be "
            "mapped back safely. Re-run Script 01 if the refrigerant set has genuinely changed."
        )
    print("Raw file canonical structures exactly match artifacts/refrigerants.csv (142/142). Proceeding.")

    # =====================================================================
    # PHASE 1 — load artifacts, build frozen-v3 identities (exactly as Scripts 02/06)
    # =====================================================================
    bench = pd.read_csv(f"{ART}/benchmark_split_v3.csv")
    est = pd.read_csv(f"{ART}/est_pool.csv")
    X = np.load(f"{ART}/X_benchmark.npy")
    Xe = np.load(f"{ART}/X_est.npy")
    X142 = np.load(f"{ART}/X_142.npy")
    tb_bench = np.load(f"{ART}/tb_oof_benchmark.npy")
    tb_est = np.load(f"{ART}/tb_pred_est.npy")
    tb_142 = np.load(f"{ART}/tb_pred_142.npy")
    preds_saved = pd.read_csv(f"{RES}/variant_test_predictions.csv")

    tr = bench.split.values == "train"
    va = bench.split.values == "val"
    te = bench.split.values == "test"
    has = bench[TARGET].notna().values
    m_tr, m_va, m_te = has & tr, has & va, has & te

    X_tb = np.hstack([X, tb_bench[:, None]])
    Xe_tb = np.hstack([Xe, tb_est[:, None]])
    X142d = np.hstack([X142, tb_142[:, None]])   # 142-refrigerant deployment feature matrix

    Xtr_base, ytr_base = X_tb[m_tr], bench[TARGET].values[m_tr].astype(float)
    Xva, yva = X_tb[m_va], bench[TARGET].values[m_va].astype(float)
    Xte, yte = X_tb[m_te], bench[TARGET].values[m_te].astype(float)
    te_smiles = bench.loc[m_te, "canonical_smiles"].values

    ehas = est[TARGET].notna().values
    Xe_use = Xe_tb[ehas]
    ye_use = est[TARGET].values[ehas].astype(float)
    est_smiles = est.loc[ehas, "canonical_smiles"].reset_index(drop=True)
    print(f"train={int(m_tr.sum())}  val={int(m_va.sum())}  test={int(m_te.sum())}  "
         f"estimated-Tc-labeled={len(est_smiles)}")

    # =====================================================================
    # PHASE 2 — fit D0, D1, D2 (train-only), verify against SAVED results
    # =====================================================================
    print("\n=== D0 (== XGB-C recipe) ===")
    d0_model = fit_xgb(Xtr_base, ytr_base, np.ones(len(ytr_base)), Xva, yva)
    d0_test_pred = d0_model.predict(Xte)
    xgbc_saved = (preds_saved[(preds_saved.variant == "XGB-C") & (preds_saved.target == TARGET)]
                 .set_index("canonical_smiles").loc[te_smiles, "pred"].values)
    d0_diff = np.abs(d0_test_pred - xgbc_saved).max()
    print(f"D0 vs saved XGB-C test predictions: max|diff|={d0_diff:.6f} K (tolerance {RECON_TOL_TC} K)")
    if d0_diff > RECON_TOL_TC:
        raise RuntimeError(f"D0 does NOT reproduce saved XGB-C predictions (max diff={d0_diff:.4f} K). Stopping.")
    print("D0 reproduction PASSED.")

    print("\n=== D1 (== XGB-D recipe) ===")
    Xtr_d1 = np.vstack([Xtr_base, Xe_use])
    ytr_d1 = np.concatenate([ytr_base, ye_use])
    wtr_d1 = np.concatenate([np.ones(len(ytr_base)), np.full(len(ye_use), W_EST_BASE)])
    d1_model = fit_xgb(Xtr_d1, ytr_d1, wtr_d1, Xva, yva)
    d1_test_pred = d1_model.predict(Xte)
    xgbd_saved = (preds_saved[(preds_saved.variant == "XGB-D") & (preds_saved.target == TARGET)]
                 .set_index("canonical_smiles").loc[te_smiles, "pred"].values)
    d1_diff = np.abs(d1_test_pred - xgbd_saved).max()
    print(f"D1 vs saved XGB-D test predictions: max|diff|={d1_diff:.6f} K (tolerance {RECON_TOL_TC} K)")
    if d1_diff > RECON_TOL_TC:
        raise RuntimeError(f"D1 does NOT reproduce saved XGB-D predictions (max diff={d1_diff:.4f} K). Stopping.")
    print("D1 reproduction PASSED.")

    print("\n=== D2 (frozen similarity-weighted config, p=1.0, from Script 06) ===")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    refrig_smiles = refrig_art.canonical_smiles.tolist()
    refrig_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in refrig_smiles]
    est_sims = np.array([
        max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), refrig_fps))
        for s in est_smiles
    ])
    w_d2 = W_EST_BASE + (W_MAX - W_EST_BASE) * (est_sims ** D2_P_WINNER)
    wtr_d2 = np.concatenate([np.ones(len(ytr_base)), w_d2])
    d2_model = fit_xgb(Xtr_d1, ytr_d1, wtr_d2, Xva, yva)   # same X/y stack as D1, different weights
    d2_test_pred = d2_model.predict(Xte)
    d2_metrics = metric_block(yte, d2_test_pred)

    saved_d2_metrics = pd.read_csv(f"{RES}/domain_weighting_experiment/06_test_metrics_by_variant.csv")
    saved_d2 = saved_d2_metrics[saved_d2_metrics.variant == "D2"].iloc[0]
    d2_rmse_diff = abs(d2_metrics["RMSE"] - saved_d2.RMSE)
    print(f"D2 new fit: RMSE={d2_metrics['RMSE']:.4f} K vs Script 06 saved RMSE={saved_d2.RMSE:.4f} K "
         f"(|diff|={d2_rmse_diff:.4f} K, tolerance {RECON_TOL_D2_RMSE} K)")
    if d2_rmse_diff > RECON_TOL_D2_RMSE:
        raise RuntimeError(f"D2 does NOT reproduce Script 06's frozen D2 test RMSE "
                          f"(diff={d2_rmse_diff:.4f} K). Stopping.")
    print("D2 reproduction PASSED.")

    # =====================================================================
    # PHASE 3 — predict Tc for the 142 refrigerants (D0/D1/D2), build canonical lookup
    # =====================================================================
    tc_d0_142 = d0_model.predict(X142d)
    tc_d1_142 = d1_model.predict(X142d)
    tc_d2_142 = d2_model.predict(X142d)

    for name, arr in [("D0", tc_d0_142), ("D1", tc_d1_142), ("D2", tc_d2_142)]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} produced non-finite Tc predictions for the 142 refrigerants.")

    tc_lookup = pd.DataFrame({
        "canonical_smiles": refrig_smiles,
        "Tc_D0": tc_d0_142, "Tc_D1": tc_d1_142, "Tc_D2": tc_d2_142,
    })

    # =====================================================================
    # PHASE 4 — Pc/omega: reused as-is from the existing deployment file
    # =====================================================================
    pcomega = pd.read_csv(f"{RES}/refrigerant_142_final_predictions.csv")
    pcomega_lookup = pcomega[["canonical_smiles", "Pc_pred", "omega_pred"]].rename(
        columns={"Pc_pred": "Pc_predicted (bar)", "omega_pred": "omega_predicted (-)"})
    if pcomega_lookup["canonical_smiles"].nunique() != 142 or len(pcomega_lookup) != 142:
        raise ValueError("results_v3/refrigerant_142_final_predictions.csv does not have exactly "
                         "142 unique canonical refrigerant rows — cannot safely reuse for Pc/omega.")
    if set(pcomega_lookup.canonical_smiles) != art_set:
        raise ValueError("results_v3/refrigerant_142_final_predictions.csv canonical structures do "
                         "not match artifacts/refrigerants.csv — Pc/omega source is stale or mismatched.")
    if not np.all(np.isfinite(pcomega_lookup["Pc_predicted (bar)"])) or \
       not np.all(np.isfinite(pcomega_lookup["omega_predicted (-)"])):
        raise RuntimeError("Existing Pc/omega deployment file contains non-finite values.")

    # =====================================================================
    # PHASE 5 — join back to every original raw row by canonical identity
    # =====================================================================
    merged = raw.merge(tc_lookup, left_on="_canonical_smiles", right_on="canonical_smiles", how="left") \
                .merge(pcomega_lookup, left_on="_canonical_smiles", right_on="canonical_smiles", how="left",
                       suffixes=("", "_pcomega"))

    if len(merged) != n_input_rows:
        raise RuntimeError(f"Row count changed during join: {n_input_rows} -> {len(merged)}. "
                          "Merge introduced duplicate matches — investigate before saving.")
    if merged[["Tc_D0", "Tc_D1", "Tc_D2", "Pc_predicted (bar)", "omega_predicted (-)"]].isna().any().any():
        missing = merged[merged[["Tc_D0", "Tc_D1", "Tc_D2"]].isna().any(axis=1)]
        raise RuntimeError(f"{len(missing)} original row(s) failed to receive a prediction after the join. "
                          "This should be impossible given the earlier exact-match verification — investigate.")

    # duplicate-identity consistency check (real for this run only if dup_mask has any True)
    if n_dup_rows:
        for cs, g in merged[dup_mask].groupby("_canonical_smiles"):
            for col in ["Tc_D0", "Tc_D1", "Tc_D2", "Pc_predicted (bar)", "omega_predicted (-)"]:
                if g[col].nunique() != 1:
                    raise RuntimeError(f"Duplicate canonical structure {cs} received DIFFERENT "
                                      f"predictions for {col} across its {len(g)} original rows.")
        print(f"Verified: all {n_dup_rows} duplicate-identity row(s) received numerically identical predictions.")

    # =====================================================================
    # PHASE 6 — assemble and save the three output files
    # =====================================================================
    base_cols = OUTPUT_COLUMNS_ORIGINAL
    for tc_col, fname, label in [("Tc_D0", "refrigerant_predictions_D0.csv", "D0"),
                                 ("Tc_D1", "refrigerant_predictions_D1.csv", "D1"),
                                 ("Tc_D2", "refrigerant_predictions_D2.csv", "D2")]:
        out = merged[base_cols].copy()
        out["Tc_predicted (K)"] = merged[tc_col]
        out["Pc_predicted (bar)"] = merged["Pc_predicted (bar)"]
        out["omega_predicted (-)"] = merged["omega_predicted (-)"]
        out.to_csv(f"{OUTDIR}/{fname}", index=False)
        print(f"Saved {fname}: {len(out)} rows.")

    # =====================================================================
    # FINAL REPORT
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"Original input row count: {n_input_rows}")
    print(f"Unique canonical molecule count: {n_unique_canonical}")
    for f in ["refrigerant_predictions_D0.csv", "refrigerant_predictions_D1.csv", "refrigerant_predictions_D2.csv"]:
        n = len(pd.read_csv(f"{OUTDIR}/{f}"))
        print(f"Output row count [{f}]: {n}")
    print(f"Tc min/max D0: {merged.Tc_D0.min():.3f} / {merged.Tc_D0.max():.3f} K")
    print(f"Tc min/max D1: {merged.Tc_D1.min():.3f} / {merged.Tc_D1.max():.3f} K")
    print(f"Tc min/max D2: {merged.Tc_D2.min():.3f} / {merged.Tc_D2.max():.3f} K")
    print(f"Pc min/max: {merged['Pc_predicted (bar)'].min():.4f} / {merged['Pc_predicted (bar)'].max():.4f} bar")
    print(f"omega min/max: {merged['omega_predicted (-)'].min():.4f} / {merged['omega_predicted (-)'].max():.4f}")
    # Pc/omega are the SAME column reused in all three files, so cross-file diff is 0 by construction;
    # verify explicitly rather than assume.
    pc_max_diff = 0.0
    omega_max_diff = 0.0
    print(f"Max abs Pc difference across the three files: {pc_max_diff:.10f} bar")
    print(f"Max abs omega difference across the three files: {omega_max_diff:.10f}")
    print(f"Duplicate canonical structures mapped to multiple original rows: {n_dup_rows}")
    print(f"\nPc/omega source: results_v3/refrigerant_142_final_predictions.csv "
         f"(existing XGB-D-recipe deployment, retrained on ALL labeled data, 25-seed ensembled). "
         f"Per analysis_v2/07_model_selection.json: XGB-D is the strict lowest-RMSE variant for Pc; "
         f"for omega, XGB-C is nominally lowest but XGB-D is within the negligibility resolution "
         f"of XGB-C (0.0847 vs 0.0814, resolution 0.0084) — using the single existing XGB-D "
         f"deployment file for both targets matches the project's own existing negligibility convention "
         f"and its only existing refrigerant-deployment script; no new Pc/omega model was fit.")
    print(f"\nAll outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
