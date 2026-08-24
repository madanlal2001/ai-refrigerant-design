"""
06_domain_weighted_xgboost.py — Controlled Tc estimated-data weighting ablation
(D0-D4), built on the frozen-v3 protocol and the existing predicted-Tb feature.
Consumes SAVED artifacts only. Preserves the frozen train/val/test split, base
feature construction (RDKit2D + Morgan2048), and predicted-Tb methodology exactly
as Script 01/02 built them. Only Tc is modeled here; Pc and omega are untouched.

Motivation (from 05_refrigerant_domain_audit.py): uniform estimated-data weighting
(XGB-D's W_EST=0.2 for all ~5554 Tc-labeled estimated rows) made every one of the
7 refrigerant-like frozen-test molecules worse relative to XGB-C. That audit's
refrigerant-like test subset is n=7 — too small to prove uniform weighting is the
root cause. This script tests whether DOMAIN-AWARE reweighting of the SAME
estimated pool recovers value for refrigerant-like chemistry, without touching the
frozen split, Script 01, or the existing Script 02 outputs.

Variants (Tc only, all built on the "C/D" feature stack = RDKit2D + Morgan2048 +
predicted Tb, i.e. these are estimated-data-weighting variants of XGB-C/XGB-D):
  D0 = experimental-only + base features + predicted Tb            -> should == XGB-C
  D1 = D0 + estimated data at uniform W_EST=0.2                    -> should == XGB-D
  D2 = D0 + estimated data weighted by a SMOOTH function of max
       Morgan-Tanimoto similarity to the 142 locked refrigerants
  D3 = D0 + estimated data weighted by a SMOOTH function of a compact
       composition-domain distance (standardized halogen_fraction + F/C ratio,
       both measured against the refrigerant population's own centroid)
  D4 = D0 + a HARD refrigerant-like SUBSET of estimated data (same composition
       distance as D3, radius selected on train/val evidence only), weight=1.0

VERIFICATION GATE: D0 and D1 must reproduce the SAVED XGB-C / XGB-D Tc test
predictions (results_v3/variant_test_predictions.csv) within RECON_TOL. If not,
the script raises and stops before touching D2/D3/D4 — per explicit instruction,
do not proceed past an unreproduced baseline.

SELECTION DISCIPLINE (no leakage / no tiny-subgroup overfitting):
  - D2/D3/D4 weighting-function FORM and anchor values (W_MIN=W_EST_BASE=0.2,
    W_MAX=1.0) are fixed by construction, not searched.
  - Each of D2/D3/D4 has exactly TWO candidate hyperparameter values, chosen from
    inspecting the actual similarity/composition-distance distributions of the
    estimated pool and the 142 refrigerants (see 02_weighting_function_design.md).
    NONE of these candidates were chosen by looking at frozen-test errors.
  - The winning candidate per D-variant is selected ONLY by global validation Tc
    RMSE (n=209, the same quantity XGB-A/B/C/D early-stopping already targets).
  - A refrigerant-domain-aware validation diagnostic is computed and reported
    (val Tc molecules mapped to refrigerant-likeness bins) but is NEVER used for
    selection — validated as too small to be stable (0 HIGH, 16 MEDIUM of 209).
  - Once a winner is picked per D-variant, its ALREADY-FITTED model is reused for
    the single, one-time frozen-test evaluation. Test predictions are computed
    exactly once per D-variant, only after all three winners are frozen.

Residual sign convention (matches 03/04/05): residual = actual - predicted.
Descriptive only — single frozen split, no causal or significance claims.

Inputs (must already exist — run Scripts 01/02 first; none are modified):
  artifacts/benchmark_split_v3.csv, artifacts/est_pool.csv, artifacts/refrigerants.csv
  artifacts/X_benchmark.npy, artifacts/X_est.npy
  artifacts/tb_oof_benchmark.npy, artifacts/tb_pred_est.npy
  results_v3/variant_test_predictions.csv
  results_v3/refrigerant_domain_audit/04_test_molecule_refrigerant_likeness.csv

Outputs: results_v3/domain_weighting_experiment/
  00_SUMMARY.md
  01_reproduction_check.csv
  02_weighting_function_design.md
  03_candidate_selection_global_val_rmse.csv
  04_val_refrigerant_likeness_diagnostic.csv
  05_est_pool_domain_weights.csv
  06_test_metrics_by_variant.csv
  07_test_metrics_by_refrigerant_likeness_bin.csv
  08_seven_refrigerant_like_molecules.csv
  plots/weighting_functions.png
  plots/seven_molecule_errors.png

Run: python 06_domain_weighted_xgboost.py
"""
# ================= CONFIG =================
ART = "artifacts"
RES = "results_v3"
OUTDIR = f"{RES}/domain_weighting_experiment"
TARGET = "Tc (K)"
FP_RADIUS, FP_BITS = 2, 2048
SIM_HIGH, SIM_MED = 0.6, 0.4

XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=7, subsample=0.8,
                  colsample_bytree=0.7, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=42, n_jobs=-1, tree_method="hist", early_stopping_rounds=150)

W_EST_BASE = 0.2     # anchor: reproduces XGB-D's existing uniform weight at zero domain-relevance
W_MAX = 1.0          # anchor: "full experimental trust" ceiling

D2_P_CANDIDATES = [1.0, 2.0]          # exponent on similarity-to-refrigerant (linear / quadratic)
D3_BW_CANDIDATES = [1.5, 2.5]         # Gaussian bandwidth, in refrigerant-population SD units
D4_RADIUS_CANDIDATES = [1.5, 2.0]     # hard-inclusion radius, same SD units as D3

RECON_TOL = 1e-2   # K; D0/D1 must reproduce saved XGB-C/XGB-D test predictions within this
# ==========================================
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input: {path}. Run Scripts 01/02/05 first.")
    return path


def sim_bin(sim):
    if sim >= SIM_HIGH:
        return "HIGH"
    if sim >= SIM_MED:
        return "MEDIUM"
    return "LOW"


def metric_block(y, p):
    e = np.abs(y - p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(n=int(len(y)), R2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                RMSE=float(np.sqrt(np.mean((y - p) ** 2))), MAE=float(e.mean()))


def fit_xgb(Xtr, ytr, wtr, Xva, yva):
    m = XGBRegressor(**XGB_PARAMS)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    return m


def val_rmse(model, Xva, yva):
    p = model.predict(Xva)
    return float(np.sqrt(np.mean((yva - p) ** 2)))


def main():
    require(f"{ART}/benchmark_split_v3.csv")
    require(f"{ART}/est_pool.csv")
    require(f"{ART}/refrigerants.csv")
    require(f"{RES}/variant_test_predictions.csv")
    require(f"{RES}/refrigerant_domain_audit/04_test_molecule_refrigerant_likeness.csv")

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)

    bench = pd.read_csv(f"{ART}/benchmark_split_v3.csv")
    est = pd.read_csv(f"{ART}/est_pool.csv")
    refrig = pd.read_csv(f"{ART}/refrigerants.csv")
    X = np.load(f"{ART}/X_benchmark.npy")
    Xe = np.load(f"{ART}/X_est.npy")
    tb_bench = np.load(f"{ART}/tb_oof_benchmark.npy")
    tb_est = np.load(f"{ART}/tb_pred_est.npy")
    preds_saved = pd.read_csv(f"{RES}/variant_test_predictions.csv")
    test_likeness = pd.read_csv(f"{RES}/refrigerant_domain_audit/04_test_molecule_refrigerant_likeness.csv")

    # ---------- frozen-v3 identities (exactly as Script 02) ----------
    tr = bench.split.values == "train"
    va = bench.split.values == "val"
    te = bench.split.values == "test"
    has = bench[TARGET].notna().values
    m_tr, m_va, m_te = has & tr, has & va, has & te
    print(f"train={int(m_tr.sum())}  val={int(m_va.sum())}  test={int(m_te.sum())}")

    X_tb = np.hstack([X, tb_bench[:, None]])   # base feature stack for ALL of D0-D4 (= XGB-C/D stack)
    Xe_tb = np.hstack([Xe, tb_est[:, None]])

    Xtr_base, ytr_base = X_tb[m_tr], bench[TARGET].values[m_tr].astype(float)
    Xva, yva = X_tb[m_va], bench[TARGET].values[m_va].astype(float)
    Xte, yte = X_tb[m_te], bench[TARGET].values[m_te].astype(float)
    te_smiles = bench.loc[m_te, "canonical_smiles"].values

    ehas = est[TARGET].notna().values
    Xe_use = Xe_tb[ehas]
    ye_use = est[TARGET].values[ehas].astype(float)
    est_smiles = est.loc[ehas, "canonical_smiles"].reset_index(drop=True)
    print(f"estimated Tc-labeled rows available for weighting: {len(est_smiles)}")

    # =====================================================================
    # PHASE 1 — D0 and D1: reproduce XGB-C and XGB-D, gate on match
    # =====================================================================
    print("\n=== PHASE 1: D0 (== XGB-C) and D1 (== XGB-D) reproduction ===")
    d0_model = fit_xgb(Xtr_base, ytr_base, np.ones(len(ytr_base)), Xva, yva)
    d0_test_pred = d0_model.predict(Xte)

    Xtr_d1 = np.vstack([Xtr_base, Xe_use])
    ytr_d1 = np.concatenate([ytr_base, ye_use])
    wtr_d1 = np.concatenate([np.ones(len(ytr_base)), np.full(len(ye_use), W_EST_BASE)])
    d1_model = fit_xgb(Xtr_d1, ytr_d1, wtr_d1, Xva, yva)
    d1_test_pred = d1_model.predict(Xte)

    xgbc_saved = preds_saved[(preds_saved.variant == "XGB-C") & (preds_saved.target == TARGET)]
    xgbd_saved = preds_saved[(preds_saved.variant == "XGB-D") & (preds_saved.target == TARGET)]
    xgbc_saved = xgbc_saved.set_index("canonical_smiles").loc[te_smiles, "pred"].values
    xgbd_saved = xgbd_saved.set_index("canonical_smiles").loc[te_smiles, "pred"].values

    d0_diff = np.abs(d0_test_pred - xgbc_saved)
    d1_diff = np.abs(d1_test_pred - xgbd_saved)
    recon_df = pd.DataFrame({
        "canonical_smiles": te_smiles, "actual": yte,
        "XGB-C_saved": xgbc_saved, "D0_new": d0_test_pred, "D0_vs_XGB-C_absdiff": d0_diff,
        "XGB-D_saved": xgbd_saved, "D1_new": d1_test_pred, "D1_vs_XGB-D_absdiff": d1_diff,
    })
    recon_df.to_csv(f"{OUTDIR}/01_reproduction_check.csv", index=False)

    print(f"D0 vs XGB-C: max|diff|={d0_diff.max():.6f} K  mean|diff|={d0_diff.mean():.6f} K")
    print(f"D1 vs XGB-D: max|diff|={d1_diff.max():.6f} K  mean|diff|={d1_diff.mean():.6f} K")

    if d0_diff.max() > RECON_TOL or d1_diff.max() > RECON_TOL:
        raise RuntimeError(
            f"REPRODUCTION FAILED (tolerance={RECON_TOL} K). "
            f"D0 max diff={d0_diff.max():.4f} K, D1 max diff={d1_diff.max():.4f} K. "
            f"See {OUTDIR}/01_reproduction_check.csv for per-molecule diagnostics. "
            "Stopping before D2/D3/D4 as instructed — diagnose the pipeline mismatch first."
        )
    print("Reproduction PASSED — proceeding to D2/D3/D4.")

    # =====================================================================
    # PHASE 2 — domain metrics for the estimated pool and the 142 refrigerants
    # =====================================================================
    print("\n=== PHASE 2: domain metrics ===")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    refrig_smiles = refrig.canonical_smiles.tolist()
    refrig_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in refrig_smiles]

    def atom_counts(smiles):
        m = Chem.MolFromSmiles(smiles)
        syms = {}
        for a in m.GetAtoms():
            syms[a.GetSymbol()] = syms.get(a.GetSymbol(), 0) + 1
        C = syms.get("C", 0); F = syms.get("F", 0); Cl = syms.get("Cl", 0)
        heavy = m.GetNumHeavyAtoms()
        hal = F + Cl + syms.get("Br", 0) + syms.get("I", 0)
        return {
            "halogen_fraction": hal / heavy if heavy > 0 else 0.0,
            "F_to_C": F / C if C > 0 else 0.0,
        }

    refrig_comp = pd.DataFrame([atom_counts(s) for s in refrig_smiles])
    mu_hf, sd_hf = float(refrig_comp.halogen_fraction.mean()), float(refrig_comp.halogen_fraction.std())
    mu_fc, sd_fc = float(refrig_comp.F_to_C.mean()), float(refrig_comp.F_to_C.std())
    print(f"Refrigerant centroid: halogen_fraction={mu_hf:.4f}+-{sd_hf:.4f}  F_to_C={mu_fc:.4f}+-{sd_fc:.4f}")

    # per estimated-training-row: similarity to refrigerants (D2) + composition distance (D3/D4)
    est_sims = np.array([
        max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), refrig_fps))
        for s in est_smiles
    ])
    est_comp = pd.DataFrame([atom_counts(s) for s in est_smiles])
    z1 = (est_comp.halogen_fraction.values - mu_hf) / sd_hf
    z2 = (est_comp.F_to_C.values - mu_fc) / sd_fc
    est_dist = np.sqrt(z1 ** 2 + z2 ** 2)

    weights_df = pd.DataFrame({
        "canonical_smiles": est_smiles.values,
        "max_tanimoto_to_142refrigerants": est_sims.round(4),
        "halogen_fraction": est_comp.halogen_fraction.values.round(4),
        "F_to_C": est_comp.F_to_C.values.round(4),
        "composition_distance_to_refrig_centroid": est_dist.round(4),
    })

    print(f"est-pool similarity-to-refrigerant: median={np.median(est_sims):.4f}, "
         f"HIGH(>=0.6)={int((est_sims>=SIM_HIGH).sum())}, MEDIUM={int(((est_sims>=SIM_MED)&(est_sims<SIM_HIGH)).sum())}")
    print(f"est-pool composition_distance: median={np.median(est_dist):.4f}")
    for r in D4_RADIUS_CANDIDATES:
        print(f"  radius {r} SD -> n included = {int((est_dist <= r).sum())}")

    # ---------- validation-side refrigerant-likeness diagnostic (report only, never used to select) ----------
    val_smiles = bench.loc[m_va, "canonical_smiles"].tolist()
    val_sims = np.array([
        max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), refrig_fps))
        for s in val_smiles
    ])
    val_bins = pd.Series([sim_bin(s) for s in val_sims]).value_counts().reindex(
        ["LOW", "MEDIUM", "HIGH"]).fillna(0).astype(int)
    val_diag_df = pd.DataFrame({"canonical_smiles": val_smiles,
                                "max_tanimoto_to_142refrigerants": val_sims.round(4),
                                "refrig_likeness_bin": [sim_bin(s) for s in val_sims]})
    val_diag_df.to_csv(f"{OUTDIR}/04_val_refrigerant_likeness_diagnostic.csv", index=False)
    print(f"VAL refrigerant-likeness (diagnostic only, n={len(val_smiles)}): {val_bins.to_dict()}")

    # =====================================================================
    # PHASE 3 — D2/D3/D4 candidate weighting functions, selection by GLOBAL val RMSE only
    # =====================================================================
    print("\n=== PHASE 3: candidate weighting functions ===")

    def fit_weighted(w_est_vec):
        wtr = np.concatenate([np.ones(len(ytr_base)), w_est_vec])
        return fit_xgb(Xtr_d1, ytr_d1, wtr, Xva, yva)   # Xtr_d1/ytr_d1 = experimental + full est stack (order fixed)

    candidate_rows = []
    fitted = {}   # (variant, candidate_label) -> model

    # ---- D2: smooth similarity-based weight, w(s) = W_MIN + (W_MAX-W_MIN) * s^p ----
    for p in D2_P_CANDIDATES:
        w = W_EST_BASE + (W_MAX - W_EST_BASE) * (est_sims ** p)
        m = fit_weighted(w)
        rmse = val_rmse(m, Xva, yva)
        fitted[("D2", p)] = (m, w)
        candidate_rows.append({"variant": "D2", "candidate": f"p={p}", "global_val_RMSE": round(rmse, 4)})
        print(f"D2 p={p}: global val RMSE={rmse:.4f}")

    # ---- D3: smooth composition-distance Gaussian kernel, w(d) = W_MIN + (W_MAX-W_MIN)*exp(-d^2/(2*bw^2)) ----
    for bw in D3_BW_CANDIDATES:
        w = W_EST_BASE + (W_MAX - W_EST_BASE) * np.exp(-(est_dist ** 2) / (2 * bw ** 2))
        m = fit_weighted(w)
        rmse = val_rmse(m, Xva, yva)
        fitted[("D3", bw)] = (m, w)
        candidate_rows.append({"variant": "D3", "candidate": f"bw={bw}", "global_val_RMSE": round(rmse, 4)})
        print(f"D3 bw={bw}: global val RMSE={rmse:.4f}")

    # ---- D4: hard refrigerant-like subset (composition distance <= radius), weight=W_MAX; others weight=0 ----
    for r in D4_RADIUS_CANDIDATES:
        include = est_dist <= r
        w = np.where(include, W_MAX, 0.0)
        m = fit_weighted(w)
        rmse = val_rmse(m, Xva, yva)
        fitted[("D4", r)] = (m, w)
        candidate_rows.append({"variant": "D4", "candidate": f"radius={r}", "n_included": int(include.sum()),
                               "global_val_RMSE": round(rmse, 4)})
        print(f"D4 radius={r} (n={int(include.sum())}): global val RMSE={rmse:.4f}")

    cand_df = pd.DataFrame(candidate_rows)
    cand_df.to_csv(f"{OUTDIR}/03_candidate_selection_global_val_rmse.csv", index=False)

    winners = {}
    for variant, group_key in [("D2", D2_P_CANDIDATES), ("D3", D3_BW_CANDIDATES), ("D4", D4_RADIUS_CANDIDATES)]:
        keys = [(variant, k) for k in group_key]
        best_key = min(keys, key=lambda k: val_rmse(fitted[k][0], Xva, yva))
        winners[variant] = best_key
        print(f"WINNER {variant}: {best_key[1]} (lowest global val RMSE)")

    # attach winner weights to the auditable weights table
    weights_df["D2_weight"] = fitted[winners["D2"]][1].round(4)
    weights_df["D3_weight"] = fitted[winners["D3"]][1].round(4)
    weights_df["D4_weight"] = fitted[winners["D4"]][1].round(4)
    weights_df.to_csv(f"{OUTDIR}/05_est_pool_domain_weights.csv", index=False)

    # ---------- design document ----------
    design_lines = [
        "# Domain-Weighting Function Design (frozen BEFORE test-set evaluation)", "",
        "All hyperparameter candidates below were chosen from distributional inspection of the "
        "estimated pool and the 142 refrigerants (unlabeled structures/composition only), and "
        "selected using GLOBAL validation Tc RMSE (n=209) only. No frozen-test labels or the "
        "tiny refrigerant-like validation subgroup (16 MEDIUM, 0 HIGH of 209) were used for selection.",
        "", f"Anchors (fixed by construction, not searched): W_MIN={W_EST_BASE} (= existing XGB-D "
        f"uniform weight), W_MAX={W_MAX} (full experimental trust).", "",
        "## D2 — similarity-based smooth weight",
        "w_i = W_MIN + (W_MAX-W_MIN) * s_i^p,  s_i = max Morgan-Tanimoto(estimated molecule i, 142 refrigerants)",
        f"Candidates: p in {D2_P_CANDIDATES} (linear / quadratic).",
        f"Est-pool s_i distribution: median={np.median(est_sims):.4f}, "
        f"MEDIUM/HIGH count={int(((est_sims>=SIM_MED)).sum())}/{len(est_sims)}.", "",
        "## D3 — composition-domain distance, Gaussian kernel",
        "z1=(halogen_fraction-mu_hf)/sd_hf, z2=(F_to_C-mu_fc)/sd_fc  [mu,sd from the 142-refrigerant population]",
        "d_i = sqrt(z1^2+z2^2);  w_i = W_MIN + (W_MAX-W_MIN) * exp(-d_i^2 / (2*bw^2))",
        f"Refrigerant centroid: halogen_fraction={mu_hf:.4f}+-{sd_hf:.4f}, F_to_C={mu_fc:.4f}+-{sd_fc:.4f}.",
        "Cl_to_C excluded from the metric: refrigerant-vs-train standardized difference was only "
        "-0.25 sigma (see 05_refrigerant_domain_audit.py output 02) — not a differentiating axis for "
        "this population, unlike halogen_fraction (+1.93 sigma) and F_to_C (+1.84 sigma).",
        f"Candidates: bandwidth in {D3_BW_CANDIDATES} SD units.", "",
        "## D4 — hard refrigerant-like subset",
        "Same d_i as D3. Include estimated molecule i iff d_i <= radius; included rows get weight=W_MAX, "
        "excluded rows get weight=0 (equivalent to removal for this target's training pass).",
        f"Candidates: radius in {D4_RADIUS_CANDIDATES} SD units "
        f"(1.0 SD was inspected and rejected pre-registration: only 21 estimated molecules, judged too "
        f"small to train a stable subset).", "",
        "## Winning candidates (by global validation RMSE)",
    ]
    for variant in ["D2", "D3", "D4"]:
        design_lines.append(f"- {variant}: {winners[variant][1]}")
    open(f"{OUTDIR}/02_weighting_function_design.md", "w", encoding="utf-8").write("\n".join(design_lines))

    # ---------- weighting-function plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    s_grid = np.linspace(0, 1, 200)
    for p in D2_P_CANDIDATES:
        axes[0].plot(s_grid, W_EST_BASE + (W_MAX - W_EST_BASE) * s_grid ** p,
                    label=f"p={p}" + (" (WINNER)" if p == winners["D2"][1] else ""))
    axes[0].set_xlabel("max Tanimoto to 142 refrigerants"); axes[0].set_ylabel("estimated-row weight")
    axes[0].set_title("D2: similarity-based weight"); axes[0].legend()
    d_grid = np.linspace(0, 6, 200)
    for bw in D3_BW_CANDIDATES:
        axes[1].plot(d_grid, W_EST_BASE + (W_MAX - W_EST_BASE) * np.exp(-(d_grid ** 2) / (2 * bw ** 2)),
                    label=f"bw={bw}" + (" (WINNER)" if bw == winners["D3"][1] else ""))
    for r in D4_RADIUS_CANDIDATES:
        axes[1].axvline(r, ls="--", color="gray", alpha=0.6)
    axes[1].set_xlabel("composition distance to refrigerant centroid (SD units)")
    axes[1].set_ylabel("estimated-row weight"); axes[1].set_title("D3 kernel (D4 radii dashed)"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plots/weighting_functions.png", dpi=150)
    plt.close()

    # =====================================================================
    # PHASE 4 — freeze test-set evaluation (touch test exactly once per D-variant)
    # =====================================================================
    print("\n=== PHASE 4: single frozen-test evaluation ===")
    final_models = {
        "D0": d0_model, "D1": d1_model,
        "D2": fitted[winners["D2"]][0], "D3": fitted[winners["D3"]][0], "D4": fitted[winners["D4"]][0],
    }
    test_preds = {v: m.predict(Xte) for v, m in final_models.items()}

    # global metrics
    global_rows = []
    for v, p in test_preds.items():
        mb = metric_block(yte, p)
        global_rows.append({"variant": v, **{k: round(x, 4) if isinstance(x, float) else x for k, x in mb.items()}})
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(f"{OUTDIR}/06_test_metrics_by_variant.csv", index=False)
    print(global_df.to_string(index=False))

    # by refrigerant-likeness bin (reuse Script05's exact test-molecule bin assignments)
    likeness_lu = test_likeness.set_index("canonical_smiles")["refrig_likeness_bin"]
    te_bins = pd.Series(te_smiles).map(likeness_lu).values

    bin_rows = []
    for v, p in test_preds.items():
        for b in ["LOW", "MEDIUM", "HIGH", "MEDIUM+HIGH", "ALL"]:
            if b == "ALL":
                mask = np.ones(len(yte), dtype=bool)
            elif b == "MEDIUM+HIGH":
                mask = np.isin(te_bins, ["MEDIUM", "HIGH"])
            else:
                mask = te_bins == b
            if mask.sum() == 0:
                continue
            mb = metric_block(yte[mask], p[mask])
            bin_rows.append({"variant": v, "refrig_likeness_bin": b,
                             **{k: round(x, 4) if isinstance(x, float) else x for k, x in mb.items()}})
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(f"{OUTDIR}/07_test_metrics_by_refrigerant_likeness_bin.csv", index=False)

    # the same 7 refrigerant-like molecules used in Script 05
    seven_smiles = test_likeness[test_likeness.refrig_likeness_bin.isin(["MEDIUM", "HIGH"])].canonical_smiles.tolist()
    idx = {s: i for i, s in enumerate(te_smiles)}
    seven_rows = []
    for s in seven_smiles:
        i = idx[s]
        row = {"canonical_smiles": s, "actual_Tc": round(float(yte[i]), 3)}
        for v in ["D0", "D1", "D2", "D3", "D4"]:
            err = abs(test_preds[v][i] - yte[i])
            row[f"{v}_pred"] = round(float(test_preds[v][i]), 3)
            row[f"{v}_abs_error"] = round(float(err), 3)
        for v in ["D2", "D3", "D4"]:
            row[f"{v}_vs_D0"] = "improved" if row[f"{v}_abs_error"] < row["D0_abs_error"] else (
                "worsened" if row[f"{v}_abs_error"] > row["D0_abs_error"] else "tied")
            row[f"{v}_vs_D1"] = "improved" if row[f"{v}_abs_error"] < row["D1_abs_error"] else (
                "worsened" if row[f"{v}_abs_error"] > row["D1_abs_error"] else "tied")
        seven_rows.append(row)
    seven_df = pd.DataFrame(seven_rows)
    seven_df.to_csv(f"{OUTDIR}/08_seven_refrigerant_like_molecules.csv", index=False)

    # ---------- seven-molecule plot ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.15
    xpos = np.arange(len(seven_smiles))
    for j, v in enumerate(["D0", "D1", "D2", "D3", "D4"]):
        ax.bar(xpos + j * width, seven_df[f"{v}_abs_error"], width=width, label=v)
    ax.set_xticks(xpos + 2 * width)
    ax.set_xticklabels([s[:14] for s in seven_smiles], rotation=30, ha="right")
    ax.set_ylabel("|error| (K)")
    ax.set_title("The 7 refrigerant-like frozen-test molecules (from Script 05): D0-D4 comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plots/seven_molecule_errors.png", dpi=150)
    plt.close()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    def g(v, b):
        r = bin_df[(bin_df.variant == v) & (bin_df.refrig_likeness_bin == b)]
        return r.iloc[0] if len(r) else None

    summary = [
        "# Domain-Weighted Tc Estimated-Data Ablation — Summary (Frozen Protocol v3)", "",
        "Read-only-in-spirit controlled experiment: frozen split, frozen features, frozen predicted-Tb "
        "methodology all reused unchanged. Only Tc estimated-data sample weights differ across D0-D4. "
        "No file under artifacts/ or the existing results_v3/ (outside this new directory) was modified.",
        "", "**Reproduction gate:** "
        f"D0 vs saved XGB-C max|diff|={d0_diff.max():.6f} K (tolerance {RECON_TOL} K) — PASSED. "
        f"D1 vs saved XGB-D max|diff|={d1_diff.max():.6f} K — PASSED.", "",
        "**Selection discipline:** D2/D3/D4 winners chosen by global validation RMSE only "
        "(refrigerant-like validation subgroup: 16 MEDIUM, 0 HIGH of 209 — reported, not used for selection).",
        "", "## Winning configurations", "",
    ]
    for variant in ["D2", "D3", "D4"]:
        summary.append(f"- {variant}: {winners[variant][1]} "
                       f"(global val RMSE={val_rmse(fitted[winners[variant]][0], Xva, yva):.4f} K)")
    summary += ["", "## Global Tc test metrics (n=209)", "", global_df.to_string(index=False), "",
               "## Refrigerant-likeness bin sizes on the frozen test set (same bins as Script 05)", ""]
    for b in ["LOW", "MEDIUM", "HIGH", "MEDIUM+HIGH"]:
        row = g("D0", b)
        if row is not None:
            summary.append(f"- {b}: n={int(row.n)}")
    summary += ["", "## RMSE by variant x refrigerant-likeness bin", "",
               bin_df.pivot(index="variant", columns="refrig_likeness_bin", values="RMSE")
               .reindex(["D0", "D1", "D2", "D3", "D4"])[["LOW", "MEDIUM", "HIGH", "MEDIUM+HIGH", "ALL"]]
               .round(3).to_string(), "",
               "## The 7 refrigerant-like frozen-test molecules — improved/worsened vs D0 and D1 "
               "(descriptive counts only, no significance claims)", ""]
    for v in ["D2", "D3", "D4"]:
        vs_d0 = seven_df[f"{v}_vs_D0"].value_counts().to_dict()
        vs_d1 = seven_df[f"{v}_vs_D1"].value_counts().to_dict()
        summary.append(f"- {v} vs D0: {vs_d0}  |  {v} vs D1: {vs_d1}")
    summary += ["", "Full molecule-level table: `08_seven_refrigerant_like_molecules.csv`", "",
               "_See `plots/weighting_functions.png` and `plots/seven_molecule_errors.png`._"]
    open(f"{OUTDIR}/00_SUMMARY.md", "w", encoding="utf-8").write("\n".join(str(s) for s in summary))

    print(f"\nAll outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
