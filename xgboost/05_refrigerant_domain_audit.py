"""
05_refrigerant_domain_audit.py — Refrigerant-domain relevance audit for Frozen Protocol v3.
READ-ONLY. Consumes SAVED artifacts and prediction CSVs only.
NO retraining, resplitting, retuning, or regeneration/modification of predictions.

Purpose: determine whether the generic Tc failure patterns found in 04_tc_outlier_audit.py
(glycol-ether / branched-alkane homologous-series extrapolation) are actually relevant to
the locked 142-refrigerant deployment domain, or whether refrigerant-like chemistry fails
(or succeeds) for different reasons.

Two independent similarity directions are computed, both with the same Morgan r2/2048
protocol used throughout the pipeline (Scripts 01/03/04):
  (1) refrigerant -> Tc-labeled experimental TRAIN pool similarity
      (are the 142 refrigerants themselves well covered by Tc training data?)
  (2) Tc TEST molecule -> 142-refrigerant-set similarity ("refrigerant-likeness")
      (of the frozen Tc test molecules, which ones structurally resemble refrigerants,
      and how do XGB-A/B/C/D actually perform on exactly those?)

A third, purely descriptive step self-clusters the 142 refrigerants (Butina, same cutoff
as Script 01's frozen split) to check whether LOW/MEDIUM-similarity refrigerants are
scattered singletons or recurring structural families.

Residual sign convention (matches 03/04): residual = actual - predicted.
Descriptive only — single frozen split, no causal or statistical-significance claims.

Inputs (must already exist — run Scripts 01/02 first):
  artifacts/benchmark_split_v3.csv
  artifacts/refrigerants.csv
  results_v3/variant_test_predictions.csv

Outputs: results_v3/refrigerant_domain_audit/
  00_SUMMARY.md
  01_refrigerant_descriptors_similarity.csv     (142 refrigerants: sim-to-train, descriptors)
  02_refrigerant_vs_train_descriptor_comparison.csv
  03_refrigerant_self_clusters.csv              (Butina clusters among the 142 refrigerants)
  04_test_molecule_refrigerant_likeness.csv     (209 Tc test molecules: sim-to-142-set)
  05_variant_metrics_by_refrigerant_likeness.csv
  06_AtoC_CtoD_by_refrigerant_likeness.csv
  plots/refrigerant_domain_diagnostic.png

Run: python 05_refrigerant_domain_audit.py
"""
# ================= CONFIG =================
ART = "artifacts"
RES = "results_v3"
OUTDIR = f"{RES}/refrigerant_domain_audit"
TARGET = "Tc (K)"
VARIANTS = ["XGB-A", "XGB-B", "XGB-C", "XGB-D"]
SIM_HIGH, SIM_MED = 0.6, 0.4          # LOW / MEDIUM / HIGH similarity bins (pipeline convention)
TIE_DELTA = 0.02                       # near-neighbor tie threshold (matches Script 03/04)
FP_RADIUS, FP_BITS = 2, 2048           # Morgan protocol (Scripts 01/03/04)
REFRIG_CLUSTER_CUTOFF = 0.60           # Butina cutoff, same value as Script 01's frozen split
HALOGENS = {"F", "Cl", "Br", "I"}
# ==========================================
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors
from rdkit.ML.Cluster import Butina
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input: {path}. Run Scripts 01/02 first.")
    return path


def sim_bin(sim):
    if sim >= SIM_HIGH:
        return "HIGH"
    if sim >= SIM_MED:
        return "MEDIUM"
    return "LOW"


def similarity_profile(fp, ref_fps, ref_ids):
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
    order = np.argsort(sims)[::-1]
    top1, id1 = float(sims[order[0]]), ref_ids[order[0]]
    top2 = float(sims[order[1]]) if len(sims) > 1 else np.nan
    tie = bool(abs(top1 - top2) <= TIE_DELTA) if len(sims) > 1 else False
    return top1, id1, top2, tie


def descriptors(smiles):
    m = Chem.MolFromSmiles(smiles)
    sym_counts = {}
    for a in m.GetAtoms():
        s = a.GetSymbol()
        sym_counts[s] = sym_counts.get(s, 0) + 1
    C = sym_counts.get("C", 0)
    F = sym_counts.get("F", 0)
    Cl = sym_counts.get("Cl", 0)
    Br = sym_counts.get("Br", 0)
    I = sym_counts.get("I", 0)
    O = sym_counts.get("O", 0)
    N = sym_counts.get("N", 0)
    S = sym_counts.get("S", 0)
    halogens = F + Cl + Br + I
    heavy = m.GetNumHeavyAtoms()
    return {
        "MolWt": round(float(Descriptors.MolWt(m)), 2),
        "C_count": C, "F_count": F, "Cl_count": Cl, "Br_count": Br, "I_count": I,
        "O_count": O, "N_count": N, "S_count": S,
        "halogen_count": halogens,
        "halogen_fraction": round(halogens / heavy, 4) if heavy > 0 else np.nan,
        "F_to_C_ratio": round(F / C, 4) if C > 0 else np.nan,
        "Cl_to_C_ratio": round(Cl / C, 4) if C > 0 else np.nan,
        "heavy_atoms": int(heavy),
        "num_rings": int(rdMolDescriptors.CalcNumRings(m)),
        "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(m)),
        "num_rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(m)),
        "TPSA": round(float(Descriptors.TPSA(m)), 3),
        "logP": round(float(Descriptors.MolLogP(m)), 3),
        "MolMR": round(float(Descriptors.MolMR(m)), 3),
        "fraction_Csp3": round(float(rdMolDescriptors.CalcFractionCSP3(m)), 4),
    }


def butina_clusters(fps, cutoff):
    n = len(fps)
    dists = []
    for i in range(1, n):
        dists.extend(1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    cid = np.zeros(n, int)
    for c, members in enumerate(clusters):
        for i in members:
            cid[i] = c
    return cid, clusters


def descriptor_comparison_table(refrig_df, train_df, desc_cols):
    rows = []
    for col in desc_cols:
        r = refrig_df[col].astype(float)
        t = train_df[col].astype(float)
        pooled_std = np.sqrt((r.var(ddof=1) + t.var(ddof=1)) / 2)
        std_diff = (r.mean() - t.mean()) / pooled_std if pooled_std > 0 else np.nan
        rows.append({
            "descriptor": col,
            "refrig_mean": round(float(r.mean()), 4), "refrig_median": round(float(r.median()), 4),
            "refrig_std": round(float(r.std()), 4), "refrig_min": round(float(r.min()), 4),
            "refrig_max": round(float(r.max()), 4),
            "train_mean": round(float(t.mean()), 4), "train_median": round(float(t.median()), 4),
            "train_std": round(float(t.std()), 4), "train_min": round(float(t.min()), 4),
            "train_max": round(float(t.max()), 4),
            "standardized_mean_diff": round(float(std_diff), 4) if np.isfinite(std_diff) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    require(f"{ART}/benchmark_split_v3.csv")
    require(f"{ART}/refrigerants.csv")
    require(f"{RES}/variant_test_predictions.csv")

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)

    bench = pd.read_csv(f"{ART}/benchmark_split_v3.csv")
    refrig = pd.read_csv(f"{ART}/refrigerants.csv")
    preds = pd.read_csv(f"{RES}/variant_test_predictions.csv")

    if refrig.canonical_smiles.nunique() != len(refrig) or len(refrig) != 142:
        raise ValueError(f"Expected 142 unique refrigerant molecules, found "
                         f"{len(refrig)} rows / {refrig.canonical_smiles.nunique()} unique.")

    preds_tc = preds[preds.target == TARGET].copy()
    if set(preds_tc.variant.unique()) != set(VARIANTS):
        raise ValueError(f"variant_test_predictions.csv variants mismatch: {preds_tc.variant.unique()}")
    preds_tc["residual"] = preds_tc.actual - preds_tc.pred
    preds_tc["abs_error"] = preds_tc.residual.abs()
    n_per_variant = preds_tc.groupby("variant").size()
    if n_per_variant.nunique() != 1:
        raise ValueError(f"Unequal Tc test-set sizes across variants: {n_per_variant.to_dict()}")
    n_test_tc = int(n_per_variant.iloc[0])
    print(f"Tc test set: {n_test_tc} molecules x {len(VARIANTS)} variants; refrigerants: {len(refrig)}")

    # ---------- fingerprints ----------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    refrig_smiles = refrig.canonical_smiles.tolist()
    refrig_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in refrig_smiles]

    tc_train_mask = (bench.split == "train") & bench[TARGET].notna()
    tc_train_smiles = bench.loc[tc_train_mask, "canonical_smiles"].tolist()
    tc_train_actual = dict(zip(bench.loc[tc_train_mask, "canonical_smiles"], bench.loc[tc_train_mask, TARGET]))
    tc_train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tc_train_smiles]
    print(f"Tc-labeled train pool: {len(tc_train_smiles)}")

    test_smiles = preds_tc.drop_duplicates("canonical_smiles").canonical_smiles.tolist()
    test_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in test_smiles]

    # =====================================================================
    # PART A — the 142 refrigerants vs the Tc-labeled experimental TRAIN pool
    # =====================================================================
    desc_cols = ["MolWt", "C_count", "F_count", "Cl_count", "Br_count", "I_count", "O_count",
                "N_count", "S_count", "halogen_count", "halogen_fraction", "F_to_C_ratio",
                "Cl_to_C_ratio", "heavy_atoms", "num_rings", "num_aromatic_rings",
                "num_rotatable_bonds", "TPSA", "logP", "MolMR", "fraction_Csp3"]

    refrig_rows = []
    for i, cs in enumerate(refrig_smiles):
        top1, id1, top2, tie = similarity_profile(refrig_fps[i], tc_train_fps, tc_train_smiles)
        row = {
            "canonical_smiles": cs,
            "IUPAC_Name": refrig.IUPAC_Name.iloc[i] if "IUPAC_Name" in refrig.columns else "",
            "max_tanimoto_to_tc_train": round(top1, 4),
            "sim_bin": sim_bin(top1),
            "nearest_tc_train_smiles": id1,
            "nearest_tc_train_actual_Tc": round(float(tc_train_actual[id1]), 3),
            "second_max_tanimoto": round(top2, 4) if np.isfinite(top2) else np.nan,
            "near_neighbor_tie": tie,
        }
        row.update(descriptors(cs))
        refrig_rows.append(row)
    refrig_df = pd.DataFrame(refrig_rows)
    refrig_df.to_csv(f"{OUTDIR}/01_refrigerant_descriptors_similarity.csv", index=False)

    sim_counts = refrig_df.sim_bin.value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).astype(int)
    print(f"Refrigerant similarity-to-Tc-train bins: {sim_counts.to_dict()}")

    # descriptor comparison: 142 refrigerants vs Tc-train pool
    train_desc_rows = [descriptors(cs) for cs in tc_train_smiles]
    train_desc_df = pd.DataFrame(train_desc_rows)
    comp_df = descriptor_comparison_table(refrig_df, train_desc_df, desc_cols)
    comp_df.to_csv(f"{OUTDIR}/02_refrigerant_vs_train_descriptor_comparison.csv", index=False)

    # self-clustering of the 142 refrigerants (descriptive only; does not touch the frozen split)
    refrig_cid, refrig_clusters_raw = butina_clusters(refrig_fps, REFRIG_CLUSTER_CUTOFF)
    refrig_df["refrig_cluster_id"] = refrig_cid
    clu_rows = []
    for c, members in enumerate(refrig_clusters_raw):
        sub = refrig_df.iloc[list(members)]
        clu_rows.append({
            "refrig_cluster_id": c,
            "n_members": len(members),
            "mean_sim_to_tc_train": round(float(sub.max_tanimoto_to_tc_train.mean()), 4),
            "min_sim_to_tc_train": round(float(sub.max_tanimoto_to_tc_train.min()), 4),
            "n_LOW": int((sub.sim_bin == "LOW").sum()),
            "n_MEDIUM": int((sub.sim_bin == "MEDIUM").sum()),
            "n_HIGH": int((sub.sim_bin == "HIGH").sum()),
            "mean_halogen_count": round(float(sub.halogen_count.mean()), 2),
            "member_smiles": ";".join(sub.canonical_smiles.tolist()),
            "member_names": ";".join(sub.IUPAC_Name.astype(str).tolist()),
        })
    refrig_cluster_df = pd.DataFrame(clu_rows).sort_values(["n_members", "mean_sim_to_tc_train"],
                                                            ascending=[False, True])
    refrig_cluster_df.to_csv(f"{OUTDIR}/03_refrigerant_self_clusters.csv", index=False)
    recurring = refrig_cluster_df[refrig_cluster_df.n_members >= 3]
    print(f"Refrigerant self-clusters with >=3 members: {len(recurring)} "
         f"(covering {int(recurring.n_members.sum())}/142 refrigerants)")

    # =====================================================================
    # PART B — Tc TEST molecules vs the 142-refrigerant set ("refrigerant-likeness")
    # =====================================================================
    test_rows = []
    for i, cs in enumerate(test_smiles):
        top1, id1, top2, tie = similarity_profile(test_fps[i], refrig_fps, refrig_smiles)
        test_rows.append({
            "canonical_smiles": cs,
            "max_tanimoto_to_142refrigerants": round(top1, 4),
            "refrig_likeness_bin": sim_bin(top1),
            "nearest_refrigerant_smiles": id1,
            "second_max_tanimoto": round(top2, 4) if np.isfinite(top2) else np.nan,
            "near_neighbor_tie": tie,
        })
    test_map_df = pd.DataFrame(test_rows)
    test_map_df.to_csv(f"{OUTDIR}/04_test_molecule_refrigerant_likeness.csv", index=False)

    likeness_counts = test_map_df.refrig_likeness_bin.value_counts().reindex(
        ["LOW", "MEDIUM", "HIGH"]).fillna(0).astype(int)
    print(f"Tc test-molecule refrigerant-likeness bins: {likeness_counts.to_dict()}")

    preds_tc = preds_tc.merge(test_map_df[["canonical_smiles", "max_tanimoto_to_142refrigerants",
                                           "refrig_likeness_bin"]], on="canonical_smiles", how="left")

    # ---------- variant metrics by refrigerant-likeness bin ----------
    def metric_block(y, p):
        e = np.abs(y - p)
        return dict(n=int(len(y)), RMSE=float(np.sqrt(np.mean((y - p) ** 2))), MAE=float(e.mean()))

    metric_rows = []
    bins_for_metrics = ["LOW", "MEDIUM", "HIGH", "MEDIUM+HIGH", "ALL"]
    for v in VARIANTS:
        dv = preds_tc[preds_tc.variant == v]
        for b in bins_for_metrics:
            if b == "ALL":
                d = dv
            elif b == "MEDIUM+HIGH":
                d = dv[dv.refrig_likeness_bin.isin(["MEDIUM", "HIGH"])]
            else:
                d = dv[dv.refrig_likeness_bin == b]
            if len(d) == 0:
                continue
            mb = metric_block(d.actual.values, d.pred.values)
            metric_rows.append({"variant": v, "refrig_likeness_bin": b, **{k: round(x, 4) if isinstance(x, float) else x
                                                                            for k, x in mb.items()}})
    metrics_by_bin = pd.DataFrame(metric_rows)
    metrics_by_bin.to_csv(f"{OUTDIR}/05_variant_metrics_by_refrigerant_likeness.csv", index=False)

    # ---------- A->C and C->D within refrigerant-likeness bins ----------
    wide_err = preds_tc.pivot(index="canonical_smiles", columns="variant", values="abs_error")
    wide_err = wide_err.join(test_map_df.set_index("canonical_smiles")["refrig_likeness_bin"])
    wide_err["AtoC_delta"] = wide_err["XGB-A"] - wide_err["XGB-C"]
    wide_err["CtoD_delta"] = wide_err["XGB-C"] - wide_err["XGB-D"]

    contrast_rows = []
    for b in ["LOW", "MEDIUM", "HIGH", "MEDIUM+HIGH", "ALL"]:
        if b == "ALL":
            d = wide_err
        elif b == "MEDIUM+HIGH":
            d = wide_err[wide_err.refrig_likeness_bin.isin(["MEDIUM", "HIGH"])]
        else:
            d = wide_err[wide_err.refrig_likeness_bin == b]
        if len(d) == 0:
            continue
        for contrast, col in [("A->C", "AtoC_delta"), ("C->D", "CtoD_delta")]:
            dd = d[col]
            contrast_rows.append({
                "refrig_likeness_bin": b, "contrast": contrast, "n": int(len(dd)),
                "median_delta": round(float(dd.median()), 4), "mean_delta": round(float(dd.mean()), 4),
                "n_improved": int((dd > 0).sum()), "n_worsened": int((dd < 0).sum()),
                "n_tied": int((dd == 0).sum()),
            })
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(f"{OUTDIR}/06_AtoC_CtoD_by_refrigerant_likeness.csv", index=False)

    # ---------- diagnostic plot ----------
    color_map = {"LOW": "tab:red", "MEDIUM": "tab:orange", "HIGH": "tab:green"}
    d_plot = preds_tc[preds_tc.variant == "XGB-D"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for b, c in color_map.items():
        d = d_plot[d_plot.refrig_likeness_bin == b]
        ax.scatter(d.max_tanimoto_to_142refrigerants, d.abs_error, s=22, alpha=0.7, color=c,
                  label=f"{b} (n={len(d)})")
    ax.axvline(SIM_MED, color="k", ls="--", lw=0.8)
    ax.axvline(SIM_HIGH, color="k", ls="--", lw=0.8)
    ax.set_xlabel("max Tanimoto: Tc test molecule -> nearest of 142 refrigerants")
    ax.set_ylabel("XGB-D |error| (K)")
    ax.set_title("Tc test-set error vs refrigerant-likeness")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plots/refrigerant_domain_diagnostic.png", dpi=150)
    plt.close()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    global_xgbd_rmse = metrics_by_bin[(metrics_by_bin.variant == "XGB-D") &
                                      (metrics_by_bin.refrig_likeness_bin == "ALL")].RMSE.iloc[0]
    high_xgbd = metrics_by_bin[(metrics_by_bin.variant == "XGB-D") &
                               (metrics_by_bin.refrig_likeness_bin == "HIGH")]
    medhigh_xgbd = metrics_by_bin[(metrics_by_bin.variant == "XGB-D") &
                                  (metrics_by_bin.refrig_likeness_bin == "MEDIUM+HIGH")]

    lines = [
        "# Refrigerant-Domain Audit — Summary (Frozen Protocol v3)", "",
        "Read-only diagnostic stage. No retraining, resplitting, retuning, or prediction "
        "modification. Fingerprints and descriptors recomputed from saved canonical SMILES "
        "(same Morgan r2/2048 protocol as Scripts 01/03/04) purely for similarity diagnostics.",
        "", "**Single frozen split.** Descriptive only — no causal or significance claims.", "",
        "## Part A — the 142 refrigerants vs the Tc-labeled experimental train pool "
        f"(n={len(tc_train_smiles)})",
        f"- LOW similarity (<0.4): {sim_counts['LOW']} / 142",
        f"- MEDIUM similarity (0.4-0.6): {sim_counts['MEDIUM']} / 142",
        f"- HIGH similarity (>=0.6): {sim_counts['HIGH']} / 142",
        f"- Recurring refrigerant self-clusters (>=3 members, Butina cutoff "
        f"{REFRIG_CLUSTER_CUTOFF}): {len(recurring)} clusters covering "
        f"{int(recurring.n_members.sum()) if len(recurring) else 0}/142 refrigerants "
        "— see `03_refrigerant_self_clusters.csv`.",
        "",
    ]
    if len(recurring):
        lines.append("Largest recurring refrigerant families (by member count):")
        for _, r in recurring.head(6).iterrows():
            lines.append(f"- cluster {int(r.refrig_cluster_id)}: n={int(r.n_members)}, "
                        f"mean sim to Tc-train={r.mean_sim_to_tc_train}, "
                        f"LOW/MED/HIGH={int(r.n_LOW)}/{int(r.n_MEDIUM)}/{int(r.n_HIGH)}, "
                        f"mean halogen count={r.mean_halogen_count}")
        lines.append("")

    lines += ["## Descriptor contrasts (142 refrigerants vs Tc-train pool, |standardized mean diff| >= 0.5)", ""]
    big_diff = comp_df.reindex(comp_df.standardized_mean_diff.abs().sort_values(ascending=False).index)
    big_diff = big_diff[big_diff.standardized_mean_diff.abs() >= 0.5]
    for _, r in big_diff.iterrows():
        lines.append(f"- **{r.descriptor}**: refrigerants mean={r.refrig_mean} vs train mean={r.train_mean} "
                    f"(standardized diff={r.standardized_mean_diff:+}) — full table in "
                    "`02_refrigerant_vs_train_descriptor_comparison.csv`")
    lines += ["", "## Part B — Tc TEST molecules mapped onto the refrigerant domain "
             f"(n={n_test_tc})",
             f"- LOW refrigerant-likeness (<0.4): {likeness_counts['LOW']} / {n_test_tc}",
             f"- MEDIUM refrigerant-likeness (0.4-0.6): {likeness_counts['MEDIUM']} / {n_test_tc}",
             f"- HIGH refrigerant-likeness (>=0.6): {likeness_counts['HIGH']} / {n_test_tc}",
             "", "## XGB-D Tc RMSE: global vs refrigerant-like subsets", "",
             f"- Global (n={int(metrics_by_bin[(metrics_by_bin.variant=='XGB-D') & (metrics_by_bin.refrig_likeness_bin=='ALL')].n.iloc[0])}): "
             f"RMSE={global_xgbd_rmse} K",
             ]
    if len(high_xgbd):
        lines.append(f"- HIGH refrigerant-likeness (n={int(high_xgbd.n.iloc[0])}): RMSE={high_xgbd.RMSE.iloc[0]} K")
    if len(medhigh_xgbd):
        lines.append(f"- MEDIUM+HIGH refrigerant-likeness (n={int(medhigh_xgbd.n.iloc[0])}): "
                    f"RMSE={medhigh_xgbd.RMSE.iloc[0]} K")
    lines += ["", "Full per-variant, per-bin RMSE/MAE/n table: `05_variant_metrics_by_refrigerant_likeness.csv`",
             "", "## A->C / C->D by refrigerant-likeness bin", "",
             "Full table: `06_AtoC_CtoD_by_refrigerant_likeness.csv` "
             "(positive delta = later variant reduces |error|)", ""]
    for _, r in contrast_df.iterrows():
        lines.append(f"- {r.refrig_likeness_bin} / {r.contrast}: n={r.n}, median={r.median_delta:+}, "
                    f"mean={r.mean_delta:+}, improved={r.n_improved}, worsened={r.n_worsened}")
    lines += ["", "_See `plots/refrigerant_domain_diagnostic.png` for XGB-D |error| vs refrigerant-likeness._"]
    open(f"{OUTDIR}/00_SUMMARY.md", "w", encoding="utf-8").write("\n".join(str(s) for s in lines))

    print(f"\nGlobal XGB-D Tc RMSE: {global_xgbd_rmse} K")
    if len(high_xgbd):
        print(f"HIGH refrigerant-likeness XGB-D Tc RMSE: {high_xgbd.RMSE.iloc[0]} K (n={int(high_xgbd.n.iloc[0])})")
    print(f"\nAll outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
