"""
04_tc_outlier_audit.py — Tc failure-diagnosis audit for XGB-A/B/C/D on Frozen Protocol v3.
READ-ONLY diagnostic stage. Consumes SAVED artifacts and prediction CSVs only.
NO retraining, resplitting, retuning, or regeneration of predictions.

Scope: Tc (K) test-set errors only, across all four variants (A/B/C/D), so failures can
be classified as SYSTEMATIC (elevated in most/all variants -> likely a data/chemistry
issue) vs MODEL-SENSITIVE (elevated in only one or two variants -> feature-dependent).

Cohorts audited:
  primary   = top 50 Tc test molecules by |error| under XGB-D
  secondary = union of the top 20 Tc test molecules by |error| under EACH of A/B/C/D
  master    = primary UNION secondary (every molecule diagnosed gets every column below)
  focused   = the top 20 of primary (by XGB-D |error|) -> narrative report

"Elevated" (for cross-variant consistency) means a molecule's |error| under a given
variant exceeds that variant's OWN P75 of |error| over its full Tc test set (n=209).
This P75 is a per-variant classification threshold only; it does NOT define which
molecules are audited (that's the top-N cohorts above). P90/P95 are also reported,
for context only, per the analysis_v2 percentile convention.

Diagnostic angles (all descriptive; single frozen split; no causal claims):
  1. Cross-variant consistency  — SYSTEMATIC / MODEL-SENSITIVE / NONE (see above)
  2. Domain of applicability    — max Tanimoto to ALL train molecules (feature-domain,
                                   same convention as 03a/03_analyze_variants: HIGH>=0.6,
                                   MEDIUM>=0.4, LOW<0.4)
  3. Near-neighbor label check  — max Tanimoto to the Tc-LABELED train pool specifically
                                   (the actual pool XGB-* were trained on for Tc), the
                                   nearest such neighbor's actual Tc, and the gap
                                   |actual - nearest_train_actual_Tc|. A close structural
                                   neighbor with a large label gap flags a possible
                                   data/labeling issue rather than a model failure.
  4. Cluster + substructure     — cluster_id hotspot aggregation (are whole Butina
                                   clusters hard, not just isolated points) and simple
                                   RDKit descriptive flags (MW, ring counts, halogen /
                                   heteroatom counts) for reporting only.

Residual sign convention (matches 03_analyze_variants.py): residual = actual - predicted.
  positive residual => model UNDER-predicts.

Inputs (must already exist — run Scripts 01/02 first):
  artifacts/benchmark_split_v3.csv
  results_v3/variant_test_predictions.csv

Outputs: results_v3/tc_outlier_audit/
  00_SUMMARY.md
  01_context_error_percentiles.csv
  02_master_outlier_table.csv
  03_top20_focused_report.md
  04_cluster_hotspots.csv
  plots/tc_outlier_diagnostic.png

Run: python 04_tc_outlier_audit.py
"""
# ================= CONFIG =================
ART = "artifacts"
RES = "results_v3"
OUTDIR = f"{RES}/tc_outlier_audit"
TARGET = "Tc (K)"
VARIANTS = ["XGB-A", "XGB-B", "XGB-C", "XGB-D"]
PRIMARY_VARIANT = "XGB-D"
PRIMARY_N = 50            # top-N Tc test molecules by |error| under XGB-D
SECONDARY_N = 20          # top-N per variant unioned into the secondary cohort
FOCUSED_N = 20             # top-N of the primary cohort -> narrative report
ELEVATED_PCTL = 75         # per-variant percentile defining "elevated" |error|
CONTEXT_PCTLS = [50, 75, 90, 95, 99]
SIM_HIGH, SIM_MED = 0.6, 0.4     # domain-of-applicability trust bins (pipeline convention)
LABEL_GAP_SIM_FLAG = 0.6         # "close neighbor" threshold for the label-gap flag
FP_RADIUS, FP_BITS = 2, 2048     # same fingerprint protocol as Scripts 01/03
HALOGENS = {"F", "Cl", "Br", "I"}
# ==========================================
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required input: {path}. Run Scripts 01 and 02 before this audit."
        )
    return path


def doa_bin(sim):
    if sim >= SIM_HIGH:
        return "HIGH"
    if sim >= SIM_MED:
        return "MEDIUM"
    return "LOW"


def nearest_neighbor(query_fp, ref_fps, ref_smiles):
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps))
    j = int(np.argmax(sims))
    return float(sims[j]), ref_smiles[j]


def substructure_flags(smiles):
    m = Chem.MolFromSmiles(smiles)
    halogen_count = sum(1 for a in m.GetAtoms() if a.GetSymbol() in HALOGENS)
    heteroatom_count = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    return {
        "MolWt": round(float(Descriptors.MolWt(m)), 2),
        "num_heavy_atoms": int(m.GetNumHeavyAtoms()),
        "num_rings": int(rdMolDescriptors.CalcNumRings(m)),
        "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(m)),
        "halogen_count": int(halogen_count),
        "heteroatom_count": int(heteroatom_count),
    }


def main():
    require(f"{ART}/benchmark_split_v3.csv")
    require(f"{RES}/variant_test_predictions.csv")

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)

    bench = pd.read_csv(f"{ART}/benchmark_split_v3.csv")
    preds = pd.read_csv(f"{RES}/variant_test_predictions.csv")

    missing_variants = set(VARIANTS) - set(preds.variant.unique())
    if missing_variants:
        raise ValueError(f"variant_test_predictions.csv is missing variants: {missing_variants}")
    if TARGET not in preds.target.unique():
        raise ValueError(f"variant_test_predictions.csv has no rows for target={TARGET!r}")

    preds_tc = preds[preds.target == TARGET].copy()
    preds_tc["residual"] = preds_tc.actual - preds_tc.pred
    preds_tc["abs_error"] = preds_tc.residual.abs()

    # actual Tc and cluster_id must be identical across variants for the same molecule
    n_actual = preds_tc.groupby("canonical_smiles")["actual"].nunique()
    bad_actual = n_actual[n_actual > 1]
    if len(bad_actual):
        raise ValueError(f"actual Tc differs across variants for {len(bad_actual)} molecules "
                         f"— predictions file corrupted: {bad_actual.index.tolist()[:5]}")
    n_cluster = preds_tc.groupby("canonical_smiles")["cluster_id"].nunique()
    bad_cluster = n_cluster[n_cluster > 1]
    if len(bad_cluster):
        raise ValueError(f"cluster_id differs across variants for {len(bad_cluster)} molecules "
                         f"— predictions file corrupted: {bad_cluster.index.tolist()[:5]}")

    n_per_variant = preds_tc.groupby("variant").size()
    if n_per_variant.nunique() != 1:
        raise ValueError(f"Unequal Tc test-set sizes across variants: {n_per_variant.to_dict()}")
    n_test_tc = int(n_per_variant.iloc[0])
    print(f"Tc test set: {n_test_tc} molecules x {len(VARIANTS)} variants")

    # ---------- (1) context error percentiles, per variant (full Tc test set) ----------
    pctl_rows = []
    pctl_by_variant = {}
    for v in VARIANTS:
        e = preds_tc.loc[preds_tc.variant == v, "abs_error"].values
        qs = np.percentile(e, CONTEXT_PCTLS)
        pctl_by_variant[v] = dict(zip(CONTEXT_PCTLS, qs))
        pctl_rows.append({"variant": v, "n": len(e),
                          **{f"P{p}": round(float(q), 4) for p, q in zip(CONTEXT_PCTLS, qs)},
                          "Max": round(float(e.max()), 4)})
    pctl_df = pd.DataFrame(pctl_rows)
    pctl_df.to_csv(f"{OUTDIR}/01_context_error_percentiles.csv", index=False)
    elevated_threshold = {v: pctl_by_variant[v][ELEVATED_PCTL] for v in VARIANTS}

    # ---------- cohorts ----------
    d_sorted = (preds_tc[preds_tc.variant == PRIMARY_VARIANT]
                .sort_values("abs_error", ascending=False).reset_index(drop=True))
    primary_ids = d_sorted.head(PRIMARY_N).canonical_smiles.tolist()
    xgbd_rank = {cs: i + 1 for i, cs in enumerate(primary_ids)}

    secondary_variants = {}  # canonical_smiles -> set of variants whose own top-20 it's in
    for v in VARIANTS:
        top_v = (preds_tc[preds_tc.variant == v]
                 .sort_values("abs_error", ascending=False).head(SECONDARY_N).canonical_smiles)
        for cs in top_v:
            secondary_variants.setdefault(cs, set()).add(v)

    master_ids = sorted(set(primary_ids) | set(secondary_variants.keys()))
    print(f"Primary cohort (XGB-D top {PRIMARY_N}): {len(primary_ids)}")
    print(f"Secondary union (top {SECONDARY_N} per variant): {len(secondary_variants)}")
    print(f"Master cohort (union): {len(master_ids)}")

    # ---------- fingerprints: all-train (feature domain) and Tc-labeled-train (label pool) ----------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    bench_fp = {cs: gen.GetFingerprint(Chem.MolFromSmiles(cs)) for cs in bench.canonical_smiles}

    all_train_smiles = bench.loc[bench.split == "train", "canonical_smiles"].tolist()
    all_train_fps = [bench_fp[cs] for cs in all_train_smiles]

    tc_train_mask = (bench.split == "train") & bench[TARGET].notna()
    tc_train_smiles = bench.loc[tc_train_mask, "canonical_smiles"].tolist()
    tc_train_actual = dict(zip(bench.loc[tc_train_mask, "canonical_smiles"],
                               bench.loc[tc_train_mask, TARGET]))
    tc_train_fps = [bench_fp[cs] for cs in tc_train_smiles]
    print(f"All-train pool: {len(all_train_smiles)} | Tc-labeled-train pool: {len(tc_train_smiles)}")

    # ---------- per-variant pred/error lookup ----------
    wide_pred = preds_tc.pivot(index="canonical_smiles", columns="variant", values="pred")
    wide_err = preds_tc.pivot(index="canonical_smiles", columns="variant", values="abs_error")
    wide_res = preds_tc.pivot(index="canonical_smiles", columns="variant", values="residual")
    actual_lu = preds_tc.drop_duplicates("canonical_smiles").set_index("canonical_smiles")["actual"]
    cluster_lu = preds_tc.drop_duplicates("canonical_smiles").set_index("canonical_smiles")["cluster_id"]

    # ---------- cluster hotspots over the FULL Tc test set (not just cohort) ----------
    bench_cluster_size = bench.cluster_id.value_counts().to_dict()
    clu_rows = []
    for cid, g in preds_tc[preds_tc.variant == PRIMARY_VARIANT].groupby("cluster_id"):
        clu_rows.append({
            "cluster_id": int(cid),
            "n_test_tc": int(g.canonical_smiles.nunique()),
            "bench_total_n_in_cluster": int(bench_cluster_size.get(cid, 0)),
            "mean_abs_error_XGB-D": round(float(g.abs_error.mean()), 4),
            "max_abs_error_XGB-D": round(float(g.abs_error.max()), 4),
            "n_in_outlier_master": int(sum(cs in master_ids for cs in g.canonical_smiles)),
        })
    cluster_df = pd.DataFrame(clu_rows).sort_values("mean_abs_error_XGB-D", ascending=False)
    cluster_df.to_csv(f"{OUTDIR}/04_cluster_hotspots.csv", index=False)
    cluster_stats = cluster_df.set_index("cluster_id").to_dict("index")

    # ---------- build master outlier table ----------
    rows = []
    for cs in master_ids:
        actual = float(actual_lu[cs])
        cid = int(cluster_lu[cs])
        fp = bench_fp[cs]

        sim_all, nn_all = nearest_neighbor(fp, all_train_fps, all_train_smiles)
        sim_tc, nn_tc = nearest_neighbor(fp, tc_train_fps, tc_train_smiles)
        nn_tc_actual = float(tc_train_actual[nn_tc])
        label_gap = abs(actual - nn_tc_actual)

        elevated = {v: bool(wide_err.loc[cs, v] > elevated_threshold[v]) for v in VARIANTS}
        n_elevated = sum(elevated.values())
        if n_elevated >= 3:
            consistency = "SYSTEMATIC"
        elif n_elevated >= 1:
            consistency = "MODEL-SENSITIVE"
        else:
            consistency = "NONE"

        row = {
            "canonical_smiles": cs,
            "cluster_id": cid,
            "actual_Tc": round(actual, 3),
            "xgbd_rank": xgbd_rank.get(cs, np.nan),
            "in_primary_top50": cs in primary_ids,
            "in_secondary_union": cs in secondary_variants,
            "secondary_union_variants": ",".join(sorted(secondary_variants.get(cs, set()))),
            "n_elevated_variants": n_elevated,
            "consistency_class": consistency,
            "max_tanimoto_all_train": round(sim_all, 4),
            "doa_bin": doa_bin(sim_all),
            "nearest_all_train_smiles": nn_all,
            "max_tanimoto_tc_train_pool": round(sim_tc, 4),
            "nearest_tc_train_smiles": nn_tc,
            "nearest_tc_train_actual": round(nn_tc_actual, 3),
            "tc_label_gap": round(label_gap, 3),
            "close_neighbor_large_label_gap": bool(sim_tc >= LABEL_GAP_SIM_FLAG and
                                                    label_gap > elevated_threshold[PRIMARY_VARIANT]),
            "cluster_n_test_tc": cluster_stats.get(cid, {}).get("n_test_tc"),
            "cluster_bench_total_n": cluster_stats.get(cid, {}).get("bench_total_n_in_cluster"),
            "cluster_mean_abs_error_XGB-D": cluster_stats.get(cid, {}).get("mean_abs_error_XGB-D"),
        }
        for v in VARIANTS:
            row[f"{v}_pred"] = round(float(wide_pred.loc[cs, v]), 3)
            row[f"{v}_residual"] = round(float(wide_res.loc[cs, v]), 3)
            row[f"{v}_abs_error"] = round(float(wide_err.loc[cs, v]), 3)
            row[f"{v}_elevated_gtP{ELEVATED_PCTL}"] = elevated[v]
        row.update(substructure_flags(cs))
        rows.append(row)

    master = pd.DataFrame(rows).sort_values("xgbd_rank", na_position="last").reset_index(drop=True)
    master.to_csv(f"{OUTDIR}/02_master_outlier_table.csv", index=False)

    # ---------- focused top-20 narrative report ----------
    focused = master[master.in_primary_top50].sort_values("xgbd_rank").head(FOCUSED_N)
    lines = [
        "# Tc Outlier Audit — Top 20 Focused Report (Frozen Protocol v3)", "",
        f"Ranked by XGB-D |error| on the frozen Tc test set (n={n_test_tc}). "
        "Descriptive diagnostics only — single frozen split, no causal claims.", "",
        f"'Elevated' = |error| under a variant exceeds that variant's own P{ELEVATED_PCTL} "
        f"of |error| over its full Tc test set. SYSTEMATIC = elevated in >=3/4 variants, "
        "MODEL-SENSITIVE = elevated in 1-2/4, NONE = elevated in 0/4 (can occur for a "
        "molecule pulled into this cohort by a different variant's own top list).", "",
    ]
    for _, r in focused.iterrows():
        lines.append(f"## Rank {int(r.xgbd_rank)} — `{r.canonical_smiles}` (cluster {r.cluster_id})")
        lines.append(f"- Actual Tc: {r.actual_Tc} K")
        for v in VARIANTS:
            flag = " [ELEVATED]" if r[f"{v}_elevated_gtP{ELEVATED_PCTL}"] else ""
            lines.append(f"- {v}: pred={r[f'{v}_pred']}  residual={r[f'{v}_residual']:+}  "
                        f"|error|={r[f'{v}_abs_error']}{flag}")
        lines.append(f"- Consistency: **{r.consistency_class}** ({r.n_elevated_variants}/4 variants elevated)")
        lines.append(f"- Domain of applicability: max Tanimoto to ANY train molecule = "
                    f"{r.max_tanimoto_all_train} (**{r.doa_bin}**), nearest = `{r.nearest_all_train_smiles}`")
        gap_flag = " **[close neighbor, large label gap]**" if r.close_neighbor_large_label_gap else ""
        lines.append(f"- Tc-labeled-train nearest neighbor: `{r.nearest_tc_train_smiles}` "
                    f"(sim={r.max_tanimoto_tc_train_pool}, actual Tc={r.nearest_tc_train_actual} K, "
                    f"gap={r.tc_label_gap} K){gap_flag}")
        lines.append(f"- Cluster: {r.cluster_n_test_tc} Tc-test molecule(s) of "
                    f"{r.cluster_bench_total_n} total benchmark molecule(s) in cluster {r.cluster_id}; "
                    f"cluster mean XGB-D |error| = {r['cluster_mean_abs_error_XGB-D']} K")
        lines.append(f"- Structure: MW={r.MolWt}, rings={r.num_rings} (aromatic={r.num_aromatic_rings}), "
                    f"halogens={r.halogen_count}, heteroatoms={r.heteroatom_count}")
        lines.append("")
    open(f"{OUTDIR}/03_top20_focused_report.md", "w", encoding="utf-8").write("\n".join(lines))

    # ---------- diagnostic plot: DOA vs error, colored by consistency ----------
    color_map = {"SYSTEMATIC": "tab:red", "MODEL-SENSITIVE": "tab:orange", "NONE": "tab:gray"}
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls, c in color_map.items():
        d = master[master.consistency_class == cls]
        ax.scatter(d.max_tanimoto_all_train, d["XGB-D_abs_error"], s=28, alpha=0.75,
                  color=c, label=f"{cls} (n={len(d)})")
    ax.axvline(SIM_MED, color="k", ls="--", lw=0.8)
    ax.axvline(SIM_HIGH, color="k", ls="--", lw=0.8)
    ax.set_xlabel("max Tanimoto to all-train (domain of applicability)")
    ax.set_ylabel("XGB-D |error| (K)")
    ax.set_title("Tc outlier master cohort: DOA vs error, by cross-variant consistency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plots/tc_outlier_diagnostic.png", dpi=150)
    plt.close()

    # ---------- summary ----------
    cls_counts = master.consistency_class.value_counts().to_dict()
    doa_counts = master.doa_bin.value_counts().to_dict()
    gap_flag_n = int(master.close_neighbor_large_label_gap.sum())
    top_clusters = cluster_df[cluster_df.n_test_tc >= 2].head(5)

    summary = [
        "# Tc Outlier Audit — Summary (Frozen Protocol v3)", "",
        "Read-only diagnostic stage. No retraining, resplitting, retuning, or prediction "
        "regeneration — all values reused from `results_v3/variant_test_predictions.csv` and "
        "`artifacts/benchmark_split_v3.csv`. Fingerprints are recomputed from saved canonical "
        "SMILES (same Morgan r2/2048 protocol as Scripts 01/03) purely for similarity diagnostics.",
        "",
        "**Single frozen split.** Descriptive only — no causal claims, no statistical-significance claims.",
        "",
        "## Cohorts",
        f"- Primary: top {PRIMARY_N} Tc test molecules by |error| under {PRIMARY_VARIANT} "
        f"({len(primary_ids)} molecules)",
        f"- Secondary union: top {SECONDARY_N} Tc test molecules by |error| under EACH variant, unioned "
        f"({len(secondary_variants)} molecules)",
        f"- Master (union of both): {len(master_ids)} unique molecules — full diagnostics in "
        "`02_master_outlier_table.csv`",
        f"- Focused report: top {FOCUSED_N} of the primary cohort — `03_top20_focused_report.md`",
        "",
        "## Cross-variant consistency (master cohort)",
        f"'Elevated' threshold = each variant's own P{ELEVATED_PCTL} of |error| over its full "
        f"Tc test set (n={n_test_tc}). See `01_context_error_percentiles.csv` for P50/75/90/95/99/Max "
        "per variant.",
    ]
    for cls in ["SYSTEMATIC", "MODEL-SENSITIVE", "NONE"]:
        summary.append(f"- {cls}: {cls_counts.get(cls, 0)} / {len(master_ids)}")
    summary += [
        "", "## Domain of applicability (master cohort, all-train fingerprint similarity)",
    ]
    for b in ["LOW", "MEDIUM", "HIGH"]:
        summary.append(f"- {b}: {doa_counts.get(b, 0)} / {len(master_ids)}")
    summary += [
        "", "## Near-neighbor label consistency",
        f"- Mean |Tc label gap| to nearest Tc-labeled-train neighbor: "
        f"{master.tc_label_gap.mean():.2f} K (median {master.tc_label_gap.median():.2f} K, "
        f"max {master.tc_label_gap.max():.2f} K)",
        f"- Flagged 'close neighbor, large label gap' (sim>={LABEL_GAP_SIM_FLAG} to a Tc-labeled train "
        f"molecule AND gap > XGB-D's P{ELEVATED_PCTL} |error|): {gap_flag_n} / {len(master_ids)} — "
        "these are candidates for a data/labeling review rather than a model-capacity issue.",
        "", "## Cluster hotspots (Tc test clusters with >=2 members, top 5 by mean XGB-D |error|)",
    ]
    for _, r in top_clusters.iterrows():
        summary.append(f"- cluster {int(r.cluster_id)}: n_test_tc={int(r.n_test_tc)}, "
                       f"bench_total={int(r.bench_total_n_in_cluster)}, "
                       f"mean |error|={r['mean_abs_error_XGB-D']} K, "
                       f"max |error|={r['max_abs_error_XGB-D']} K "
                       f"(full table: `04_cluster_hotspots.csv`)")
    try:
        pctl_block = pctl_df.to_markdown(index=False)
    except ImportError:
        pctl_block = pctl_df.to_string(index=False)
    summary += ["", "## Context: |error| percentiles per variant (full Tc test set)", "",
               pctl_block,
               "", "_See `plots/tc_outlier_diagnostic.png` for DOA vs error, colored by consistency class._"]
    open(f"{OUTDIR}/00_SUMMARY.md", "w", encoding="utf-8").write("\n".join(str(s) for s in summary))

    print(f"\nConsistency breakdown: {cls_counts}")
    print(f"DOA breakdown: {doa_counts}")
    print(f"Close-neighbor/large-label-gap flags: {gap_flag_n}")
    print(f"\nAll outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
