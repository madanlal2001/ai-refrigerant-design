"""
Independent XGBoost training experiment on unique_molecules_Tc_Pc_omega.csv.

Reuses the repo's VALIDATED Frozen-Protocol-v3 methodology (from 01_prepare_data.py /
02_train_variants.py, notebook cells 1-2) exactly where applicable:
  - canonical identity: Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
  - features: RDKit-2D descriptors (Descriptors.descList) + Morgan(r=2, 2048 bits, counts=False)
  - split: Morgan r2/2048 -> Butina distance cutoff 0.60 -> deterministic
    normalized-deficit 80/10/10 assignment, computed on ALL benchmark molecules
    before target-specific label filtering
  - XGBoost hyperparameters: identical to XGB_PARAMS in 02_train_variants.py
  - val used ONLY for early stopping; test evaluated once
  - refrigerant-leakage check: 142 locked refrigerants excluded from training (same
    invariant enforced in 01_prepare_data.py), so this experiment cannot contaminate
    refrigerant-domain evaluation.

This is a completely independent experiment: writes only to results/unique_molecules_xgboost/,
never touches artifacts/, results_v3/, or existing *.json models.

No hyperparameter search is performed. No GNN/NN. XGBoost only.
"""
import os, json, hashlib, warnings
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

# ================= CONFIG (reused verbatim from the validated pipeline) =================
IN_FILE       = "unique_molecules_Tc_Pc_omega.csv"
REFRIG_FILE   = "molecules_144_refrigerant.xlsx"
TARGETS       = ["Tc (K)", "Pc (bar)", "omega (-)"]
CLUSTER_CUTOFF, SPLIT_FP_BITS = 0.60, 2048
MODEL_FP_BITS = 2048
SEED = 42
N_CV_FOLDS = 5
XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=7, subsample=0.8,
                  colsample_bytree=0.7, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, tree_method="hist",
                  early_stopping_rounds=150)
ORGANIC_ELEMENTS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}  # matches pipeline's
                                                                          # halogen/organic scope
OUT = "results/unique_molecules_splitfix_xgboost"
CLUSTER_SIZE_CAP = 200        # NEW: no Butina cluster may exceed this many molecules
CLUSTER_STEP = 0.15           # NEW: cutoff decrement applied when recursively de-clumping
CLUSTER_MIN_CUTOFF = 0.10     # NEW: floor cutoff for recursive de-clumping
BINS = [0, 0.4, 0.6, 0.8, 1.01]; BIN_LABELS = ["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"]
# ==========================================================================================

DESC = [(n, f) for n, f in Descriptors.descList]
MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MODEL_FP_BITS)
SGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=SPLIT_FP_BITS)


def canonical(s):
    m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m is not None else None


def classify_molecule(smi):
    """Flags non-organic / non-molecular entries unsuitable for a Morgan+RDKit QSPR model."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "invalid_smiles"
    if "." in smi:
        return "multi_fragment_salt_or_mixture"
    mh = Chem.AddHs(m)
    elems = set(a.GetSymbol() for a in mh.GetAtoms())
    if not elems.issubset(ORGANIC_ELEMENTS):
        return "inorganic_or_metal_containing"
    if mh.GetNumAtoms() == 1:
        return "bare_element_no_hydrogens"
    return "ok"


def featurize(canonical_smiles):
    rows = []
    for s in canonical_smiles:
        m = Chem.MolFromSmiles(s)
        d = []
        for _, fn in DESC:
            try:
                v = fn(m); d.append(v if np.isfinite(v) else 0.0)
            except Exception:
                d.append(0.0)
        row = np.concatenate([np.asarray(d, np.float32),
                              np.asarray(MGEN.GetFingerprint(m), np.float32)])
        rows.append(row)
    return np.vstack(rows)


def _butina_at(fps, cutoff):
    n = len(fps); dists = []
    for i in range(1, n):
        dists.extend(1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    return Butina.ClusterData(dists, n, cutoff, isDistData=True)


def _declump(idx, fps_all, cutoff, cap, step, min_cutoff):
    """Recursively re-cluster an oversized cluster at a tighter cutoff until every
    resulting sub-cluster is <= cap (or the cutoff floor is hit). idx = original
    indices (into fps_all) belonging to this cluster. Returns list of index-lists."""
    if len(idx) <= cap or cutoff <= min_cutoff:
        return [idx]
    sub_fps = [fps_all[i] for i in idx]
    sub_clusters = _butina_at(sub_fps, cutoff)
    out = []
    next_cutoff = max(cutoff - step, min_cutoff)
    for members in sub_clusters:
        sub_idx = [idx[m] for m in members]
        if len(sub_idx) > cap and cutoff > min_cutoff:
            out.extend(_declump(sub_idx, fps_all, next_cutoff, cap, step, min_cutoff))
        else:
            out.append(sub_idx)
    return out


def frozen_cluster_split(canonical_smiles, cutoff=CLUSTER_CUTOFF, fp_bits=SPLIT_FP_BITS,
                         fractions=(0.8, 0.1, 0.1), size_cap=CLUSTER_SIZE_CAP,
                         declump_step=CLUSTER_STEP, min_cutoff=CLUSTER_MIN_CUTOFF):
    fps = [SGEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in canonical_smiles]
    n = len(fps)
    base_clusters = _butina_at(fps, cutoff)

    # NEW: recursively de-clump any cluster larger than size_cap so no single Butina
    # cluster can dominate a partition or a CV fold. Clusters at/under the cap are
    # returned unchanged (identical behaviour to the original frozen split for the
    # ~934 normal-sized clusters observed in this dataset).
    clusters = []
    for members in base_clusters:
        idx = list(members)
        if len(idx) > size_cap:
            clusters.extend(_declump(idx, fps, max(cutoff - declump_step, min_cutoff),
                                     size_cap, declump_step, min_cutoff))
        else:
            clusters.append(idx)

    names = ("train", "val", "test")
    target = {k: f * n for k, f in zip(names, fractions)}
    cnt = {k: 0 for k in names}
    split = np.empty(n, object); cid = np.zeros(n, int)
    for c, members in sorted(enumerate(clusters), key=lambda z: (-len(z[1]), z[0])):
        deficit = {k: (target[k] - cnt[k]) / target[k] for k in names}
        q = max(names, key=lambda k: deficit[k])
        for i in members:
            split[i] = q; cid[i] = c
        cnt[q] += len(members)
    return split, cid, fps


def metric_row(y, p):
    e = np.abs(y - p)
    return dict(n=len(y), R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=mean_absolute_error(y, p), MedAE=float(np.median(e)), MaxAE=float(e.max()),
                MAPE=float(np.mean(e / np.maximum(np.abs(y), 1e-9)) * 100))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for sub in ["metrics", "predictions", "figures", "models"]:
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    report = {}

    # ---------- 1. INSPECT ----------
    raw = pd.read_csv(IN_FILE)
    report["raw_rows"] = len(raw)
    report["raw_missing"] = {c: int(raw[c].isna().sum()) for c in TARGETS}

    raw["canonical_smiles"] = raw["smiles"].map(canonical)
    n_invalid = int(raw["canonical_smiles"].isna().sum())
    raw = raw[raw["canonical_smiles"].notna()].reset_index(drop=True)

    dup_mask = raw["canonical_smiles"].duplicated(keep=False)
    n_dup_structures = int(raw.loc[dup_mask, "canonical_smiles"].nunique())
    conflicts = []
    if dup_mask.any():
        for cs, g in raw[dup_mask].groupby("canonical_smiles"):
            for t in TARGETS:
                vals = g[t].dropna().unique()
                if len(vals) > 1:
                    conflicts.append({"canonical_smiles": cs, "target": t, "values": vals.tolist()})
        raw = raw.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)

    raw["_class"] = raw["smiles"].map(classify_molecule)
    excl_counts = raw["_class"].value_counts().to_dict()
    excluded = raw[raw["_class"] != "ok"].copy()
    excluded.to_csv(f"{OUT}/excluded_rows.csv", index=False)
    df = raw[raw["_class"] == "ok"].drop(columns=["_class"]).reset_index(drop=True)

    # refrigerant-leakage check (same invariant as 01_prepare_data.py)
    refrig = pd.read_excel(REFRIG_FILE)
    refrig["canonical_smiles"] = refrig["SMILES"].map(canonical)
    refrig_set = set(refrig["canonical_smiles"])
    leak_mask = df["canonical_smiles"].isin(refrig_set)
    n_leak = int(leak_mask.sum())
    df[leak_mask].to_csv(f"{OUT}/excluded_refrigerant_overlap.csv", index=False)
    df = df[~leak_mask].reset_index(drop=True)

    report["invalid_smiles_dropped"] = n_invalid
    report["duplicate_structures_found"] = n_dup_structures
    report["label_conflicts_in_duplicates"] = conflicts
    report["excluded_by_class"] = excl_counts
    report["excluded_refrigerant_overlap"] = n_leak
    report["usable_rows_final"] = len(df)
    report["usable_per_target"] = {t: int(df[t].notna().sum()) for t in TARGETS}
    report["target_ranges"] = {
        t: {"min": float(df[t].dropna().min()), "max": float(df[t].dropna().max()),
            "mean": float(df[t].dropna().mean()), "median": float(df[t].dropna().median())}
        for t in TARGETS
    }
    print("=== INSPECTION ===")
    print(json.dumps(report, indent=2, default=str))

    # ---------- 2. BENCHMARK SET + FROZEN SPLIT ----------
    bench = df[df[TARGETS].notna().any(axis=1)].reset_index(drop=True)
    split, cid, split_fps = frozen_cluster_split(bench.canonical_smiles.tolist())
    bench["split"], bench["cluster_id"] = split, cid
    sizes = pd.Series(split).value_counts().to_dict()
    if bench.groupby("cluster_id")["split"].nunique().max() != 1:
        raise ValueError("Cluster leakage across partitions!")
    bench.to_csv(f"{OUT}/benchmark_split.csv", index=False)

    # ---------- 3. FEATURES ----------
    X = featurize(bench.canonical_smiles)
    feature_dim = X.shape[1]
    feat_names = [n for n, _ in DESC] + [f"morgan_{i}" for i in range(MODEL_FP_BITS)]
    json.dump(feat_names, open(f"{OUT}/feature_names.json", "w"))

    tr = bench.split.values == "train"
    va = bench.split.values == "val"
    te = bench.split.values == "test"

    split_manifest = {
        "protocol": "unique_molecules_splitfix_independent_experiment",
        "based_on": "frozen_v3 methodology (01_prepare_data.py / 02_train_variants.py), "
                    "MODIFIED ONLY in the clustering step: any Butina cluster larger than "
                    "CLUSTER_SIZE_CAP is recursively re-clustered at a tighter cutoff so no "
                    "single cluster can dominate a partition or a CV fold. Features, XGBoost "
                    "hyperparameters, and the 80/10/10 deficit-assignment algorithm are unchanged.",
        "canonical_definition": "Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)",
        "split_cutoff": CLUSTER_CUTOFF, "split_fp_bits": SPLIT_FP_BITS,
        "cluster_size_cap": CLUSTER_SIZE_CAP, "declump_step": CLUSTER_STEP,
        "declump_min_cutoff": CLUSTER_MIN_CUTOFF,
        "assignment": "deterministic normalized-deficit 80/10/10",
        "seed": SEED,
        "total_benchmark_molecules": len(bench),
        "split_sizes": sizes,
        "usable_per_target": report["usable_per_target"],
        "feature_dim": int(feature_dim),
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": MODEL_FP_BITS, "counts": False},
        "descriptor_set": "RDKit Descriptors.descList (all)",
        "n_rdkit_descriptors": len(DESC),
        "refrigerant_molecules_excluded_for_leakage": n_leak,
        "input_hashes": {IN_FILE: sha256(IN_FILE), REFRIG_FILE: sha256(REFRIG_FILE)},
    }
    json.dump(split_manifest, open(f"{OUT}/split_manifest.json", "w"), indent=2)
    print("\n=== SPLIT MANIFEST ===")
    print(json.dumps(split_manifest, indent=2, default=str))

    # ---------- 4. TRAIN XGB-A-EQUIVALENT MODEL PER TARGET ----------
    all_rows = []
    all_test_preds = []
    cv_rows = []

    for tgt in TARGETS:
        has = bench[tgt].notna().values
        m_tr, m_va, m_te = has & tr, has & va, has & te
        Xtr, ytr = X[m_tr], bench[tgt].values[m_tr].astype(float)
        Xva, yva = X[m_va], bench[tgt].values[m_va].astype(float)
        Xte, yte = X[m_te], bench[tgt].values[m_te].astype(float)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        short = tgt.split(" ")[0]
        model.save_model(f"{OUT}/models/xgb_unique_{short}.json")

        ptr = model.predict(Xtr); pva = model.predict(Xva); pte = model.predict(Xte)
        row = {"target": tgt, "best_iteration": int(model.best_iteration),
               "n_train": int(m_tr.sum()), "n_val": int(m_va.sum()), "n_test": int(m_te.sum())}
        for split_name, y, p in [("train", ytr, ptr), ("val", yva, pva), ("test", yte, pte)]:
            mr = metric_row(y, p)
            for k, v in mr.items():
                row[f"{split_name}_{k}"] = round(v, 5) if isinstance(v, float) else v
        all_rows.append(row)

        f = bench.loc[m_te, ["canonical_smiles", "name", "cluster_id"]].copy()
        f["target"], f["actual"], f["pred"] = tgt, yte, pte
        f["abs_error"] = np.abs(yte - pte)
        all_test_preds.append(f)

        # ---- cluster-aware GroupKFold CV on TRAIN partition only (mirrors the pipeline's
        #      existing validated Tb-OOF CV design in 01_prepare_data.py) ----
        groups_tr = bench.loc[m_tr, "cluster_id"].values
        n_groups = len(set(groups_tr))
        k = min(N_CV_FOLDS, n_groups)
        gkf = GroupKFold(n_splits=k)
        fold_metrics = []
        for a, b in gkf.split(Xtr, ytr, groups=groups_tr):
            cvm = XGBRegressor(**{**XGB_PARAMS, "early_stopping_rounds": None})
            cvm.set_params(n_estimators=model.best_iteration or XGB_PARAMS["n_estimators"])
            cvm.fit(Xtr[a], ytr[a])
            pb = cvm.predict(Xtr[b])
            fold_metrics.append(metric_row(ytr[b], pb))
        fm = pd.DataFrame(fold_metrics)
        cv_rows.append({
            "target": tgt, "n_folds": k,
            "RMSE_mean": round(float(fm.RMSE.mean()), 4), "RMSE_std": round(float(fm.RMSE.std()), 4),
            "MAE_mean": round(float(fm.MAE.mean()), 4), "MAE_std": round(float(fm.MAE.std()), 4),
            "R2_mean": round(float(fm.R2.mean()), 4), "R2_std": round(float(fm.R2.std()), 4),
        })
        print(f"\n[{tgt}] train n={m_tr.sum()} val n={m_va.sum()} test n={m_te.sum()} "
              f"best_iter={model.best_iteration}")
        print(f"  test  RMSE={row['test_RMSE']:.4f} MAE={row['test_MAE']:.4f} R2={row['test_R2']:.4f}")
        print(f"  CV({k}-fold, train only)  RMSE={cv_rows[-1]['RMSE_mean']:.4f}"
              f"±{cv_rows[-1]['RMSE_std']:.4f}  R2={cv_rows[-1]['R2_mean']:.4f}±{cv_rows[-1]['R2_std']:.4f}")

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(f"{OUT}/metrics/train_val_test_metrics.csv", index=False)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(f"{OUT}/metrics/cv_metrics.csv", index=False)
    preds_df = pd.concat(all_test_preds, ignore_index=True)
    preds_df.to_csv(f"{OUT}/predictions/test_predictions.csv", index=False)

    # ---------- 5. WORST-N TABLES + FIGURES ----------
    train_fp_by_cs = {cs: fp for cs, fp, sp in zip(bench.canonical_smiles, split_fps, bench.split) if sp == "train"}
    train_fps_list = list(train_fp_by_cs.values())

    def max_tanimoto_to_train(cs_list):
        out = []
        for cs in cs_list:
            fp = SGEN.GetFingerprint(Chem.MolFromSmiles(cs))
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps_list)
            out.append(max(sims) if sims else 0.0)
        return np.array(out)

    for tgt in TARGETS:
        short = tgt.split(" ")[0]
        sub = preds_df[preds_df.target == tgt].copy()
        sub["max_tanimoto_train"] = max_tanimoto_to_train(sub.canonical_smiles.tolist())
        sub.sort_values("abs_error", ascending=False).head(20).to_csv(
            f"{OUT}/predictions/worst20_{short}.csv", index=False)

        # parity
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(sub.actual, sub.pred, c=sub.max_tanimoto_train, cmap="viridis", s=14, alpha=0.8)
        lims = [min(sub.actual.min(), sub.pred.min()), max(sub.actual.max(), sub.pred.max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel(f"Actual {tgt}"); ax.set_ylabel(f"Predicted {tgt}")
        ax.set_title(f"Parity — {short} (test, n={len(sub)})")
        plt.colorbar(sc, label="max Tanimoto to train")
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/parity_{short}.png", dpi=130); plt.close(fig)

        # residuals
        fig, ax = plt.subplots(figsize=(5, 4))
        resid = sub.actual - sub.pred
        ax.scatter(sub.pred, resid, s=12, alpha=0.7)
        ax.axhline(0, color="r", lw=1, ls="--")
        ax.set_xlabel(f"Predicted {tgt}"); ax.set_ylabel("Residual (actual - pred)")
        ax.set_title(f"Residuals — {short} (test)")
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/residuals_{short}.png", dpi=130); plt.close(fig)

        # error distribution
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(sub.abs_error, bins=30, color="steelblue", edgecolor="k", alpha=0.8)
        ax.set_xlabel(f"|error| ({tgt})"); ax.set_ylabel("count")
        ax.set_title(f"Error distribution — {short} (test)")
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/error_dist_{short}.png", dpi=130); plt.close(fig)

    # ---------- 6. DIVERSITY / DOA STATS FOR AUDIT ----------
    n_clusters = bench["cluster_id"].nunique()
    cluster_sizes = bench["cluster_id"].value_counts()
    doa = {}
    for tgt in TARGETS:
        sub = preds_df[preds_df.target == tgt]
        bins = pd.cut(sub.max_tanimoto_train if "max_tanimoto_train" in sub else max_tanimoto_to_train(sub.canonical_smiles.tolist()),
                      BINS, labels=BIN_LABELS)
        doa[tgt] = sub.groupby(bins, observed=False)["abs_error"].agg(["count", "mean"]).to_dict()

    audit = {
        "n_benchmark_molecules": len(bench),
        "n_butina_clusters": int(n_clusters),
        "largest_cluster_size": int(cluster_sizes.max()),
        "median_cluster_size": float(cluster_sizes.median()),
        "singleton_clusters": int((cluster_sizes == 1).sum()),
        "train_val_test_sizes": sizes,
    }
    json.dump(audit, open(f"{OUT}/metrics/diversity_audit.json", "w"), indent=2, default=str)

    experiment_config = {
        "input_file": IN_FILE,
        "targets": TARGETS,
        "xgb_params": XGB_PARAMS,
        "seed": SEED,
        "cv_folds": N_CV_FOLDS,
        "feature_dim": int(feature_dim),
        "organic_element_filter": sorted(ORGANIC_ELEMENTS),
    }
    json.dump(experiment_config, open(f"{OUT}/experiment_config.json", "w"), indent=2, default=str)
    json.dump(report, open(f"{OUT}/data_quality_report.json", "w"), indent=2, default=str)

    print("\n=== METRICS SUMMARY ===")
    print(metrics_df[["target", "n_train", "n_val", "n_test", "train_RMSE", "val_RMSE", "test_RMSE",
                      "train_R2", "val_R2", "test_R2"]].to_string(index=False))
    print("\n=== CV SUMMARY ===")
    print(cv_df.to_string(index=False))
    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
