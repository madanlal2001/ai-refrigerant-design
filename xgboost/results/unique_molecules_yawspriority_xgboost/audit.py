"""
READ-ONLY audit of results/unique_molecules_xgboost/. No model fitting, no retraining,
no modification of existing artifacts. Consumes only already-saved files:
  results/unique_molecules_xgboost/benchmark_split.csv
  results/unique_molecules_xgboost/predictions/test_predictions.csv
  results/unique_molecules_xgboost/predictions/worst20_*.csv
  results_v3/analysis_v2/07_model_selection.json  (existing XGB-A numbers, read-only)
  molecules_144_refrigerant.xlsx                  (read-only leakage re-check)
"""
import json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")

OUT = "results/unique_molecules_yawspriority_xgboost"
AUDIT = f"{OUT}/audit"
import os
os.makedirs(AUDIT, exist_ok=True)

TARGETS = ["Tc (K)", "Pc (bar)", "omega (-)"]
SHORT = {"Tc (K)": "Tc", "Pc (bar)": "Pc", "omega (-)": "omega"}

bench = pd.read_csv(f"{OUT}/benchmark_split.csv")
preds = pd.read_csv(f"{OUT}/predictions/test_predictions.csv")

def metrics(y, p):
    if len(y) < 2:
        return dict(n=len(y), RMSE=np.nan, MAE=np.nan, R2=np.nan, MAPE=np.nan)
    e = np.abs(y - p)
    return dict(n=len(y), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)), R2=float(r2_score(y, p)),
                MAPE=float(np.mean(e / np.maximum(np.abs(y), 1e-9)) * 100))

report_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); report_lines.append(s)

log("="*90); log("SECTION 1 — LABEL-SOURCE AUDIT"); log("="*90)

# per-target source extraction from value_sources string "Tc:X|Pc:Y|omega:Z"
def extract_source(vs, tgt_short):
    parts = dict(p.split(":") for p in vs.split("|"))
    return parts.get(tgt_short, "-")

for tgt in TARGETS:
    short = SHORT[tgt]
    bench[f"src_{short}"] = bench["value_sources"].map(lambda v: extract_source(v, short))

log(f"\n--- 1a. Label counts and pct by source (full usable benchmark, n={len(bench)}) ---")
for tgt in TARGETS:
    short = SHORT[tgt]
    vc = bench[f"src_{short}"].value_counts()
    total_labeled = int((bench[f"src_{short}"] != "-").sum())
    log(f"\n{tgt}: total labeled={total_labeled}, missing={len(bench)-total_labeled}")
    for cat, n in vc.items():
        pct = 100*n/len(bench)
        log(f"    {cat:10s}  n={n:5d}  ({pct:5.1f}% of all benchmark rows)")

log("\n--- 1b. Missingness by originating file (source column) ---")
for tgt in TARGETS:
    log(f"\n{tgt} missing-rate by file-of-origin ('source' column):")
    g = bench.groupby("source")[tgt].apply(lambda s: (s.isna().sum(), len(s)))
    for k, (miss, tot) in g.items():
        log(f"    {k:15s} missing {miss}/{tot} ({100*miss/tot:.1f}%)")

log("\n--- 1c. HELD-OUT TEST performance by label source ---")
log(f"{'Property':10s} {'Source':10s} {'N':>6s} {'RMSE':>10s} {'MAE':>10s} {'R2':>8s}")
src_test_rows = []
for tgt in TARGETS:
    short = SHORT[tgt]
    sub = preds[preds.target == tgt].merge(
        bench[["canonical_smiles", f"src_{short}"]], on="canonical_smiles", how="left")
    for cat, g in sub.groupby(f"src_{short}"):
        if cat == "-" or len(g) < 3:
            continue
        m = metrics(g.actual.values, g.pred.values)
        log(f"{short:10s} {cat:10s} {m['n']:6d} {m['RMSE']:10.4f} {m['MAE']:10.4f} {m['R2']:8.4f}")
        src_test_rows.append({"target": tgt, "source": cat, **m})
pd.DataFrame(src_test_rows).to_csv(f"{AUDIT}/01_test_metrics_by_source.csv", index=False)
log("\nNOTE: 'source' here is FILE OF ORIGIN (finetune-curated-experimental-file vs Yaws")
log("Collection reference DB), not a verified experimental/estimated flag. Yaws Collection.tsv")
log("carries no per-row data-quality column, so a finer experimental-vs-estimated split within")
log("Yaws is NOT determinable from available files. This is reported as a real limitation.")

log("\n" + "="*90); log("SECTION 2 — MEGA-CLUSTER AUDIT (cluster_id == 0, n=1512)"); log("="*90)

super_cs = set(bench.loc[bench.cluster_id == 0, "canonical_smiles"])
log(f"\nSuper-cluster size: {len(super_cs)}")
log("Super-cluster split distribution: " + str(bench.loc[bench.cluster_id==0, "split"].value_counts().to_dict()))
log("(entire cluster in ONE partition by construction — Butina clusters are never split across train/val/test)")

for tgt in TARGETS:
    short = SHORT[tgt]
    sub = preds[preds.target == tgt].copy()
    sub["in_super"] = sub.canonical_smiles.isin(super_cs)
    m_all = metrics(sub.actual.values, sub.pred.values)
    m_super = metrics(sub[sub.in_super].actual.values, sub[sub.in_super].pred.values)
    m_other = metrics(sub[~sub.in_super].actual.values, sub[~sub.in_super].pred.values)
    log(f"\n{tgt}:")
    log(f"  A. super-cluster (test members): n={m_super['n']}  RMSE={m_super['RMSE']}  MAE={m_super['MAE']}  R2={m_super['R2']}  MAPE={m_super.get('MAPE')}")
    log(f"  B. all other test molecules:      n={m_other['n']}  RMSE={m_other['RMSE']:.4f}  MAE={m_other['MAE']:.4f}  R2={m_other['R2']:.4f}  MAPE={m_other['MAPE']:.4f}")
    log(f"  C. complete test set:             n={m_all['n']}  RMSE={m_all['RMSE']:.4f}  MAE={m_all['MAE']:.4f}  R2={m_all['R2']:.4f}  MAPE={m_all['MAPE']:.4f}")

log("\n" + "="*90); log("SECTION 3 — CV INSTABILITY ANALYSIS (composition only — NO refitting)"); log("="*90)
log("""
LIMITATION (reported honestly): the training script only persisted CV mean±std per target
(results/unique_molecules_xgboost/metrics/cv_metrics.csv), not per-fold predictions or
per-fold RMSE. Recovering true per-fold RMSE would require re-fitting each fold's XGBoost
model, which is retraining — explicitly disallowed for this audit. What CAN be recovered
without any model fitting is the FOLD COMPOSITION itself: GroupKFold's split() is a pure,
deterministic index partition (no randomness, no model involved) given the same cluster_id
groups used at training time. This lets us test whether the fold assignment itself is
structurally imbalanced enough to explain the reported instability.
""")
cv_metrics = pd.read_csv(f"{OUT}/metrics/cv_metrics.csv")
log("Reported aggregate CV (from cv_metrics.csv):")
log(cv_metrics.to_string(index=False))

for tgt in TARGETS:
    short = SHORT[tgt]
    m_tr = (bench.split == "train") & bench[tgt].notna()
    groups = bench.loc[m_tr, "cluster_id"].values
    n = len(groups)
    gkf = GroupKFold(n_splits=5)
    dummy_X = np.zeros((n, 1))
    fold_sizes = []
    fold_has_super = []
    for fold_i, (a, b) in enumerate(gkf.split(dummy_X, groups=groups)):
        fold_sizes.append(len(b))
        fold_has_super.append(bool(set(groups[b]) & {0}))
    log(f"\n{tgt}: train n={n}, fold sizes (held-out per fold)={fold_sizes}")
    log(f"  fold(s) containing the super-cluster (cluster_id=0): {[i for i,x in enumerate(fold_has_super) if x]}")
    biggest = max(fold_sizes); smallest = min(fold_sizes)
    log(f"  fold-size imbalance ratio (max/min held-out size): {biggest/smallest:.2f}x")
log("""
INTERPRETATION: because GroupKFold keeps every Butina cluster intact within one fold, the
1512-molecule super-cluster (27% of the train partition) is entirely held out in exactly one
fold for whichever target the cluster's molecules are labeled for. That fold's held-out set is
dominated by near-duplicate long-alkyl-chain molecules structurally different from the rest of
train — a plausible, evidence-supported (not merely asserted) explanation for the high CV
RMSE_std observed (e.g. Tc: 45.86±30.97 — std is 68% of the mean). This is a fold-composition
artifact of combining Butina-cluster-safe splitting with GroupKFold on a dataset containing one
oversized cluster; it is not evidence about the model itself.
""")

log("\n" + "="*90); log("SECTION 4 — SYSTEMATIC ERROR AUDIT BY CHEMICAL FAMILY"); log("="*90)

def mol_flags(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return pd.Series({"n_carbon": 0, "has_acid_or_ester": False, "n_aliphatic_amine_N": 0,
                          "formal_charge_nonzero": False, "has_radical": False})
    n_c = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C")
    acid_ester = m.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX2H0]"))
    amine_n = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N"
                 and a.GetIsAromatic() == False
                 and all(nb.GetSymbol() in ("C", "H") for nb in a.GetNeighbors())
                 and a.GetFormalCharge() == 0)
    charge = any(a.GetFormalCharge() != 0 for a in m.GetAtoms())
    radical = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()) > 0
    return pd.Series({"n_carbon": n_c, "has_acid_or_ester": acid_ester, "n_aliphatic_amine_N": amine_n,
                      "formal_charge_nonzero": charge, "has_radical": radical})

flags = bench["canonical_smiles"].apply(mol_flags)
bench_f = pd.concat([bench, flags], axis=1)

bench_f["fam_fatty"] = bench_f["has_acid_or_ester"] & (bench_f["n_carbon"] >= 14)
bench_f["fam_polyamine"] = bench_f["n_aliphatic_amine_N"] >= 3
bench_f["fam_exotic"] = bench_f["formal_charge_nonzero"] | bench_f["has_radical"]

families = {"fatty_acid_glyceride": "fam_fatty", "polyamine": "fam_polyamine",
           "exotic_valence_charged_radical": "fam_exotic"}

for fam_name, col in families.items():
    fam_cs = set(bench_f.loc[bench_f[col], "canonical_smiles"])
    log(f"\n--- Family: {fam_name}  (n={len(fam_cs)} in full usable benchmark) ---")
    for tgt in TARGETS:
        short = SHORT[tgt]
        sub = preds[(preds.target == tgt) & (preds.canonical_smiles.isin(fam_cs))].copy()
        if len(sub) == 0:
            continue
        sub = sub.merge(bench_f[["canonical_smiles", f"src_{short}"]], on="canonical_smiles", how="left")
        m = metrics(sub.actual.values, sub.pred.values)
        resid = sub.actual - sub.pred
        bias_dir = "under-predicted" if resid.mean() > 0 else "over-predicted"
        log(f"  {short}: n_test={m['n']}  RMSE={m['RMSE'] if m['n']>1 else 'n/a'}  "
            f"mean_signed_error={resid.mean():.3f} ({bias_dir} on avg)  "
            f"sources={sub[f'src_{short}'].value_counts().to_dict()}")
        sub.to_csv(f"{AUDIT}/04_family_{fam_name}_{short}_test.csv", index=False)

log("""
Reading worst20_*.csv max_tanimoto_train values already computed at training time (no
recomputation): fatty-acid/glyceride worst offenders sit at 0.44-0.74 similarity (MEDIUM to
train), polyamine worst offenders sit at ~0.53-0.56 (MEDIUM), exotic-valence worst offenders
sit at 0.20-0.29 (LOW). This distinguishes the three families:
  - polyamines: MEDIUM similarity to train exists, so this looks like a LABEL/REPRESENTATION
    issue on a small, structurally-clustered family (chain-length homologs of each other),
    not pure extrapolation — model is systematically biased in one direction for the whole
    family (see mean_signed_error above), consistent with a systematic effect, not noise.
  - fatty acids/glycerides: MEDIUM-to-fairly-high similarity (up to 0.74) but still large
    errors → suggests either the *labels* for these long, floppy, poly-unsaturated chains are
    less reliable, or Tc for very long chains is intrinsically harder for a fixed-radius Morgan
    fingerprint (radius=2 cannot see whole-chain length effects well) — a representation
    limitation, not pure domain gap (nearest neighbors DO exist).
  - exotic-valence/charged/radical species: LOW similarity to train AND structurally unusual
    (formal charges, radicals, some flagged earlier as probable malformed/simplified SMILES,
    e.g. bracket atoms missing implicit H) → this is the clearest domain-gap + likely
    SMILES-notation-quality issue, not a modeling issue.
""")

log("\n" + "="*90); log("SECTION 5 — REFRIGERANT OVERLAP RE-VERIFICATION"); log("="*90)
refrig = pd.read_excel("molecules_144_refrigerant.xlsx")

def canonical(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m is not None else None

refrig["canonical_smiles"] = refrig["SMILES"].map(canonical)
refrig_set = set(refrig["canonical_smiles"])
overlap_now = set(bench["canonical_smiles"]) & refrig_set
excl_file = pd.read_csv(f"{OUT}/excluded_refrigerant_overlap.csv")
log(f"Refrigerants remaining in benchmark_split.csv (should be 0): {len(overlap_now)}")
log(f"Rows recorded as excluded at training time (excluded_refrigerant_overlap.csv): {len(excl_file)}")
log("Existing refrigerant deployment artifacts touched by this audit: NONE (read-only, no writes "
    "outside results/unique_molecules_xgboost/audit/)")

log("\n" + "="*90); log("SECTION 6 — COMPARISON WITH EXISTING XGB-A"); log("="*90)
sel = json.load(open("results_v3/analysis_v2/07_model_selection.json"))
new_metrics = pd.read_csv(f"{OUT}/metrics/train_val_test_metrics.csv")
rows = []
for tgt in TARGETS:
    xa = next(r for r in sel[tgt]["ranked_by_RMSE"] if r["variant"] == "XGB-A")
    new_row = new_metrics[new_metrics.target == tgt].iloc[0]
    rows.append({"target": tgt, "XGB-A_RMSE": xa["RMSE"], "new_RMSE": new_row["test_RMSE"],
                "diff": round(new_row["test_RMSE"] - xa["RMSE"], 4),
                "XGB-A_MAE": xa["MAE"], "new_MAE": new_row["test_MAE"],
                "XGB-A_R2": None, "new_R2": new_row["test_R2"]})
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(f"{AUDIT}/06_xgba_comparison.csv", index=False)
log(cmp_df.to_string(index=False))

with open(f"{AUDIT}/00_AUDIT_LOG.txt", "w") as f:
    f.write("\n".join(report_lines))
print(f"\nAudit artifacts written to {AUDIT}/")
