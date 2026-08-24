# Validation status and known limitations

This document exists so that no result in this repo gets cited as more validated than it actually is. Read it before using anything here for a downstream decision.

## xgboost/ — Frozen Protocol v3 (the most rigorously validated stage)

**Validated, with explicit caveats stated by the project's own analysis:**

- Single frozen train/val/test split; every metric report explicitly states "no error bars, no statistical-significance claims — observed on this frozen split only."
- Model selection (`results_v3/analysis_v2/07_model_selection.json`) uses a negligibility-resolution rule rather than picking a single "winner" whenever deltas are within noise (see `docs/methodology.md`).
- `results_v3/tc_outlier_audit/`: 38/54 of the worst Tc errors are classified SYSTEMATIC (elevated across all four ablation variants → likely a data/chemistry issue, not fixable by feature engineering alone); 10/54 are flagged "close structural neighbor, large label gap" — candidates for a label review, not a model-capacity issue.
- `results_v3/refrigerant_domain_audit/`: only 5/142 locked refrigerants have HIGH structural similarity to the Tc training pool; 104/142 are LOW. Refrigerants are systematically smaller and far more halogenated than the bulk of the training data (halogen_fraction 0.50 vs train's 0.11). **Predictions for LOW-similarity refrigerants in `results_v3/refrigerant_142_final_predictions.csv` carry a LOW trust_flag for exactly this reason — check it before using a specific molecule's prediction.**
- `results_v3/domain_weighting_experiment/`: tested whether domain-aware reweighting of estimated Tc data recovers value for refrigerant-like chemistry specifically. Selection was done by global validation RMSE only, explicitly not by peeking at the 7-molecule refrigerant-like test subset (too small to select on safely) — that subset's results are reported, not used for selection.

## xgboost/results/ — the newer, larger `unique_molecules_*` experiments

**Diagnostic, not deployment-ready — stated explicitly everywhere these are reported.** Key findings (full detail in each experiment's `audit/00_AUDIT_LOG.txt`):

1. Initial degradation vs the curated XGB-A benchmark (Tc RMSE 38.6 vs 31.2 K) was traced primarily to a **split defect**: a 1,512-molecule Butina cluster (long-chain homologs, all sharing near-saturated Morgan fingerprints regardless of functional group) dominated the train partition and destabilized 5-fold cross-validation (RMSE std of 31.0 K, 68% of the mean).
2. Capping cluster size at 200 (recursive re-clustering of oversized clusters only) fixed this: CV RMSE std dropped ~5x, and Tc/omega test performance **surpassed** the curated XGB-A benchmark despite training on a broader, noisier dataset.
3. Pc's instability was **not** resolved by the split fix (CV std barely moved) — its problem looks like genuine label/domain noise, not clustering, and remains open.
4. Label-source comparison (finetune-file vs Yaws Collection) shows real signal for Tc (Yaws-sourced test predictions ~37% worse RMSE) but **no such gap for omega** — "Yaws data is noisier" is true for Tc specifically, not a blanket rule.
5. Three chemical families show systematic (not noise-like) errors: long-chain fatty acids/glycerides (Tc under-predicted), polyamines (Pc over-predicted, Tc under-predicted), and a handful of charge/radical/hypervalent species with likely SMILES-notation issues (e.g. diatomic elements written as bare bracket atoms missing implicit hydrogens).

## gnn/ — known train/test overlap issue

**This is the most serious open validation gap in the repo.** Per the platform's own README (`cop/PLATFORM_README.md`): "~90% of the fine-tuning molecules also appear in the pre-training file, so the reported test MAE is not a generalization estimate." This was explicitly set aside during platform development to ship the app, not resolved. It means:

- GNN test-set metrics reported anywhere in `gnn/results/` should **not** be read as a clean generalization estimate.
- The **142 locked deployment refrigerants are stated to be clean** of this overlap — protect that set; don't let future data additions compromise it.
- This is a different, likely more severe issue than the XGBoost frozen-v3 protocol's leakage filtering, which explicitly removes val/test/refrigerant overlap from the estimated pool before training (see `xgboost/README.md`). The two pipelines' validity claims are **not equivalent** — XGBoost's are stronger.

## pr_eos/ + cop/

The `refrig` package has a pytest suite (`cop/tests/`) and is validated against CoolProp for the pure-component PR-EOS path (`pr_eos/validation/validate_pr_binary_vs_coolprop.py` for the binary case). Its own README states its limitations plainly (no uncertainty quantification, no applicability domain, no volume translation, no Tc/Pc/omega self-consistency check — see "Documented V1 limitations" in `cop/PLATFORM_README.md`). The most under-appreciated one: predicting Tc, Pc, and omega independently (as both `xgboost/` and `gnn/` do) can produce a triplet whose own PR saturation curve doesn't reproduce the omega you fed in, since omega is *defined* by Psat at Tr=0.7 — there is currently no consistency check catching this.

## Bottom line for anyone using this repo's numbers

- Trust order, highest to lowest: `results_v3/` (Frozen Protocol v3) > `xgboost/results/unique_molecules_splitfix_xgboost/` (diagnostic but split-fixed) > `xgboost/results/unique_molecules_{xgboost,yawspriority_xgboost}/` (diagnostic, known split defect) > `gnn/` (known train/test overlap, unresolved).
- Always check a prediction's `trust_flag` / structural-similarity-to-train before treating a specific molecule's number as reliable — this applies to every stage.
- No COP or screening result in this repo should be cited as a validated thesis result without reading this file and the specific stage's own limitations section first.
