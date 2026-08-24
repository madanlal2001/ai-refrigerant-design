# Methodology

## Pipeline overview

```
Molecular structure (SMILES)
    │
    ├── xgboost/  ─┐
    └── gnn/      ─┴─→  Tc, Pc, omega (critical temperature, pressure, acentric factor)
                              │
                        pr_eos/ (Peng-Robinson EOS, cop/refrig/eos/)
                              │
                        cop/  (vapour-compression cycle → COP + state table)
                              │
                  refrigerant / mixture screening (pure, binary, ternary)
```

Two independent property-prediction approaches (XGBoost, GNN) feed the same downstream thermodynamics — this was a deliberate cross-check, not redundancy: if both models roughly agree on a candidate refrigerant's Tc/Pc/omega, that agreement is weak evidence against a model-specific artifact.

## xgboost/ — Frozen Protocol v3

- **Canonical identity**: `Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)` everywhere; duplicate canonical structures within a file are a hard failure.
- **Features**: RDKit-2D descriptors (`Descriptors.descList`, 217 of them) concatenated with a Morgan fingerprint (radius 2, 2048 bits, no counts) — 2265-dim total.
- **Split**: Morgan r2/2048 fingerprints → Butina clustering at distance cutoff 0.60 → deterministic normalized-deficit assignment to 80/10/10 train/val/test, computed on **all** benchmark molecules before any target-specific label filtering (so a molecule's split doesn't depend on which property it happens to have labeled).
- **Leakage discipline**: the 142 locked refrigerants are excluded from every training file (hard failure if found); the "estimated" supplementary pool is filtered to remove anything overlapping val/test/refrigerants before it's allowed to contribute (even at low sample weight) to training.
- **Ablation** (`02_train_variants.py`): XGB-A (experimental only) → XGB-B (+ weighted estimated data) → XGB-C (+ predicted-Tb auxiliary feature) → XGB-D (both). Val is used only for early stopping; test is evaluated exactly once per (variant, target).
- **Negligibility rule**: for each ablation contrast, `resolution = 0.5 * median(pooled |error|)` of the two variants being compared; `|ΔRMSE|` or `|ΔMAE|` below that resolution is reported as NEGLIGIBLE rather than a "winner" — this project explicitly avoids over-reading small deltas on a single frozen split as if they were statistically significant.

## gnn/ — Hybrid D-MPNN pretrain → fine-tune

A PyTorch Geometric message-passing network, pretrained on the larger estimated-property pool then fine-tuned on the curated experimental set, with separate specialist heads per target (Tc; Pc+omega jointly). See `gnn/README.md` for the known pretrain/fine-tune overlap caveat — this is the main methodological weak point in the GNN stage relative to the XGBoost stage's stricter leakage filtering.

## pr_eos/ + cop/ — Thermodynamics

Peng-Robinson EOS (PR76) with full-double-precision constants, wrapped in a pure `run_cycle(fluid, spec) -> CycleResult` function so that property-sensitivity analysis (e.g. "how does a 2% Tc error propagate into COP?") is a second, free call to the same function rather than a new subsystem — see `pr_eos/README.md` and `cop/PLATFORM_README.md` for the full design rationale.

## Refrigerant / mixture screening

`gnn/predict_61_candidates.py` and `xgboost/results_v3/final_refrigerant_predictions/` produce candidate Tc/Pc/omega predictions for screening sets (61 GNN candidates; 142 locked refrigerants via XGBoost). `cop/mixtures/` runs binary and ternary mixture COP sweeps over these candidates using the McLinden reference dataset for comparison.

## Iterative-development note

Several stages (GNN, mixture COP scripts) went through multiple numbered iterations during development (GNN1-4; several `cop_calculation_for_binary_*` script variants). Only the current/canonical version of each is committed here — see each stage's own README for which iteration was chosen and why, and what happened to the others (kept locally, not deleted, just not duplicated into git).
