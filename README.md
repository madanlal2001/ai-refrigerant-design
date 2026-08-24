# ai-refrigerant-design

Research objective: screen novel refrigerant candidates by predicting their thermodynamic properties from molecular structure alone, then evaluating them in a real vapour-compression cycle — without needing to synthesize or experimentally measure each candidate first.

```
Molecular structure (SMILES)
    │
    ├── XGBoost  ─┐   critical-property prediction
    └── GNN      ─┴─→  (Tc, Pc, omega)
                              │
                        Peng-Robinson EOS
                              │
                    vapour-compression cycle → COP
                              │
              refrigerant / mixture screening & ranking
```

## 1. XGBoost critical-property prediction (`xgboost/`)

Gradient-boosted trees (RDKit-2D descriptors + Morgan fingerprints) predicting critical temperature (Tc), critical pressure (Pc), and acentric factor (omega). The primary result — "Frozen Protocol v3" — is a four-way ablation (XGB-A/B/C/D) on a single frozen, cluster-aware train/val/test split, with extensive read-only diagnostic audits (outlier classification, refrigerant-domain coverage, domain-weighted reweighting experiments). A separate set of experiments trains the same methodology on a larger, broader dataset and diagnoses (and partly fixes) why it initially underperformed the curated benchmark. See `xgboost/README.md`.

## 2. GNN / D-MPNN approach (`gnn/`)

A hybrid message-passing neural network predicting the same three properties, pretrained on a larger estimated-property pool then fine-tuned on curated experimental data. Used to screen a 61-molecule candidate set. Has a known, documented train/test overlap caveat — see `docs/validation.md` before trusting its reported test metrics. See `gnn/README.md`.

## 3. PR-EOS implementation (`pr_eos/`)

Peng-Robinson equation of state, full double-precision constants (a documented fix over the textbook-rounded values, which fail at the critical point). Implemented as part of the `cop/refrig/` package rather than standalone — see `pr_eos/README.md` for why.

## 4. COP calculation (`cop/`)

Predicted Tc/Pc/omega → Peng-Robinson EOS → vapour-compression cycle → COP and a full thermodynamic state table, via a clean, tested Python package (`cop/refrig/`) with a Streamlit front-end. `run_cycle()` is a pure function by design, so property-sensitivity analysis is a second function call, not a new subsystem. See `cop/README.md` and `cop/PLATFORM_README.md`.

## 5. Mixture calculations (`cop/mixtures/`)

Binary and ternary mixture COP scripts, built on the McLinden reference dataset. Mixing rules are not yet part of the core `refrig` package (that's a documented roadmap item) — these are the working prototype for it.

## 6. Validation against CoolProp / literature

`pr_eos/validation/validate_pr_binary_vs_coolprop.py` validates the binary-mixture PR-EOS implementation against CoolProp. `cop/tests/` is a pytest suite covering the core thermodynamics. See `docs/validation.md` for the full validation status of every stage — including what is **not yet validated**.

## 7. Refrigerant screening

`xgboost/results_v3/final_refrigerant_predictions/` (142 locked candidates, three Tc recipes) and `gnn/results/` (61-candidate screen) are the two screening outputs this project has produced so far.

## 8. Repository structure

```
ai-refrigerant-design/
├── xgboost/          Critical-property prediction (validated Frozen Protocol v3 + diagnostic experiments)
├── gnn/               D-MPNN critical-property prediction
├── pr_eos/            Peng-Robinson EOS validation/utility scripts (core implementation lives in cop/refrig/)
├── cop/               Integrated PR-EOS + cycle + COP package, Streamlit app, mixture scripts
├── data/               Input and derived datasets (small files only — see data/README.md)
├── docs/
│   ├── methodology.md      How each stage works and how they connect
│   ├── validation.md       What's validated, what isn't, and the known caveats
│   └── reproducibility.md  What's excluded from git and how to regenerate it
└── requirements.txt
```

No top-level `experiments/`, `results/`, or `notebooks/` folders — results and notebooks live inside each pipeline stage's own folder (`xgboost/results_v3/`, `xgboost/results/`, `gnn/results/`, `xgboost/XGBoost.ipynb`, `gnn/GNN_model.ipynb`) since each is only meaningful alongside the code that produced it. This is a deliberate adaptation of the originally sketched flat layout.

## 9. Reproducibility instructions

See `docs/reproducibility.md` for the full environment setup, what's excluded from git and how to regenerate it, determinism notes, and the exact order of operations to reproduce every result in this repo from the committed data files.

## 10. Limitations

This is active, unpublished research — treat every number here accordingly:

- **The `xgboost/results/unique_molecules_*` experiments are diagnostic, not deployment-ready.** They exist to answer "does more/different data help, and why" — not to replace the Frozen Protocol v3 models.
- **The GNN pipeline has a known, unresolved train/test overlap issue** (~90% of fine-tuning molecules also appear in the pretraining file) — its reported test metrics are not a clean generalization estimate. The 142 locked refrigerants are confirmed clean of this specific issue.
- **No stage propagates prediction uncertainty into the final COP number**, and there is no check that independently-predicted Tc/Pc/omega triplets are self-consistent with each other (omega is *defined* by the other two via the saturation curve).
- **Refrigerant candidates are structurally unlike most of the training data** (smaller, far more halogenated) — most predictions for the 142 locked candidates carry a LOW or MEDIUM trust flag by design; check it before treating any single molecule's number as reliable.
- Single frozen data splits throughout — reported metrics have no error bars and should not be read as statistically significant differences between model variants unless a stage's own docs say otherwise (see the negligibility-resolution convention in `docs/methodology.md`).

Full detail, including the evidence behind each of the above, is in `docs/validation.md`.
