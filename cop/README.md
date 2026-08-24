# cop/ — Vapour-compression cycle COP, from Tc/Pc/omega to a full state table

This folder is the integrated PR-EOS + cycle + COP platform. **Full architecture, quickstart, and design rationale are in `PLATFORM_README.md`** (the original project README, kept intact) — read that first. This file only orients you and documents what was folded in on top of it.

```
SMILES → (Tc, Pc, omega) → Peng-Robinson EOS → vapour-compression cycle → COP + state table
```

## Layout

- `refrig/` — the core physics package (pure Python, no UI/ML imports at module scope). `refrig/eos/peng_robinson.py` is the Peng-Robinson EOS implementation — see `pr_eos/README.md` for why it lives here rather than duplicated into a separate top-level folder.
- `app.py` — the Streamlit UI.
- `tests/` — pytest suite (`pytest tests/ -v`).
- `.env.example` / `requirements.txt` — platform-specific config and deps (also rolled into the top-level `requirements.txt`).
- `pure_refrigerant/` — single-component COP calculation and validation against experimental Tc (`cop_calculation_for_pure_refrigerant.py`, parity plots, comparison results). Fed by GNN/XGBoost property predictions.
- `mixtures/` — binary and ternary mixture COP scripts (`cop_calculation_for_binary_mixtures.py`, `cop_calculation_for_ternary.py`) plus supporting McLinden binary-dataset generation scripts. These are **standalone scripts, not yet part of the `refrig/` package** — the platform's own roadmap (see `PLATFORM_README.md` → "Roadmap hooks") notes mixtures (`Fluid.composition`, `kij`) are wired into the data structures but the mixing rules aren't implemented in `refrig/` yet. Treat `mixtures/` as the working prototype for that V3 milestone.

## What was excluded from git here

- `checkpoints/` and `merged_pipeline/` (the platform's own README already documents these as external — see "Pointing the app at your trained model" in `PLATFORM_README.md`)
- Two large mixture-sweep result CSVs (`ternary_mixture_cop_9comp.csv` ~690MB, `binary_mixture_cop_data_81comp.csv` ~95MB) — regenerate with the scripts in `mixtures/`; smaller summary CSVs (`binary_mixture_cop_sweep.csv`, `COP_comparison_results.csv`, etc.) are kept.
- Several earlier iterations of the mixture COP scripts (`cop_comparison.py`, `cop_calculations_binary_mixtures.py`, `COP_calculation_for_binary_mixtures_1.py`) — superseded by the versions kept here; still on local disk at `2_Refrigrant/PR_EOS_Validation/` if needed.

## Before treating any COP number here as a thesis result

Read "Before any COP here is treated as a thesis result" in `PLATFORM_README.md` and `docs/validation.md` — the property models feeding this platform have known train/test overlap issues (see `gnn/README.md`), and this platform's own limitations section (no uncertainty, no applicability domain, no volume translation, no Tc/Pc/omega self-consistency check) applies to every number this code produces.
