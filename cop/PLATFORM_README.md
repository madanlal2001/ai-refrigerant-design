# Refrigerant Screening Platform — V1

SMILES → (Tc, Pc, ω) → Peng–Robinson EOS → vapour-compression cycle → COP
and the full state table.

Pure components only in V1.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with R134a (`FCC(F)(F)F`) and literature properties, so you can
check it reproduces something you already trust before you point it at a novel
molecule.

Run the tests:

```bash
pytest tests/ -v
```

---

## What runs without what

| You want | You need |
|---|---|
| COP, states, plots, run records | `numpy`, `scipy` (+ `streamlit`, `pandas`, `matplotlib` for the UI) |
| Canonical SMILES, molar mass, formula | `rdkit` (optional) |
| Predict Tc/Pc/ω from structure | `torch`, `torch_geometric`, `rdkit`, **and** `checkpoints/merged_pipeline.pt` |

The GNN option is **hidden** in the UI until the checkpoint exists. The core
package never imports torch — which is why the thermodynamics stays installable
and testable in seconds.

---

## Pointing the app at your trained model

`gnn_pipeline.py` (the DMPNN training pipeline) and its `merged_pipeline.pt`
checkpoint are not part of this repo — they live wherever you trained them.
`refrig/predictors/gnn.py` needs both to be reachable:

- **Checkpoint**: defaults to `checkpoints/merged_pipeline.pt` inside this
  project. Override with:
  ```bash
  export REFRIG_GNN_CHECKPOINT=/path/to/your/checkpoints/merged_pipeline.pt
  ```
- **`gnn_pipeline` module**: by default it must already be importable (copied
  or symlinked into this project, or already on `PYTHONPATH`). If it lives
  somewhere else, point at the directory that contains it:
  ```bash
  export REFRIG_GNN_PIPELINE_DIR=/path/to/GNN
  ```

See `.env.example`. If neither variable is set, the app falls back to the
project-relative default above; if that's also missing, the GNN option stays
hidden and the sidebar explains exactly which variable to set.

---

## Architecture

The rule that everything else follows from:

```
app.py  ──imports──>  refrig/        (physics, pure Python)
refrig/ ──imports──>  NOTHING UI     (never streamlit, never torch at module scope)
```

```
refrig/
  core/
    property.py     Property: a value + provenance + (future) uncertainty
    component.py    Component, Fluid (composition + kij carried from day 1)
    cp.py           CpProvider interface; Poly4Cp with analytic integrals
    chem.py         optional RDKit helpers
    constants.py    R, unit conversions — one definition, SI everywhere
    errors.py       physics failure vs solver failure, kept distinct
  eos/
    peng_robinson.py  PR76. Holds no operating conditions; methods take (T, P)
  equilibrium/
    psat.py         fugacity-equality Psat, returns convergence diagnostics
    dome.py         saturation envelope for the diagrams
  cycle/
    solvers.py      bracket-then-Brent root finders at fixed S or H
    vcc.py          run_cycle(fluid, spec) -> CycleResult   <-- PURE FUNCTION
  predictors/
    base.py         Predictor interface + applicability-domain hook
    manual.py       user-entered properties (the reference implementation)
    gnn.py          DMPNN adapter, optional, activates on checkpoint presence
  runio/
    run_record.py   the reproducibility unit
```

### Why `run_cycle` is a pure function

Because the thesis contribution is not the COP number. It's *how property error
propagates into the COP*. With `run_cycle(fluid, spec)` free of globals and I/O:

```python
base = run_cycle(fluid, spec).COP
fluid.component.override("Tc", Tc * 1.02)
perturbed = run_cycle(fluid, spec).COP        # sensitivity, for free
```

Uncertainty propagation is the same call inside a Monte-Carlo loop. Neither
needs a new subsystem. `tests/test_thermo.py::test_cop_sensitivity_to_tc_is_computable`
is that experiment, expressed as a test.

### Why `Property` instead of a float

A float cannot tell you whether 374.21 K was measured, predicted, or typed in
by hand at 2 a.m. Every value carries its source, the model that produced it,
and — if edited — the value it replaced. That chain lands in the run record.

### Why `uncertainty` exists but is always `None`

So that adding UQ later changes zero call sites. Same reasoning for
`Fluid.composition` / `kij` (mixtures) and `Predictor.applicability` (AD).

---

## Documented V1 limitations

These are stated in the app itself, in an expander, on purpose.

- **No uncertainty.** Every number is a bare point estimate.
- **No applicability domain.** Nothing warns you that a molecule is unlike the
  training set.
- **No mixtures.** The data structures carry composition and k_ij; the mixing
  rules are not written.
- **No volume translation.** PR is poor at liquid density; state 3 and the
  subcooling enthalpy inherit that.
- **No (Tc, Pc, ω) consistency check.** ω is *defined* by Psat at Tr = 0.7.
  Predicting the three independently can yield a triplet whose own PR
  saturation curve does not reproduce the ω you fed in. This is the most
  under-appreciated error source in the whole pipeline.

---

## Notes on the port from `COP.ipynb`

The physics is unchanged. Three things were corrected:

1. **Operating conditions were module globals** (`T_EVAP`, `T_COND`, `ETA_C`…).
   They are now `CycleSpec` fields. This is what makes the function pure.
2. **`R = 8.314` → `R = 8.314462618`** (CODATA). ~0.005 % — free to get right.
3. **PR constants `0.45724 / 0.07780` → full double precision.** These are fine
   everywhere *except* at the critical point, where the cubic has a triple root
   and is cube-root sensitive to coefficient error: the rounding displaces Zc to
   0.3214 instead of 0.30740 and produces a spurious complex pair. Effect on COP
   away from the critical point: 0.002 %. Effect on the critical-point test:
   pass instead of fail.

---

## Roadmap hooks already in place

| Version | Work | What already exists |
|---|---|---|
| V2 | ML Cp(T) | `CpProvider` interface — implement `cp_ig`, `h_ig`, `s_ig` |
| V3 | Mixtures | `Fluid.composition`, `Fluid.kij`, `PengRobinson` guard |
| V4 | Uncertainty | `Property.uncertainty`, pure `run_cycle` |
| V5 | Applicability domain | `Predictor.applicability` → `ADResult` |
| V6 | Ranking | `RunRecord` JSON is already the row format |

---

## Before any COP here is treated as a thesis result

The property models feeding this platform have a train/test contamination
issue that was set aside to build the app: ~90 % of the fine-tuning molecules
also appear in the pre-training file, so the reported test MAE is not a
generalization estimate. The 142 deployment molecules are clean — protect that
set. The platform is correct regardless of what goes in; the numbers coming out
are only as good as the properties.
