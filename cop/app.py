"""Streamlit interface for the refrigerant screening platform.

This file is a PRESENTATION LAYER. It contains no thermodynamics, no solver,
and no correlation. It reads widgets, builds core objects, calls run_cycle(),
and renders the result. If you find yourself writing physics here, it belongs
in `refrig/` instead -- that is the rule that keeps the science testable,
scriptable, and citable without Streamlit in the loop.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from refrig import (Component, CycleSpec, CycleStatus, EOSSettings, Fluid,
                    PengRobinson, Poly4Cp, Property, Source, __version__,
                    run_cycle)
from refrig import config
from refrig.core.chem import analyze_smiles, rdkit_available
from refrig.equilibrium.dome import saturation_dome
from refrig.predictors.gnn import GNNPredictor
from refrig.predictors.manual import ManualPredictor
from refrig.runio import RunRecord

st.set_page_config(page_title="Refrigerant Screening Platform",
                   page_icon="❄", layout="wide")

SOURCE_BADGE = {
    Source.PREDICTED: "🔮 predicted",
    Source.USER_OVERRIDE: "✏️ user override",
    Source.LITERATURE: "📖 literature",
    Source.EXPERIMENTAL: "🧪 experimental",
    Source.ESTIMATED: "≈ estimated",
    Source.DEFAULT: "· default",
}


# ---------------------------------------------------------------- helpers ---

@st.cache_resource
def get_gnn_predictor(checkpoint: str, pipeline_dir: Optional[str]) -> GNNPredictor:
    """Cached so the checkpoint is loaded once per process, not per rerun."""
    return GNNPredictor(checkpoint=Path(checkpoint),
                        pipeline_dir=Path(pipeline_dir) if pipeline_dir else None)


@st.cache_data(show_spinner=False)
def cached_predict(checkpoint: str, pipeline_dir: Optional[str], smiles: str) -> Dict[str, dict]:
    """Cache predictions by (checkpoint, pipeline_dir, smiles).

    3D conformer embedding costs ~0.5-3 s per molecule; without this the app
    feels broken while sliders are being dragged.
    """
    props = get_gnn_predictor(checkpoint, pipeline_dir).predict(smiles)
    return {k: v.to_dict() for k, v in props.items()}


def badge(p: Property) -> str:
    b = SOURCE_BADGE.get(p.source, p.source.value)
    if p.was_overridden:
        orig = p.overrides
        b += f" (was {orig.value:.6g} {orig.unit}, {SOURCE_BADGE.get(orig.source, '')})"
    return b


def fig_ph(result, eos, cp, spec):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    dome = saturation_dome(eos, cp, spec.T_ref, spec.P_ref)
    if len(dome.T):
        ax.plot(dome.h_liq / 1000, dome.P / 1e5, "-", color="#4c72b0", lw=1.5)
        ax.plot(dome.h_vap / 1000, dome.P / 1e5, "-", color="#4c72b0", lw=1.5,
                label="saturation dome")

    order = ["1", "2", "3", "4", "1"]
    h = [result.state(k).state.h / 1000 for k in order]
    P = [result.state(k).state.P / 1e5 for k in order]
    ax.plot(h, P, "o-", color="#c44e52", lw=2, ms=6, label="cycle")
    h2s = result.state("2s").state.h / 1000
    ax.plot([h[0], h2s], [P[0], result.state("2s").state.P / 1e5], "--",
            color="#dd8452", lw=1.4, label="isentropic (2s)")
    for k in ["1", "2", "3", "4"]:
        s = result.state(k)
        ax.annotate(k, (s.state.h / 1000, s.state.P / 1e5),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("Specific enthalpy  h  [kJ/mol]")
    ax.set_ylabel("Pressure  P  [bar]")
    ax.set_title(f"P–h diagram — {result.fluid_label}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_ts(result, eos, cp, spec):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    dome = saturation_dome(eos, cp, spec.T_ref, spec.P_ref)
    if len(dome.T):
        ax.plot(dome.s_liq, dome.T, "-", color="#4c72b0", lw=1.5)
        ax.plot(dome.s_vap, dome.T, "-", color="#4c72b0", lw=1.5,
                label="saturation dome")

    order = ["1", "2", "3", "4", "1"]
    s = [result.state(k).state.s for k in order]
    T = [result.state(k).state.T for k in order]
    ax.plot(s, T, "o-", color="#c44e52", lw=2, ms=6, label="cycle")
    st2s = result.state("2s").state
    ax.plot([s[0], st2s.s], [T[0], st2s.T], "--", color="#dd8452", lw=1.4,
            label="isentropic (2s)")
    for k in ["1", "2", "3", "4"]:
        p = result.state(k)
        ax.annotate(k, (p.state.s, p.state.T), textcoords="offset points",
                    xytext=(7, 5), fontsize=10, fontweight="bold")
    ax.axhline(eos.Tc, color="grey", ls=":", lw=1)
    ax.annotate(f"Tc = {eos.Tc:.1f} K", (ax.get_xlim()[0], eos.Tc),
                fontsize=8, color="grey", va="bottom")
    ax.set_xlabel("Specific entropy  s  [J/(mol·K)]")
    ax.set_ylabel("Temperature  T  [K]")
    ax.set_title(f"T–s diagram — {result.fluid_label}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_cp(cp, T_lo, T_hi):
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    T = np.linspace(T_lo - 20, T_hi + 60, 200)
    ax.plot(T, [cp.cp_ig(t) for t in T], color="#55a868", lw=2)
    ax.axvspan(T_lo, T_hi, alpha=0.12, color="#c44e52",
               label="cycle temperature span")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("cp_ig [J/(mol·K)]")
    ax.set_title("Ideal-gas heat capacity (POLY4)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar ---

st.sidebar.title("❄ Refrigerant Screening")
st.sidebar.caption(f"core v{__version__}")

mode = st.sidebar.radio("Interface mode", ["Basic", "Expert"], horizontal=True,
                        help="Expert exposes the reference state, solver "
                             "tolerances and every predicted property.")
expert = mode == "Expert"

st.sidebar.header("1 · Molecule")
smiles = st.sidebar.text_input("SMILES", value="FCC(F)(F)F",
                               help="Structure of the working fluid.")
mol = analyze_smiles(smiles)
if not mol.valid:
    st.sidebar.error(mol.message)
elif mol.formula:
    st.sidebar.caption(f"{mol.formula} · MW {mol.mw*1000:.2f} g/mol · "
                       f"canonical: `{mol.canonical_smiles}`")
elif mol.message:
    st.sidebar.caption(mol.message)

# ---- property source -------------------------------------------------------

st.sidebar.header("2 · Tc, Pc, ω")

CHECKPOINT = str(config.gnn_checkpoint_path())
PIPELINE_DIR = config.gnn_pipeline_dir()
PIPELINE_DIR_STR = str(PIPELINE_DIR) if PIPELINE_DIR else None
gnn = get_gnn_predictor(CHECKPOINT, PIPELINE_DIR_STR)
gnn_ok = gnn.available()

options = ["Enter manually"] + (["Predict from SMILES (GNN)"] if gnn_ok else [])
prop_mode = st.sidebar.radio("Source", options, index=0)

if not gnn_ok:
    st.sidebar.info("**GNN prediction unavailable.** " + gnn.why_unavailable(),
                    icon="ℹ️")

base_props: Optional[Dict[str, Property]] = None
predictor_desc: Optional[dict] = None

if prop_mode.startswith("Predict"):
    try:
        with st.spinner("Featurizing molecule and running the GNN…"):
            raw = cached_predict(CHECKPOINT, PIPELINE_DIR_STR, mol.canonical_smiles)
        base_props = {
            "Tc": Property("Tc", raw["Tc"]["value"], "K", Source.PREDICTED,
                           model_id=raw["Tc"]["model_id"]),
            "Pc": Property("Pc", raw["Pc"]["value"], "Pa", Source.PREDICTED,
                           model_id=raw["Pc"]["model_id"]),
            "omega": Property("omega", raw["omega"]["value"], "-",
                              Source.PREDICTED, model_id=raw["omega"]["model_id"]),
        }
        predictor_desc = gnn.describe()
        st.sidebar.success("Predicted. Values remain editable below.")
    except Exception as exc:
        st.sidebar.error(f"Prediction failed: {exc}")

# ---- editable values -------------------------------------------------------
# "Every predicted value must remain editable." In GNN mode an edit becomes a
# real Property.override() so the prediction survives in the audit chain. In
# manual mode there is no prediction to override, so the value is simply the
# user's -- chaining an override onto itself would be noise.

if base_props is not None:
    st.sidebar.caption("Edit any value to override the prediction. The original "
                       "is kept in the run record.")
    default_Tc = float(base_props["Tc"].value)
    default_Pc = float(base_props["Pc"].value * 1e-5)
    default_w = float(base_props["omega"].value)
else:
    default_Tc, default_Pc, default_w = 374.21, 40.593, 0.32684

Tc_in = st.sidebar.number_input("Tc  [K]", value=default_Tc, min_value=1.0,
                                max_value=2000.0, step=1.0, format="%.4f")
Pc_in = st.sidebar.number_input("Pc  [bar]", value=default_Pc, min_value=0.01,
                                max_value=1000.0, step=0.1, format="%.4f")
w_in = st.sidebar.number_input("ω  [-]", value=default_w, min_value=-0.5,
                               max_value=2.0, step=0.001, format="%.5f")

if base_props is None:
    mp = ManualPredictor(Tc_K=Tc_in, Pc_bar=Pc_in, omega=w_in,
                         source=Source.USER_OVERRIDE)
    base_props = mp.predict(mol.canonical_smiles)
    predictor_desc = mp.describe()

st.sidebar.header("3 · Cp_ig coefficients")
st.sidebar.caption("cp_ig = a + b·T + c·T² + d·T³ + e·T⁴   [J/(mol·K)], T in K")
a = st.sidebar.number_input("a", value=19.4, format="%.6g")
b = st.sidebar.number_input("b", value=0.2585, format="%.6g")
c = st.sidebar.number_input("c", value=-1.29e-4, format="%.6g")
d = st.sidebar.number_input("d", value=2.05e-8, format="%.6g")
e = st.sidebar.number_input("e", value=0.0, format="%.6g")

st.sidebar.header("4 · Cycle conditions")
T_evap = st.sidebar.number_input("Evaporator T  [K]", value=283.15, step=1.0,
                                 format="%.2f")
T_cond = st.sidebar.number_input("Condenser T  [K]", value=313.15, step=1.0,
                                 format="%.2f")
superheat = st.sidebar.number_input("Superheat  [K]", value=0.0, min_value=0.0,
                                    step=1.0)
subcool = st.sidebar.number_input("Subcooling  [K]", value=0.0, min_value=0.0,
                                  step=1.0)
eta = st.sidebar.slider("Compressor isentropic efficiency  [-]",
                        min_value=0.05, max_value=1.0, value=1.0, step=0.01)

if expert:
    st.sidebar.header("5 · Expert settings")
    st.sidebar.caption("Reference state — arbitrary datum. COP is invariant to "
                       "it; absolute h and s are not.")
    T_ref = st.sidebar.number_input("T_ref  [K]", value=233.15, format="%.2f")
    P_ref = st.sidebar.number_input("P_ref  [Pa]", value=101325.0, format="%.1f")
    psat_tol = st.sidebar.number_input("Psat tolerance  |ln φL − ln φV|",
                                       value=1e-8, format="%.1e")
    psat_iter = st.sidebar.number_input("Psat max iterations", value=150,
                                        min_value=10, max_value=10000, step=10)
    m_dot_on = st.sidebar.checkbox("Specify molar flow (report duties in W)")
    m_dot = st.sidebar.number_input("ṁ  [mol/s]", value=1.0, min_value=1e-6,
                                    format="%.4g") if m_dot_on else None
    st.sidebar.number_input("k_ij  [-]", value=0.0, disabled=True,
                            help="Binary interaction parameter. Pure components "
                                 "only in V1 — this activates with mixtures.")
else:
    T_ref, P_ref = 233.15, 101325.0
    psat_tol, psat_iter, m_dot = 1e-8, 150, None


# ------------------------------------------------------------------ build ---

comp = Component(canonical_smiles=mol.canonical_smiles,
                 name=mol.formula or mol.canonical_smiles,
                 inchikey=mol.inchikey)
comp.properties = dict(base_props)

# Apply overrides only where the user actually changed something.
if not np.isclose(Tc_in, base_props["Tc"].value, rtol=0, atol=1e-9):
    comp.override("Tc", Tc_in)
if not np.isclose(Pc_in * 1e5, base_props["Pc"].value, rtol=0, atol=1e-4):
    comp.override("Pc", Pc_in * 1e5)
if not np.isclose(w_in, base_props["omega"].value, rtol=0, atol=1e-12):
    comp.override("omega", w_in)

comp.cp = Poly4Cp(a, b, c, d, e, source=Source.USER_OVERRIDE)
if mol.mw:
    comp.set("MW", mol.mw, Source.EXPERIMENTAL)

fluid = Fluid.pure(comp)
spec = CycleSpec(T_evap=T_evap, T_cond=T_cond, superheat=superheat,
                 subcool=subcool, eta_isen=eta, T_ref=T_ref, P_ref=P_ref,
                 m_dot=m_dot,
                 eos=EOSSettings(psat_tol=psat_tol, psat_max_iter=int(psat_iter)))

result = run_cycle(fluid, spec)


# ----------------------------------------------------------------- render ---

st.title("Molecular Refrigerant Screening Platform")
st.caption("SMILES → Tc / Pc / ω → Peng–Robinson EOS → vapour-compression cycle")

prov = st.columns(3)
for col, key, disp in zip(prov, ["Tc", "Pc", "omega"], ["Tc", "Pc", "ω"]):
    p = comp.get(key)
    shown = p.value * 1e-5 if key == "Pc" else p.value
    unit = "bar" if key == "Pc" else ("K" if key == "Tc" else "")
    col.metric(f"{disp} [{unit}]" if unit else disp, f"{shown:.4f}")
    col.caption(badge(p))

st.divider()

if not result.ok:
    icon = {"transcritical": "🌡️", "infeasible": "🚫",
            "not_converged": "🔁"}.get(result.status.value, "⚠️")
    st.error(f"{icon}  **{result.status.value.replace('_', ' ').title()}** — "
             f"{result.message}")
    st.info("This is a reported outcome, not a crash. In a screening run this "
            "molecule would be flagged and the batch would continue.",
            icon="ℹ️")
else:
    for w in result.warnings:
        st.warning(w, icon="⚠️")

    m = st.columns(4)
    m[0].metric("COP", f"{result.COP:.4f}")
    m[1].metric("Compressor work", f"{result.W_comp/1000:.3f} kJ/mol")
    m[2].metric("Cooling capacity", f"{result.Q_evap/1000:.3f} kJ/mol")
    m[3].metric("Heat rejected", f"{result.Q_cond/1000:.3f} kJ/mol")

    m2 = st.columns(4)
    m2[0].metric("Pressure ratio", f"{result.pressure_ratio:.3f}")
    m2[1].metric("Discharge T", f"{result.T_discharge:.2f} K",
                 delta=f"{result.T_discharge - 273.15:.1f} °C",
                 delta_color="off")
    m2[2].metric("Volumetric capacity",
                 f"{result.volumetric_capacity/1e6:.3f} MJ/m³")
    m2[3].metric("Throttle outlet quality", f"{result.quality_evap_inlet:.4f}")

    if result.Q_evap_W is not None:
        m3 = st.columns(3)
        m3[0].metric("Cooling duty", f"{result.Q_evap_W/1000:.3f} kW")
        m3[1].metric("Compressor power", f"{result.W_comp_W/1000:.3f} kW")
        m3[2].metric("Condenser duty", f"{result.Q_cond_W/1000:.3f} kW")

    carnot = spec.T_evap / (spec.T_cond - spec.T_evap)
    st.caption(f"Carnot COP for these reservoir temperatures is "
               f"{carnot:.3f}; second-law efficiency "
               f"{100*result.COP/carnot:.1f}%.")

    tab_states, tab_plots, tab_sat, tab_cp, tab_rec = st.tabs(
        ["State table", "P–h & T–s", "Saturation", "Cp_ig", "Run record"])

    with tab_states:
        rows = []
        for s in result.states:
            rows.append({
                "State": s.label,
                "Description": s.description,
                "T [K]": s.state.T,
                "P [bar]": s.state.P * 1e-5,
                "Z [-]": s.state.Z,
                "v [cm³/mol]": s.state.v * 1e6,
                "ρ [mol/m³]": s.state.rho,
                "h [kJ/mol]": s.state.h / 1000,
                "s [J/mol·K]": s.state.s,
                "Phase": s.state.phase.value,
                "Quality [-]": s.quality,
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={
                         c: st.column_config.NumberColumn(c, format="%.5g")
                         for c in df.columns if df[c].dtype.kind == "f"})
        st.caption("h and s are relative to the reference state "
                   f"({spec.T_ref:.2f} K, {spec.P_ref:.0f} Pa). The datum is "
                   "arbitrary and cancels out of COP.")
        st.download_button("⬇ States CSV", df.to_csv(index=False),
                           f"states_{mol.canonical_smiles}.csv", "text/csv")

    with tab_plots:
        eos = PengRobinson(fluid, spec.eos)
        c1, c2 = st.columns(2)
        with st.spinner("Building saturation dome…"):
            c1.pyplot(fig_ph(result, eos, comp.cp, spec))
            c2.pyplot(fig_ts(result, eos, comp.cp, spec))

    with tab_sat:
        sc = st.columns(2)
        for col, sat, label in ((sc[0], result.sat_evap, "Evaporator"),
                                (sc[1], result.sat_cond, "Condenser")):
            col.markdown(f"**{label}**")
            col.write({
                "T [K]": round(sat.T, 4),
                "Psat [bar]": round(sat.P * 1e-5, 6),
                "Z vapour": round(sat.Z_vapor, 6),
                "Z liquid": round(sat.Z_liquid, 6),
                "iterations": sat.iterations,
                "residual |ln φL − ln φV|": f"{sat.residual:.2e}",
                "converged": sat.converged,
            })
        st.caption("Convergence diagnostics are shown rather than hidden. If "
                   "the iteration count is near the limit, distrust the result.")

    with tab_cp:
        st.pyplot(fig_cp(comp.cp, spec.T_evap, result.T_discharge))
        st.write(comp.cp.to_dict())
        st.info("These coefficients are entered by hand in V1. The Cp module is "
                "an interface — a future ML cp_ig(T) model drops in behind it "
                "with no change to the EOS or the cycle.", icon="🔌")

    with tab_rec:
        record = RunRecord(
            fluid=fluid, result=result,
            inputs={"smiles_input": smiles,
                    "canonical_smiles": mol.canonical_smiles,
                    "interface_mode": mode},
            models_used=[predictor_desc] if predictor_desc else [],
        )
        js = record.to_json()
        st.download_button("⬇ Run record (JSON)", js,
                           f"run_{record.run_id}.json", "application/json",
                           type="primary")
        st.caption("Contains every input, every property with its provenance, "
                   "the model that produced it, solver settings, and package "
                   "versions — enough to reproduce this result without the app.")
        st.json(record.to_dict(), expanded=False)

with st.expander("What this V1 does and does not do"):
    st.markdown(f"""
**Does**
- Pure components only, via Peng–Robinson (PR76 κ correlation).
- Psat by fugacity equality with a Wilson initial guess; departure functions
  for h and s; analytic da/dT.
- Reports transcritical and non-converged cases as *results*, not crashes.

**Does not — by design, and stated here rather than discovered later**
- **No uncertainty.** `Property.uncertainty` exists and is always `None`. Every
  number shown is a point estimate with no error bar.
- **No applicability domain.** Nothing tells you whether this molecule
  resembles the training data.
- **No mixtures.** `Fluid` carries composition and k_ij, but the mixing rules
  are not implemented.
- **No volume translation.** PR is poor at liquid density, so state 3 and the
  subcooling enthalpy inherit that error.
- **Thermodynamic consistency of (Tc, Pc, ω) is not checked.** ω is *defined*
  by Psat at Tr = 0.7. Predicting the three independently can produce a triplet
  whose own PR saturation curve does not reproduce the given ω.
- RDKit present: **{rdkit_available()}** · GNN checkpoint present: **{gnn_ok}**
""")
