"""Vapor-compression cycle simulator.

    run_cycle(fluid, spec) -> CycleResult

This is a PURE FUNCTION. No globals, no file I/O, no Streamlit. That property
is what makes the later work cheap:
  - sensitivity of COP to Tc error  = perturb Property.value, re-run
  - uncertainty propagation         = Monte-Carlo over the same call
Both become loops over this function instead of new subsystems.

Cycle (states follow the standard numbering):
    1  : compressor inlet   -- vapor at (T_evap + superheat, P_evap)
    2s : isentropic outlet  -- at P_cond with s = s1
    2  : actual outlet      -- h2 = h1 + (h2s - h1)/eta_isen
    3  : condenser outlet   -- liquid at (T_cond - subcool, P_cond)
    4  : evaporator inlet   -- h4 = h3 (isenthalpic throttle) at P_evap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.component import Fluid
from ..core.constants import R
from ..core.errors import ConvergenceError, EOSRootError, InfeasibleCycleError, PropertyError
from ..eos.peng_robinson import EOSSettings, EOSState, PengRobinson, Phase
from ..equilibrium.psat import SaturationResult, psat
from .solvers import solve_T_at_H, solve_T_at_S


class CycleStatus(str, Enum):
    """Outcome of a cycle run.

    Note TRANSCRITICAL and INFEASIBLE are *results*, not exceptions. When you
    screen 142 molecules, "this one is transcritical at 40 C" is information.
    """

    OK = "ok"
    TRANSCRITICAL = "transcritical"
    INFEASIBLE = "infeasible"
    NOT_CONVERGED = "not_converged"


@dataclass(frozen=True)
class CycleSpec:
    """Everything that was a module-level global in COP.ipynb.

    Expert Mode edits this object. Nothing else.
    """

    T_evap: float = 283.15        # K, evaporator saturation temperature
    T_cond: float = 313.15        # K, condenser saturation temperature
    superheat: float = 0.0        # K, T1 = T_evap + superheat
    subcool: float = 0.0          # K, T3 = T_cond - subcool
    eta_isen: float = 1.0         # -, compressor isentropic efficiency

    # Enthalpy/entropy datum. Arbitrary but must be consistent within a run;
    # the cycle only ever uses differences, so the datum cancels out of COP.
    T_ref: float = 233.15         # K
    P_ref: float = 101325.0       # Pa

    m_dot: Optional[float] = None  # mol/s. If given, duties are reported in W.

    eos: EOSSettings = field(default_factory=EOSSettings)

    def validate(self) -> List[str]:
        p: List[str] = []
        if self.T_cond <= self.T_evap:
            p.append(f"T_cond ({self.T_cond:.2f} K) must exceed T_evap "
                     f"({self.T_evap:.2f} K)")
        if self.superheat < 0:
            p.append("superheat must be >= 0")
        if self.subcool < 0:
            p.append("subcool must be >= 0")
        if not (0.0 < self.eta_isen <= 1.0):
            p.append(f"eta_isen must be in (0, 1] (got {self.eta_isen})")
        if self.T_ref <= 0 or self.P_ref <= 0:
            p.append("reference state must be positive")
        if self.m_dot is not None and self.m_dot <= 0:
            p.append("m_dot must be > 0 if given")
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {
            "T_evap_K": self.T_evap, "T_cond_K": self.T_cond,
            "superheat_K": self.superheat, "subcool_K": self.subcool,
            "eta_isentropic": self.eta_isen,
            "T_ref_K": self.T_ref, "P_ref_Pa": self.P_ref,
            "m_dot_mol_per_s": self.m_dot,
            "eos": self.eos.to_dict(),
        }


@dataclass(frozen=True)
class CycleState:
    """A labelled point on the cycle."""

    label: str                 # "1", "2s", "2", "3", "4"
    description: str
    state: EOSState
    quality: Optional[float] = None   # vapor fraction, None if single-phase

    def to_dict(self) -> Dict[str, Any]:
        d = {"label": self.label, "description": self.description,
             "quality": self.quality}
        d.update(self.state.to_dict())
        return d


@dataclass(frozen=True)
class CycleResult:
    """Everything a run produces, successful or not."""

    status: CycleStatus
    fluid_label: str
    spec: CycleSpec

    states: List[CycleState] = field(default_factory=list)
    sat_evap: Optional[SaturationResult] = None
    sat_cond: Optional[SaturationResult] = None

    COP: Optional[float] = None
    W_comp: Optional[float] = None            # J/mol
    Q_evap: Optional[float] = None            # J/mol
    Q_cond: Optional[float] = None            # J/mol
    pressure_ratio: Optional[float] = None
    T_discharge: Optional[float] = None       # K
    volumetric_capacity: Optional[float] = None   # J/m^3 (per m^3 of suction vapor)
    quality_evap_inlet: Optional[float] = None

    W_comp_W: Optional[float] = None          # W, only if m_dot given
    Q_evap_W: Optional[float] = None
    Q_cond_W: Optional[float] = None

    warnings: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status is CycleStatus.OK

    def state(self, label: str) -> CycleState:
        for s in self.states:
            if s.label == label:
                return s
        raise KeyError(label)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "fluid": self.fluid_label,
            "spec": self.spec.to_dict(),
            "saturation": {
                "evaporator": self.sat_evap.to_dict() if self.sat_evap else None,
                "condenser": self.sat_cond.to_dict() if self.sat_cond else None,
            },
            "states": [s.to_dict() for s in self.states],
            "performance": {
                "COP": self.COP,
                "W_comp_J_per_mol": self.W_comp,
                "Q_evap_J_per_mol": self.Q_evap,
                "Q_cond_J_per_mol": self.Q_cond,
                "pressure_ratio": self.pressure_ratio,
                "T_discharge_K": self.T_discharge,
                "volumetric_capacity_J_per_m3": self.volumetric_capacity,
                "quality_evap_inlet": self.quality_evap_inlet,
                "W_comp_W": self.W_comp_W,
                "Q_evap_W": self.Q_evap_W,
                "Q_cond_W": self.Q_cond_W,
            },
            "warnings": self.warnings,
        }


def _failed(status: CycleStatus, fluid: Fluid, spec: CycleSpec, msg: str,
            warnings: List[str]) -> CycleResult:
    return CycleResult(status=status, fluid_label=fluid.label, spec=spec,
                       warnings=warnings, message=msg)


def run_cycle(fluid: Fluid, spec: CycleSpec) -> CycleResult:
    """Simulate one vapor-compression cycle. Never raises for physics reasons."""

    warnings: List[str] = []

    problems = fluid.validate() + spec.validate()
    if problems:
        return _failed(CycleStatus.INFEASIBLE, fluid, spec,
                       "; ".join(problems), warnings)

    comp = fluid.component
    cp = comp.cp
    eos = PengRobinson(fluid, spec.eos)
    T_ref, P_ref = spec.T_ref, spec.P_ref

    # ---- feasibility ---------------------------------------------------
    if spec.T_cond >= eos.Tc:
        return _failed(
            CycleStatus.TRANSCRITICAL, fluid, spec,
            f"T_cond = {spec.T_cond:.2f} K exceeds Tc = {eos.Tc:.2f} K. "
            "The condenser cannot condense; this is a transcritical cycle, "
            "which V1 does not model.", warnings)
    if spec.T_evap >= eos.Tc:
        return _failed(
            CycleStatus.TRANSCRITICAL, fluid, spec,
            f"T_evap = {spec.T_evap:.2f} K exceeds Tc = {eos.Tc:.2f} K.",
            warnings)
    if spec.T_cond > 0.98 * eos.Tc:
        warnings.append(
            f"T_cond = {spec.T_cond:.1f} K is within 2% of Tc = {eos.Tc:.1f} K. "
            "PR is unreliable this close to the critical point.")

    T1 = spec.T_evap + spec.superheat
    T3 = spec.T_cond - spec.subcool

    # Cp range check -- reported, never silently enforced.
    if cp.valid_range is not None:
        for T, what in ((T1, "compressor inlet"), (spec.T_cond, "condenser")):
            if not cp.in_range(T):
                warnings.append(
                    f"{what} T = {T:.1f} K is outside the Cp correlation range "
                    f"{cp.valid_range}. Extrapolating.")

    try:
        # ---- saturation ------------------------------------------------
        sat_e = psat(eos, spec.T_evap)
        sat_c = psat(eos, spec.T_cond)
        P_evap, P_cond = sat_e.P, sat_c.P

        if P_cond <= P_evap:
            return _failed(CycleStatus.INFEASIBLE, fluid, spec,
                           "P_cond <= P_evap; the saturation curve is not "
                           "monotonic here, which means the property set is "
                           "inconsistent.", warnings)

        # ---- state 1: superheated (or saturated) vapor at P_evap --------
        s1 = eos.state(T1, P_evap, Phase.VAPOR, cp, T_ref, P_ref)
        st1 = CycleState("1", "Compressor inlet (vapor)", s1)

        # ---- state 3: subcooled (or saturated) liquid at P_cond ---------
        s3 = eos.state(T3, P_cond, Phase.LIQUID, cp, T_ref, P_ref)
        st3 = CycleState("3", "Condenser outlet (liquid)", s3)

        # ---- state 2s: isentropic compression to P_cond ------------------
        # Saturated vapor at the condenser defines the dome boundary. If the
        # isentrope lands at or below it, the outlet is wet.
        A_c, B_c = eos.AB(spec.T_cond, P_cond)
        Zv_c, Zl_c = sat_c.Z_vapor, sat_c.Z_liquid
        s_vap_c = eos.entropy(spec.T_cond, P_cond, Zv_c, cp, T_ref, P_ref)
        h_vap_c = eos.enthalpy(spec.T_cond, P_cond, Zv_c, cp, T_ref)

        if s1.s <= s_vap_c + 1e-9:
            # Wet compression: outlet sits inside the dome at T_cond.
            s_liq_c = eos.entropy(spec.T_cond, P_cond, Zl_c, cp, T_ref, P_ref)
            h_liq_c = eos.enthalpy(spec.T_cond, P_cond, Zl_c, cp, T_ref)
            denom = s_vap_c - s_liq_c
            x2s = 1.0 if abs(denom) < 1e-12 else (s1.s - s_liq_c) / denom
            x2s = float(np.clip(x2s, 0.0, 1.0))
            h2s = h_liq_c + x2s * (h_vap_c - h_liq_c)
            v2s = (Zl_c + x2s * (Zv_c - Zl_c)) * R * spec.T_cond / P_cond
            s2s_state = EOSState(T=spec.T_cond, P=P_cond,
                                 Z=Zl_c + x2s * (Zv_c - Zl_c), v=v2s,
                                 h=h2s, s=s1.s, phase=Phase.TWO_PHASE,
                                 rho=1.0 / v2s)
            st2s = CycleState("2s", "Isentropic outlet (wet)", s2s_state, quality=x2s)
            warnings.append(
                f"Isentropic compression ends inside the two-phase dome "
                f"(quality {x2s:.3f}). Real compressors would be damaged; "
                "add superheat.")
        else:
            T2s = solve_T_at_S(eos, P_cond, s1.s, cp, T_ref, P_ref,
                               T_start=spec.T_cond)
            s2s_state = eos.state(T2s, P_cond, Phase.VAPOR, cp, T_ref, P_ref)
            st2s = CycleState("2s", "Isentropic outlet (superheated)", s2s_state)

        # ---- state 2: actual outlet -------------------------------------
        h2 = s1.h + (st2s.state.h - s1.h) / spec.eta_isen
        if spec.eta_isen == 1.0:
            st2 = CycleState("2", "Compressor outlet (actual)", st2s.state,
                             quality=st2s.quality)
        else:
            T2 = solve_T_at_H(eos, P_cond, h2, cp, T_ref, T_start=spec.T_cond)
            s2_state = eos.state(T2, P_cond, Phase.VAPOR, cp, T_ref, P_ref)
            st2 = CycleState("2", "Compressor outlet (actual)", s2_state)

        # ---- state 4: isenthalpic throttle to P_evap --------------------
        h4 = st3.state.h
        Zv_e, Zl_e = sat_e.Z_vapor, sat_e.Z_liquid
        h_vap_e = eos.enthalpy(spec.T_evap, P_evap, Zv_e, cp, T_ref)
        h_liq_e = eos.enthalpy(spec.T_evap, P_evap, Zl_e, cp, T_ref)
        s_vap_e = eos.entropy(spec.T_evap, P_evap, Zv_e, cp, T_ref, P_ref)
        s_liq_e = eos.entropy(spec.T_evap, P_evap, Zl_e, cp, T_ref, P_ref)

        denom = h_vap_e - h_liq_e
        x4 = 0.0 if abs(denom) < 1e-12 else (h4 - h_liq_e) / denom
        x4_clipped = float(np.clip(x4, 0.0, 1.0))
        if not (0.0 <= x4 <= 1.0):
            warnings.append(
                f"Throttle outlet quality computed as {x4:.4f}, clipped to "
                f"{x4_clipped:.4f}. This usually means the property set is "
                "thermodynamically inconsistent.")
        Z4 = Zl_e + x4_clipped * (Zv_e - Zl_e)
        s4 = s_liq_e + x4_clipped * (s_vap_e - s_liq_e)
        v4 = Z4 * R * spec.T_evap / P_evap
        s4_state = EOSState(T=spec.T_evap, P=P_evap, Z=Z4, v=v4, h=h4, s=s4,
                            phase=Phase.TWO_PHASE, rho=1.0 / v4)
        st4 = CycleState("4", "Evaporator inlet (after throttle)", s4_state,
                         quality=x4_clipped)

        # ---- energy balance ---------------------------------------------
        Q_evap = s1.h - h4
        W_comp = h2 - s1.h
        Q_cond = h2 - st3.state.h

        if W_comp <= 0:
            return _failed(CycleStatus.INFEASIBLE, fluid, spec,
                           f"Compressor work is non-positive "
                           f"({W_comp:.4g} J/mol).", warnings)
        if Q_evap <= 0:
            return _failed(CycleStatus.INFEASIBLE, fluid, spec,
                           f"Cooling effect is non-positive "
                           f"({Q_evap:.4g} J/mol).", warnings)

        COP = Q_evap / W_comp
        vol_cap = Q_evap / s1.v
        states = [st1, st2s, st2, st3, st4]

        # Energy balance closure check -- Q_cond should equal Q_evap + W_comp.
        imbalance = abs(Q_cond - (Q_evap + W_comp))
        rel = imbalance / max(abs(Q_cond), 1e-12)
        if rel > 1e-8:
            warnings.append(
                f"Energy balance closes to only {rel:.2e} relative error.")

        m = spec.m_dot
        return CycleResult(
            status=CycleStatus.OK, fluid_label=fluid.label, spec=spec,
            states=states, sat_evap=sat_e, sat_cond=sat_c,
            COP=float(COP), W_comp=float(W_comp), Q_evap=float(Q_evap),
            Q_cond=float(Q_cond),
            pressure_ratio=float(P_cond / P_evap),
            T_discharge=float(st2.state.T),
            volumetric_capacity=float(vol_cap),
            quality_evap_inlet=x4_clipped,
            W_comp_W=float(W_comp * m) if m else None,
            Q_evap_W=float(Q_evap * m) if m else None,
            Q_cond_W=float(Q_cond * m) if m else None,
            warnings=warnings,
        )

    except InfeasibleCycleError as exc:
        return _failed(CycleStatus.TRANSCRITICAL, fluid, spec, str(exc), warnings)
    except ConvergenceError as exc:
        return _failed(CycleStatus.NOT_CONVERGED, fluid, spec, str(exc), warnings)
    except (EOSRootError, PropertyError) as exc:
        return _failed(CycleStatus.INFEASIBLE, fluid, spec, str(exc), warnings)
