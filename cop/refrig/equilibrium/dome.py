"""Saturation dome for P-h and T-s diagrams.

Sweeps T from a low bound up toward Tc, solving Psat at each step and
evaluating both roots. Points that fail to converge are skipped rather than
raising -- a dome with a gap is still a useful plot; a crashed plot is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.cp import CpProvider
from ..eos.peng_robinson import PengRobinson
from .psat import psat


@dataclass
class Dome:
    T: np.ndarray          # K
    P: np.ndarray          # Pa
    h_liq: np.ndarray      # J/mol
    h_vap: np.ndarray
    s_liq: np.ndarray      # J/(mol*K)
    s_vap: np.ndarray
    n_failed: int = 0


def saturation_dome(eos: PengRobinson, cp: CpProvider, T_ref: float,
                    P_ref: float, T_min: float | None = None,
                    n: int = 120, T_max_frac: float = 0.999) -> Dome:
    """Build the two-phase envelope between T_min and T_max_frac*Tc."""
    Tc = eos.Tc
    if T_min is None:
        T_min = 0.45 * Tc
    T_max = T_max_frac * Tc
    Ts = np.linspace(T_min, T_max, n)

    T_ok: List[float] = []
    P_ok: List[float] = []
    hl: List[float] = []
    hv: List[float] = []
    sl: List[float] = []
    sv: List[float] = []
    failed = 0

    for T in Ts:
        try:
            r = psat(eos, float(T))
            hl.append(eos.enthalpy(T, r.P, r.Z_liquid, cp, T_ref))
            hv.append(eos.enthalpy(T, r.P, r.Z_vapor, cp, T_ref))
            sl.append(eos.entropy(T, r.P, r.Z_liquid, cp, T_ref, P_ref))
            sv.append(eos.entropy(T, r.P, r.Z_vapor, cp, T_ref, P_ref))
            T_ok.append(float(T))
            P_ok.append(r.P)
        except Exception:
            failed += 1
            continue

    return Dome(T=np.array(T_ok), P=np.array(P_ok),
                h_liq=np.array(hl), h_vap=np.array(hv),
                s_liq=np.array(sl), s_vap=np.array(sv), n_failed=failed)
