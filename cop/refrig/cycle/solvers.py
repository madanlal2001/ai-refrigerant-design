"""Temperature root-finders at fixed pressure.

Used by the compressor (fixed S) and to recover the actual discharge
temperature (fixed H). Bracket-then-Brent, because Newton on the PR entropy
surface near the dome is a good way to end up on the wrong root.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.optimize import brentq

from ..core.cp import CpProvider
from ..core.errors import ConvergenceError
from ..eos.peng_robinson import PengRobinson, Phase


def _bracket_upward(f: Callable[[float], float], T_lo: float, f_lo: float,
                    T_max: float, step: float = 10.0,
                    max_steps: int = 300) -> Tuple[float, float]:
    """March T upward from T_lo until f changes sign. Returns (T_hi, f_hi)."""
    T_hi = T_lo + step
    f_hi = f(T_hi)
    n = 0
    while f_lo * f_hi > 0.0 and n < max_steps:
        T_hi += step
        if T_hi > T_max:
            break
        try:
            f_hi = f(T_hi)
        except Exception:
            T_hi -= step
            break
        n += 1
    return T_hi, f_hi


def solve_T_at_S(eos: PengRobinson, P: float, S_target: float, cp: CpProvider,
                 T_ref: float, P_ref: float, T_start: float) -> float:
    """Find T such that S_vapor(T, P) = S_target. Superheated branch."""

    def resid(T: float) -> float:
        Zv, _ = eos.Z_roots(T, P)
        return eos.entropy(T, P, Zv, cp, T_ref, P_ref) - S_target

    f_lo = resid(T_start)
    T_hi, f_hi = _bracket_upward(resid, T_start, f_lo, T_max=5.0 * eos.Tc)
    if f_lo * f_hi > 0.0:
        raise ConvergenceError(
            f"Could not bracket the isentropic root at P = {P*1e-5:.4f} bar "
            f"between {T_start:.1f} K and {T_hi:.1f} K")
    return float(brentq(resid, T_start, T_hi, xtol=1e-7, rtol=1e-12, maxiter=300))


def solve_T_at_H(eos: PengRobinson, P: float, H_target: float, cp: CpProvider,
                 T_ref: float, T_start: float) -> float:
    """Find T such that H_vapor(T, P) = H_target. Superheated branch."""

    def resid(T: float) -> float:
        Zv, _ = eos.Z_roots(T, P)
        return eos.enthalpy(T, P, Zv, cp, T_ref) - H_target

    f_lo = resid(T_start)
    T_hi, f_hi = _bracket_upward(resid, T_start, f_lo, T_max=5.0 * eos.Tc)
    if f_lo * f_hi > 0.0:
        raise ConvergenceError(
            f"Could not bracket the isenthalpic root at P = {P*1e-5:.4f} bar")
    return float(brentq(resid, T_start, T_hi, xtol=1e-7, rtol=1e-12, maxiter=300))
