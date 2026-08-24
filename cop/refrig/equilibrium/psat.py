"""Saturation pressure by fugacity equality.

Promoted to a first-class module: this is the solver most likely to fail, and
its convergence behaviour is a *result* the user should be able to see, not an
implementation detail buried inside the EOS.

Method (unchanged from COP.ipynb):
    1. Wilson correlation for the initial pressure guess
    2. iterate  P <- P * exp(ln(phi_L) - ln(phi_V))  until |diff| < tol
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..core.errors import ConvergenceError, InfeasibleCycleError
from ..eos.peng_robinson import PengRobinson


@dataclass(frozen=True)
class SaturationResult:
    """Outcome of a Psat solve, including how hard it was."""

    T: float             # K
    P: float             # Pa
    Z_vapor: float
    Z_liquid: float
    iterations: int
    residual: float      # final |ln(phi_L) - ln(phi_V)|
    converged: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "T_K": self.T, "P_Pa": self.P, "P_bar": self.P * 1e-5,
            "Z_vapor": self.Z_vapor, "Z_liquid": self.Z_liquid,
            "iterations": self.iterations, "residual": self.residual,
            "converged": self.converged,
        }


def wilson_guess(T: float, Tc: float, Pc: float, omega: float) -> float:
    """Wilson correlation initial guess for Psat [Pa]."""
    return Pc * np.exp(5.373 * (1.0 + omega) * (1.0 - Tc / T))


def psat(eos: PengRobinson, T: float) -> SaturationResult:
    """Saturation pressure of a pure fluid at temperature T.

    Raises
    ------
    InfeasibleCycleError : if T >= Tc (no saturation line exists)
    ConvergenceError     : if the fugacity iteration runs out of budget
    """
    if T >= eos.Tc:
        raise InfeasibleCycleError(
            f"T = {T:.2f} K is at or above Tc = {eos.Tc:.2f} K; "
            "no saturation pressure exists (supercritical).")

    st = eos.settings
    P = wilson_guess(T, eos.Tc, eos.Pc, eos.omega)
    diff = np.inf

    for it in range(1, st.psat_max_iter + 1):
        A, B = eos.AB(T, P)
        Zv, Zl = eos.Z_roots(T, P)
        lnphi_v = eos.ln_phi(Zv, A, B)
        lnphi_l = eos.ln_phi(Zl, A, B)
        diff = lnphi_l - lnphi_v

        if abs(diff) < st.psat_tol:
            return SaturationResult(T=T, P=float(P), Z_vapor=Zv, Z_liquid=Zl,
                                    iterations=it, residual=float(abs(diff)),
                                    converged=True)
        P *= np.exp(diff)
        if not np.isfinite(P) or P <= 0:
            raise ConvergenceError(
                f"Psat iteration diverged at T = {T:.2f} K (P became {P:.6g} Pa)")

    raise ConvergenceError(
        f"Psat did not converge at T = {T:.2f} K after {st.psat_max_iter} "
        f"iterations (residual {abs(diff):.3g}, tol {st.psat_tol:.3g})")
