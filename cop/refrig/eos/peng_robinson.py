"""Peng-Robinson equation of state (pure component, V1).

    P = R*T/(v - b) - a(T) / [v*(v + b) + b*(v - b)]

Ported from COP.ipynb with the physics unchanged. What changed:
  - every operating condition is now a parameter, not a module global
  - settings live in an EOSSettings object
  - states are returned as immutable EOSState objects, not loose tuples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.component import Fluid
from ..core.constants import R
from ..core.cp import CpProvider
from ..core.errors import EOSRootError

SQRT2 = np.sqrt(2.0)

# PR critical constants, at full double precision.
#
# COP.ipynb (and most textbooks) use the rounded 0.45724 / 0.07780. Those are
# fine everywhere except AT the critical point, where the cubic has a triple
# root and is therefore cube-root sensitive to coefficient error: the rounding
# at ~5e-6 relative displaces Zc by ~(5e-6)^(1/3) ~ 0.014, i.e. the cubic
# returns Zc = 0.3214 instead of the correct 0.30740, plus a spurious complex
# pair. Away from the critical point the roots are well separated and the
# difference is ~1e-5 relative -- invisible in COP. Using the exact values
# costs nothing and makes the critical-point test pass honestly.
OMEGA_A = 0.4572355289213822
OMEGA_B = 0.0777960739038885
Z_CRIT = 0.3074013086987039


class Phase(str, Enum):
    VAPOR = "vapor"
    LIQUID = "liquid"
    TWO_PHASE = "two_phase"
    SUPERCRITICAL = "supercritical"


@dataclass(frozen=True)
class EOSSettings:
    """Numerical + model settings. Everything the Expert Mode may touch."""

    psat_tol: float = 1e-8          # |ln(phi_L) - ln(phi_V)| convergence
    psat_max_iter: int = 150
    root_tol: float = 1e-8          # imaginary-part tolerance for real roots
    kappa_correlation: str = "PR76"  # PR76 = 0.37464 + 1.54226w - 0.26992w^2
    volume_translation: bool = False  # not implemented in V1; see note below

    def to_dict(self) -> Dict[str, Any]:
        return {
            "psat_tol": self.psat_tol,
            "psat_max_iter": self.psat_max_iter,
            "root_tol": self.root_tol,
            "kappa_correlation": self.kappa_correlation,
            "volume_translation": self.volume_translation,
        }


@dataclass(frozen=True)
class EOSState:
    """A fully determined single-phase thermodynamic state.

    Immutable on purpose: states get stored in CycleResult and run records,
    and nothing downstream should be able to mutate a solved state.
    """

    T: float            # K
    P: float            # Pa
    Z: float            # -
    v: float            # m^3/mol
    h: float            # J/mol   (relative to the cycle's reference state)
    s: float            # J/(mol*K)
    phase: Phase
    rho: float          # mol/m^3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "T_K": self.T, "P_Pa": self.P, "Z": self.Z,
            "v_m3_per_mol": self.v, "rho_mol_per_m3": self.rho,
            "h_J_per_mol": self.h, "s_J_per_mol_K": self.s,
            "phase": self.phase.value,
        }


class PengRobinson:
    """PR EOS bound to one Fluid. Pure-component only in V1.

    The class holds no operating conditions. Every method takes (T, P).
    """

    def __init__(self, fluid: Fluid, settings: Optional[EOSSettings] = None):
        self.fluid = fluid
        self.settings = settings or EOSSettings()
        if not fluid.is_pure:
            raise NotImplementedError(
                "V1 is pure-component. Mixture mixing rules are a later version.")
        c = fluid.component
        self.Tc = c.Tc
        self.Pc = c.Pc
        self.omega = c.omega
        self.a_c, self.b, self.kappa = self._params()

    # ---- parameters -------------------------------------------------------

    def _params(self) -> Tuple[float, float, float]:
        """a_c [J*m^3/mol^2], b [m^3/mol], kappa [-]."""
        a_c = OMEGA_A * R**2 * self.Tc**2 / self.Pc
        b = OMEGA_B * R * self.Tc / self.Pc
        if self.settings.kappa_correlation.upper() != "PR76":
            raise NotImplementedError(
                f"kappa_correlation={self.settings.kappa_correlation!r} not implemented")
        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        return a_c, b, kappa

    def alpha(self, T: float) -> float:
        """alpha(T) = [1 + kappa*(1 - sqrt(T/Tc))]^2"""
        return (1.0 + self.kappa * (1.0 - np.sqrt(T / self.Tc))) ** 2

    def a_of_T(self, T: float) -> float:
        return self.a_c * self.alpha(T)

    def dadT(self, T: float) -> float:
        """Analytic da/dT = -a_c * kappa * sqrt(alpha / (T*Tc))."""
        return -self.a_c * self.kappa * np.sqrt(self.alpha(T) / (T * self.Tc))

    def AB(self, T: float, P: float) -> Tuple[float, float]:
        """Dimensionless A and B of the PR cubic."""
        A = self.a_of_T(T) * P / (R**2 * T**2)
        B = self.b * P / (R * T)
        return A, B

    # ---- cubic ------------------------------------------------------------

    def Z_roots(self, T: float, P: float) -> Tuple[float, float]:
        """Return (Z_vapor, Z_liquid) = (largest, smallest) admissible root.

        Admissible means real and Z > B, i.e. molar volume exceeds the
        co-volume. When only one real root survives, both returns are equal --
        that is the single-phase case and the caller must not read anything
        into "two roots came back".
        """
        A, B = self.AB(T, P)
        coeffs = [1.0,
                  -(1.0 - B),
                  (A - 3 * B**2 - 2 * B),
                  -(A * B - B**2 - B**3)]
        roots = np.roots(coeffs)
        real = np.array([r.real for r in roots if abs(r.imag) < self.settings.root_tol])
        valid = real[real > B]
        if valid.size == 0:
            raise EOSRootError(
                f"No admissible PR root at T={T:.3f} K, P={P:.6g} Pa "
                f"(A={A:.5g}, B={B:.5g})")
        return float(valid.max()), float(valid.min())

    def Z(self, T: float, P: float, phase: Phase) -> float:
        Zv, Zl = self.Z_roots(T, P)
        if phase is Phase.LIQUID:
            return Zl
        return Zv

    # ---- fugacity ---------------------------------------------------------

    def ln_phi(self, Z: float, A: float, B: float) -> float:
        """ln(fugacity coefficient) from PR."""
        arg = (Z + (1 + SQRT2) * B) / (Z + (1 - SQRT2) * B)
        if arg <= 0:
            raise EOSRootError(f"Non-positive log argument in ln_phi: Z={Z:.6g}, B={B:.6g}")
        return (Z - 1) - np.log(Z - B) - A / (2 * SQRT2 * B) * np.log(arg)

    # ---- departures -------------------------------------------------------

    def _log_term(self, Z: float, B: float) -> float:
        num = Z + (1 + SQRT2) * B
        den = Z + (1 - SQRT2) * B
        if den <= 0 or num / den <= 0:
            raise EOSRootError(f"Invalid departure log term: Z={Z:.6g}, B={B:.6g}")
        return float(np.log(num / den))

    def h_departure(self, T: float, Z: float, B: float) -> float:
        """H - H_ig  [J/mol]."""
        a = self.a_of_T(T)
        da = self.dadT(T)
        prefac = (T * da - a) / (2 * SQRT2 * self.b)
        return R * T * (Z - 1) + prefac * self._log_term(Z, B)

    def s_departure(self, T: float, Z: float, B: float) -> float:
        """S - S_ig  [J/(mol*K)]."""
        da = self.dadT(T)
        prefac = da / (2 * SQRT2 * self.b)
        return R * np.log(Z - B) + prefac * self._log_term(Z, B)

    # ---- totals -----------------------------------------------------------

    def enthalpy(self, T: float, P: float, Z: float, cp: CpProvider,
                 T_ref: float) -> float:
        A, B = self.AB(T, P)
        return cp.h_ig(T, T_ref) + self.h_departure(T, Z, B)

    def entropy(self, T: float, P: float, Z: float, cp: CpProvider,
                T_ref: float, P_ref: float) -> float:
        A, B = self.AB(T, P)
        return cp.s_ig(T, P, T_ref, P_ref) + self.s_departure(T, Z, B)

    def state(self, T: float, P: float, phase: Phase, cp: CpProvider,
              T_ref: float, P_ref: float) -> EOSState:
        """Solve one single-phase state completely."""
        Z = self.Z(T, P, phase)
        A, B = self.AB(T, P)
        h = cp.h_ig(T, T_ref) + self.h_departure(T, Z, B)
        s = cp.s_ig(T, P, T_ref, P_ref) + self.s_departure(T, Z, B)
        v = Z * R * T / P
        actual_phase = Phase.SUPERCRITICAL if T >= self.Tc and P >= self.Pc else phase
        return EOSState(T=T, P=P, Z=Z, v=v, h=h, s=s, phase=actual_phase, rho=1.0 / v)
