"""Ideal-gas heat capacity providers.

The whole point of this module is that the *cycle* never knows where Cp came
from. V1 ships Poly4Cp (user types 5 numbers). V2 will ship MLCp. Neither the
EOS nor the cycle changes.

The interface exposes the INTEGRALS, not just cp_ig(T). This matters:
  - Poly4 has an exact antiderivative -> no quadrature error, no cost.
  - A future ML Cp(T) can implement the integrals numerically.
Forcing everything through numerical integration would tax the analytic case
for nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import R
from .property import Source


class CpProvider(ABC):
    """Interface for ideal-gas heat capacity as a function of temperature."""

    #: Tag identifying the functional form, e.g. "POLY4". Stored in run records
    #: so a future ML model trained on a different form cannot be silently
    #: substituted.
    form: str = "abstract"

    #: Provenance of the underlying parameters.
    source: Source = Source.DEFAULT

    #: (T_min, T_max) in K over which the correlation is claimed valid.
    #: None means "unknown" -- reported as a warning, never enforced silently.
    valid_range: Optional[Tuple[float, float]] = None

    @abstractmethod
    def cp_ig(self, T: float) -> float:
        """Ideal-gas molar heat capacity [J/(mol*K)]."""

    @abstractmethod
    def h_ig(self, T: float, T_ref: float) -> float:
        """Ideal-gas molar enthalpy relative to T_ref [J/mol].

        H_ig = integral(T_ref -> T) cp_ig dT
        """

    @abstractmethod
    def s_ig(self, T: float, P: float, T_ref: float, P_ref: float) -> float:
        """Ideal-gas molar entropy relative to (T_ref, P_ref) [J/(mol*K)].

        S_ig = integral(T_ref -> T) cp_ig/T dT - R*ln(P/P_ref)
        """

    def in_range(self, T: float) -> bool:
        if self.valid_range is None:
            return True
        lo, hi = self.valid_range
        return lo <= T <= hi

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializable description, for the run record."""


@dataclass(frozen=True)
class Poly4Cp(CpProvider):
    """Quartic polynomial in T.

        cp_ig(T) = a + b*T + c*T^2 + d*T^3 + e*T^4     [J/(mol*K)]

    with T in KELVIN. This is the exact form used by molecules_smiles_BEST1.xlsx
    (columns a..e) and by COP.ipynb.

    The units of each coefficient follow from the form:
        a [J/(mol*K)], b [J/(mol*K^2)], ... e [J/(mol*K^5)]

    Do not feed this class coefficients from a different correlation (DIPPR-100,
    Aly-Lee, Poling's a+bT+cT^2+dT^3 with T in degC, ...). They will produce
    plausible, wrong numbers silently. That is why `form` is recorded.
    """

    a: float
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    source: Source = Source.LITERATURE
    valid_range: Optional[Tuple[float, float]] = None

    form: str = "POLY4"

    @classmethod
    def from_sequence(cls, coeffs: Sequence[float], **kw) -> "Poly4Cp":
        vals = list(coeffs) + [0.0] * (5 - len(coeffs))
        return cls(*vals[:5], **kw)

    @property
    def coeffs(self) -> Tuple[float, float, float, float, float]:
        return (self.a, self.b, self.c, self.d, self.e)

    def cp_ig(self, T: float) -> float:
        a, b, c, d, e = self.coeffs
        return a + b * T + c * T**2 + d * T**3 + e * T**4

    def _H(self, t: float) -> float:
        """Antiderivative of cp_ig."""
        a, b, c, d, e = self.coeffs
        return a * t + (b / 2) * t**2 + (c / 3) * t**3 + (d / 4) * t**4 + (e / 5) * t**5

    def _S(self, t: float) -> float:
        """Antiderivative of cp_ig / T."""
        a, b, c, d, e = self.coeffs
        return a * np.log(t) + b * t + (c / 2) * t**2 + (d / 3) * t**3 + (e / 4) * t**4

    def h_ig(self, T: float, T_ref: float) -> float:
        return float(self._H(T) - self._H(T_ref))

    def s_ig(self, T: float, P: float, T_ref: float, P_ref: float) -> float:
        return float(self._S(T) - self._S(T_ref) - R * np.log(P / P_ref))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "form": self.form,
            "expression": "a + b*T + c*T^2 + d*T^3 + e*T^4",
            "cp_unit": "J/(mol*K)",
            "T_unit": "K",
            "coefficients": {
                "a": self.a, "b": self.b, "c": self.c, "d": self.d, "e": self.e
            },
            "source": self.source.value,
            "valid_range_K": list(self.valid_range) if self.valid_range else None,
        }


@dataclass(frozen=True)
class ConstantCp(CpProvider):
    """Constant cp_ig. Useful for tests and sanity checks only."""

    cp: float
    source: Source = Source.DEFAULT
    valid_range: Optional[Tuple[float, float]] = None
    form: str = "CONSTANT"

    def cp_ig(self, T: float) -> float:
        return self.cp

    def h_ig(self, T: float, T_ref: float) -> float:
        return self.cp * (T - T_ref)

    def s_ig(self, T: float, P: float, T_ref: float, P_ref: float) -> float:
        return float(self.cp * np.log(T / T_ref) - R * np.log(P / P_ref))

    def to_dict(self) -> Dict[str, Any]:
        return {"form": self.form, "cp": self.cp, "cp_unit": "J/(mol*K)",
                "source": self.source.value, "valid_range_K": None}
