"""Component and Fluid.

Fluid carries a composition vector and a kij matrix even though V1 is pure-
component only (n = 1, x = [1.0], kij = [[0.0]]). This costs nothing today and
means the mixture version is an extension rather than a rewrite of the EOS,
the flash, and the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .constants import BAR_TO_PA
from .cp import CpProvider
from .errors import PropertyError
from .property import Property, Source

#: Properties every component must have before it can enter the EOS.
REQUIRED = ("Tc", "Pc", "omega")

UNITS = {"Tc": "K", "Pc": "Pa", "omega": "-", "MW": "kg/mol", "Tb": "K"}


@dataclass
class Component:
    """A single pure substance: identity + properties + a Cp provider."""

    canonical_smiles: str
    name: str = ""
    inchikey: Optional[str] = None
    cas: Optional[str] = None
    properties: Dict[str, Property] = field(default_factory=dict)
    cp: Optional[CpProvider] = None

    # ---- construction helpers -------------------------------------------

    @staticmethod
    def make_property(name: str, value: float, source: Source,
                      model_id: Optional[str] = None,
                      uncertainty: Optional[float] = None) -> Property:
        return Property(name=name, value=value, unit=UNITS.get(name, "-"),
                        source=source, model_id=model_id, uncertainty=uncertainty)

    def set(self, name: str, value: float, source: Source,
            model_id: Optional[str] = None) -> None:
        self.properties[name] = self.make_property(name, value, source, model_id)

    def override(self, name: str, new_value: float) -> None:
        """Replace a property value, preserving the original in the audit chain."""
        if name in self.properties:
            self.properties[name] = self.properties[name].override(new_value)
        else:
            self.properties[name] = self.make_property(name, new_value, Source.USER_OVERRIDE)

    def get(self, name: str) -> Property:
        try:
            return self.properties[name]
        except KeyError:
            raise PropertyError(f"{self.label!r} has no property {name!r}") from None

    # ---- SI accessors (the EOS uses these, never raw dict lookups) -------

    @property
    def Tc(self) -> float:
        """Critical temperature [K]."""
        return self.get("Tc").value

    @property
    def Pc(self) -> float:
        """Critical pressure [Pa]."""
        return self.get("Pc").value

    @property
    def omega(self) -> float:
        """Acentric factor [-]."""
        return self.get("omega").value

    @property
    def label(self) -> str:
        return self.name or self.canonical_smiles

    # ---- validation ------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of problems. Empty list == usable."""
        problems: List[str] = []
        for k in REQUIRED:
            if k not in self.properties:
                problems.append(f"missing {k}")
        if self.cp is None:
            problems.append("missing Cp provider")
        if "Tc" in self.properties and self.Tc <= 0:
            problems.append(f"Tc must be > 0 (got {self.Tc:g} K)")
        if "Pc" in self.properties and self.Pc <= 0:
            problems.append(f"Pc must be > 0 (got {self.Pc:g} Pa)")
        if "omega" in self.properties and not (-0.5 <= self.omega <= 2.0):
            problems.append(f"omega = {self.omega:g} is outside any sane range")
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles,
            "name": self.name,
            "inchikey": self.inchikey,
            "cas": self.cas,
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "cp": self.cp.to_dict() if self.cp else None,
        }


@dataclass
class Fluid:
    """A (possibly multi-component) working fluid.

    V1 constructs these via `Fluid.pure(component)`.
    """

    components: List[Component]
    composition: np.ndarray = None          # type: ignore[assignment]
    kij: np.ndarray = None                  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.components)
        if n == 0:
            raise PropertyError("Fluid must have at least one component")
        if self.composition is None:
            self.composition = np.ones(n) / n
        self.composition = np.asarray(self.composition, dtype=float)
        if self.composition.shape != (n,):
            raise PropertyError(
                f"composition has shape {self.composition.shape}, expected ({n},)")
        total = self.composition.sum()
        if not np.isclose(total, 1.0, atol=1e-8):
            raise PropertyError(f"composition must sum to 1.0 (got {total:g})")
        if self.kij is None:
            self.kij = np.zeros((n, n))
        self.kij = np.asarray(self.kij, dtype=float)
        if self.kij.shape != (n, n):
            raise PropertyError(f"kij has shape {self.kij.shape}, expected ({n},{n})")

    @classmethod
    def pure(cls, component: Component) -> "Fluid":
        return cls(components=[component], composition=np.array([1.0]),
                   kij=np.zeros((1, 1)))

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def is_pure(self) -> bool:
        return self.n_components == 1

    @property
    def component(self) -> Component:
        """The only component. Raises for mixtures -- V1 guard."""
        if not self.is_pure:
            raise PropertyError(
                "Fluid.component is pure-only. Mixture support is a later version.")
        return self.components[0]

    @property
    def label(self) -> str:
        if self.is_pure:
            return self.component.label
        return " / ".join(c.label for c in self.components)

    def validate(self) -> List[str]:
        problems: List[str] = []
        for c in self.components:
            problems += [f"{c.label}: {p}" for p in c.validate()]
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "composition": self.composition.tolist(),
            "kij": self.kij.tolist(),
        }
