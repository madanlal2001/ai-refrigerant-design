"""Property: a scientific value that knows where it came from.

This is the single most important class in the package. A bare float loses
provenance; once lost, a result is unreproducible and an override is
indistinguishable from a prediction.

`uncertainty` is optional and unused in V1. It exists now so that adding UQ
in a later version does not change a single call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Source(str, Enum):
    """Where a property value came from. Ordered loosely by trust."""

    EXPERIMENTAL = "experimental"
    LITERATURE = "literature"
    USER_OVERRIDE = "user_override"
    PREDICTED = "predicted"
    ESTIMATED = "estimated"      # e.g. group contribution / Joback
    DEFAULT = "default"

    @property
    def label(self) -> str:
        return {
            Source.EXPERIMENTAL: "Experimental",
            Source.LITERATURE: "Literature",
            Source.USER_OVERRIDE: "User override",
            Source.PREDICTED: "Predicted (ML)",
            Source.ESTIMATED: "Estimated",
            Source.DEFAULT: "Default",
        }[self]


@dataclass(frozen=True)
class Property:
    """An immutable scientific quantity, always stored in SI.

    Attributes
    ----------
    name        : canonical short name, e.g. "Tc", "Pc", "omega"
    value       : magnitude in SI units
    unit        : SI unit string, e.g. "K", "Pa", "-"
    source      : provenance tag
    uncertainty : optional 1-sigma, same unit as `value`. None in V1.
    model_id    : identifier of the model that produced it, if predicted
    overrides   : the Property this one replaced, if any (audit chain)
    """

    name: str
    value: float
    unit: str
    source: Source
    uncertainty: Optional[float] = None
    model_id: Optional[str] = None
    overrides: Optional["Property"] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))

    def override(self, new_value: float) -> "Property":
        """Return a new Property replacing this one, keeping the audit chain.

        The original is retained in `.overrides`, so a run record can show
        "user typed 378 K over a predicted 374 K" rather than just "378 K".
        """
        return Property(
            name=self.name,
            value=float(new_value),
            unit=self.unit,
            source=Source.USER_OVERRIDE,
            uncertainty=None,
            model_id=None,
            overrides=self,
        )

    @property
    def was_overridden(self) -> bool:
        return self.overrides is not None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source.value,
            "uncertainty": self.uncertainty,
            "model_id": self.model_id,
        }
        if self.overrides is not None:
            d["overrides"] = self.overrides.to_dict()
        return d

    def __str__(self) -> str:
        u = "" if self.unit == "-" else f" {self.unit}"
        return f"{self.name} = {self.value:.6g}{u} [{self.source.label}]"
