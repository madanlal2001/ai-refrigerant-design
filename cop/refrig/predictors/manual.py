"""Manual "predictor": the user supplies Tc, Pc, omega directly.

This is not a fallback hack. It is the reference implementation of the
Predictor interface, and it is what lets the platform be validated against
known fluids (R134a, propane, ...) with literature properties, independently
of whether the GNN is any good.
"""

from __future__ import annotations

from typing import Dict

from ..core.component import UNITS
from ..core.constants import BAR_TO_PA
from ..core.property import Property, Source
from .base import Predictor


class ManualPredictor(Predictor):
    """Wraps user-entered values in Property objects with honest provenance."""

    predicts = ("Tc", "Pc", "omega")
    model_id = "manual/user-entered"

    def __init__(self, Tc_K: float, Pc_bar: float, omega: float,
                 source: Source = Source.USER_OVERRIDE):
        self.Tc_K = float(Tc_K)
        self.Pc_bar = float(Pc_bar)
        self.omega = float(omega)
        self.source = source

    def available(self) -> bool:
        return True

    def predict(self, smiles: str) -> Dict[str, Property]:
        def mk(name: str, value: float) -> Property:
            return Property(name=name, value=value, unit=UNITS[name],
                            source=self.source, model_id=self.model_id)

        return {
            "Tc": mk("Tc", self.Tc_K),
            # Note the unit conversion happens HERE, at the boundary.
            # Everything downstream is SI.
            "Pc": mk("Pc", self.Pc_bar * BAR_TO_PA),
            "omega": mk("omega", self.omega),
        }
