"""Predictor interface.

Deliberately multi-target: your Model A predicts Tc, Model B predicts Pc and
omega. A `predict_tc(smiles) -> float` signature could not express that, and
would have to be thrown away when the GNN arrives.

`applicability` is a stub in V1 returning UNKNOWN. It exists so the applicability
-domain version is an implementation, not an API change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from ..core.property import Property


class ADVerdict(str, Enum):
    INSIDE = "inside"
    BORDERLINE = "borderline"
    OUTSIDE = "outside"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ADResult:
    """Applicability-domain verdict for one molecule."""

    verdict: ADVerdict = ADVerdict.UNKNOWN
    score: Optional[float] = None
    detail: str = "Applicability domain analysis is not implemented in V1."

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value, "score": self.score,
                "detail": self.detail}


class Predictor(ABC):
    """Maps molecular structure to Property objects."""

    #: Property names this predictor supplies, e.g. ["Tc", "Pc", "omega"].
    predicts: Sequence[str] = ()

    #: Stable identifier written into every Property and RunRecord it touches.
    model_id: str = "abstract"

    @abstractmethod
    def available(self) -> bool:
        """Can this predictor actually run right now? (weights present, etc.)"""

    @abstractmethod
    def predict(self, smiles: str) -> Dict[str, Property]:
        """Return {property_name: Property} for one canonical SMILES."""

    def applicability(self, smiles: str) -> ADResult:
        return ADResult()

    def describe(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "predicts": list(self.predicts),
                "available": self.available()}
