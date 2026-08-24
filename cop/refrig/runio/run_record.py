"""RunRecord: the reproducibility unit.

Every simulation emits one. It contains enough to reconstruct the result
without the app: the exact inputs, every property WITH its provenance and the
model that produced it, the solver settings, and the environment.

This is the artifact you cite in the thesis, and the substrate on which
sensitivity and uncertainty studies are later built.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.component import Fluid
from ..cycle.vcc import CycleResult

SCHEMA_VERSION = "1.0.0"


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> Dict[str, str]:
    vers: Dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "pandas", "rdkit", "torch", "streamlit"):
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return vers


@dataclass
class RunRecord:
    """A complete, self-describing simulation record."""

    fluid: Fluid
    result: CycleResult
    inputs: Dict[str, Any] = field(default_factory=dict)
    models_used: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp,
            "notes": self.notes,
            "environment": {
                "git_sha": _git_sha(),
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
            "inputs": self.inputs,
            "models_used": self.models_used,
            "fluid": self.fluid.to_dict(),
            "result": self.result.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, directory: Path = Path("runs")) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"run_{self.run_id}.json"
        path.write_text(self.to_json())
        return path
