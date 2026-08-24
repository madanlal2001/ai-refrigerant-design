"""GNN predictor adapter (optional).

This wraps the DMPNN pipeline behind the same Predictor interface as
ManualPredictor. It activates ONLY if:
    1. torch / torch_geometric / rdkit are importable, and
    2. the checkpoint (default checkpoints/merged_pipeline.pt, override via
       REFRIG_GNN_CHECKPOINT) exists, and
    3. gnn_pipeline.py is importable (already on sys.path, or its directory
       is given via REFRIG_GNN_PIPELINE_DIR -- see ../config.py).

Otherwise `available()` returns False and the app hides the option. The core
package never imports torch at module scope -- that is what keeps the thermo
engine installable with numpy + scipy alone.

Two things this adapter fixes about the current pipeline:
  * `gnn_pipeline.mol_scaler` is a module-level global mutated via
    globals()[...]. Under Streamlit's rerun model that is a live bug. Here it
    is set once, at load, and the loaded bundle owns it.
  * There was no single-molecule entry point. There is now.

NOTE: model A predicts Tc; model B predicts Pc and omega. Two artifacts, one
facade. That is why Predictor.predict returns a dict rather than a scalar.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, Optional

from .. import config
from ..core.component import UNITS
from ..core.constants import BAR_TO_PA
from ..core.property import Property, Source
from .base import ADResult, ADVerdict, Predictor

DEFAULT_CHECKPOINT = config.DEFAULT_GNN_CHECKPOINT


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class GNNPredictor(Predictor):
    """Adapter over gnn_pipeline's merged bundle."""

    predicts = ("Tc", "Pc", "omega")

    def __init__(self, checkpoint: Optional[Path] = None,
                 pipeline_module: str = "gnn_pipeline",
                 pipeline_dir: Optional[Path] = None,
                 device: str = "cpu"):
        self.checkpoint = Path(checkpoint) if checkpoint is not None \
            else config.gnn_checkpoint_path()
        self.pipeline_module = pipeline_module
        self.pipeline_dir = Path(pipeline_dir) if pipeline_dir is not None \
            else config.gnn_pipeline_dir()
        self.device_str = device
        self._loaded = False
        self._model_a = None
        self._model_b = None
        self._scaler_a = None
        self._scaler_b = None
        self._gp = None
        self._torch = None
        self._hash: Optional[str] = None

    # ---- availability ---------------------------------------------------

    @property
    def model_id(self) -> str:  # type: ignore[override]
        if self._hash:
            return f"dmpnn/merged_pipeline@{self._hash}"
        return "dmpnn/merged_pipeline@unloaded"

    def _ensure_pipeline_dir_on_path(self) -> None:
        """Make `pipeline_dir` importable, idempotently.

        A no-op when `pipeline_dir` is unset (the default): that means
        gnn_pipeline.py is assumed to already be on sys.path.
        """
        if self.pipeline_dir is None:
            return
        p = str(self.pipeline_dir)
        if p not in sys.path:
            sys.path.insert(0, p)

    def available(self) -> bool:
        if not self.checkpoint.exists():
            return False
        try:
            import importlib.util
            for mod in ("torch", "torch_geometric", "rdkit"):
                if importlib.util.find_spec(mod) is None:
                    return False
            self._ensure_pipeline_dir_on_path()
            if importlib.util.find_spec(self.pipeline_module) is None:
                return False
        except Exception:
            return False
        return True

    def why_unavailable(self) -> str:
        """Human-readable reason, for the UI."""
        import importlib.util
        if not self.checkpoint.exists():
            return (f"Checkpoint not found at {self.checkpoint}. Either train "
                    f"the model (`python gnn_pipeline.py --mode finetune`) and "
                    f"place merged_pipeline.pt there, or set the "
                    f"{config.ENV_CHECKPOINT} environment variable to its path.")
        missing = [m for m in ("torch", "torch_geometric", "rdkit")
                   if importlib.util.find_spec(m) is None]
        if missing:
            return f"Missing packages: {', '.join(missing)}"
        self._ensure_pipeline_dir_on_path()
        if importlib.util.find_spec(self.pipeline_module) is None:
            return (f"{self.pipeline_module}.py is not importable from here. "
                    f"Set the {config.ENV_PIPELINE_DIR} environment variable "
                    f"to the directory that contains it.")
        return "Available."

    # ---- loading --------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        import importlib

        import torch

        self._torch = torch
        self._ensure_pipeline_dir_on_path()
        gp = importlib.import_module(self.pipeline_module)
        self._gp = gp

        device = torch.device(self.device_str)
        (model_a, model_b, scaler_a, scaler_b,
         mol_scaler) = gp.load_merged_bundle(self.checkpoint, device)

        # Own the scaler explicitly instead of relying on a mutated global.
        gp.mol_scaler = mol_scaler

        self._model_a, self._model_b = model_a, model_b
        self._scaler_a, self._scaler_b = scaler_a, scaler_b
        self._device = device
        self._hash = _file_hash(self.checkpoint)
        self._loaded = True

    # ---- inference ------------------------------------------------------

    def predict(self, smiles: str) -> Dict[str, Property]:
        self._load()
        import pandas as pd

        gp = self._gp
        df = pd.DataFrame({"smiles": [smiles]})

        pred_a = gp.predict_multitask(self._model_a, df, gp.TARGETS_A,
                                      self._scaler_a, self._device, batch_size=1)
        pred_b = gp.predict_multitask(self._model_b, df, gp.TARGETS_B,
                                      self._scaler_b, self._device, batch_size=1)

        if len(pred_a) == 0 or len(pred_b) == 0:
            raise ValueError(
                f"The featurizer rejected {smiles!r} (invalid SMILES, or 3D "
                "conformer embedding failed).")

        def mk(name: str, value: float) -> Property:
            return Property(name=name, value=float(value), unit=UNITS[name],
                            source=Source.PREDICTED, model_id=self.model_id,
                            uncertainty=None)  # UQ arrives in a later version

        return {
            "Tc": mk("Tc", pred_a["Tc (K)"].iloc[0]),
            "Pc": mk("Pc", pred_b["Pc (bar)"].iloc[0] * BAR_TO_PA),
            "omega": mk("omega", pred_b["omega (-)"].iloc[0]),
        }

    def applicability(self, smiles: str) -> ADResult:
        return ADResult(
            verdict=ADVerdict.UNKNOWN,
            detail="Applicability domain is scheduled for a later version. "
                   "Treat predictions for unusual chemistries with suspicion.")
