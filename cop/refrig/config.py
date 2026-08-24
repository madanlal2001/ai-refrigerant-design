"""Configuration for the optional GNN predictor.

Pure stdlib -- no torch, no streamlit, no rdkit. This module must stay
importable even on a machine that has none of those installed, since
`refrig.predictors.gnn` imports it unconditionally at module scope while
deferring the heavy imports themselves to inside functions.

Two things are environment-specific and therefore not hardcoded here or in
`gnn.py`:

  REFRIG_GNN_CHECKPOINT    path to merged_pipeline.pt
  REFRIG_GNN_PIPELINE_DIR  directory containing gnn_pipeline.py, if it does
                           not already live on sys.path / next to this repo

Both default to locations inside the project so a checkout with the
checkpoint and gnn_pipeline.py copied in place needs no environment at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GNN_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "merged_pipeline.pt"

ENV_CHECKPOINT = "REFRIG_GNN_CHECKPOINT"
ENV_PIPELINE_DIR = "REFRIG_GNN_PIPELINE_DIR"


def gnn_checkpoint_path() -> Path:
    """Where to find merged_pipeline.pt.

    `REFRIG_GNN_CHECKPOINT` overrides the default of
    `<project root>/checkpoints/merged_pipeline.pt`.
    """
    env = os.environ.get(ENV_CHECKPOINT)
    return Path(env).expanduser() if env else DEFAULT_GNN_CHECKPOINT


def gnn_pipeline_dir() -> Optional[Path]:
    """Extra directory to add to sys.path so `gnn_pipeline` is importable.

    None means: don't touch sys.path, assume gnn_pipeline.py is already
    importable (e.g. it lives at the project root, or PYTHONPATH already
    covers it).
    """
    env = os.environ.get(ENV_PIPELINE_DIR)
    return Path(env).expanduser() if env else None
