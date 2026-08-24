#!/usr/bin/env python
"""
gnn_pipeline.py
================
Production-quality, single-file research pipeline that reproduces the
"Hybrid Specialist" DMPNN critical-property notebook (GNN model.ipynb, V2)
using the finalized datasets:

    Pretraining : final_pretraining_estimated_critprop.csv
    Fine-tuning : final_master_experimental_finetuning_6644.csv
    Deployment  : molecules_144_refrigerant.xlsx

This file is a straight *conversion* of the notebook: the GNN architecture,
message passing, descriptor generation, hidden dimensions, loss function,
optimizer, scheduler and hyperparameters are copied verbatim. Nothing here
was rewritten, tuned, or "improved" -- only reorganized into functions,
given a CLI, logging, checkpointing and validation.

IMPORTANT ALGORITHMIC NOTE (read before touching --mode semantics)
--------------------------------------------------------------------
The notebook's "hybrid specialist" design has a real data dependency:
Model B (Pc/omega specialist) does NOT start from a fresh encoder or from
Model A's *pretrained* encoder -- it starts from Model A's *fine-tuned*
encoder (`model_B.encoder.load_state_dict(model_A_ft.encoder.state_dict())`).
That means "Model B pretraining" cannot happen before "Model A fine-tuning"
in the original algorithm. This file preserves that exact order:

    --mode pretrain : Model A pretraining ONLY (the one stage that has no
                       dependency on anything else). Produces
                       best_tc_pretrained.pt + a shared scalers bundle.
    --mode finetune : Model A fine-tuning, THEN Model B pretraining
                       (bootstrapped from A's fine-tuned encoder, frozen),
                       THEN Model B fine-tuning. Produces
                       best_tc_finetuned.pt, best_pcomega_pretrained.pt,
                       best_pcomega_finetuned.pt and merged_pipeline.pt.
    --mode test     : Loads the fine-tuned bundle and evaluates on the
                       held-out test split.
    --mode predict  : Loads the fine-tuned bundle and scores the 142/144
                       deployment refrigerant molecules.
    --mode all      : pretrain -> finetune -> test -> predict, in-process.

See the module docstring above each stage function for more detail.

Usage
-----
    python gnn_pipeline.py --mode pretrain
    python gnn_pipeline.py --mode finetune
    python gnn_pipeline.py --mode test
    python gnn_pipeline.py --mode predict
    python gnn_pipeline.py --mode all --device cuda --seed 42
"""

# ======================================================================================
# SECTION 0: IMPORTS
# ======================================================================================
import argparse
import logging
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, Descriptors3D

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler  # kept for fidelity with the notebook's imports (unused, as in V1/V2)

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_add_pool  # MessagePassing kept for fidelity (unused directly, as in the notebook)

import matplotlib
matplotlib.use("Agg")  # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt


# ======================================================================================
# SECTION 1: CONFIGURATION
# ======================================================================================
# All values below are copied verbatim from the notebook. Nothing in this section
# changes the model, the loss, the optimizer, or any training hyperparameter.

# --- Finalized dataset filenames (V2 migration) ---
PRETRAIN_CSV_DEFAULT = "final_pretraining_estimated_critprop.csv"
FINETUNE_CSV_DEFAULT = "final_master_experimental_finetuning_6644.csv"
MCLINDEN_CSV_DEFAULT = "McLinden-clean-with-smiles.csv"   # untouched auxiliary dataset, not part of the V2 migration
DEPLOY_XLSX_DEFAULT = "molecules_144_refrigerant.xlsx"

# --- Output directory names (exactly as required) ---
CHECKPOINT_DIR_DEFAULT = "checkpoints"
PREDICTIONS_DIR_DEFAULT = "predictions"
RESULTS_DIR_DEFAULT = "results"
LOGS_DIR_DEFAULT = "logs"
PLOTS_SUBDIR = "plots"  # results/plots

# --- Checkpoint filenames (exactly as required) ---
CKPT_TC_PRETRAINED = "best_tc_pretrained.pt"
CKPT_PCOMEGA_PRETRAINED = "best_pcomega_pretrained.pt"
CKPT_TC_FINETUNED = "best_tc_finetuned.pt"
CKPT_PCOMEGA_FINETUNED = "best_pcomega_finetuned.pt"
CKPT_MERGED = "merged_pipeline.pt"
CKPT_SCALERS = "scalers.pt"  # internal cross-mode handoff artifact (mol_scaler + scaler_A + scaler_B)

# --- Target columns (verbatim) ---
TARGET_COLS = ["Tc (K)", "Pc (bar)", "omega (-)"]

# --- Domain specialization filter thresholds (verbatim) ---
DOMAIN_TC_MIN, DOMAIN_TC_MAX = 150, 800
DOMAIN_PC_MAX = 100
DOMAIN_OMEGA_MAX = 0.8

# --- Split random states (verbatim; NOT touched by --seed, these are notebook constants) ---
EXPT_SPLIT_RANDOM_STATE = 192431
MCL_SPLIT_RANDOM_STATE = 42

# --- Model architecture hyperparameters (verbatim -- DO NOT CHANGE) ---
DMPNN_DEPTH = 3
HIDDEN_SIZE = 512
FNN_HIDDEN = 256
FNN_DEPTH = 3
DROPOUT = 0.10

ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
HYB_LIST = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]

NUM_ATOM_FEATURES = 24
NUM_BOND_FEATURES = 8
NUM_RDKIT_FEATURES = 29
NUM_JOBACK_FEATURES = 3
NUM_MACCS_FEATURES = 167
NUM_MOL_FEATURES = NUM_RDKIT_FEATURES + NUM_JOBACK_FEATURES + NUM_MACCS_FEATURES  # 199

TC_BAND_LOSS_BOOST = 2.0  # mild loss weight boost for Tc in [300, 500] K

# --- Model A (Tc specialist) task config (verbatim) ---
TARGETS_A = ["Tc (K)"]
WEIGHTS_A = np.array([3.0], dtype=np.float32)

# --- Model B (Pc + omega specialist) task config (verbatim) ---
TARGETS_B = ["Pc (bar)", "omega (-)"]
WEIGHTS_B = np.array([2.5, 2.0], dtype=np.float32)

# --- Pretraining hyperparameters (verbatim defaults from train_multitask_model) ---
PRETRAIN_BATCH_SIZE_DEFAULT = 50
PRETRAIN_LR = 5e-5
PRETRAIN_WEIGHT_DECAY = 1e-5
PRETRAIN_MAX_EPOCHS_DEFAULT = 100000
PRETRAIN_PATIENCE = 150
PRETRAIN_MIN_DELTA = 1e-12
PRETRAIN_SCHED_FACTOR = 0.5
PRETRAIN_SCHED_PATIENCE = 30
PRETRAIN_INTERNAL_VAL_FRACTION = 0.1
PRETRAIN_INTERNAL_SPLIT_SEED = 897742

# --- Fine-tuning hyperparameters (verbatim defaults from fine_tune_multitask_model) ---
FINETUNE_BATCH_SIZE_DEFAULT = 32
FINETUNE_LR = 1e-5
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_MAX_EPOCHS_DEFAULT = 50000
FINETUNE_PATIENCE = 200
FINETUNE_MIN_DELTA = 1e-12
FINETUNE_SCHED_FACTOR = 0.5
FINETUNE_SCHED_PATIENCE = 50

GRAD_CLIP_MAX_NORM = 5.0
HUBER_DELTA = 1.0

# ======================================================================================
# SECTION 2: UTILITY FUNCTIONS
# ======================================================================================

def set_seed(seed: int) -> None:
    """Global reproducibility layer (new in V2 -- the notebook never seeded these
    global RNGs explicitly). This does NOT touch the notebook's own hardcoded
    random_state values used in the train/val/test splits (those are dataset-split
    constants, preserved verbatim in Section 6); this only seeds the RNGs that
    control weight initialization, dropout, and DataLoader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels where practical. We deliberately do NOT call
    # torch.use_deterministic_algorithms(True): DirectedMessagePassing's
    # index_add_ scatter-aggregation has no deterministic CUDA kernel in this
    # PyTorch version and would raise at runtime. cuDNN determinism flags are
    # the practical middle ground requested ("where practical").
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_path: Path) -> logging.Logger:
    """Configures a logger that writes to both the console and logs/training.log.
    All former print() calls inside the training loops are routed through this
    logger (see Sections 7-8) so a full run is captured in the log file."""
    ensure_dir(log_path.parent)
    logger = logging.getLogger("gnn_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def read_csv_robust(path, logger: logging.Logger = None, **kwargs) -> pd.DataFrame:
    """pd.read_csv wrapper that tolerates non-UTF-8 source files.

    Chemistry data exported from Excel/older tools frequently contains
    Windows-1252/Latin-1 bytes (degree signs, plus-minus, Greek letters, etc.)
    that are not valid UTF-8, and/or malformed rows (an unescaped comma inside
    a name field is the classic cause of pandas' "Expected N fields, saw M").
    Pandas' default encoding is strictly UTF-8 on every platform and its
    default row parsing is strict, so a file that happens to "work" under one
    Python/OS/locale/pandas-version combination can hard-fail on another (this
    is NOT a GPU/CUDA issue -- it is pure file parsing, and happens before any
    device or model code runs). This tries utf-8-sig, then cp1252, then
    latin-1 (which never raises on decoding, since every byte value is a
    valid Latin-1 code point); for each encoding, if strict row-parsing fails
    with a small number of malformed rows, it retries once with those rows
    skipped (logging exactly what was skipped) rather than hard-failing the
    whole file."""
    last_err = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            if enc != "utf-8-sig" and logger:
                logger.warning(f"{Path(path).name} is not UTF-8; read successfully using '{enc}' encoding instead.")
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.ParserError as e:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    df = pd.read_csv(path, encoding=enc, on_bad_lines="warn", **kwargs)
                except UnicodeDecodeError as e2:
                    last_err = e2
                    continue
            if logger:
                logger.warning(f"{Path(path).name}: malformed row(s) detected ({e}); "
                                f"re-read with the offending line(s) skipped ({len(caught)} skipped) "
                                f"-- verify they weren't needed.")
                for w in caught:
                    logger.warning(f"  {Path(path).name} skipped line detail: {w.message}")
            return df
    raise last_err


def resolve_data_file(filename: str, data_dir: Path, logger: logging.Logger = None) -> Path:
    """Locates an input dataset file. Simplified, local-filesystem equivalent of
    the notebook's find_uploaded_file/find_kaggle_file (which searched
    /kaggle/input); this pipeline runs outside Kaggle, so it searches only
    `data_dir` (recursively) and raises a clear error if nothing is found."""
    direct = data_dir / filename
    if direct.exists():
        return direct
    matches = sorted(data_dir.rglob(filename))
    if matches:
        if logger:
            logger.info(f"Found '{filename}' at {matches[0]}")
        return matches[0]
    raise FileNotFoundError(
        f"Required data file '{filename}' was not found under '{data_dir}'. "
        f"Pass --data-dir to point at the folder containing the finalized datasets."
    )


def is_halocarbon(smiles):
    """Verbatim from the notebook (Cell 3)."""
    if not isinstance(smiles, str):
        return False
    if "F" in smiles or "Cl" in smiles or "Br" in smiles or "I" in smiles:
        return True
    return False


def count_atoms(smiles):
    """Verbatim from the notebook (Cell 4)."""
    if pd.isna(smiles):
        return np.nan
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms()


def count_molecules_with_atom(df, smiles_col="smiles", atoms=("F", "Cl", "Br", "O", "N")):
    """Verbatim from the notebook (Cell 4)."""
    counts = {atom: 0 for atom in atoms}
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        present = {atom.GetSymbol() for atom in mol.GetAtoms()}
        for atom in atoms:
            if atom in present:
                counts[atom] += 1
    return counts


def joback_predictions(smiles):
    """Verbatim from the notebook (Cell 4). Joback group-contribution estimate of
    Tb/Tc/Pc, used both as an input feature (Section 3) and to fill residual
    pretraining gaps (Section 6)."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_with_hs = Chem.AddHs(mol)
    num_atoms = mol_with_hs.GetNumAtoms()

    joback_groups = {
        "[CH3;X4;!R]": (23.58, 0.0141, 0.0012),
        "[CH2;X4;!R]": (22.88, 0.0189, 0.0000),
        "[CH;X4;!R]":  (21.74, 0.0164, 0.0020),
        "[C;X4;!R]":   (18.25, 0.0067, 0.0043),
        "[CH2X3;!R;$([CH2]=*)]": (18.18, 0.0113, -0.0028),
        "[CHX3;!R;$([CH]=*)]": (24.96, 0.0129, -0.0006),
        "[CX3;!R;$([CX3]=*);!$([CX3]=O)]": (24.14, 0.0117, 0.0011),
        "[CH2;X4;R]":  (27.15, 0.0100, 0.0025),
        "[CH;X4;R]":   (21.78, 0.0122, 0.0039),
        "[C;X4;R]":    (21.32, 0.0042, 0.0061),
        "[OH;X2;!R]":  (92.88, 0.0741, 0.0112),
        "[O;X2;!R]":   (22.42, 0.0168, 0.0015),
        "[C;X3;!R]=O": (76.75, 0.0400, 0.0085),
        "[CX3;!R](=[OX1])[OX2H1]": (155.5, 0.0791, 0.0077),
        "[F]":  (-0.03, 0.0111, -0.0002),
        "[Cl]": (38.13, 0.0105, 0.0040),
        "[Br]": (66.86, 0.0133, 0.0057),
        "[I]":  (93.84, 0.0230, 0.0072),
        "[NH2;X3;!R]": (73.23, 0.0243, 0.0109),
    }

    sum_dtb, sum_dtc, sum_dpc = 0.0, 0.0, 0.0
    used_atoms = set()

    for smarts, inc in joback_groups.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            if match[0] in used_atoms:
                continue
            sum_dtb += inc[0]
            sum_dtc += inc[1]
            sum_dpc += inc[2]
            used_atoms.update(match)

    try:
        Tb = 198 + sum_dtb
        Tc = Tb / (0.584 + 0.965 * sum_dtc - sum_dtc ** 2)
        Pc_bar = (0.113 + 0.0032 * num_atoms - sum_dpc) ** (-2)
        return {
            "SMILES": smiles, "Tb (K)": Tb, "Tc (K)": Tc,
            "Pc (bar)": Pc_bar, "Pc (MPa)": Pc_bar / 10
        }
    except Exception:
        return None


def iupac_to_smiles(name):
    """Verbatim from the notebook (Cell 4). Unused in the active V2 pipeline
    (its only caller, the Critical_Properties_dataset_exp.xlsx merge, was removed
    during the V2 migration -- see gnn_model_v2 migration notes). Kept for
    fidelity/possible future reuse."""
    if pd.isna(name) or not isinstance(name, str):
        return None
    try:
        import pubchempy as pcp
        results = pcp.get_compounds(name, namespace="name")
        if not results:
            return None
        smiles = results[0].connectivity_smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def fill_missing_Tc_with_joback(df, smiles_col="smiles", tc_col="Tc (K)"):
    """Verbatim from the notebook (Cell 4). Unused in the active pipeline (kept
    for fidelity -- it was already dead code in the original notebook)."""
    df = df.copy()
    missing_indices = df[df[tc_col].isna()].index
    for i in missing_indices:
        smiles = df.loc[i, smiles_col]
        if pd.isna(smiles):
            continue
        try:
            result = joback_predictions(smiles)
            if result is None:
                continue
            df.loc[i, tc_col] = result["Tc (K)"]
        except Exception as e:
            print(f"Failed for index {i}, SMILES {smiles}: {e}")
    return df


def element_bond_tc_prediction(smiles, tb_experimental):
    """Verbatim from the notebook (Cell 4)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or tb_experimental is None:
        return np.nan
    mol = Chem.AddHs(mol)
    a_const = 1.5530
    b_const = 18.999

    element_delta = {
        "C": -0.0003, "H": -0.0016, "O": -0.0472, "N": -0.0533,
        "S": 0.0179, "F": -0.0478, "Cl": -0.0380, "Br": 0.0071,
        "I": 0.0091, "Si": 0.0224
    }
    bond_delta = {
        (("C", "C"), Chem.rdchem.BondType.SINGLE, False): -0.0168,
        (("C", "C"), Chem.rdchem.BondType.DOUBLE, False): -0.0080,
        (("C", "C"), Chem.rdchem.BondType.TRIPLE, False): -0.0079,
        (("C", "O"), Chem.rdchem.BondType.SINGLE, False): 0.0059,
        (("C", "O"), Chem.rdchem.BondType.DOUBLE, False): 0.0200,
        (("O", "O"), Chem.rdchem.BondType.SINGLE, False): 0.0000,
        (("O", "H"), Chem.rdchem.BondType.SINGLE, False): -0.0114,
        (("C", "H"), Chem.rdchem.BondType.SINGLE, False): -0.0001,
        (("C", "F"), Chem.rdchem.BondType.SINGLE, False): 0.0285,
        (("C", "Cl"), Chem.rdchem.BondType.SINGLE, False): 0.0358,
        (("C", "Br"), Chem.rdchem.BondType.SINGLE, False): 0.0071,
        (("C", "I"), Chem.rdchem.BondType.SINGLE, False): 0.0091,
        (("C", "S"), Chem.rdchem.BondType.SINGLE, False): -0.0153,
        (("C", "S"), Chem.rdchem.BondType.DOUBLE, False): 0.0360,
        (("C", "Si"), Chem.rdchem.BondType.SINGLE, False): -0.0316,
        (("Si", "Cl"), Chem.rdchem.BondType.SINGLE, False): 0.0209,
        (("Si", "O"), Chem.rdchem.BondType.SINGLE, False): 0.0268,
        (("Si", "H"), Chem.rdchem.BondType.SINGLE, False): 0.0029,
        (("S", "S"), Chem.rdchem.BondType.SINGLE, False): -0.0192,
        (("S", "H"), Chem.rdchem.BondType.SINGLE, False): 0.0152,
        (("C", "N"), Chem.rdchem.BondType.SINGLE, False): -0.0024,
        (("C", "N"), Chem.rdchem.BondType.DOUBLE, False): 0.0180,
        (("C", "N"), Chem.rdchem.BondType.TRIPLE, False): 0.0000,
        (("N", "H"), Chem.rdchem.BondType.SINGLE, False): 0.0171,
        (("N", "N"), Chem.rdchem.BondType.SINGLE, False): 0.0000,
        (("N", "O"), Chem.rdchem.BondType.DOUBLE, False): 0.0528,
        (("C", "C"), Chem.rdchem.BondType.SINGLE, True): -0.0061,
        (("C", "C"), Chem.rdchem.BondType.DOUBLE, True): -0.0055,
        (("C", "O"), Chem.rdchem.BondType.SINGLE, True): 0.0238,
        (("C", "S"), Chem.rdchem.BondType.SINGLE, True): 0.0022,
        (("C", "N"), Chem.rdchem.BondType.SINGLE, True): 0.0209,
        (("C", "N"), Chem.rdchem.BondType.DOUBLE, True): 0.0285,
    }
    aromatic_ring_delta = -0.0398
    sum_delta = 0.0

    for atom in mol.GetAtoms():
        sum_delta += element_delta.get(atom.GetSymbol(), 0.0)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        pair = tuple(sorted((a1, a2)))
        sum_delta += bond_delta.get(((pair, bond.GetBondType(), bond.IsInRing())), 0.0)

    ring_info = mol.GetRingInfo()
    aromatic_count = sum(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) for ring in ring_info.AtomRings())
    sum_delta += aromatic_count * aromatic_ring_delta

    Tc = tb_experimental * (a_const + sum_delta) + b_const
    return Tc


def joback_predictions_Pc(smiles):
    """Verbatim from the notebook (Cell 4)."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molH = Chem.AddHs(mol)
    Na = molH.GetNumAtoms()

    joback_table = {
        "[CH3;X4;!R]": (23.58, 0.0141, -0.0012, 65),
        "[CH2;X4;!R]": (22.88, 0.0189, 0.0, 56),
        "[CH1;X4;!R]": (21.74, 0.0164, 0.0020, 41),
        "[CH0;X4;!R]": (18.25, 0.0067, 0.0043, 27),
        "[CH2;X3;!R]": (18.18, 0.0113, -0.0028, 56),
        "[CH1;X3;!R]": (24.96, 0.0129, -0.0006, 46),
        "[CH0;X3;!R]": (24.14, 0.0117, 0.0011, 38),
        "[$([C;X2](=[C;X2])=[C;X2])]": (26.15, 0.0026, 0.0028, 36),
        "[CH1;X2;!R]": (27.38, 0.0027, -0.0008, 46),
        "[CH0;X2;!R]": (31.01, 0.0020, 0.0016, 37),
        "[CH2;X4;R]":  (27.15, 0.0100, 0.0025, 48),
        "[CH1;X4;R]":  (21.78, 0.0122, 0.0004, 38),
        "[CH0;X4;R]":  (21.32, 0.0042, 0.0061, 27),
        "[CH1;X3;R]":  (26.73, 0.0082, 0.0011, 41),
        "[CH0;X3;R]":  (31.01, 0.0143, 0.0008, 32),
        "[F]":   (-15.78, 0.0111, -0.0057, 27),
        "[Cl]":  (35.99, 0.0105, -0.0049, 58),
        "[Br]":  (63.56, 0.0133, 0.0057, 71),
        "[I]":   (88.43, 0.0068, -0.0034, 97),
        "[OX2H;!$(c-O);!R]": (92.88, 0.0741, 0.0112, 28),
        "[OX2H;$(c-O)]":     (76.34, 0.0240, 0.0184, -25),
        "[OX2;!R]":          (22.42, 0.0168, 0.0015, 18),
        "[OX2;R]":           (31.22, 0.0098, 0.0048, 13),
        "[CX3;!R]=O":        (76.75, 0.0380, 0.0031, 62),
        "[CX3;R]=O":         (94.97, 0.0284, 0.0028, 55),
        "[CX3H1;!R]=O":      (72.24, 0.0379, 0.0030, 82),
        "[CX3;!R](=O)[OX2H1]": (169.13, 0.0791, 0.0077, 89),
        "[CX3;!R](=O)[OX2H0]": (81.10, 0.0481, 0.0005, 82),
        "[OX1;!$(C=O)]":     (36.90, 0.0143, 0.0101, 36),
        "[NH2;X3;!R]": (73.23, 0.0243, 0.0109, 38),
        "[NH1;X3;!R]": (77.11, 0.0295, 0.0077, 35),
        "[NH1;X3;R]":  (52.82, 0.0130, 0.0114, 29),
        "[NH0;X3;!R]": (54.30, 0.0169, 0.0074, 9),
        "[NH0;X2;!R]=*": (45.37, 0.0255, -0.0099, 0),
        "[NH0;X2;R]=*":  (45.37, 0.0085, 0.0076, 34),
        "C#N":           (125.66, 0.0496, -0.0101, 91),
        "[N+](=[OX1])-[OX1-]": (152.54, 0.0437, 0.0064, 91),
        "[SX2H]":    (63.56, 0.0031, 0.0084, 63),
        "[SX2;!R]":  (68.78, 0.0119, 0.0049, 54),
        "[SX2;R]":   (68.78, 0.0019, 0.0051, 38)
    }

    sums = {"tb": 0, "tc": 0, "pc": 0, "vc": 0}
    matched = set()

    for smarts in joback_table:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for m in matches:
            if m[0] in matched:
                continue
            inc = joback_table[smarts]
            sums["tb"] += inc[0]
            sums["tc"] += inc[1]
            sums["pc"] += inc[2]
            sums["vc"] += inc[3]
            matched.add(m[0])

    Tb = 198 + sums["tb"]
    Tc = element_bond_tc_prediction(smiles, Tb)
    Pc_bar = (0.113 + 0.0032 * Na - sums["pc"]) ** (-2)

    return {
        "SMILES": smiles, "Tb (K)": Tb, "Tc (K)": Tc,
        "Pc (MPa)": Pc_bar, "Vc (cm3/mol)": 17.5 + sums["vc"]
    }


def fill_missing_Pc(df, smiles_col="smiles", Pc_col="Pc (bar)"):
    """Verbatim from the notebook (Cell 4). Unused in the active pipeline (kept
    for fidelity -- it was already dead code in the original notebook)."""
    df = df.copy()
    mask = df[Pc_col].isna() & df[smiles_col].notna()
    for idx, smiles in df.loc[mask, smiles_col].items():
        try:
            result = joback_predictions_Pc(smiles)
            if result is not None and "Pc (MPa)" in result:
                df.at[idx, Pc_col] = result["Pc (MPa)"]
        except Exception as e:
            print(f"Failed for SMILES {smiles}: {e}")
    return df


def fill_missing_properties_from_estimates(
    df_miss, df_est,
    smiles_col_miss="smiles", smiles_col_est="smiles",
    Tc_miss="Tc (K)", Pc_miss="Pc (bar)", omega_miss="omega (-)",
    Tc_est="Tc (K)", Pc_est="Pc (bar)", omega_est="omega (-)",
):
    """Verbatim from the notebook (Cell 5). Fills missing Tc/Pc/omega in df_miss
    (fine-tuning rows) from df_est (pretraining/estimated rows) by SMILES."""
    df_est_temp = df_est[[smiles_col_est, Tc_est, Pc_est, omega_est]].rename(columns={
        Tc_est: f"{Tc_miss}_est", Pc_est: f"{Pc_miss}_est", omega_est: f"{omega_miss}_est"
    })

    if smiles_col_miss == smiles_col_est:
        merged = df_miss.merge(df_est_temp, on=smiles_col_miss, how="outer")
    else:
        merged = df_miss.merge(df_est_temp, left_on=smiles_col_miss, right_on=smiles_col_est, how="outer")
        merged[smiles_col_miss] = merged[smiles_col_miss].fillna(merged[smiles_col_est])

    merged[Tc_miss] = merged[Tc_miss].fillna(merged[f"{Tc_miss}_est"])
    merged[Pc_miss] = merged[Pc_miss].fillna(merged[f"{Pc_miss}_est"])
    merged[omega_miss] = merged[omega_miss].fillna(merged[f"{omega_miss}_est"])

    cols_to_drop = [f"{Tc_miss}_est", f"{Pc_miss}_est", f"{omega_miss}_est"]
    if smiles_col_miss != smiles_col_est:
        cols_to_drop.append(smiles_col_est)
    merged = merged.drop(columns=cols_to_drop, errors="ignore")
    return merged


def filter_refrigerant_domain(dataset_df):
    """Verbatim from the notebook (Cell 5). Safely drops non-refrigerant massive
    outliers without dropping NaN targets."""
    filtered = dataset_df[
        (dataset_df["Tc (K)"].isna() | ((dataset_df["Tc (K)"] >= DOMAIN_TC_MIN) & (dataset_df["Tc (K)"] <= DOMAIN_TC_MAX))) &
        (dataset_df["Pc (bar)"].isna() | (dataset_df["Pc (bar)"] <= DOMAIN_PC_MAX)) &
        (dataset_df["omega (-)"].isna() | (dataset_df["omega (-)"] <= DOMAIN_OMEGA_MAX))
    ].copy()
    return filtered


def split_train_val_test(df_source, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42):
    """Verbatim from the notebook (Cell 5). Reproducible 70/15/15 split for
    high-trust experimental-style data."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    train_df = df_source.sample(frac=train_frac, random_state=random_state)
    remaining = df_source.drop(train_df.index)
    val_fraction_of_remaining = val_frac / (val_frac + test_frac)
    val_df = remaining.sample(frac=val_fraction_of_remaining, random_state=random_state + 1)
    test_df = remaining.drop(val_df.index)
    return train_df.copy(), val_df.copy(), test_df.copy()


def fill_pretraining_tc_pc_with_joback(df_train, logger=None):
    """Verbatim from the notebook (Cell 5). Fills remaining Tc/Pc gaps with
    Joback estimates for pretraining rows only."""
    df_train = df_train.copy()
    before_tc = df_train["Tc (K)"].isna().sum()
    before_pc = df_train["Pc (bar)"].isna().sum()

    for idx, row in df_train.iterrows():
        if pd.isna(row["smiles"]):
            continue
        if pd.notna(row["Tc (K)"]) and pd.notna(row["Pc (bar)"]):
            continue
        try:
            result = joback_predictions(row["smiles"])
        except Exception:
            result = None
        if result is None:
            continue
        if pd.isna(row["Tc (K)"]) and "Tc (K)" in result:
            df_train.at[idx, "Tc (K)"] = result["Tc (K)"]
        if pd.isna(row["Pc (bar)"]) and "Pc (bar)" in result:
            df_train.at[idx, "Pc (bar)"] = result["Pc (bar)"]

    filled_tc = before_tc - df_train["Tc (K)"].isna().sum()
    filled_pc = before_pc - df_train["Pc (bar)"].isna().sum()
    msg = f"  Joback pretraining fill: Tc filled {filled_tc}, Pc filled {filled_pc}"
    logger.info(msg) if logger else print(msg)
    return df_train

# ======================================================================================
# SECTION 3: FEATURE GENERATION (descriptor generation -- UNCHANGED from notebook)
# ======================================================================================

def generate_3d_mol(smiles):
    """Verbatim from the notebook (Cell 6). Generates a 3D-optimized molecule
    with Gasteiger charges. Returns (mol_3d, conformer) or (None, None) on
    failure. Coordinates are centered around the molecule's centroid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # Reproducibility (notebook constant, not the CLI --seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status == -1:
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status == -1:
            return None, None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    conf = mol.GetConformer()

    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    centroid = positions.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x - centroid[0], pos.y - centroid[1], pos.z - centroid[2]))

    return mol, conf


def get_rdkit_features(mol, mol_3d=None):
    """Verbatim from the notebook (Cell 6). 29 molecular descriptors: 19 curated
    2D descriptors + 10 3D shape descriptors. Falls back to zeros for 3D
    features if mol_3d is None."""
    mol_2d = Chem.RemoveHs(mol) if mol.GetNumAtoms() > 0 else mol

    feats_2d = [
        Descriptors.MolWt(mol_2d),
        Descriptors.NumValenceElectrons(mol_2d),
        Descriptors.BertzCT(mol_2d),
        Descriptors.TPSA(mol_2d),
        Descriptors.NumRotatableBonds(mol_2d),
        Descriptors.RingCount(mol_2d),
        Descriptors.NumHDonors(mol_2d),
        Descriptors.NumHAcceptors(mol_2d),
        Descriptors.NumHeteroatoms(mol_2d),
        Descriptors.NumAliphaticRings(mol_2d),
        Descriptors.NumAromaticRings(mol_2d),
        Descriptors.FractionCSP3(mol_2d),
        Descriptors.MolLogP(mol_2d),
        Descriptors.MolMR(mol_2d),
        Descriptors.LabuteASA(mol_2d),
        Descriptors.BalabanJ(mol_2d),
        Descriptors.HallKierAlpha(mol_2d),
        Descriptors.Chi0v(mol_2d),
        Descriptors.Chi1v(mol_2d),
    ]

    if mol_3d is not None and mol_3d.GetNumConformers() > 0:
        try:
            feats_3d = [
                Descriptors3D.Asphericity(mol_3d),
                Descriptors3D.Eccentricity(mol_3d),
                Descriptors3D.InertialShapeFactor(mol_3d),
                Descriptors3D.NPR1(mol_3d),
                Descriptors3D.NPR2(mol_3d),
                Descriptors3D.PMI1(mol_3d),
                Descriptors3D.PMI2(mol_3d),
                Descriptors3D.PMI3(mol_3d),
                Descriptors3D.RadiusOfGyration(mol_3d),
                Descriptors3D.SpherocityIndex(mol_3d),
            ]
        except Exception:
            feats_3d = [0.0] * 10
    else:
        feats_3d = [0.0] * 10

    all_feats = feats_2d + feats_3d
    return np.nan_to_num(np.array(all_feats, dtype=np.float32))


def get_joback_features(smiles):
    """Verbatim from the notebook (Cell 6). Returns [Tb_joback, Tc_joback,
    Pc_joback] as physics-based priors, [0, 0, 0] on failure."""
    try:
        result = joback_predictions(smiles)
        if result is not None:
            return np.array([
                result["Tb (K)"],
                result["Tc (K)"],
                result["Pc (bar)"],
            ], dtype=np.float32)
    except Exception:
        pass
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def atom_features(atom, conformer=None, atom_idx=None):
    """Verbatim from the notebook (Cell 6). 24-dim atom features: 10 one-hot
    atom type + 5 one-hot hybridisation + 5 continuous + 3 spatial coords
    (centered) + 1 Gasteiger charge."""
    atom_type = [int(atom.GetAtomicNum() == a) for a in ATOM_LIST]
    hyb = atom.GetHybridization()
    hyb_enc = [int(hyb == h) for h in HYB_LIST]
    continuous = [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        atom.GetMass() * 0.01,
    ]

    if conformer is not None and atom_idx is not None:
        pos = conformer.GetAtomPosition(atom_idx)
        coords = [pos.x * 0.1, pos.y * 0.1, pos.z * 0.1]
    else:
        coords = [0.0, 0.0, 0.0]

    try:
        charge = float(atom.GetDoubleProp("_GasteigerCharge"))
        if np.isnan(charge) or np.isinf(charge):
            charge = 0.0
    except Exception:
        charge = 0.0

    return np.array(atom_type + hyb_enc + continuous + coords + [charge], dtype=np.float32)


def bond_features(bond, conformer=None):
    """Verbatim from the notebook (Cell 6). 8-dim bond features: 7 original + 1
    3D bond length."""
    bt = bond.GetBondType()

    if conformer is not None:
        pos_i = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
        pos_j = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        dist = pos_i.Distance(pos_j)
        bond_length = dist * 0.5
    else:
        bond_length = 0.75

    return np.array([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        int(bond.GetStereo()),
        bond_length,
    ], dtype=np.float32)


class TargetScaler:
    """Verbatim from the notebook (Cell 6). Scales targets to Mean=0, Std=1."""

    def __init__(self):
        self.means = None
        self.stds = None
        self.columns = None

    def fit(self, df, columns):
        self.columns = columns
        self.means = df[columns].mean().values
        self.stds = df[columns].std().values
        self.stds = np.where(self.stds == 0, 1e-6, self.stds)

    def transform(self, values):
        return (values - self.means) / self.stds

    def inverse_transform(self, values_scaled):
        return (values_scaled * self.stds) + self.means


class MolFeatureScaler:
    """Verbatim from the notebook (Cell 6). Pre-computes per-feature mean/std
    from a list of SMILES. Standardises the combined 199 descriptors."""

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, smiles_list, logger=None):
        features = []
        n_fail = 0
        for i, smi in enumerate(smiles_list):
            if pd.isna(smi):
                continue
            mol_3d, conf = generate_3d_mol(smi)
            mol_2d = Chem.MolFromSmiles(smi)
            if mol_2d is not None:
                rdkit_feats = get_rdkit_features(mol_2d, mol_3d)
                joback_feats = get_joback_features(smi)
                maccs = MACCSkeys.GenMACCSKeys(mol_2d)
                maccs_feats = np.array([int(x) for x in maccs.ToBitString()])
                combined = np.concatenate([rdkit_feats, joback_feats, maccs_feats])
                features.append(combined)
            else:
                n_fail += 1
            if (i + 1) % 500 == 0:
                msg = f"    MolFeatureScaler: processed {i + 1}/{len(smiles_list)} molecules..."
                logger.info(msg) if logger else print(msg)
        features = np.array(features)
        self.means = features.mean(axis=0)
        self.stds = features.std(axis=0)
        self.stds = np.where(self.stds == 0, 1e-6, self.stds)
        msg = f"  MolFeatureScaler fit on {len(features)} molecules ({features.shape[1]} features), {n_fail} failed"
        logger.info(msg) if logger else print(msg)

    def transform(self, raw_features):
        return ((raw_features - self.means) / self.stds).astype(np.float32)


# ======================================================================================
# SECTION 4: GRAPH GENERATION (UNCHANGED from notebook)
# ======================================================================================
# `mol_scaler` is a module-level singleton, matching the notebook's own pattern
# (smiles_to_graph_multitask references it as a global). It is fit once per
# process in Section 7/8 and reused everywhere -- exactly as in the notebook.
mol_scaler = MolFeatureScaler()


def smiles_to_graph_multitask(smiles, targets_scaled):
    """Verbatim from the notebook (Cell 6). Builds a single torch_geometric
    Data object: atom features (24-dim), bond features (8-dim), scaled
    molecular descriptors (199-dim), and the (already-scaled) targets."""
    mol_2d = Chem.MolFromSmiles(smiles)
    if mol_2d is None:
        return None

    mol_3d, conf = generate_3d_mol(smiles)

    if mol_3d is not None and conf is not None:
        atom_feats = []
        for atom in mol_2d.GetAtoms():
            idx = atom.GetIdx()
            if idx < mol_3d.GetNumAtoms():
                atom_3d = mol_3d.GetAtomWithIdx(idx)
                feat = atom_features(atom_3d, conf, idx)
            else:
                feat = atom_features(atom)
            atom_feats.append(feat)
    else:
        atom_feats = [atom_features(atom) for atom in mol_2d.GetAtoms()]

    x = torch.tensor(np.array(atom_feats), dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol_2d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond, conf)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    y = torch.tensor([targets_scaled], dtype=torch.float)

    rdkit_feats = get_rdkit_features(mol_2d, mol_3d)
    joback_feats = get_joback_features(smiles)
    maccs = MACCSkeys.GenMACCSKeys(mol_2d)
    maccs_feats = np.array([int(x) for x in maccs.ToBitString()])
    raw_feats = np.concatenate([rdkit_feats, joback_feats, maccs_feats])
    scaled_feats = mol_scaler.transform(raw_feats)
    mol_attr = torch.tensor(scaled_feats, dtype=torch.float).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mol_attr=mol_attr)

# ======================================================================================
# SECTION 5: DATASET PREPARATION (torch Dataset classes -- UNCHANGED from notebook)
# ======================================================================================

class MultitaskDataset(Dataset):
    """Verbatim from the notebook (Cell 6). Keeps molecules even if some targets
    are NaN. NaN targets are filled with 0 and a binary mask is stored so the
    loss function can ignore them."""

    def __init__(self, df, target_cols, target_scaler):
        super().__init__()
        self.graphs = []
        for _, row in df.iterrows():
            smiles = row["smiles"]
            if pd.isna(smiles):
                continue
            raw_targets = row[target_cols].values.astype(np.float32)
            mask = (~np.isnan(raw_targets)).astype(np.float32)
            if mask.sum() == 0:
                continue
            raw_targets_filled = np.nan_to_num(raw_targets, nan=0.0)
            scaled_targets = target_scaler.transform(raw_targets_filled)
            graph = smiles_to_graph_multitask(smiles, scaled_targets)
            if graph is not None:
                loss_weight = mask.copy()
                if "Tc (K)" in target_cols:
                    tc_idx = target_cols.index("Tc (K)")
                    tc_val = raw_targets[tc_idx]
                    if np.isfinite(tc_val) and 300.0 <= tc_val <= 500.0:
                        loss_weight[tc_idx] *= TC_BAND_LOSS_BOOST
                graph.target_mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
                graph.loss_weight = torch.tensor(loss_weight, dtype=torch.float).unsqueeze(0)
                self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class MultitaskPredictionDataset(Dataset):
    """Verbatim from the notebook (Cell 6). For inference -- does NOT skip rows
    with missing targets."""

    def __init__(self, df, target_cols, target_scaler):
        super().__init__()
        self.graphs = []
        self.true_targets = []
        self.smiles = []
        self.row_ids = []
        n_targets = len(target_cols)

        for row_pos, row in df.iterrows():
            smiles = row["smiles"]
            if pd.isna(smiles):
                continue
            dummy_scaled_targets = np.zeros(n_targets, dtype=np.float32)
            graph = smiles_to_graph_multitask(smiles, dummy_scaled_targets)
            if graph is None:
                continue
            self.graphs.append(graph)
            self.smiles.append(smiles)
            self.row_ids.append(row["row_id"] if "row_id" in df.columns else row_pos)

            actuals = []
            for col in target_cols:
                if col in df.columns:
                    actuals.append(row[col])
                else:
                    actuals.append(np.nan)
            self.true_targets.append(actuals)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ======================================================================================
# MODEL ARCHITECTURE (UNCHANGED from notebook -- DMPNN + FNN + loss)
# ======================================================================================
# Grouped here, between dataset prep and pretraining, exactly where Cell 6 placed
# it in the notebook. Nothing in this block was modified: hidden sizes, depth,
# dropout, message passing, and the loss function are copied verbatim.

class DirectedMessagePassing(nn.Module):
    def __init__(self, hidden_size, edge_size, dropout=DROPOUT):
        super().__init__()
        self.W_msg = nn.Linear(hidden_size + edge_size, hidden_size)
        self.W_update = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg_input = torch.cat([x[src], edge_attr], dim=1)
        message = self.dropout(F.relu(self.W_msg(msg_input)))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, message)
        updated = self.dropout(F.relu(self.W_update(torch.cat([x, agg], dim=1))))
        return updated


class DMPNN(nn.Module):
    def __init__(self, atom_size=NUM_ATOM_FEATURES, bond_size=NUM_BOND_FEATURES,
                 hidden_size=HIDDEN_SIZE, depth=DMPNN_DEPTH, dropout=DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(atom_size, hidden_size)
        self.layers = nn.ModuleList([
            DirectedMessagePassing(hidden_size, bond_size, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        x = F.relu(self.input_proj(x))
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        mol_embedding = global_add_pool(x, batch)
        return mol_embedding


class FNN(nn.Module):
    def __init__(self, input_size=HIDDEN_SIZE + NUM_MOL_FEATURES,
                 hidden_size=FNN_HIDDEN, depth=FNN_DEPTH, output_size=3, dropout=DROPOUT):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultitaskModel(nn.Module):
    def __init__(self, num_targets=3, num_additional_features=NUM_MOL_FEATURES, dropout=DROPOUT):
        super().__init__()
        self.encoder = DMPNN(hidden_size=HIDDEN_SIZE, dropout=dropout)
        self.predictor = FNN(
            input_size=HIDDEN_SIZE + num_additional_features,
            output_size=num_targets,
            dropout=dropout,
        )

    def forward(self, data):
        embedding = self.encoder(data)
        extra_feats = data.mol_attr.view(embedding.size(0), -1)
        combined = torch.cat([embedding, extra_feats], dim=1)
        predictions = self.predictor(combined)
        return predictions


def masked_weighted_huber_loss(pred, target, mask, target_weights=None, sample_weights=None, delta=HUBER_DELTA):
    """Verbatim from the notebook (Cell 6). Masked, target-weighted Huber loss
    in scaled target space."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    element_loss = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta),
    )

    if target_weights is None:
        weights = torch.ones_like(mask)
    else:
        weights = torch.tensor(target_weights, dtype=pred.dtype, device=pred.device).view(1, -1)
        weights = weights.expand_as(mask)

    if sample_weights is None:
        sample_weights = torch.ones_like(mask)

    weighted_mask = mask * weights * sample_weights
    n_valid = weighted_mask.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return (element_loss * weighted_mask).sum() / n_valid


def masked_mse_loss(pred, target, mask, target_weights=None, sample_weights=None):
    """Verbatim from the notebook (Cell 6). Backward-compatible name used by
    training loops (it actually calls the Huber loss, exactly as in V1/V2)."""
    return masked_weighted_huber_loss(
        pred, target, mask,
        target_weights=target_weights,
        sample_weights=sample_weights,
        delta=HUBER_DELTA,
    )


# ======================================================================================
# SECTION 6: DATA PREPROCESSING (dataset curation pipeline)
# ======================================================================================

def curate_datasets(args, logger):
    """Faithful port of the notebook's Cell 3 + Cell 5 (V2 migrated). Loads the
    finalized pretraining/fine-tuning CSVs plus the untouched McLinden
    auxiliary CSV, applies the same cleaning/filtering/splitting logic as the
    notebook, and returns the four dataframes the rest of the pipeline needs.

    Removed vs. the V1 notebook (see migration notes, unchanged from the V2
    notebook migration already applied):
      - Critical_Properties_dataset_exp.xlsx merge (superseded by the
        finalized fine-tuning file, which already carries this data forward).
      - Towhee Tc-only pretraining augmentation (explicitly excluded; the
        finalized pretraining file is authoritative and does not include it).
    """
    data_dir = Path(args.data_dir)
    plots_dir = ensure_dir(Path(args.results_dir) / PLOTS_SUBDIR)

    # --- Cell 3: halocarbon analysis on the fine-tuning file ---
    finetune_path = resolve_data_file(args.finetune_csv, data_dir, logger)
    df = read_csv_robust(finetune_path, logger=logger)
    df["is_halocarbon"] = df["smiles"].apply(is_halocarbon)
    counts = df["is_halocarbon"].value_counts()
    halocarbons_count = counts.get(True, 0)
    non_halocarbons_count = counts.get(False, 0)
    logger.info(f"Halocarbon analysis: {halocarbons_count} halocarbons, {non_halocarbons_count} non-halocarbons")

    fig = plt.figure(figsize=(8, 8))
    plt.pie(
        [non_halocarbons_count, halocarbons_count],
        labels=["Non-Halocarbons", "Halocarbons"],
        autopct="%1.4f%%", startangle=140, colors=["#66b3ff", "#ff9999"],
    )
    plt.title("Proportion of Halocarbons vs Rest")
    plt.axis("equal")
    fig.savefig(plots_dir / "halocarbons_pie_chart.png")
    plt.close(fig)

    # --- Cell 5: dataset curation pipeline ---
    df = read_csv_robust(finetune_path, logger=logger, header=0)

    tc_counts = df["Tc (K)"].value_counts()
    suspicious_tcs = tc_counts[tc_counts >= 5].index
    if len(suspicious_tcs) > 0:
        logger.info(f"[DATA CLEANING] Found {len(suspicious_tcs)} suspicious 'default' Tc values appearing 5+ times")
        logger.info(f"  values: {suspicious_tcs.values.tolist()}")
        n_drop = df["Tc (K)"].isin(suspicious_tcs).sum()
        logger.info(f"  Dropping {n_drop} artificial rows to clean the ground truth...")
        df = df[~df["Tc (K)"].isin(suspicious_tcs)].copy()

    df["NumAtoms"] = df["smiles"].apply(count_atoms)
    count = (df["NumAtoms"] < 31).sum()
    logger.info(f"Number of molecules with NumAtoms <= 30: {count}")

    df_flt = df[df["NumAtoms"] <= 30].copy()

    atom_counts = count_molecules_with_atom(df, "smiles")
    logger.info(f"Atom presence counts: {atom_counts}")

    df_expt = df_flt[
        df_flt["Tc (K)"].notna() |
        df_flt["Pc (bar)"].notna() |
        df_flt["omega (-)"].notna()
    ].copy()

    df_miss = df_flt[df_flt["Tc (K)"].isna() | df_flt["Pc (bar)"].isna() | df_flt["omega (-)"].isna()].copy()

    pretrain_path = resolve_data_file(args.pretrain_csv, data_dir, logger)
    df_est = read_csv_robust(pretrain_path, logger=logger, header=0)

    df_est_fill = fill_missing_properties_from_estimates(
        df_miss, df_est,
        smiles_col_miss="smiles", smiles_col_est="smiles",
        Tc_miss="Tc (K)", Pc_miss="Pc (bar)", omega_miss="omega (-)",
        Tc_est="Tc (K)", Pc_est="Pc (bar)", omega_est="omega (-)",
    )
    df_est_fill["NumAtoms"] = df_est_fill["smiles"].apply(count_atoms)
    df_est_fill = df_est_fill[df_est_fill["NumAtoms"] <= 30].copy()

    mcl_path = resolve_data_file(args.mclinden_csv, data_dir, logger)
    df_mcl_clean = read_csv_robust(mcl_path, logger=logger, header=0)
    df_mcl_clean = df_mcl_clean[df_mcl_clean["smiles"].notna()].copy()
    df_mcl_clean = df_mcl_clean[df_mcl_clean["Tc (K)"].notna()].copy()
    for col in TARGET_COLS:
        if col not in df_mcl_clean.columns:
            df_mcl_clean[col] = np.nan

    logger.info("Applying Domain Specialization Filters (Tc 150-800K, Pc <= 100 bar)...")
    n_expt = len(df_expt)
    df_expt = filter_refrigerant_domain(df_expt)
    logger.info(f"  Experimental Domain: kept {len(df_expt)}/{n_expt} molecules")

    n_est = len(df_est_fill)
    df_est_fill = filter_refrigerant_domain(df_est_fill)
    logger.info(f"  Joback Domain: kept {len(df_est_fill)}/{n_est} molecules")

    n_mcl = len(df_mcl_clean)
    df_mcl_clean = filter_refrigerant_domain(df_mcl_clean)
    logger.info(f"  McLinden Domain: kept {len(df_mcl_clean)}/{n_mcl} molecules")
    for col in TARGET_COLS:
        if col not in df_mcl_clean.columns:
            df_mcl_clean[col] = np.nan

    df_expt_train, df_expt_val, df_expt_test = split_train_val_test(df_expt, random_state=EXPT_SPLIT_RANDOM_STATE)
    df_mcl_train, df_mcl_val, df_mcl_test = split_train_val_test(df_mcl_clean, random_state=MCL_SPLIT_RANDOM_STATE)

    df_est_with_targets = df_est_fill[["smiles"] + TARGET_COLS].copy()
    df_est_with_targets = df_est_with_targets[df_est_with_targets["Tc (K)"].notna()]

    df_pre_train = df_est_with_targets[["smiles"] + TARGET_COLS].copy()
    df_pre_train = df_pre_train.drop_duplicates(subset="smiles", keep="first")
    df_pre_train = fill_pretraining_tc_pc_with_joback(df_pre_train, logger=logger)

    df_fine_train = pd.concat([
        df_expt_train[["smiles"] + TARGET_COLS],
        df_mcl_train[["smiles"] + TARGET_COLS],
    ], ignore_index=True)
    df_fine_train = df_fine_train.drop_duplicates(subset="smiles")

    df_fine_val = pd.concat([
        df_expt_val[["smiles"] + TARGET_COLS],
        df_mcl_val[["smiles"] + TARGET_COLS],
    ], ignore_index=True)
    df_fine_val = df_fine_val.drop_duplicates(subset="smiles")

    df_test = pd.concat([
        df_mcl_test[["smiles"] + TARGET_COLS],
        df_expt_test[["smiles"] + TARGET_COLS],
    ], ignore_index=True)

    logger.info("Final Dataset Size:")
    logger.info(f"  Pre-Train Data: {df_pre_train.shape[0]} molecules")
    logger.info(f"  Fine-Tuning Train: {df_fine_train.shape[0]} molecules")
    logger.info(f"  Fine-Tuning Val: {df_fine_val.shape[0]} molecules")
    logger.info(f"  Final Test Data: {df_test.shape[0]} molecules "
                f"(McLinden: {df_mcl_test.shape[0]}, Expt: {df_expt_test.shape[0]}, Estimated: 0)")

    n_full = df_test.dropna(subset=TARGET_COLS).shape[0]
    logger.info("Test set ground-truth coverage:")
    logger.info(f"  Tc:    {df_test['Tc (K)'].notna().sum()}/{len(df_test)}")
    logger.info(f"  Pc:    {df_test['Pc (bar)'].notna().sum()}/{len(df_test)}")
    logger.info(f"  omega: {df_test['omega (-)'].notna().sum()}/{len(df_test)}")
    logger.info(f"  All 3: {n_full}/{len(df_test)}")

    all_test_smiles = set(df_test["smiles"].dropna().values)
    logger.info(f"Total unique test SMILES to exclude from training: {len(all_test_smiles)}")

    return {
        "df_pre_train": df_pre_train,
        "df_fine_train": df_fine_train,
        "df_fine_val": df_fine_val,
        "df_test": df_test,
        "all_test_smiles": all_test_smiles,
    }


def build_task_subsets(datasets, logger):
    """Splits the curated dataframes into the Model A (Tc) and Model B
    (Pc/omega) subsets, and fits scaler_A/scaler_B -- verbatim logic lifted
    from the top of the notebook's Cell 7."""
    df_pre_train = datasets["df_pre_train"]
    df_fine_train = datasets["df_fine_train"]
    df_fine_val = datasets["df_fine_val"]
    all_test_smiles = datasets["all_test_smiles"]

    df_pre_A = df_pre_train[df_pre_train["Tc (K)"].notna()].copy()
    df_pre_A = df_pre_A[~df_pre_A["smiles"].isin(all_test_smiles)].copy()

    df_fine_A = df_fine_train[df_fine_train["Tc (K)"].notna()].copy()
    df_fine_A = df_fine_A[~df_fine_A["smiles"].isin(all_test_smiles)].copy()

    df_val_A = df_fine_val[df_fine_val["Tc (K)"].notna()].copy()
    df_val_A = df_val_A[~df_val_A["smiles"].isin(all_test_smiles)].copy()

    scaler_A = TargetScaler()
    scaler_A.fit(df_fine_A, TARGETS_A)

    df_pre_B = df_pre_train[df_pre_train[TARGETS_B].notna().any(axis=1)].copy()
    df_pre_B = df_pre_B[~df_pre_B["smiles"].isin(all_test_smiles)].copy()

    df_fine_B = df_fine_train[df_fine_train[TARGETS_B].notna().any(axis=1)].copy()
    df_fine_B = df_fine_B[~df_fine_B["smiles"].isin(all_test_smiles)].copy()

    df_val_B = df_fine_val[df_fine_val[TARGETS_B].notna().any(axis=1)].copy()
    df_val_B = df_val_B[~df_val_B["smiles"].isin(all_test_smiles)].copy()

    scaler_B = TargetScaler()
    scaler_B.fit(df_fine_B, TARGETS_B)

    logger.info(f"Model A (Tc) subsets      : pretrain={len(df_pre_A)}, finetune={len(df_fine_A)}, val={len(df_val_A)}")
    logger.info(f"Model B (Pc/omega) subsets : pretrain={len(df_pre_B)}, finetune={len(df_fine_B)}, val={len(df_val_B)}")

    return {
        "df_pre_A": df_pre_A, "df_fine_A": df_fine_A, "df_val_A": df_val_A, "scaler_A": scaler_A,
        "df_pre_B": df_pre_B, "df_fine_B": df_fine_B, "df_val_B": df_val_B, "scaler_B": scaler_B,
    }


def fit_mol_scaler(datasets, logger):
    """Verbatim logic from the notebook (Cell 7): mol_scaler is fit ONCE on the
    union of all pretraining + fine-tuning SMILES, excluding test SMILES, and
    reused everywhere afterwards."""
    all_test_smiles = datasets["all_test_smiles"]
    all_train_smiles = list(
        set(datasets["df_pre_train"]["smiles"].dropna().tolist()) |
        set(datasets["df_fine_train"]["smiles"].dropna().tolist())
    )
    all_train_smiles = [s for s in all_train_smiles if s not in all_test_smiles]
    logger.info(f"Fitting mol_scaler on {len(all_train_smiles)} training SMILES...")
    mol_scaler.fit(all_train_smiles, logger=logger)
    return mol_scaler

# ======================================================================================
# SECTION 7: PRETRAINING
# ======================================================================================
# train_multitask_model is a verbatim port of the notebook's Cell 6 function.
# The ONLY changes are: (1) print() -> logger, (2) optional train/val metrics
# list append (for results/train_metrics.csv & validation_metrics.csv), and
# (3) checkpoint path is now an explicit Path under checkpoints/. No
# architecture, loss, optimizer, scheduler, or hyperparameter was touched.

def train_multitask_model(
    df,
    target_cols,
    target_scaler,
    device,
    logger,
    batch_size=PRETRAIN_BATCH_SIZE_DEFAULT,
    lr=PRETRAIN_LR,
    weight_decay=PRETRAIN_WEIGHT_DECAY,
    max_epochs=PRETRAIN_MAX_EPOCHS_DEFAULT,
    patience=PRETRAIN_PATIENCE,
    min_delta=PRETRAIN_MIN_DELTA,
    model_save_path="best_multitask_model.pt",
    target_weights=None,
    stage_name="pretrain",
    train_metrics=None,
    val_metrics=None,
):
    train_df, val_df = train_test_split(df, test_size=PRETRAIN_INTERNAL_VAL_FRACTION,
                                         random_state=PRETRAIN_INTERNAL_SPLIT_SEED)

    train_dataset = MultitaskDataset(train_df, target_cols, target_scaler)
    val_dataset = MultitaskDataset(val_df, target_cols, target_scaler)

    n_targets = len(target_cols)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"[{stage_name}] Train size: {len(train_dataset)}")
    logger.info(f"[{stage_name}] Val size: {len(val_dataset)}")

    model = MultitaskModel(num_targets=n_targets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PRETRAIN_SCHED_FACTOR, patience=PRETRAIN_SCHED_PATIENCE
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.view(-1, n_targets)
            mask = batch.target_mask.view(-1, n_targets)
            sample_weights = getattr(batch, "loss_weight", torch.ones_like(mask)).view(-1, n_targets)
            loss = masked_mse_loss(pred, target, mask, target_weights=target_weights, sample_weights=sample_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.y.view(-1, n_targets)
                mask = batch.target_mask.view(-1, n_targets)
                sample_weights = getattr(batch, "loss_weight", torch.ones_like(mask)).view(-1, n_targets)
                loss = masked_mse_loss(pred, target, mask, target_weights=target_weights, sample_weights=sample_weights)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if train_metrics is not None:
            train_metrics.append({"stage": stage_name, "epoch": epoch + 1, "train_loss": train_loss, "lr": current_lr})
        if val_metrics is not None:
            val_metrics.append({"stage": stage_name, "epoch": epoch + 1, "val_loss": val_loss})

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"[{stage_name}] Epoch {epoch + 1:4d} | Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"[{stage_name}] Early stopping triggered")
            break

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, train_losses, val_losses


def predict_multitask(model, df_exp, target_cols, target_scaler, device, batch_size=64):
    """Verbatim from the notebook (Cell 6)."""
    model.eval()
    dataset = MultitaskPredictionDataset(df_exp, target_cols, target_scaler)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_scaled = model(batch).cpu().numpy()
            pred_real = target_scaler.inverse_transform(pred_scaled)
            predictions.extend(pred_real)

    predictions = np.array(predictions)
    trues = np.array(dataset.true_targets)

    res_dict = {"row_id": dataset.row_ids, "smiles": dataset.smiles}
    for i, col in enumerate(target_cols):
        res_dict[col + "_exp"] = trues[:, i]
        res_dict[col] = predictions[:, i]

    return pd.DataFrame(res_dict)


def run_pretrain_stage(args, logger):
    """Orchestrates the ONE stage that has no cross-model dependency: Model A
    (Tc specialist) pretraining on the estimated dataset. Produces
    checkpoints/best_tc_pretrained.pt and checkpoints/scalers.pt (mol_scaler +
    scaler_A + scaler_B, needed by `--mode finetune` even when run as a
    separate process)."""
    logger.info("=" * 70)
    logger.info("STAGE: PRETRAINING (Model A -- Tc specialist)")
    logger.info("=" * 70)

    ckpt_dir = ensure_dir(Path(args.checkpoint_dir))
    results_dir = ensure_dir(Path(args.results_dir))
    plots_dir = ensure_dir(results_dir / PLOTS_SUBDIR)

    datasets = curate_datasets(args, logger)
    run_preflight_graph_checks(datasets["df_pre_train"]["smiles"].head(20).tolist(), logger)

    task_subsets = build_task_subsets(datasets, logger)
    fit_mol_scaler(datasets, logger)

    train_metrics, val_metrics = [], []

    batch_size = args.batch_size if args.batch_size else PRETRAIN_BATCH_SIZE_DEFAULT
    max_epochs = args.epochs if args.epochs else PRETRAIN_MAX_EPOCHS_DEFAULT

    logger.info(f"Pre-training Tc model on {len(task_subsets['df_pre_A'])} molecules")
    model_A_pre, pre_train_losses_A, pre_val_losses_A = train_multitask_model(
        df=task_subsets["df_pre_A"],
        target_cols=TARGETS_A,
        target_scaler=task_subsets["scaler_A"],
        device=args.device,
        logger=logger,
        batch_size=batch_size,
        max_epochs=max_epochs,
        model_save_path=str(ckpt_dir / CKPT_TC_PRETRAINED),
        target_weights=WEIGHTS_A,
        stage_name="pretrain_A_tc",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )

    save_scalers_bundle(mol_scaler, task_subsets["scaler_A"], task_subsets["scaler_B"], ckpt_dir / CKPT_SCALERS)
    write_metrics_csv(train_metrics, results_dir / "train_metrics.csv", append=True)
    write_metrics_csv(val_metrics, results_dir / "validation_metrics.csv", append=True)
    plot_loss_curve(pre_train_losses_A, pre_val_losses_A, "Model A Pretraining (Tc)", plots_dir / "pretrain_A_loss.png")

    logger.info(f"Pretraining stage complete. Checkpoint: {ckpt_dir / CKPT_TC_PRETRAINED}")
    return {"model_A_pre": model_A_pre, "datasets": datasets, "task_subsets": task_subsets}

# ======================================================================================
# SECTION 8: FINE-TUNING
# ======================================================================================
# fine_tune_multitask_model is a verbatim port of the notebook's Cell 6
# function (same print->logger / metrics-list additions as Section 7).
# IMPORTANT: this single function is reused for THREE different roles in the
# hybrid pipeline, exactly as in the notebook: (1) Model A fine-tuning,
# (2) Model B "pretraining" with a frozen, transfer-learned encoder, and
# (3) Model B fine-tuning. See run_finetune_stage() for the exact order.

def fine_tune_multitask_model(
    pretrained_model,
    df_finetune,
    target_cols,
    target_scaler,
    device,
    logger,
    batch_size=FINETUNE_BATCH_SIZE_DEFAULT,
    lr=FINETUNE_LR,
    weight_decay=FINETUNE_WEIGHT_DECAY,
    max_epochs=FINETUNE_MAX_EPOCHS_DEFAULT,
    patience=FINETUNE_PATIENCE,
    min_delta=FINETUNE_MIN_DELTA,
    freeze_encoder=True,
    model_save_path="best_multitask_finetuned.pt",
    target_weights=None,
    val_df=None,
    stage_name="finetune",
    train_metrics=None,
    val_metrics=None,
):
    logger.info(f"[{stage_name}] Starting Multitask Fine-Tuning...")

    if val_df is None:
        train_df, val_df = train_test_split(df_finetune, test_size=0.1, random_state=42)
    else:
        train_df = df_finetune.copy()
        val_df = val_df.copy()

    train_dataset = MultitaskDataset(train_df, target_cols, target_scaler)
    val_dataset = MultitaskDataset(val_df, target_cols, target_scaler)
    n_targets = len(target_cols)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    logger.info(f"[{stage_name}] Fine-tune Train size: {len(train_dataset)}")
    logger.info(f"[{stage_name}] Fine-tune Val size: {len(val_dataset)}")

    model = pretrained_model.to(device)

    if freeze_encoder:
        logger.info(f"[{stage_name}] Freezing encoder weights")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        logger.info(f"[{stage_name}] Training entire model")
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.predictor.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=FINETUNE_SCHED_FACTOR, patience=FINETUNE_SCHED_PATIENCE
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.view(-1, n_targets)
            mask = batch.target_mask.view(-1, n_targets)
            sample_weights = getattr(batch, "loss_weight", torch.ones_like(mask)).view(-1, n_targets)
            loss = masked_mse_loss(pred, target, mask, target_weights=target_weights, sample_weights=sample_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.y.view(-1, n_targets)
                mask = batch.target_mask.view(-1, n_targets)
                sample_weights = getattr(batch, "loss_weight", torch.ones_like(mask)).view(-1, n_targets)
                loss = masked_mse_loss(pred, target, mask, target_weights=target_weights, sample_weights=sample_weights)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if train_metrics is not None:
            train_metrics.append({"stage": stage_name, "epoch": epoch + 1, "train_loss": train_loss, "lr": current_lr})
        if val_metrics is not None:
            val_metrics.append({"stage": stage_name, "epoch": epoch + 1, "val_loss": val_loss})

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"[{stage_name}] Fine-Tune Epoch {epoch + 1:4d} | Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"[{stage_name}] Fine-tuning early stopped")
            break

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, train_losses, val_losses


def run_finetune_stage(args, logger, pretrain_outputs=None):
    """Orchestrates the three dependent stages, in the exact order the
    notebook's Cell 7 executes them:
      1. Model A fine-tuning (encoder unfrozen) on experimental data.
      2. Model B built from Model A's *fine-tuned* encoder (frozen) and
         "pretrained" (via fine_tune_multitask_model) on the estimated pool.
      3. Model B fine-tuning (encoder unfrozen) on experimental data.
    Then saves the portable merged_pipeline.pt bundle.

    If `pretrain_outputs` is provided (i.e. this is being called in-process
    from --mode all), the already-curated datasets/task-subsets and the
    already-pretrained Model A are reused directly instead of reloading from
    disk -- this exactly matches a single continuous notebook run."""
    logger.info("=" * 70)
    logger.info("STAGE: FINE-TUNING (Model A finetune -> Model B pretrain -> Model B finetune)")
    logger.info("=" * 70)

    ckpt_dir = ensure_dir(Path(args.checkpoint_dir))
    results_dir = ensure_dir(Path(args.results_dir))
    plots_dir = ensure_dir(results_dir / PLOTS_SUBDIR)
    device = args.device

    if pretrain_outputs is not None:
        datasets = pretrain_outputs["datasets"]
        task_subsets = pretrain_outputs["task_subsets"]
        model_A_pre = pretrain_outputs["model_A_pre"]
        scaler_A, scaler_B = task_subsets["scaler_A"], task_subsets["scaler_B"]
    else:
        datasets = curate_datasets(args, logger)
        task_subsets = build_task_subsets(datasets, logger)
        scaler_A, scaler_B, loaded_mol_scaler = load_scalers_bundle(ckpt_dir / CKPT_SCALERS)
        globals()["mol_scaler"] = loaded_mol_scaler
        model_A_pre = MultitaskModel(num_targets=len(TARGETS_A)).to(device)
        pretrained_path = ckpt_dir / CKPT_TC_PRETRAINED
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"{pretrained_path} not found. Run `python gnn_pipeline.py --mode pretrain` first."
            )
        model_A_pre.load_state_dict(torch.load(pretrained_path, map_location=device))

    train_metrics, val_metrics = [], []
    batch_size = args.batch_size if args.batch_size else FINETUNE_BATCH_SIZE_DEFAULT
    max_epochs = args.epochs if args.epochs else FINETUNE_MAX_EPOCHS_DEFAULT

    # --- 1. Model A fine-tuning ---
    logger.info(f"Fine-tuning Tc model on {len(task_subsets['df_fine_A'])} molecules")
    model_A_ft, ft_train_losses_A, ft_val_losses_A = fine_tune_multitask_model(
        pretrained_model=model_A_pre,
        df_finetune=task_subsets["df_fine_A"],
        target_cols=TARGETS_A,
        target_scaler=scaler_A,
        device=device,
        logger=logger,
        batch_size=batch_size,
        max_epochs=max_epochs,
        freeze_encoder=False,
        lr=FINETUNE_LR,
        model_save_path=str(ckpt_dir / CKPT_TC_FINETUNED),
        target_weights=WEIGHTS_A,
        val_df=task_subsets["df_val_A"],
        stage_name="finetune_A_tc",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    plot_loss_curve(ft_train_losses_A, ft_val_losses_A, "Model A Fine-Tuning (Tc)", plots_dir / "finetune_A_loss.png")

    # --- 2. Model B "pretraining" bootstrapped from A's fine-tuned encoder ---
    logger.info(f"Pre-training Pc/omega model on {len(task_subsets['df_pre_B'])} molecules")
    model_B = MultitaskModel(num_targets=len(TARGETS_B)).to(device)
    model_B.encoder.load_state_dict(model_A_ft.encoder.state_dict())

    model_B_pre, pre_train_losses_B, pre_val_losses_B = fine_tune_multitask_model(
        pretrained_model=model_B,
        df_finetune=task_subsets["df_pre_B"],
        target_cols=TARGETS_B,
        target_scaler=scaler_B,
        device=device,
        logger=logger,
        batch_size=batch_size,
        max_epochs=max_epochs,
        freeze_encoder=True,
        lr=FINETUNE_LR,
        model_save_path=str(ckpt_dir / CKPT_PCOMEGA_PRETRAINED),
        target_weights=WEIGHTS_B,
        stage_name="pretrain_B_pcomega",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    plot_loss_curve(pre_train_losses_B, pre_val_losses_B, "Model B Pretraining (Pc/omega)", plots_dir / "pretrain_B_loss.png")

    # --- 3. Model B fine-tuning ---
    logger.info(f"Fine-tuning Pc/omega model on {len(task_subsets['df_fine_B'])} molecules")
    model_B_ft, ft_train_losses_B, ft_val_losses_B = fine_tune_multitask_model(
        pretrained_model=model_B_pre,
        df_finetune=task_subsets["df_fine_B"],
        target_cols=TARGETS_B,
        target_scaler=scaler_B,
        device=device,
        logger=logger,
        batch_size=batch_size,
        max_epochs=max_epochs,
        freeze_encoder=False,
        lr=FINETUNE_LR,
        model_save_path=str(ckpt_dir / CKPT_PCOMEGA_FINETUNED),
        target_weights=WEIGHTS_B,
        val_df=task_subsets["df_val_B"],
        stage_name="finetune_B_pcomega",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    plot_loss_curve(ft_train_losses_B, ft_val_losses_B, "Model B Fine-Tuning (Pc/omega)", plots_dir / "finetune_B_loss.png")

    write_metrics_csv(train_metrics, results_dir / "train_metrics.csv", append=True)
    write_metrics_csv(val_metrics, results_dir / "validation_metrics.csv", append=True)

    save_merged_bundle(model_A_ft, model_B_ft, scaler_A, scaler_B, mol_scaler, ckpt_dir / CKPT_MERGED)
    logger.info(f"Fine-tuning stage complete. Merged bundle: {ckpt_dir / CKPT_MERGED}")

    return {
        "model_A_ft": model_A_ft, "model_B_ft": model_B_ft,
        "scaler_A": scaler_A, "scaler_B": scaler_B,
        "datasets": datasets,
    }

# ======================================================================================
# SECTION 9: EVALUATION (test-set metrics + scatter plots)
# ======================================================================================

def run_test_stage(args, logger, finetune_outputs=None):
    """Faithful port of the notebook's Cell 7 "TESTING" block: evaluates Model
    A (Tc) and Model B (Pc/omega) on the held-out test split, computes
    R2/RMSE/MAE (overall + the Tc 300-500K band), saves scatter plots, saves
    the Tc-band diagnostics CSV, and writes results/test_metrics.csv."""
    logger.info("=" * 70)
    logger.info("STAGE: TESTING")
    logger.info("=" * 70)

    ckpt_dir = ensure_dir(Path(args.checkpoint_dir))
    results_dir = ensure_dir(Path(args.results_dir))
    plots_dir = ensure_dir(results_dir / PLOTS_SUBDIR)
    device = args.device

    if finetune_outputs is not None:
        model_A_ft = finetune_outputs["model_A_ft"]
        model_B_ft = finetune_outputs["model_B_ft"]
        scaler_A = finetune_outputs["scaler_A"]
        scaler_B = finetune_outputs["scaler_B"]
        df_test = finetune_outputs["datasets"]["df_test"]
    else:
        model_A_ft, model_B_ft, scaler_A, scaler_B, loaded_mol_scaler = load_merged_bundle(ckpt_dir / CKPT_MERGED, device)
        globals()["mol_scaler"] = loaded_mol_scaler
        datasets = curate_datasets(args, logger)
        df_test = datasets["df_test"]

    df_test_eval = df_test.reset_index(drop=True).copy()
    df_test_eval["row_id"] = np.arange(len(df_test_eval))

    logger.info("Evaluating Model A (Tc)...")
    df_res_A = predict_multitask(model_A_ft, df_test_eval, TARGETS_A, scaler_A, device)
    logger.info("Evaluating Model B (Pc, omega)...")
    df_res_B = predict_multitask(model_B_ft, df_test_eval, TARGETS_B, scaler_B, device)

    df_results_ft = df_test_eval[["row_id", "smiles", "Tc (K)", "Pc (bar)", "omega (-)"]].copy()
    df_results_ft = df_results_ft.rename(columns={
        "Tc (K)": "Tc_exp", "Pc (bar)": "Pc_exp", "omega (-)": "omega_exp",
    })
    df_results_ft = df_results_ft.merge(
        df_res_A[["row_id", "Tc (K)"]].rename(columns={"Tc (K)": "Tc_pred"}), on="row_id", how="left"
    )
    df_results_ft = df_results_ft.merge(
        df_res_B[["row_id", "Pc (bar)", "omega (-)"]].rename(columns={
            "Pc (bar)": "Pc_pred", "omega (-)": "omega_pred",
        }), on="row_id", how="left"
    )

    properties = [
        ("Tc (K)", "Tc_exp", "Tc_pred", "crimson", "Actual Tc (K)", "Predicted Tc (K)"),
        ("Pc (bar)", "Pc_exp", "Pc_pred", "steelblue", "Actual Pc (bar)", "Predicted Pc (bar)"),
        ("omega (-)", "omega_exp", "omega_pred", "forestgreen", "Actual omega (-)", "Predicted omega (-)"),
    ]

    test_metric_rows = []
    for prop_name, exp_col, pred_col, color, xlabel, ylabel in properties:
        mask = df_results_ft[exp_col].notna() & df_results_ft[pred_col].notna()
        n_valid = mask.sum()

        if n_valid == 0:
            logger.info(f"{prop_name}: No ground truth in test set to evaluate.")
            continue

        y_true = df_results_ft.loc[mask, exp_col].values
        y_pred = df_results_ft.loc[mask, pred_col].values

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        logger.info(f"--- METRICS: {prop_name} (n={n_valid}) --- R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        test_metric_rows.append({"property": prop_name, "band": "all", "n": int(n_valid), "r2": r2, "rmse": rmse, "mae": mae})

        if prop_name == "Tc (K)":
            band_mask = mask & df_results_ft[exp_col].between(300, 500, inclusive="both")
            n_band = band_mask.sum()
            if n_band > 0:
                y_true_band = df_results_ft.loc[band_mask, exp_col].values
                y_pred_band = df_results_ft.loc[band_mask, pred_col].values
                r2_band = r2_score(y_true_band, y_pred_band) if n_band >= 2 else np.nan
                rmse_band = np.sqrt(mean_squared_error(y_true_band, y_pred_band))
                mae_band = mean_absolute_error(y_true_band, y_pred_band)
                logger.info(f"Tc 300-500 K band (n={n_band}) | R2: {r2_band:.4f} | RMSE: {rmse_band:.4f} | MAE: {mae_band:.4f}")
                test_metric_rows.append({"property": prop_name, "band": "300-500K", "n": int(n_band),
                                          "r2": r2_band, "rmse": rmse_band, "mae": mae_band})

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color=color, edgecolors="k", linewidths=0.3)
        minv = min(y_true.min(), y_pred.min())
        maxv = max(y_true.max(), y_pred.max())
        margin = (maxv - minv) * 0.05
        plt.plot([minv - margin, maxv + margin], [minv - margin, maxv + margin], "--", color="black", linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Hybrid specialist {prop_name} Prediction (R2={r2:.3f})")
        plt.gca().set_aspect("equal")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = prop_name.split(" ")[0]
        fig.savefig(plots_dir / f"test_scatter_{safe_name}.png")
        plt.close(fig)

    df_results_ft["Tc_abs_error"] = (df_results_ft["Tc_pred"] - df_results_ft["Tc_exp"]).abs()
    df_results_ft["Tc_band_300_500"] = df_results_ft["Tc_exp"].between(300, 500, inclusive="both")
    df_results_ft[df_results_ft["Tc_band_300_500"]].sort_values("Tc_abs_error", ascending=False).to_csv(
        results_dir / "tc_band_diagnostics_hybrid_specialists.csv", index=False
    )
    logger.info(f"Saved Tc band diagnostics to {results_dir / 'tc_band_diagnostics_hybrid_specialists.csv'}")

    write_metrics_csv(test_metric_rows, results_dir / "test_metrics.csv", append=False)
    logger.info(f"Testing stage complete. Metrics: {results_dir / 'test_metrics.csv'}")

    return {"df_results": df_results_ft, "test_metrics": test_metric_rows}

# ======================================================================================
# SECTION 10: DEPLOYMENT PREDICTION
# ======================================================================================
# Faithful port of the notebook's deployment cell (V2-migrated to
# molecules_144_refrigerant.xlsx). The standardization logic is generic
# (auto-detects a smiles/SMILES/Smiles/Structure column, and optional
# ground-truth Tc/Pc/omega columns for error reporting) -- unchanged from the
# notebook, since it already worked with the new file with zero edits.

def _standardize_prediction_input(df, smiles_column_hint="smiles"):
    df = df.copy()

    if "smiles" not in df.columns:
        if smiles_column_hint in df.columns:
            df["smiles"] = df[smiles_column_hint]
        elif "SMILES" in df.columns:
            df["smiles"] = df["SMILES"]
        elif "Smiles" in df.columns:
            df["smiles"] = df["Smiles"]
        elif "Structure" in df.columns:
            df["smiles"] = df["Structure"]
        else:
            raise ValueError("Input file must contain one of: smiles, SMILES, Smiles, Structure.")

    if "Tc (K)" not in df.columns and "Tcrit_K" in df.columns:
        df["Tc (K)"] = pd.to_numeric(df["Tcrit_K"], errors="coerce")

    for col in ["Tc (K)", "Pc (bar)", "omega (-)"]:
        if col not in df.columns:
            df[col] = np.nan

    df["__row_id"] = np.arange(len(df))
    return df


def _add_occurrence_key(df, smiles_col="smiles"):
    df = df.copy()
    df["__smiles_occurrence"] = df.groupby(smiles_col, dropna=False).cumcount()
    return df


def _safe_predict_with_occurrence(model, df_model_input, target_cols, scaler, device, batch_size=64):
    pred = predict_multitask(model, df_model_input, target_cols, scaler, device, batch_size=batch_size)
    pred = _add_occurrence_key(pred, "smiles")
    return pred


def predict_deployment_file(args, logger, model_A, model_B, scaler_A, scaler_B,
                             input_path=None, output_path=None):
    """
    Predicts Tc, Pc, and omega for a deployment-style Excel file, e.g.:
      - molecules_144_refrigerant.xlsx: IUPAC_Name, Structure, COP, SMILES
        (no ground-truth Tc/Pc/omega columns -- pure inference).
      - McLinden-style: PubChem_Number, IUPAC_Name, Structure, smiles,
        ASHRAE_Designation, Tcrit_K (has ground truth -> error columns filled).
    """
    device = args.device
    if input_path is None:
        input_path = resolve_data_file(args.deploy_xlsx, Path(args.data_dir), logger)
    if output_path is None:
        output_path = Path(args.predictions_dir) / "refrigerant_predictions.csv"

    logger.info(f"--- Predicting deployment file: {input_path} ---")
    raw = pd.read_excel(input_path)
    df_infer = _standardize_prediction_input(raw)

    df_model_input = df_infer[df_infer["smiles"].notna()].copy()
    df_model_input = df_model_input[df_model_input["smiles"].astype(str).str.len() > 0].copy()
    df_model_input = _add_occurrence_key(df_model_input, "smiles")

    logger.info(f"Input rows: {len(df_infer)}")
    logger.info(f"Rows with usable SMILES: {len(df_model_input)}")

    logger.info("Predicting Tc with Model A...")
    pred_a = _safe_predict_with_occurrence(model_A, df_model_input, TARGETS_A, scaler_A, device, batch_size=50)
    pred_a = pred_a[["smiles", "__smiles_occurrence", "Tc (K)"]].rename(columns={"Tc (K)": "Tc_predicted (K)"})

    logger.info("Predicting Pc and omega with Model B...")
    pred_b = _safe_predict_with_occurrence(model_B, df_model_input, TARGETS_B, scaler_B, device, batch_size=50)
    pred_b = pred_b[["smiles", "__smiles_occurrence", "Pc (bar)", "omega (-)"]].rename(columns={
        "Pc (bar)": "Pc_predicted (bar)", "omega (-)": "omega_predicted (-)",
    })

    out = _add_occurrence_key(df_infer, "smiles")
    out = out.merge(pred_a, on=["smiles", "__smiles_occurrence"], how="left")
    out = out.merge(pred_b, on=["smiles", "__smiles_occurrence"], how="left")

    out["Tc_abs_error_K"] = np.where(
        out["Tc (K)"].notna() & out["Tc_predicted (K)"].notna(),
        (out["Tc_predicted (K)"] - out["Tc (K)"]).abs(), np.nan,
    )
    out["Tc_abs_percent_error"] = np.where(
        out["Tc (K)"].notna() & out["Tc_predicted (K)"].notna(),
        out["Tc_abs_error_K"] / out["Tc (K)"].abs() * 100, np.nan,
    )
    out["Pc_abs_error_bar"] = np.where(
        out["Pc (bar)"].notna() & out["Pc_predicted (bar)"].notna(),
        (out["Pc_predicted (bar)"] - out["Pc (bar)"]).abs(), np.nan,
    )
    out["Pc_abs_percent_error"] = np.where(
        out["Pc (bar)"].notna() & out["Pc_predicted (bar)"].notna(),
        out["Pc_abs_error_bar"] / out["Pc (bar)"].abs() * 100, np.nan,
    )

    out = out.drop(columns=["__row_id", "__smiles_occurrence"], errors="ignore")

    ensure_dir(Path(output_path).parent)
    out.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to: {output_path}")

    # Added visualization (not in the original notebook): distribution of
    # predicted Tc/Pc/omega across the deployment set, for quick QA.
    plots_dir = ensure_dir(Path(args.results_dir) / PLOTS_SUBDIR)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(
        axes,
        ["Tc_predicted (K)", "Pc_predicted (bar)", "omega_predicted (-)"],
        ["Predicted Tc (K)", "Predicted Pc (bar)", "Predicted omega (-)"],
    ):
        vals = out[col].dropna()
        ax.hist(vals, bins=20, color="steelblue", edgecolor="k", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
    fig.suptitle(f"Deployment prediction distribution (n={len(out)})")
    plt.tight_layout()
    fig.savefig(plots_dir / "deployment_prediction_distribution.png")
    plt.close(fig)

    return out


def run_predict_stage(args, logger):
    """Loads the fine-tuned merged bundle and scores the deployment file."""
    logger.info("=" * 70)
    logger.info("STAGE: DEPLOYMENT PREDICTION")
    logger.info("=" * 70)

    ckpt_dir = Path(args.checkpoint_dir)
    merged_path = ckpt_dir / CKPT_MERGED
    if not merged_path.exists():
        raise FileNotFoundError(
            f"{merged_path} not found. Run `python gnn_pipeline.py --mode finetune` first."
        )
    model_A_ft, model_B_ft, scaler_A, scaler_B, loaded_mol_scaler = load_merged_bundle(merged_path, args.device)
    globals()["mol_scaler"] = loaded_mol_scaler

    return predict_deployment_file(args, logger, model_A_ft, model_B_ft, scaler_A, scaler_B)

# ======================================================================================
# SECTION 11: SAVE / LOAD CHECKPOINTS
# ======================================================================================

def _state_dict_cpu(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _target_scaler_to_dict(scaler):
    return {
        "means": np.asarray(scaler.means, dtype=np.float32).tolist(),
        "stds": np.asarray(scaler.stds, dtype=np.float32).tolist(),
        "columns": list(getattr(scaler, "columns", [])),
    }


def _target_scaler_from_dict(d):
    s = TargetScaler()
    s.means = np.array(d["means"], dtype=np.float32)
    s.stds = np.array(d["stds"], dtype=np.float32)
    s.columns = d["columns"]
    return s


def _mol_scaler_to_dict(scaler):
    return {
        "means": np.asarray(scaler.means, dtype=np.float32).tolist(),
        "stds": np.asarray(scaler.stds, dtype=np.float32).tolist(),
    }


def _mol_scaler_from_dict(d):
    s = MolFeatureScaler()
    s.means = np.array(d["means"], dtype=np.float32)
    s.stds = np.array(d["stds"], dtype=np.float32)
    return s


def save_scalers_bundle(mol_scaler_obj, scaler_A, scaler_B, path: Path):
    """Internal cross-mode handoff artifact (checkpoints/scalers.pt): lets
    `--mode finetune` run as a separate process from `--mode pretrain` while
    reusing the EXACT SAME fitted mol_scaler/scaler_A/scaler_B the notebook's
    single continuous run would have used (mol_scaler and the target scalers
    are each fit exactly once in the notebook, then reused everywhere)."""
    ensure_dir(path.parent)
    torch.save({
        "mol_scaler": _mol_scaler_to_dict(mol_scaler_obj),
        "scaler_A": _target_scaler_to_dict(scaler_A),
        "scaler_B": _target_scaler_to_dict(scaler_B),
    }, path)


def load_scalers_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python gnn_pipeline.py --mode pretrain` first "
            f"(it fits and saves the shared scalers)."
        )
    d = torch.load(path, map_location="cpu", weights_only=False)
    scaler_A = _target_scaler_from_dict(d["scaler_A"])
    scaler_B = _target_scaler_from_dict(d["scaler_B"])
    mol_scaler_obj = _mol_scaler_from_dict(d["mol_scaler"])
    return scaler_A, scaler_B, mol_scaler_obj


def save_merged_bundle(model_A_ft, model_B_ft, scaler_A, scaler_B, mol_scaler_obj, path: Path):
    """Verbatim port of the notebook's "SAVE PORTABLE MERGED CRITICAL-PROPERTY
    MODEL" cell. Saves one portable bundle: both fine-tuned models, both
    target scalers, the molecular feature scaler, and the architecture
    config -- everything `--mode test` / `--mode predict` need to run
    standalone, without re-curating data or re-fitting scalers."""
    ensure_dir(path.parent)
    bundle = {
        "bundle_version": 1,
        "model_type": "split_dmpnn_fnn_critical_property_pipeline",
        "model_A_task": TARGETS_A,
        "model_B_tasks": TARGETS_B,
        "model_A_state_dict": _state_dict_cpu(model_A_ft),
        "model_B_state_dict": _state_dict_cpu(model_B_ft),
        "scaler_A": _target_scaler_to_dict(scaler_A),
        "scaler_B": _target_scaler_to_dict(scaler_B),
        "mol_scaler": _mol_scaler_to_dict(mol_scaler_obj),
        "model_config": {
            "dmpnn_depth": int(DMPNN_DEPTH),
            "hidden_size": int(HIDDEN_SIZE),
            "fnn_hidden": int(FNN_HIDDEN),
            "fnn_depth": int(FNN_DEPTH),
            "dropout": float(DROPOUT),
            "num_atom_features": int(NUM_ATOM_FEATURES),
            "num_bond_features": int(NUM_BOND_FEATURES),
            "num_mol_features": int(NUM_MOL_FEATURES),
        },
    }
    torch.save(bundle, path)


def load_merged_bundle(path: Path, device):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python gnn_pipeline.py --mode finetune` first."
        )
    bundle = torch.load(path, map_location=device, weights_only=False)

    model_A = MultitaskModel(num_targets=len(bundle["model_A_task"])).to(device)
    model_A.load_state_dict(bundle["model_A_state_dict"])

    model_B = MultitaskModel(num_targets=len(bundle["model_B_tasks"])).to(device)
    model_B.load_state_dict(bundle["model_B_state_dict"])

    scaler_A = _target_scaler_from_dict(bundle["scaler_A"])
    scaler_B = _target_scaler_from_dict(bundle["scaler_B"])
    mol_scaler_obj = _mol_scaler_from_dict(bundle["mol_scaler"])

    return model_A, model_B, scaler_A, scaler_B, mol_scaler_obj


# ======================================================================================
# SECTION 12: SAVE METRICS (CSV writers + plotting helpers)
# ======================================================================================

def write_metrics_csv(records, path: Path, append: bool):
    """Writes/append a list of dict records to a CSV under results/."""
    ensure_dir(path.parent)
    if not records:
        if not path.exists():
            pd.DataFrame(records).to_csv(path, index=False)
        return
    df = pd.DataFrame(records)
    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def plot_loss_curve(train_losses, val_losses, title, out_path: Path):
    """Added visualization (not present in the original notebook, which only
    printed loss values every 10 epochs) -- pure observability, no effect on
    training. Plots train/val loss vs. epoch for one training stage."""
    if not train_losses:
        return
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(7, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked Huber Loss (scaled targets)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ======================================================================================
# PRE-FLIGHT VALIDATION CHECKS (required before training/prediction)
# ======================================================================================
# These checks are additive (new in V2's production conversion) -- they do not
# change any preprocessing step, they only verify the inputs are sane BEFORE
# any model runs, and stop with a clear error otherwise.

REQUIRED_TABULAR_COLS = ["smiles", "Tc (K)", "Pc (bar)", "omega (-)"]


def validate_dataset_file(path: Path, required_cols, logger, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] Dataset file does not exist: {path}")

    df = read_csv_robust(path, logger=logger) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    logger.info(f"[{label}] Loaded {len(df)} rows from {path.name}")

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{label}] Missing required column(s) {missing_cols} in {path.name}. "
                          f"Found columns: {list(df.columns)}")
    logger.info(f"[{label}] Required columns present: {required_cols}")
    return df


def validate_smiles_column(df: pd.DataFrame, logger, label: str, min_valid_fraction: float = 0.5) -> None:
    smiles_col = "smiles" if "smiles" in df.columns else next(
        (c for c in ("SMILES", "Smiles", "Structure") if c in df.columns), None
    )
    if smiles_col is None:
        raise ValueError(f"[{label}] No SMILES-like column found.")

    n_total = len(df)
    n_valid = 0
    for s in df[smiles_col]:
        if isinstance(s, str) and s.strip() and Chem.MolFromSmiles(s) is not None:
            n_valid += 1
    frac = n_valid / n_total if n_total else 0.0
    logger.info(f"[{label}] Valid SMILES: {n_valid}/{n_total} ({frac:.1%})")

    if frac < min_valid_fraction:
        raise ValueError(
            f"[{label}] Only {frac:.1%} of rows have a valid, RDKit-parseable SMILES "
            f"(threshold {min_valid_fraction:.0%}). This usually means the wrong file was "
            f"pointed at, or the file is corrupted."
        )

    n_dup_raw = df[smiles_col].duplicated().sum()
    logger.info(f"[{label}] Duplicate raw SMILES strings: {n_dup_raw}")

    for col in ("Tc (K)", "Pc (bar)", "omega (-)"):
        if col in df.columns:
            n_missing = df[col].isna().sum()
            logger.info(f"[{label}] Missing target '{col}': {n_missing}/{n_total}")


def run_preflight_graph_checks(sample_smiles, logger, min_success_fraction: float = 0.3) -> None:
    """Smoke-tests the full descriptor + graph generation pipeline on a small
    sample before committing to a full run. A LOW (but nonzero) success rate
    is normal (a handful of exotic structures legitimately fail 3D embedding
    and are skipped downstream, exactly as in the notebook) -- a success rate
    of exactly zero means the pipeline itself is broken, which IS fatal."""
    if not sample_smiles:
        logger.warning("Graph-generation check: no sample SMILES provided, skipping.")
        return

    scratch_scaler = MolFeatureScaler()
    scratch_scaler.fit(sample_smiles, logger=None)

    global mol_scaler
    saved_means, saved_stds = mol_scaler.means, mol_scaler.stds
    mol_scaler.means, mol_scaler.stds = scratch_scaler.means, scratch_scaler.stds

    n_ok = 0
    try:
        for smi in sample_smiles:
            try:
                g = smiles_to_graph_multitask(smi, [0.0, 0.0, 0.0])
                if g is not None and g.x.shape[1] == NUM_ATOM_FEATURES and g.mol_attr.shape[1] == NUM_MOL_FEATURES:
                    n_ok += 1
            except Exception as e:
                logger.warning(f"Graph generation failed for {smi}: {e}")
    finally:
        mol_scaler.means, mol_scaler.stds = saved_means, saved_stds

    frac = n_ok / len(sample_smiles)
    logger.info(f"Graph generation smoke test: {n_ok}/{len(sample_smiles)} succeeded ({frac:.1%})")
    if n_ok == 0:
        raise RuntimeError(
            "Graph generation pipeline produced zero valid graphs on the sample. "
            "Descriptor/graph generation is broken -- check RDKit installation and inputs."
        )
    if frac < min_success_fraction:
        logger.warning(
            f"Graph generation success rate ({frac:.1%}) is low. This can be normal for "
            f"exotic structures (3D embedding failures are skipped downstream), but verify "
            f"the input file is correct."
        )


def run_preflight_checks(args, logger, need_training_data: bool, need_deploy_data: bool) -> None:
    logger.info("=" * 70)
    logger.info("PRE-FLIGHT VALIDATION")
    logger.info("=" * 70)
    data_dir = Path(args.data_dir)

    if need_training_data:
        for label, filename in [
            ("pretraining", args.pretrain_csv),
            ("fine-tuning", args.finetune_csv),
            ("mclinden (auxiliary, untouched)", args.mclinden_csv),
        ]:
            path = resolve_data_file(filename, data_dir, logger)
            df = validate_dataset_file(path, REQUIRED_TABULAR_COLS, logger, label)
            validate_smiles_column(df, logger, label)

        pretrain_path = resolve_data_file(args.pretrain_csv, data_dir, logger)
        pretrain_smiles = read_csv_robust(pretrain_path, logger=logger)["smiles"].dropna()
        sample = pretrain_smiles.sample(min(20, len(pretrain_smiles)), random_state=args.seed).tolist()
        run_preflight_graph_checks(sample, logger)

    if need_deploy_data:
        deploy_path = resolve_data_file(args.deploy_xlsx, data_dir, logger)
        if not deploy_path.exists():
            raise FileNotFoundError(f"[deployment] Dataset file does not exist: {deploy_path}")
        raw = pd.read_excel(deploy_path)
        logger.info(f"[deployment] Loaded {len(raw)} rows from {deploy_path.name}")
        std = _standardize_prediction_input(raw)
        validate_smiles_column(std, logger, "deployment")

    logger.info("Pre-flight validation PASSED.")

# ======================================================================================
# SECTION 13: MAIN()
# ======================================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production pipeline for the Hybrid Specialist DMPNN critical-property "
                     "model (Tc/Pc/omega), converted from GNN model.ipynb (V2 datasets). "
                     "Architecture, message passing, descriptors, loss, optimizer, scheduler "
                     "and hyperparameters are unchanged from the notebook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""Examples:
  python gnn_pipeline.py --mode pretrain
  python gnn_pipeline.py --mode finetune
  python gnn_pipeline.py --mode test
  python gnn_pipeline.py --mode predict
  python gnn_pipeline.py --mode all --device cuda --seed 42
""",
    )
    parser.add_argument("--mode", choices=["pretrain", "finetune", "test", "predict", "all"],
                         default="all", help="Which pipeline stage(s) to run.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                         help="Compute device. Default: auto-detect (cuda if available, else cpu), "
                              "exactly as the notebook did.")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Override batch size for ALL stages run in this invocation. "
                              "Default: None -> use the notebook's exact per-stage values "
                              "(50 for pretraining, 32 for fine-tuning).")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override max_epochs for ALL stages run in this invocation "
                              "(early stopping still applies). Default: None -> use the "
                              "notebook's exact per-stage values (100000 pretrain / 50000 finetune, "
                              "both effectively governed by early-stopping patience).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Global reproducibility seed (random/numpy/torch). Does not alter "
                              "the notebook's own hardcoded dataset-split random_state values.")

    parser.add_argument("--data-dir", default=".", help="Directory containing the finalized dataset files.")
    parser.add_argument("--pretrain-csv", default=PRETRAIN_CSV_DEFAULT)
    parser.add_argument("--finetune-csv", default=FINETUNE_CSV_DEFAULT)
    parser.add_argument("--mclinden-csv", default=MCLINDEN_CSV_DEFAULT,
                         help="Untouched auxiliary dataset (not part of the V2 migration).")
    parser.add_argument("--deploy-xlsx", default=DEPLOY_XLSX_DEFAULT)

    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR_DEFAULT)
    parser.add_argument("--predictions-dir", default=PREDICTIONS_DIR_DEFAULT)
    parser.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--logs-dir", default=LOGS_DIR_DEFAULT)

    parser.add_argument("--skip-checks", action="store_true",
                         help="Skip pre-flight validation checks (not recommended).")
    return parser


def resolve_device(requested: str, logger: logging.Logger) -> torch.device:
    if requested is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"--device not specified, auto-detected: {device}")
        return device
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but CUDA is not available; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_dir(Path(args.checkpoint_dir))
    ensure_dir(Path(args.predictions_dir))
    ensure_dir(Path(args.results_dir))
    ensure_dir(Path(args.logs_dir))

    logger = setup_logging(Path(args.logs_dir) / "training.log")
    logger.info("#" * 70)
    logger.info(f"gnn_pipeline.py starting -- mode={args.mode}")
    logger.info("#" * 70)

    set_seed(args.seed)
    args.device = resolve_device(args.device, logger)
    logger.info(f"Using device: {args.device}")

    t_start = time.time()
    try:
        if args.mode == "pretrain":
            if not args.skip_checks:
                run_preflight_checks(args, logger, need_training_data=True, need_deploy_data=False)
            run_pretrain_stage(args, logger)

        elif args.mode == "finetune":
            if not args.skip_checks:
                run_preflight_checks(args, logger, need_training_data=True, need_deploy_data=False)
            run_finetune_stage(args, logger)

        elif args.mode == "test":
            if not args.skip_checks:
                run_preflight_checks(args, logger, need_training_data=True, need_deploy_data=False)
            run_test_stage(args, logger)

        elif args.mode == "predict":
            if not args.skip_checks:
                run_preflight_checks(args, logger, need_training_data=False, need_deploy_data=True)
            run_predict_stage(args, logger)

        elif args.mode == "all":
            if not args.skip_checks:
                run_preflight_checks(args, logger, need_training_data=True, need_deploy_data=True)
            pretrain_outputs = run_pretrain_stage(args, logger)
            finetune_outputs = run_finetune_stage(args, logger, pretrain_outputs=pretrain_outputs)
            run_test_stage(args, logger, finetune_outputs=finetune_outputs)
            run_predict_stage(args, logger)

    except Exception:
        logger.exception(f"Pipeline FAILED in mode='{args.mode}'")
        raise

    elapsed = time.time() - t_start
    logger.info(f"gnn_pipeline.py finished mode='{args.mode}' in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
