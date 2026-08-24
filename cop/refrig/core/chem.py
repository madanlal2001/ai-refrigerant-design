"""Light chemistry helpers.

RDKit is optional. Without it the platform still runs -- SMILES is then treated
as an opaque identifier and molar mass is unavailable. With it, you get
canonicalization (so "C(F)(F)F" and "FC(F)F" are the same molecule in a run
record) and mass-basis results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class MolInfo:
    input_smiles: str
    canonical_smiles: str
    inchikey: Optional[str] = None
    formula: Optional[str] = None
    mw: Optional[float] = None      # kg/mol (SI)
    valid: bool = True
    message: str = ""


def analyze_smiles(smiles: str) -> MolInfo:
    """Canonicalize and describe a SMILES string. Never raises."""
    s = (smiles or "").strip()
    if not s:
        return MolInfo(s, s, valid=False, message="Empty SMILES.")

    if not rdkit_available():
        return MolInfo(s, s, valid=True,
                       message="RDKit not installed: SMILES not canonicalized, "
                               "molar mass unavailable.")

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return MolInfo(s, s, valid=False,
                       message=f"RDKit could not parse {s!r}.")
    return MolInfo(
        input_smiles=s,
        canonical_smiles=Chem.MolToSmiles(mol),
        inchikey=Chem.MolToInchiKey(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        mw=float(Descriptors.MolWt(mol)) / 1000.0,   # g/mol -> kg/mol
        valid=True,
    )
