"""
================================================================================
 Joback Group Contribution vs CoolProp Ideal Gas Cp — Validation Script
================================================================================
 Purpose:
     Benchmark the Joback method (predicted from SMILES) against CoolProp's
     high-accuracy ideal gas Cp (Cp0molar) at T = 300 K for all fluids in
     the refrigerant database CSV.

 Input CSV columns used:
     - Fluid  : CoolProp-compatible fluid name (e.g. 'R134a', 'Propane')
     - SMILES : Molecular SMILES string  (e.g. 'CC(F)(F)F')

 Output:
     - Cp_CoolProp_J_molK  : Ideal gas molar Cp from CoolProp [J/mol·K]
     - Cp_Joback_J_molK    : Ideal gas molar Cp from Joback method [J/mol·K]
     - Deviation_%         : |Joback - CoolProp| / CoolProp × 100

 Methods:
     Method A (ground truth): CoolProp 'Cp0molar' at P = 101325 Pa
     Method B (prediction)  : thermo.Joback group contribution from RDKit mol

 Sources:
     [Joback]  Joback & Reid (1987), Chem. Eng. Comm. 57, 233–243
     [PPO]     Poling, Prausnitz & O'Connell (2001), Table 2-2
     [CoolProp] Bell et al. (2014), Ind. Eng. Chem. Res.
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
from rdkit import Chem
from thermo import Joback   # thermo.Joback accepts an RDKit mol object


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Path to input CSV
#   Required columns: 'Fluid' (CoolProp name), 'SMILES' (molecular SMILES)
INPUT_CSV  = '/mnt/user-data/uploads/coolprop_fluids_Tc_Pc_omega_database.csv'

# Output CSV path
OUTPUT_CSV = '/mnt/user-data/outputs/Cp_Validation_Results.csv'

# Reference temperature for Cp evaluation [K]
T_REF = 300.0

# CoolProp evaluates ideal-gas Cp at any pressure; use 1 atm
P_REF = 101325.0   # Pa


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_cp_coolprop(fluid: str, T: float = T_REF, P: float = P_REF) -> float:
    """
    Return ideal-gas molar heat capacity Cp⁰ [J/mol·K] from CoolProp.

    Uses 'Cp0molar' — the ideal-gas limit of the molar heat capacity.
    This is independent of pressure but CoolProp still requires a P argument.

    Parameters
    ----------
    fluid : str   — CoolProp fluid name (e.g. 'R134a')
    T     : float — Temperature [K]
    P     : float — Pressure [Pa] (irrelevant for Cp0molar but required)

    Returns
    -------
    float — Cp0molar [J/mol·K], or np.nan on failure

    Source: CoolProp 'Cp0molar' parameter; Bell et al. (2014)
    """
    try:
        return CP.PropsSI('Cp0molar', 'T', T, 'P', P, fluid)
    except Exception:
        return np.nan


def get_cp_joback(smiles: str, T: float = T_REF) -> float:
    """
    Return ideal-gas molar heat capacity Cp⁰ [J/mol·K] using the
    Joback group contribution method.

    Steps:
      1. Parse SMILES → RDKit molecule
      2. Pass molecule to thermo.Joback to identify functional groups
      3. Evaluate Cpig(T) using Joback group increments

    Joback polynomial form (Poling et al. Table 2-2):
        Cp⁰(T) = ΣNₖ·[aₖ + bₖT + cₖT² + dₖT³] − 37.93

    Parameters
    ----------
    smiles : str   — Molecular SMILES string
    T      : float — Temperature [K]

    Returns
    -------
    float — Cpig [J/mol·K], or np.nan on failure

    Source: Joback & Reid (1987); Poling et al. (2001) Table 2-2
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan   # invalid SMILES

        j = Joback(mol)

        # Check that Joback found at least one group
        if j.status is not None and 'missing' in str(j.status).lower():
            return np.nan

        cp = j.Cpig(T)
        if cp is None or np.isnan(cp) or cp <= 0:
            return np.nan
        return cp

    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  Joback vs CoolProp — Ideal Gas Cp Validation")
    print(f"  T = {T_REF} K  |  Input: {INPUT_CSV}")
    print("=" * 68)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df)} fluids from CSV.")
    print(f"Columns: {df.columns.tolist()}\n")

    # ── Storage lists ─────────────────────────────────────────────────────────
    cp_coolprop_list = []   # Method A: CoolProp ground truth
    cp_joback_list   = []   # Method B: Joback prediction
    deviation_list   = []   # Absolute percentage error

    # ── Row-by-row evaluation ─────────────────────────────────────────────────
    for _, row in df.iterrows():
        fluid  = str(row['Fluid'])
        smiles = str(row['SMILES'])

        # ── Method A: CoolProp ideal gas Cp ──────────────────────────────────
        cp_cp = get_cp_coolprop(fluid)

        # ── Method B: Joback from SMILES ──────────────────────────────────────
        cp_jb = get_cp_joback(smiles)

        # ── Absolute percentage deviation ─────────────────────────────────────
        # Only compute if both values are valid
        if np.isnan(cp_cp) or np.isnan(cp_jb):
            dev = np.nan
        else:
            dev = abs(cp_jb - cp_cp) / cp_cp * 100.0

        cp_coolprop_list.append(cp_cp)
        cp_joback_list.append(cp_jb)
        deviation_list.append(dev)

        # Console progress
        if np.isnan(dev):
            status = f"SKIP  (CP={cp_cp:.2f}, JB={cp_jb:.2f})"
        else:
            status = f"OK    CP={cp_cp:.2f}  JB={cp_jb:.2f}  dev={dev:.2f}%"
        print(f"  {fluid:<30} {status}")

    # ── Append new columns to DataFrame ──────────────────────────────────────
    df['Cp_CoolProp_J_molK'] = cp_coolprop_list
    df['Cp_Joback_J_molK']   = cp_joback_list
    df['Deviation_%']        = deviation_list

    # ── Drop rows where deviation is NaN (either method failed) ──────────────
    df_valid = df.dropna(subset=['Deviation_%']).copy()

    # ── Summary statistics ────────────────────────────────────────────────────
    n_total   = len(df)
    n_success = len(df_valid)
    n_skipped = n_total - n_success
    mape      = df_valid['Deviation_%'].mean()
    median_e  = df_valid['Deviation_%'].median()
    max_e     = df_valid['Deviation_%'].max()
    max_fluid = df_valid.loc[df_valid['Deviation_%'].idxmax(), 'Fluid']
    min_e     = df_valid['Deviation_%'].min()
    min_fluid = df_valid.loc[df_valid['Deviation_%'].idxmin(), 'Fluid']

    print("\n" + "=" * 68)
    print("  RESULTS SUMMARY")
    print("=" * 68)
    print(f"  Total fluids in CSV       : {n_total}")
    print(f"  Successfully compared     : {n_success}")
    print(f"  Skipped (method failed)   : {n_skipped}")
    print(f"\n  MAPE (Mean Abs % Error)   : {mape:.3f}%")
    print(f"  Median absolute error     : {median_e:.3f}%")
    print(f"  Max error                 : {max_e:.3f}%  ({max_fluid})")
    print(f"  Min error                 : {min_e:.3f}%  ({min_fluid})")
    print(f"\n  Fluids with error < 5%    : {(df_valid['Deviation_%'] < 5).sum()}")
    print(f"  Fluids with error 5–15%   : {((df_valid['Deviation_%'] >= 5) & (df_valid['Deviation_%'] < 15)).sum()}")
    print(f"  Fluids with error > 15%   : {(df_valid['Deviation_%'] >= 15).sum()}")

    # ── Save output CSV ───────────────────────────────────────────────────────
    # Save the full DataFrame (including skipped rows with NaN) for transparency
    df_out = df.copy()
    df_out.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')
    print(f"\n  Results saved → {OUTPUT_CSV}")
    print("=" * 68)

    # ── Top 10 worst predictions (for review) ────────────────────────────────
    print("\n  Top 10 largest deviations:")
    top10 = df_valid.nlargest(10, 'Deviation_%')[
        ['Fluid', 'SMILES', 'Cp_CoolProp_J_molK', 'Cp_Joback_J_molK', 'Deviation_%']
    ]
    print(top10.to_string(index=False))

    return df_valid


if __name__ == '__main__':
    df_result = main()
