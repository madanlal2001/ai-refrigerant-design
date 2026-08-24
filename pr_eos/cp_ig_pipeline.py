"""
=============================================================================
  Cp_ig PIPELINE FOR REFRIGERANT SCREENING
=============================================================================
  Purpose : Compute ideal-gas heat capacity polynomial coefficients [a,b,c,d]
            for all 142 McLinden refrigerants.

  Strategy:
    Group 1 — CoolProp   : Fit Cp0molar data → polynomial [a,b,c,d]
    Group 2 — Joback      : Compute from SMILES using Joback GC method
    Group 3 — Missing     : No SMILES and not in CoolProp → flagged as NaN

  Output  : Sorted CSV with columns:
            Name, ASHRAE, SMILES, Tc, Pc, omega, source, a, b, c, d

  Cp polynomial form (J/mol·K):
            Cp_ig(T) = a + b*T + c*T^2 + d*T^3     [T in Kelvin]

  Integration formulas (used later for H_ig and S_ig):
            H_ig(T2) - H_ig(T1) = a(T2-T1) + b/2*(T2^2-T1^2)
                                 + c/3*(T2^3-T1^3) + d/4*(T2^4-T1^4)

            S_ig(T2,P2) - S_ig(T1,P1) = a*ln(T2/T1) + b*(T2-T1)
                                        + c/2*(T2^2-T1^2) + d/3*(T2^3-T1^3)
                                        - R*ln(P2/P1)

  Author  : Refrigerant Screening Project
  Inputs  : refrigerants-gwp-Tc-Pc-El-Bo-contr.csv
  Outputs : refrigerants_cpig_coefficients.csv
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy  as np
import pandas as pd

from CoolProp.CoolProp import PropsSI   # for Group 1 fluids
from thermo import Joback               # for Group 2 fluids


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Temperature range for polynomial fitting (covers all 4 cycle state points)
T_MIN    = 250.0   # K  (well below T_evap = 283 K)
T_MAX    = 450.0   # K  (well above T_cond = 313 K, and compressor outlet ~370 K)
N_POINTS = 50      # number of points for CoolProp fitting


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — COOLPROP NAME MAPPING
# ─────────────────────────────────────────────────────────────────────────────
# CoolProp uses specific internal names that differ from ASHRAE designations.
# This dictionary maps our CSV names → CoolProp accepted names.

COOLPROP_NAME_MAP = {
    "R170"          : "Ethane",
    "R-1270"        : "Propylene",
    "R-290"         : "Propane",
    "R-C270"        : "CycloPropane",
    "R-600a"        : "Isobutane",
    "R-41"          : "R41",
    "R-32"          : "R32",
    "R-161"         : "R161",
    "R-152a"        : "R152A",
    "R-1234yf"      : "R1234yf",
    "R-1243zf"      : "R1243zf",
    "R-1234ze(E)"   : "R1234ze(E)",
    "R-1234ze(Z)"   : "R1234ze(Z)",
    "R-E143a"       : "RE143a",
    "R-744"         : "CO2",
    "R-717"         : "Ammonia",
    "R-227ea"       : "R227EA",
    "R-125"         : "R125",
    "R-143a"        : "R143a",
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_coolprop_name(csv_name: str) -> str | None:
    """
    Return the CoolProp-accepted name for a fluid, or None if not available.

    Tries:
      1. Direct lookup in our hand-mapped dictionary
      2. Strip 'R-' prefix variations
      3. Direct PropsSI call as a final check
    """
    # Try hand-mapped name first (most reliable)
    cp_name = COOLPROP_NAME_MAP.get(csv_name.strip())
    if cp_name:
        return cp_name

    # Try a few automatic variations
    candidates = [
        csv_name.strip(),
        csv_name.strip().replace("R-", "R"),
        csv_name.strip().replace("-", ""),
    ]
    for candidate in candidates:
        try:
            val = PropsSI("Cp0molar", "T", 300.0, "P", 101325.0, candidate)
            if val > 0:
                return candidate
        except Exception:
            continue

    return None   # not found in CoolProp


def fit_coolprop_polynomial(cp_name: str) -> tuple[float, float, float, float] | None:
    """
    Evaluate CoolProp Cp0molar at N_POINTS temperatures between T_MIN and T_MAX,
    then fit a cubic polynomial: Cp = a + b*T + c*T^2 + d*T^3

    Returns (a, b, c, d) or None if fitting fails.
    """
    T_array = np.linspace(T_MIN, T_MAX, N_POINTS)

    # Collect Cp values from CoolProp
    cp_array = []
    for T in T_array:
        try:
            cp = PropsSI("Cp0molar", "T", T, "P", 101325.0, cp_name)
            cp_array.append(cp)
        except Exception:
            return None   # if any point fails, skip this fluid

    cp_array = np.array(cp_array)

    # np.polyfit returns coefficients highest degree first: [d, c, b, a]
    poly_coeffs = np.polyfit(T_array, cp_array, deg=3)
    d, c, b, a = poly_coeffs

    return float(a), float(b), float(c), float(d)


def fit_joback_polynomial(smiles: str) -> tuple[float, float, float, float] | None:
    """
    Use Joback group contribution method on a SMILES string to get
    Cp_ig polynomial coefficients.

    Joback gives: Cp = a + b*T + c*T^2 + d*T^3   (J/mol·K, T in K)

    Returns (a, b, c, d) or None if Joback fails (unknown groups).
    """
    try:
        j = Joback(smiles)

        # Check if Joback successfully parsed all groups
        if not j.success:
            return None

        # Get coefficients directly from Joback
        coeffs = j.Cpig_coeffs(j.counts)   # returns [a, b, c, d]
        if coeffs is None or len(coeffs) != 4:
            return None

        a, b, c, d = coeffs
        return float(a), float(b), float(c), float(d)

    except Exception:
        return None


def evaluate_polynomial(a, b, c, d, T):
    """
    Evaluate Cp_ig at temperature T using the polynomial:
        Cp_ig(T) = a + b*T + c*T^2 + d*T^3
    T in Kelvin, returns Cp in J/mol·K
    """
    return a + b*T + c*T**2 + d*T**3


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(csv_path: str) -> pd.DataFrame:
    """
    Full pipeline:
      1. Load CSV
      2. Classify each fluid (CoolProp / Joback / Missing)
      3. Compute Cp_ig polynomial coefficients
      4. Sort by group (CoolProp first, Missing last)
      5. Return final DataFrame
    """

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("=" * 65)
    print("  LOADING DATA")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} fluids from: {csv_path}")

    # Keep only relevant columns, rename for clarity
    df = df.rename(columns={
        "Tc(K)"   : "Tc_K",
        "Pc(MPa)" : "Pc_MPa",
        "w"       : "omega",
    })

    # ── Step 2 & 3: Classify and compute Cp_ig ───────────────────────────────
    print("\n" + "=" * 65)
    print("  CLASSIFYING FLUIDS AND COMPUTING Cp_ig COEFFICIENTS")
    print("=" * 65)

    records = []   # will collect one dict per fluid

    count_cp  = 0   # CoolProp group counter
    count_jb  = 0   # Joback group counter
    count_mis = 0   # Missing group counter

    for idx, row in df.iterrows():

        name   = str(row["Name"]).strip()
        smiles = str(row.get("SMILES", "")).strip()
        ashrae = str(row.get("ASHRAE_Designation", "")).strip()

        # ── Try Group 1: CoolProp ────────────────────────────────────────────
        cp_name = get_coolprop_name(name)

        if cp_name is not None:
            coeffs = fit_coolprop_polynomial(cp_name)

            if coeffs is not None:
                a, b, c, d = coeffs
                source = "CoolProp"
                count_cp += 1
                print(f"  [CoolProp] {name:30s}  a={a:.4f}  b={b:.6f}")

                records.append({
                    "Name"         : name,
                    "ASHRAE"       : ashrae if ashrae != "nan" else "",
                    "SMILES"       : smiles if smiles != "nan" else "",
                    "Tc_K"         : row.get("Tc_K"),
                    "Pc_MPa"       : row.get("Pc_MPa"),
                    "omega"        : row.get("omega"),
                    "MW"           : row.get("MW"),
                    "source"       : source,
                    "cp_fluid_name": cp_name,
                    "a"            : a,
                    "b"            : b,
                    "c"            : c,
                    "d"            : d,
                    "sort_order"   : 1,   # CoolProp = first in sort
                })
                continue   # move to next fluid

        # ── Try Group 2: Joback (needs valid SMILES) ─────────────────────────
        if smiles and smiles != "nan":
            coeffs = fit_joback_polynomial(smiles)

            if coeffs is not None:
                a, b, c, d = coeffs
                source = "Joback"
                count_jb += 1
                print(f"  [Joback  ] {name:30s}  a={a:.4f}  b={b:.6f}")

                records.append({
                    "Name"         : name,
                    "ASHRAE"       : ashrae if ashrae != "nan" else "",
                    "SMILES"       : smiles,
                    "Tc_K"         : row.get("Tc_K"),
                    "Pc_MPa"       : row.get("Pc_MPa"),
                    "omega"        : row.get("omega"),
                    "MW"           : row.get("MW"),
                    "source"       : source,
                    "cp_fluid_name": "",
                    "a"            : a,
                    "b"            : b,
                    "c"            : c,
                    "d"            : d,
                    "sort_order"   : 2,   # Joback = second in sort
                })
                continue   # move to next fluid

        # ── Group 3: Missing — no CoolProp, no valid SMILES ──────────────────
        count_mis += 1
        print(f"  [MISSING ] {name:30s}  — no SMILES, not in CoolProp")

        records.append({
            "Name"         : name,
            "ASHRAE"       : ashrae if ashrae != "nan" else "",
            "SMILES"       : smiles if smiles != "nan" else "",
            "Tc_K"         : row.get("Tc_K"),
            "Pc_MPa"       : row.get("Pc_MPa"),
            "omega"        : row.get("omega"),
            "MW"           : row.get("MW"),
            "source"       : "Missing",
            "cp_fluid_name": "",
            "a"            : np.nan,
            "b"            : np.nan,
            "c"            : np.nan,
            "d"            : np.nan,
            "sort_order"   : 3,   # Missing = last in sort
        })

    # ── Step 4: Sort ─────────────────────────────────────────────────────────
    result_df = pd.DataFrame(records)
    result_df  = result_df.sort_values("sort_order").reset_index(drop=True)
    result_df  = result_df.drop(columns=["sort_order"])

    return result_df, count_cp, count_jb, count_mis


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SUMMARY & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df, count_cp, count_jb, count_mis):
    """Print a clean summary of results."""

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total fluids processed : {len(df)}")
    print(f"  ├── CoolProp           : {count_cp}")
    print(f"  ├── Joback (SMILES)    : {count_jb}")
    print(f"  └── Missing            : {count_mis}")

    # Quick validation — check Cp at T=300K for a few known fluids
    print("\n  VALIDATION — Cp_ig at T=300 K")
    print("  " + "-" * 50)

    validate_fluids = {
        "R-290" : 73.7,    # propane, known value J/mol·K
        "R-32"  : 37.3,    # R32
        "R-717" : 35.5,    # ammonia
    }

    for name, expected in validate_fluids.items():
        row = df[df["Name"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        if pd.isna(r["a"]):
            continue
        cp_calc = evaluate_polynomial(r["a"], r["b"], r["c"], r["d"], T=300.0)
        err_pct = abs(cp_calc - expected) / expected * 100
        src     = r["source"]
        print(f"  {name:12s} [{src:8s}]  "
              f"Calculated={cp_calc:7.2f}  Expected≈{expected:7.2f}  "
              f"Err={err_pct:.2f}%")

    print("=" * 65)


def print_group_tables(df):
    """Print the three groups as neat tables."""

    for source, label in [("CoolProp","GROUP 1 — COOLPROP"),
                           ("Joback",  "GROUP 2 — JOBACK (SMILES)"),
                           ("Missing", "GROUP 3 — MISSING")]:
        sub = df[df["source"] == source][["Name","ASHRAE","source","a","b","c","d"]]
        print(f"\n  {label}  ({len(sub)} fluids)")
        print("  " + "-" * 90)
        print(sub.to_string(index=True))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    INPUT_CSV  = "/mnt/user-data/uploads/refrigerants-gwp-Tc-Pc-El-Bo-contr.csv"
    OUTPUT_CSV = "/mnt/user-data/outputs/refrigerants_cpig_coefficients.csv"

    # Run the pipeline
    result_df, n_cp, n_jb, n_mis = run_pipeline(INPUT_CSV)

    # Print summary and group tables
    print_summary(result_df, n_cp, n_jb, n_mis)
    print_group_tables(result_df)

    # Save output
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Output saved → {OUTPUT_CSV}\n")
