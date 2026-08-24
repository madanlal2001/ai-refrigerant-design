
"""
================================================================================
 COP Comparison: Experimental Tc  vs  Predicted Tc  vs  Literature COP
================================================================================
 Inputs  :
   File A (XLSX) — IUPAC_Name, Structure, SMILES, COP (literature),
                   Cp_ig polynomial coefficients a, b, c, d, e
   File B (CSV)  — IUPAC_Name, Tcrit_K (experimental Tc),
                   Tc_predicted (K), Pc_predicted (bar), omega_predicted (-)

 Cp_ig polynomial (4th order, 5 coefficients):
   Cp = a + b·T + c·T² + d·T³ + e·T⁴     [J/mol·K]
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from sklearn.metrics import r2_score
from scipy.optimize import brentq
warnings.filterwarnings("ignore")


# ================================================================================
#  CONFIG  ──
# ================================================================================

# ── Input files ──────────────────────────────────────────────────────────────────

FILE_XLSX     = "refrigerant_candidates_cp_61.xlsx"          # Cp coefficients + lit. COP
FILE_CSV      = "predicted_Tc_Pc_omega_61_candidates.csv"           # Tc/Pc/omega (GNN predictions)

# ── Output files ─────────────────────────────────────────────────────────────────
OUTPUT_CSV    = "COP_comparison_results.csv"
PLOT_TC_EXP   = "COP_scatter_Tc_exp.png"
PLOT_TC_PRED  = "COP_scatter_Tc_pred.png"

# ── Column names in XLSX (File A) ───────────────────────────────────────────────

COL_NAME_A    = "IUPAC_Name"                   # Molecule identifier (used to merge both files)
COL_STRUCT    = "Structure"                    # Structural formula string
COL_SMILES    = "SMILES"                       # SMILES string
COL_COP_LIT   = "COP"                          # Literature / experimental COP
COL_CP_A      = "a"                            # Cp polynomial coeff a  [J/mol·K]
COL_CP_B      = "b"                            # Cp polynomial coeff b  [J/mol·K²]
COL_CP_C      = "c"                            # Cp polynomial coeff c  [J/mol·K³]
COL_CP_D      = "d"                            # Cp polynomial coeff d  [J/mol·K⁴]
COL_CP_E      = "e"                            # Cp polynomial coeff e  [J/mol·K⁵]

# ── Column names in CSV (File B) ────────────────────────────────────────────────
COL_NAME_B    = "IUPAC_Name"                   # Must match COL_NAME_A for merging
COL_TC_EXP    = "Tcrit_K"                      # Experimental critical temperature [K]
COL_TC_PRED   = "Tc_predicted (K)"              # GNN-predicted Tc [K]
COL_PC_PRED   = "Pc_predicted (bar)"            # GNN-predicted Pc [bar]  ← always used
COL_OMEGA     = "omega_predicted (-)"           # GNN-predicted acentric factor  ← always used

# ── Optional exclusions ─────────────────────────────────────────────────────────

EXCLUDE_FLUIDS = []
# EXCLUDE_FLUIDS = [
#     "deuterium oxide",       # heavy water — H-bonding + quantum effects
#     "hydrogen chloride",     # strongly polar
#     "fluoromethane",         # polar
#     "sulfur hexafluoride",   # octahedral, highly non-spherical ]

# ── Cycle operating conditions ───────────────────────────────────────────────────

T_EVAP   = 283.15                              # [K] Evaporator saturation temperature   (10°C)
T_COND   = 313.15                              # [K] Condenser saturation temperature    (40°C)
DT_SH    =   0                              # [K] Superheat  → T1 = T_evap + DT_SH
DT_SC    =   0                              # [K] Subcooling → T3 = T_cond - DT_SC
ETA_C    =   1                              # [-] Isentropic compressor efficiency (1.0 = ideal)

# ── Reference state for ideal-gas H and S ────────────────────────────────────────
T_REF    = 233.15                              # [K] Reference temperature  (−40°C)
P_REF    = 101325                              # [Pa] Reference pressure     (1 atm)

# ── Numerical settings ────────────────────────────────────────────────────────────
P_SAT_TOL = 1e-8                               # P_sat flash convergence tolerance  |φ_L − φ_V|
NR_TOL    = 1e-6                               # Newton-Raphson convergence tolerance  |S − S_target|
MAX_ITER  = 150                                # Maximum iterations (both P_sat and N-R loops)

# ── Physical constant ─────────────────────────────────────────────────────────────
R = 8.314                                      # [J/mol·K] Universal gas constant


# ================================================================================
#  SECTION 1: IDEAL-GAS PROPERTIES
#  Cp_ig = a + b·T + c·T² + d·T³ + e·T⁴    [J/mol·K]
# ================================================================================

def cp_ig(T, a, b, c, d, e):
    """Ideal-gas heat capacity [J/mol·K]."""
    return a + b*T + c*T**2 + d*T**3 + e*T**4


def h_ig(T, T_ref, a, b, c, d, e):
    """
    Ideal-gas molar enthalpy relative to T_ref [J/mol].
    H_ig = ∫(T_ref → T) Cp dT
    """
    def F(t):
        return (a*t
                + (b/2)*t**2
                + (c/3)*t**3
                + (d/4)*t**4
                + (e/5)*t**5)
    return F(T) - F(T_ref)


def s_ig(T, P, T_ref, P_ref, a, b, c, d, e):
    """
    Ideal-gas molar entropy relative to (T_ref, P_ref) [J/mol·K].
    S_ig = ∫(T_ref → T) (Cp/T) dT  − R·ln(P/P_ref)
    """
    def G(t):
        return (a*np.log(t)
                + b*t
                + (c/2)*t**2
                + (d/3)*t**3
                + (e/4)*t**4)
    return G(T) - G(T_ref) - R * np.log(P / P_ref)


# ================================================================================
#  SECTION 2: PENG-ROBINSON EOS PARAMETERS
#  P = RT/(V−b) − a(T)/[V(V+b) + b(V−b)]
# ================================================================================

def pr_params(Tc, Pc_Pa, omega):
    """
    Pure-component PR EOS parameters.
    Returns: a_c [J·m³/mol²],  b [m³/mol],  kappa [-]
    """
    a_c   = 0.45724 * R**2 * Tc**2 / Pc_Pa
    b_eos = 0.07780 * R * Tc / Pc_Pa
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    return a_c, b_eos, kappa


def alpha(T, Tc, kappa):
    """α(T) = [1 + κ·(1 − √(T/Tc))]²"""
    return (1.0 + kappa * (1.0 - np.sqrt(T / Tc)))**2


def a_of_T(T, Tc, Pc_Pa, omega):
    """a(T) = a_c · α(T)  [J·m³/mol²]"""
    a_c, _, kappa = pr_params(Tc, Pc_Pa, omega)
    return a_c * alpha(T, Tc, kappa)


def dadt(T, Tc, Pc_Pa, omega):
    """
    Analytical da/dT  [J·m³/(mol²·K)]
    da/dT = −a_c · κ · √(α / (T·Tc))
    """
    a_c, _, kappa = pr_params(Tc, Pc_Pa, omega)
    alp = alpha(T, Tc, kappa)
    return -a_c * kappa * np.sqrt(alp / (T * Tc))


# ================================================================================
#  SECTION 3: CUBIC Z-FACTOR SOLVER
#  Z³ − (1−B)Z² + (A−3B²−2B)Z − (AB−B²−B³) = 0
# ================================================================================

def AB(T, P, Tc, Pc_Pa, omega):
    """Dimensionless A and B for PR cubic equation."""
    a = a_of_T(T, Tc, Pc_Pa, omega)
    _, b, _ = pr_params(Tc, Pc_Pa, omega)
    A = a * P / (R**2 * T**2)
    B = b * P / (R * T)
    return A, B


def solve_Z(A, B):
    """
    Solve PR cubic for Z.
    Returns Z_V (vapor, largest root) and Z_L (liquid, smallest root > B).
    Raises ValueError if no valid roots found.
    """
    coeffs = [1.0, -(1.0 - B), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)]
    roots  = np.roots(coeffs)

    # Keep only real roots (discard imaginary parts)
    real_roots = np.array([r.real for r in roots if abs(r.imag) < 1e-8])

    # Physical constraint: Z > B  (molar volume must exceed co-volume b)
    valid = real_roots[real_roots > B]

    if len(valid) == 0:
        raise ValueError(f"No valid Z roots: A={A:.4f}, B={B:.4f}")

    return max(valid), min(valid)    # Z_V, Z_L


# ================================================================================
#  SECTION 4: FUGACITY COEFFICIENT
#  Used for vapor-liquid equilibrium (P_sat iteration)
# ================================================================================

def ln_phi(Z, A, B):
    """
    ln(φ) from PR EOS:
    ln(φ) = (Z−1) − ln(Z−B) − A/(2√2·B) · ln[(Z+(1+√2)B)/(Z+(1−√2)B)]
    """
    s2 = np.sqrt(2.0)
    arg = (Z + (1 + s2)*B) / (Z + (1 - s2)*B)
    if arg <= 0:
        raise ValueError(f"Negative argument in ln_phi: Z={Z:.6f}, B={B:.6f}")
    return (Z - 1) - np.log(Z - B) - A / (2*s2*B) * np.log(arg)


# ================================================================================
#  SECTION 5: SATURATION PRESSURE
#  Wilson equation initial guess → iterate until φ_L = φ_V
# ================================================================================

def psat(T, Tc, Pc_Pa, omega):
    """
    Saturation pressure [Pa] via fugacity-equality iteration.
    Convergence: |ln(φ_L) − ln(φ_V)| < P_SAT_TOL
    """
    # Wilson guess
    P = Pc_Pa * np.exp(5.373 * (1 + omega) * (1 - Tc / T))

    for _ in range(MAX_ITER):
        A_, B_ = AB(T, P, Tc, Pc_Pa, omega)
        Z_V, Z_L = solve_Z(A_, B_)

        lnphi_V = ln_phi(Z_V, A_, B_)
        lnphi_L = ln_phi(Z_L, A_, B_)
        diff = lnphi_L - lnphi_V

        if abs(diff) < P_SAT_TOL:
            return P, Z_V, Z_L

        P *= np.exp(diff)    # Update pressure to drive fugacities equal

    raise RuntimeError(f"P_sat did not converge at T={T:.2f} K")


# ================================================================================
#  SECTION 6: DEPARTURE FUNCTIONS  (H_dep, S_dep)
#  Correct thermodynamic properties for real-gas behavior.
# ================================================================================

def _log_term(Z, B):
    """
    ln[(Z + (1+√2)·B) / (Z + (1−√2)·B)]
    Shared by both H_dep and S_dep.
    """
    s2  = np.sqrt(2.0)
    num = Z + (1 + s2)*B
    den = Z + (1 - s2)*B
    if den <= 0 or num/den <= 0:
        raise ValueError(f"Invalid log term: Z={Z:.6f}, B={B:.6f}")
    return np.log(num / den)


def h_dep(T, Z, B, Tc, Pc_Pa, omega):
    """
    Enthalpy departure [J/mol]:
    H_dep = RT(Z−1) + [T·(da/dT) − a(T)] / (2√2·b) · log_term
    """
    _, b, _  = pr_params(Tc, Pc_Pa, omega)
    a        = a_of_T(T, Tc, Pc_Pa, omega)
    da       = dadt(T, Tc, Pc_Pa, omega)
    prefac   = (T * da - a) / (2 * np.sqrt(2) * b)
    return R * T * (Z - 1) + prefac * _log_term(Z, B)


def s_dep(T, Z, B, Tc, Pc_Pa, omega):
    """
    Entropy departure [J/mol·K]:
    S_dep = R·ln(Z−B) + (da/dT) / (2√2·b) · log_term
    """
    _, b, _ = pr_params(Tc, Pc_Pa, omega)
    da      = dadt(T, Tc, Pc_Pa, omega)
    prefac  = da / (2 * np.sqrt(2) * b)
    return R * np.log(Z - B) + prefac * _log_term(Z, B)


# ================================================================================
#  SECTION 7: TOTAL ENTHALPY AND ENTROPY
#  H_total = H_ig + H_dep
#  S_total = S_ig + S_dep
# ================================================================================

def calc_H(T, P, Z, B, Tc, Pc_Pa, omega, cp_coeffs):
    """Total molar enthalpy [J/mol]."""
    a, b, c, d, e = cp_coeffs
    H_ideal  = h_ig(T, T_REF, a, b, c, d, e)
    H_depart = h_dep(T, Z, B, Tc, Pc_Pa, omega)
    return H_ideal + H_depart


def calc_S(T, P, Z, B, Tc, Pc_Pa, omega, cp_coeffs):
    """Total molar entropy [J/mol·K]."""
    a, b, c, d, e = cp_coeffs
    S_ideal  = s_ig(T, P, T_REF, P_REF, a, b, c, d, e)
    S_depart = s_dep(T, Z, B, Tc, Pc_Pa, omega)
    return S_ideal + S_depart


# ================================================================================
#  SECTION 8: ISENTROPIC COMPRESSOR OUTLET TEMPERATURE (T2s)
#  Newton-Raphson: find T2s at P_cond such that S(T2s, P_cond) = S1
# ================================================================================

def find_T2s(S_target, P_cond, Tc, Pc_Pa, omega, cp_coeffs):
    """
    Returns: T2s [K], Z at the outlet, H2s [J/mol]
    """
    # Saturated liquid & vapour properties at the condenser pressure (T = T_cond).
    A_sat, B_sat   = AB(T_COND, P_cond, Tc, Pc_Pa, omega)
    Z_V, Z_L       = solve_Z(A_sat, B_sat)
    S_vap          = calc_S(T_COND, P_cond, Z_V, B_sat, Tc, Pc_Pa, omega, cp_coeffs)
    H_vap          = calc_H(T_COND, P_cond, Z_V, B_sat, Tc, Pc_Pa, omega, cp_coeffs)

    # Regime 2 — isentrope lands in/at the dome → wet (two-phase) outlet.
    if S_target <= S_vap + 1e-9:
        S_liq = calc_S(T_COND, P_cond, Z_L, B_sat, Tc, Pc_Pa, omega, cp_coeffs)
        H_liq = calc_H(T_COND, P_cond, Z_L, B_sat, Tc, Pc_Pa, omega, cp_coeffs)
        denom = (S_vap - S_liq)
        x = 1.0 if abs(denom) < 1e-12 else (S_target - S_liq) / denom
        x = min(max(x, 0.0), 1.0)                      # clip to physical [0, 1]
        H2s = H_liq + x * (H_vap - H_liq)
        return T_COND, Z_V, H2s

    # Regime 1 — superheated outlet: bracket and solve with Brent.
    def resid(T):
        A_, B_ = AB(T, P_cond, Tc, Pc_Pa, omega)
        ZV_, _ = solve_Z(A_, B_)
        return calc_S(T, P_cond, ZV_, B_, Tc, Pc_Pa, omega, cp_coeffs) - S_target

    T_lo, T_hi = T_COND, T_COND + 10.0
    f_lo = S_vap - S_target                            # < 0 in this regime
    f_hi = resid(T_hi)
    n = 0
    while f_lo * f_hi > 0.0 and n < 300:
        T_hi += 10.0
        if T_hi > 5.0 * Tc:
            break
        try:
            f_hi = resid(T_hi)
        except Exception:
            T_hi -= 10.0
            break
        n += 1

    if f_lo * f_hi > 0.0:
        raise RuntimeError(f"T2s: could not bracket isentropic root up to {T_hi:.1f} K")

    T2s = brentq(resid, T_lo, T_hi, xtol=1e-7, rtol=1e-12, maxiter=300)
    A2s, B2s = AB(T2s, P_cond, Tc, Pc_Pa, omega)
    Z_2s, _  = solve_Z(A2s, B2s)
    H2s      = calc_H(T2s, P_cond, Z_2s, B2s, Tc, Pc_Pa, omega, cp_coeffs)
    return T2s, Z_2s, H2s


# ================================================================================
#  SECTION 9: FULL CYCLE COP FOR ONE MOLECULE
#  Standard vapor-compression refrigeration cycle:
#    State 1  → Superheated vapor at (T1, P_evap)        compressor inlet
#    State 2s → Isentropic compressor outlet at (T2s, P_cond)
#    State 3  → Subcooled liquid at (T3, P_cond)          condenser outlet
#    State 4  → Isenthalpic expansion  H4 = H3            evaporator inlet
# ================================================================================

def calc_COP(Tc, Pc_bar, omega, cp_coeffs):
    """
    Calculate COP for a single refrigerant.

    """
    Pc_Pa = Pc_bar * 1e5     # bar → Pa

    T1 = T_EVAP + DT_SH      # Compressor inlet temperature (superheated)
    T3 = T_COND - DT_SC      # Condenser exit temperature   (subcooled)

    # ── Guard: subcritical cycle check ─────────────────────────────────────────
    if T_EVAP >= Tc:
        raise ValueError(f"T_evap ({T_EVAP:.2f} K) ≥ Tc ({Tc:.2f} K) — transcritical.")
    if T_COND >= Tc:
        raise ValueError(f"T_cond ({T_COND:.2f} K) ≥ Tc ({Tc:.2f} K) — transcritical.")

    # ── Saturation pressures ────────────────────────────────────────────────────
    P_evap, _, _ = psat(T_EVAP, Tc, Pc_Pa, omega)
    P_cond, _, _ = psat(T_COND, Tc, Pc_Pa, omega)

    # ── State 1: Superheated vapor at (T1, P_evap) ─────────────────────────────
    A1, B1    = AB(T1, P_evap, Tc, Pc_Pa, omega)
    Z_V1, _   = solve_Z(A1, B1)
    H1        = calc_H(T1, P_evap, Z_V1, B1, Tc, Pc_Pa, omega, cp_coeffs)
    S1        = calc_S(T1, P_evap, Z_V1, B1, Tc, Pc_Pa, omega, cp_coeffs)

    # ── State 3: Subcooled liquid at (T3, P_cond) ──────────────────────────────
    A3, B3    = AB(T3, P_cond, Tc, Pc_Pa, omega)
    _, Z_L3   = solve_Z(A3, B3)
    H3        = calc_H(T3, P_cond, Z_L3, B3, Tc, Pc_Pa, omega, cp_coeffs)

    # ── State 4: Isenthalpic expansion (throttle valve) ────────────────────────
    H4 = H3    # Enthalpy conserved across throttle

    # ── State 2s: Isentropic compressor outlet ──────────────────────────────────
    # find_T2s now returns H2s directly (handles superheated AND wet/two-phase outlets)
    T2s, Z_2s, H2s = find_T2s(S1, P_cond, Tc, Pc_Pa, omega, cp_coeffs)

    # ── Actual compressor outlet (isentropic efficiency) ────────────────────────
    H2 = H1 + (H2s - H1) / ETA_C    # η=1 → H2 = H2s

    # ── Energy balance ──────────────────────────────────────────────────────────
    Q_evap = H1 - H4
    W_comp = H2 - H1

    if W_comp <= 0:
        raise ValueError(f"Non-positive compressor work W_comp={W_comp:.2f} J/mol")
    if Q_evap <= 0:
        raise ValueError(f"Non-positive cooling effect Q_evap={Q_evap:.2f} J/mol")

    return Q_evap / W_comp


# ================================================================================
#  SECTION 10: BATCH RUNNER
#  Runs COP twice per molecule: once with Tc_exp, once with Tc_pred.
#  Pc_predicted and omega_predicted are used for BOTH runs.
# ================================================================================

def run_batch(df):
    """
    Process all rows in the merged DataFrame.
    Returns list of result dicts (one per molecule).
    """
    results = []

    for idx, row in df.iterrows():
        name    = str(row.get("IUPAC_Name",  "Unknown")).strip()
        struct  = str(row.get("Structure",   ""))
        smiles  = str(row.get("SMILES",      ""))
        cop_lit = row.get("COP_lit")          # Literature COP (may be NaN)

        # Cp polynomial coefficients (always available after cleaning)
        cp = (float(row["a"]), float(row["b"]),
              float(row["c"]), float(row["d"]), float(row["e"]))

        Pc_bar = row.get("Pc_predicted (bar)")
        omega  = row.get("omega_predicted (-)")
        Tc_exp = row.get("Tcrit_K")
        Tc_pred= row.get("Tc_predicted (K)")

        result = {
            "IUPAC_Name"  : name,
            "Structure"   : struct,
            "SMILES"      : smiles,
            "COP"         : cop_lit,             # Literature COP
            "COP_Tc_exp"  : np.nan,
            "COP_Tc_pred" : np.nan,
            "error_Tc_exp": np.nan,
            "error_Tc_pred": np.nan,
            "note"        : ""
        }

        notes = []

        # ── COP with Tc_exp ─────────────────────────────────────────────────────
        if pd.notna(Tc_exp) and pd.notna(Pc_bar) and pd.notna(omega):
            try:
                cop_exp = calc_COP(float(Tc_exp), float(Pc_bar), float(omega), cp)
                result["COP_Tc_exp"] = round(cop_exp, 4)
                if pd.notna(cop_lit):
                    result["error_Tc_exp"] = round(abs(cop_exp - cop_lit) / cop_lit * 100, 3)
            except Exception as ex:
                notes.append(f"Tc_exp failed: {ex}")
        else:
            notes.append("Tc_exp/Pc/omega missing")

        # ── COP with Tc_pred ────────────────────────────────────────────────────
        if pd.notna(Tc_pred) and pd.notna(Pc_bar) and pd.notna(omega):
            try:
                cop_pred = calc_COP(float(Tc_pred), float(Pc_bar), float(omega), cp)
                result["COP_Tc_pred"] = round(cop_pred, 4)
                if pd.notna(cop_lit):
                    result["error_Tc_pred"] = round(abs(cop_pred - cop_lit) / cop_lit * 100, 3)
            except Exception as ex:
                notes.append(f"Tc_pred failed: {ex}")
        else:
            notes.append("Tc_pred/Pc/omega missing")

        result["note"] = "; ".join(notes) if notes else "OK"

        # ── Console progress ────────────────────────────────────────────────────
        tag_exp  = f"{result['COP_Tc_exp']:.4f}" if pd.notna(result["COP_Tc_exp"])  else "  N/A  "
        tag_pred = f"{result['COP_Tc_pred']:.4f}" if pd.notna(result["COP_Tc_pred"]) else "  N/A  "
        tag_lit  = f"{cop_lit:.2f}" if pd.notna(cop_lit) else " N/A"
        print(f"  [{idx:3d}]  {name[:42]:<42s}  "
              f"COP_lit={tag_lit}  Tc_exp={tag_exp}  Tc_pred={tag_pred}")

        results.append(result)

    return results


# ================================================================================
#  SECTION 11: SCATTER PLOTS WITH R²
#  One plot per Tc source (exp and pred) — comparing calculated vs literature COP
# ================================================================================

def scatter_plot(x_vals, y_vals, labels, xlabel, ylabel, title, filename, color):
    """
    Scatter plot of x (literature COP) vs y (calculated COP).

    """
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    # Keep only paired (non-NaN) values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    lab  = np.array(labels)[mask]

    if len(x) < 2:
        print(f"  [WARNING] Not enough data points for plot: {filename}")
        return None

    r2 = r2_score(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, color=color, edgecolors="white", linewidths=0.5,
               s=60, alpha=0.85, zorder=3, label="Molecules")

    # 1:1 perfect-prediction line
    lo = min(x.min(), y.min()) - 0.3
    hi = max(x.max(), y.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, zorder=2, label="1:1 line")

    # R² text box
    ax.text(0.05, 0.93,
            f"$R^2$ = {r2:.4f}\n$n$ = {len(x)}",
            transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {filename}  (R² = {r2:.4f},  n = {len(x)})")
    return r2


# ================================================================================
#  SECTION 12: MAIN ENTRY POINT
# ================================================================================

def main():
    print("=" * 75)
    print("  COP Comparison: Tc_experimental vs Tc_predicted  (PR EOS)")
    print("=" * 75)
    print(f"\n  Cycle conditions:")
    print(f"    T_evap = {T_EVAP} K ({T_EVAP-273.15:.1f}°C)   "
          f"T_cond = {T_COND} K ({T_COND-273.15:.1f}°C)")
    print(f"    Superheat ΔT_SH = {DT_SH} K  →  T1 = {T_EVAP+DT_SH:.2f} K")
    print(f"    Subcool   ΔT_SC = {DT_SC} K  →  T3 = {T_COND-DT_SC:.2f} K")
    print(f"    Compressor efficiency η = {ETA_C}")
    print(f"    Reference: T_ref={T_REF} K,  P_ref={P_REF} Pa")
    print(f"\n  NOTE: Pc_predicted and omega_predicted are used for BOTH COP runs.")
    print(f"        Only Tc switches between experimental and predicted.\n")

    # ── Load files ─────────────────────────────────────────────────────────────
    print(f"  Loading: {FILE_XLSX}")
    df_a = pd.read_excel(FILE_XLSX)
    print(f"  Loading: {FILE_CSV}")
    df_b = pd.read_csv(FILE_CSV)

    # Drop fully-empty trailing rows (XLSX often reports ~1e6 blank rows)
    df_a = df_a.dropna(subset=[COL_NAME_A]).copy()

    # Normalise the merge key (strip stray whitespace) in BOTH files
    df_a[COL_NAME_A] = df_a[COL_NAME_A].astype(str).str.strip()
    df_b[COL_NAME_B] = df_b[COL_NAME_B].astype(str).str.strip()

    # ── Clean a known bad duplicate ────────────────────────────────────────────
    # '1,1,2,2-tetrafluoroethane' (R-134) appears twice in the CSV: one row carries
    # an erroneous Tcrit_K = 138 K (real/literature value ≈ 391.8 K, per McLinden).
    # Drop the bad-data copy so the valid row survives de-duplication below.
    bad = (df_b[COL_NAME_B] == "1,1,2,2-tetrafluoroethane") & (df_b[COL_TC_EXP] < 200)
    if bad.any():
        print(f"  Dropping {int(bad.sum())} erroneous tetrafluoroethane row(s) "
              f"(Tcrit < 200 K data error)")
        df_b = df_b[~bad].copy()

    print(f"\n  Loaded: {len(df_a)} rows from XLSX,  {len(df_b)} rows from CSV")

    # ── Optional: drop PR-EOS-incompatible fluids (by IUPAC_Name) ──────────────
    if EXCLUDE_FLUIDS:
        ex = {s.strip().lower() for s in EXCLUDE_FLUIDS}
        before = len(df_a)
        df_a = df_a[~df_a[COL_NAME_A].str.lower().isin(ex)].copy()
        print(f"  Excluded {before - len(df_a)} fluid(s) listed in EXCLUDE_FLUIDS")

    # ── Rename literature COP column before merge (avoid collision) ────────────
    df_a = df_a.rename(columns={COL_COP_LIT: "COP_lit",
                                 COL_SMILES:  "SMILES"})

    # ── Merge on IUPAC_Name ────────────────────────────────────────────────────
    df_b_keep = df_b[[COL_NAME_B, "Structure", COL_TC_EXP,
                       COL_TC_PRED, COL_PC_PRED, COL_OMEGA]].copy()

    merged = pd.merge(df_a, df_b_keep, left_on=COL_NAME_A,
                      right_on=COL_NAME_B, how="left", suffixes=("_xlsx", "_csv"))

    # Handle duplicate Structure columns from merge
    if "Structure_xlsx" in merged.columns:
        merged["Structure"] = merged["Structure_xlsx"].fillna(merged.get("Structure_csv", ""))
        merged.drop(columns=["Structure_xlsx", "Structure_csv"], errors="ignore", inplace=True)

    # Drop duplicates (same IUPAC_Name appearing twice in either file)
    merged = merged.drop_duplicates(subset=COL_NAME_A, keep="first").reset_index(drop=True)

    # Drop rows where Cp coefficients are missing (cannot calculate COP at all)
    cp_cols = [COL_CP_A, COL_CP_B, COL_CP_C, COL_CP_D, COL_CP_E]
    n_before = len(merged)
    merged   = merged.dropna(subset=cp_cols).reset_index(drop=True)
    n_after  = len(merged)
    print(f"  After dedup + Cp filter: {n_after} rows  "
          f"({n_before - n_after} dropped for missing Cp coefficients)")

    print(f"\n  Running COP calculations...\n" + "-"*75)

    # ── Batch calculation ───────────────────────────────────────────────────────
    results = run_batch(merged)

    # ── Build output DataFrame ──────────────────────────────────────────────────
    df_out = pd.DataFrame(results)[[
        "IUPAC_Name", "Structure", "SMILES",
        "COP", "COP_Tc_exp", "COP_Tc_pred",
        "error_Tc_exp", "error_Tc_pred", "note"
    ]]

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'─'*75}")
    print(f"  Results saved → {OUTPUT_CSV}")

    # ── Summary statistics ──────────────────────────────────────────────────────
    ok_exp  = df_out["COP_Tc_exp"].notna().sum()
    ok_pred = df_out["COP_Tc_pred"].notna().sum()
    both    = df_out[df_out["COP_Tc_exp"].notna() & df_out["COP_Tc_pred"].notna()]
    lit_ok  = df_out["COP"].notna().sum()
    err_exp  = df_out["error_Tc_exp"].dropna()
    err_pred = df_out["error_Tc_pred"].dropna()

    print(f"\n  SUMMARY")
    print(f"  {'─'*40}")
    print(f"  Total molecules        : {len(df_out)}")
    print(f"  Literature COP avail.  : {lit_ok}")
    print(f"  COP_Tc_exp  computed   : {ok_exp}")
    print(f"  COP_Tc_pred computed   : {ok_pred}")
    if len(err_exp):
        print(f"  Mean |error| Tc_exp    : {err_exp.mean():.2f}%  (max {err_exp.max():.2f}%)")
    if len(err_pred):
        print(f"  Mean |error| Tc_pred   : {err_pred.mean():.2f}%  (max {err_pred.max():.2f}%)")

    # ── Scatter plots ───────────────────────────────────────────────────────────
    print(f"\n  Generating scatter plots...")

    # Pairs where both lit COP and calculated COP exist
    plot_data_exp  = df_out[df_out["COP"].notna() & df_out["COP_Tc_exp"].notna()]
    plot_data_pred = df_out[df_out["COP"].notna() & df_out["COP_Tc_pred"].notna()]

    r2_exp = scatter_plot(
        x_vals   = plot_data_exp["COP"].values,
        y_vals   = plot_data_exp["COP_Tc_exp"].values,
        labels   = plot_data_exp["IUPAC_Name"].values,
        xlabel   = "COP  (Literature)",
        ylabel   = "COP  (Calculated — Tc experimental)",
        title    = "COP: Literature vs Calculated (Tc experimental)",
        filename = PLOT_TC_EXP,
        color    = "#2196F3"   # blue
    )

    r2_pred = scatter_plot(
        x_vals   = plot_data_pred["COP"].values,
        y_vals   = plot_data_pred["COP_Tc_pred"].values,
        labels   = plot_data_pred["IUPAC_Name"].values,
        xlabel   = "COP  (Literature)",
        ylabel   = "COP  (Calculated — Tc predicted)",
        title    = "COP: Literature vs Calculated (Tc predicted)",
        filename = PLOT_TC_PRED,
        color    = "#E91E63"   # pink
    )

    print(f"\n  R²  Tc_exp  = {r2_exp:.4f}" if r2_exp is not None else "")
    print(f"  R²  Tc_pred = {r2_pred:.4f}" if r2_pred is not None else "")
    print(f"\n{'='*75}")
    print("  Done.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
