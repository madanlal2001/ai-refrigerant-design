"""
PR EOS Vapor Compression Cycle Engine
======================================
Computes COP and volumetric capacity (Q_vol) for pure refrigerants using:
    - Peng-Robinson EOS (Tc, Pc, ω)
    - Exact ideal-gas Cp polynomial from CoolProp fit (A, B, C, D)
    - Standard vapor compression cycle with superheat/subcooling

Operating conditions (configurable):
    T_evap = 10°C, T_cond = 40°C, ΔT_sh = 5 K, ΔT_sc = 5 K, η_is = 1.0

References:
    Peng & Robinson (1976); Smith, Van Ness & Abbott (2018);
    Bell et al. (2019), Int. J. Refrig. 104, 484-495
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CYCLE CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
R = 8.314462          # J / (mol·K)
SQRT2 = np.sqrt(2)

@dataclass
class CycleConditions:
    T_evap_C  : float = 10.0    # evaporator temperature [°C]
    T_cond_C  : float = 40.0    # condenser temperature [°C]
    dT_sh     : float = 5.0     # superheat at evaporator outlet [K]
    dT_sc     : float = 5.0     # subcooling at condenser outlet [K]
    eta_is    : float = 1.0     # isentropic compressor efficiency

    @property
    def T_evap(self): return self.T_evap_C + 273.15

    @property
    def T_cond(self): return self.T_cond_C + 273.15

    @property
    def T1(self): return self.T_evap + self.dT_sh    # compressor inlet

    @property
    def T3(self): return self.T_cond - self.dT_sc    # condenser outlet

DEFAULT_CONDITIONS = CycleConditions()

# Reference state for ideal-gas integration (arbitrary, cancels in cycle)
T_REF = 298.15
P_REF = 101325.0


# ─────────────────────────────────────────────────────────────────────────────
# PR EOS CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def pr_params(T: float, Tc: float, Pc: float, omega: float):
    """
    Compute PR EOS parameters at temperature T.

    Returns: a(T), b, da/dT
    """
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    alpha = (1.0 + kappa*(1.0 - np.sqrt(T/Tc)))**2
    a     = 0.45724 * R**2 * Tc**2 / Pc * alpha
    b     = 0.07780 * R * Tc / Pc
    dadt  = -0.45724 * R**2 * Tc**2 / Pc * kappa * np.sqrt(alpha / (T * Tc))
    return a, b, dadt


def solve_cubic_Z(A_dim: float, B_dim: float):
    """
    Solve PR EOS cubic: Z³ − (1−B)Z² + (A−3B²−2B)Z − (AB−B²−B³) = 0

    Returns (Z_V, Z_L):
        Two-phase:   Z_V = largest real root, Z_L = smallest real root
        Single-phase: one root is physical; other returned as None
    """
    p = [1.0,
         -(1.0 - B_dim),
          A_dim - 3.0*B_dim**2 - 2.0*B_dim,
         -(A_dim*B_dim - B_dim**2 - B_dim**3)]
    roots = np.roots(p)
    # Keep only real roots above B (unphysical below B)
    real_roots = sorted([r.real for r in roots
                         if abs(r.imag) < 1e-6 and r.real > B_dim + 1e-10])

    if len(real_roots) >= 3:
        return real_roots[-1], real_roots[0]    # Z_V, Z_L (discard middle)
    elif len(real_roots) == 1:
        return real_roots[0], None
    elif len(real_roots) == 2:
        # Rare: two valid roots — return larger as vapor
        return real_roots[-1], real_roots[0]
    return None, None


def ln_fugacity_coeff(Z: float, A_dim: float, B_dim: float) -> float:
    """
    Natural log of fugacity coefficient for PR EOS.
    Eq. 18 from Peng & Robinson (1976).
    """
    return (Z - 1.0
            - np.log(Z - B_dim)
            - A_dim / (2.0*SQRT2*B_dim)
            * np.log((Z + (1.0+SQRT2)*B_dim) / (Z + (1.0-SQRT2)*B_dim)))


def psat_iter(T: float, Tc: float, Pc: float, omega: float,
              tol: float = 1e-9, max_iter: int = 200) -> Optional[float]:
    """
    Saturation pressure from equal-fugacity iteration:
        φ_L(T, P_sat) = φ_V(T, P_sat)

    Returns P_sat in Pa, or None if not converged (outside two-phase region).
    """
    # Wilson equation initial guess
    P = Pc * np.exp(5.373 * (1.0 + omega) * (1.0 - Tc/T))
    P = np.clip(P, 1e3, 0.99*Pc)

    for _ in range(max_iter):
        a, b, _ = pr_params(T, Tc, Pc, omega)
        A_d = a * P / (R*T)**2
        B_d = b * P / (R*T)
        Z_V, Z_L = solve_cubic_Z(A_d, B_d)

        if Z_V is None or Z_L is None or Z_V <= B_d or Z_L <= B_d:
            return None

        ln_phi_L = ln_fugacity_coeff(Z_L, A_d, B_d)
        ln_phi_V = ln_fugacity_coeff(Z_V, A_d, B_d)

        P_new = P * np.exp(ln_phi_L - ln_phi_V)
        if not np.isfinite(P_new) or P_new <= 0:
            return None
        if abs(P_new/P - 1.0) < tol:
            return P_new
        P = np.clip(P_new, 1e3, 0.99*Pc)

    return None    # did not converge


# ─────────────────────────────────────────────────────────────────────────────
# DEPARTURE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _log_term(Z: float, B_dim: float) -> float:
    num = Z + (1.0+SQRT2)*B_dim
    den = Z + (1.0-SQRT2)*B_dim
    if den <= 0 or num <= 0:
        return 0.0
    return np.log(num/den)


def h_departure(T: float, P: float, Z: float,
                Tc: float, Pc: float, omega: float) -> float:
    """Enthalpy departure H_dep [J/mol]. Smith et al. Eq. 6.61"""
    a, b, dadt = pr_params(T, Tc, Pc, omega)
    B_dim = b * P / (R*T)
    return (R*T*(Z - 1.0)
            + (T*dadt - a) / (2.0*SQRT2*b) * _log_term(Z, B_dim))


def s_departure(T: float, P: float, Z: float,
                Tc: float, Pc: float, omega: float) -> float:
    """Entropy departure S_dep [J/mol·K]. Smith et al. Eq. 6.62"""
    a, b, dadt = pr_params(T, Tc, Pc, omega)
    B_dim = b * P / (R*T)
    if Z <= B_dim:
        return 0.0
    return (R * np.log(Z - B_dim)
            + dadt / (2.0*SQRT2*b) * _log_term(Z, B_dim))


# ─────────────────────────────────────────────────────────────────────────────
# IDEAL GAS PROPERTIES  (Cp polynomial integration)
# ─────────────────────────────────────────────────────────────────────────────

def h_ideal(T: float, A: float, B: float, C: float, D: float,
            T_ref: float = T_REF) -> float:
    """
    Ideal-gas enthalpy change from T_ref to T [J/mol]:
        ΔH_ig = ∫[T_ref→T] Cp dT  with  Cp = A + BT + CT² + DT³
    """
    return (A*(T - T_ref)
            + B/2*(T**2 - T_ref**2)
            + C/3*(T**3 - T_ref**3)
            + D/4*(T**4 - T_ref**4))


def s_ideal(T: float, P: float, A: float, B: float, C: float, D: float,
            T_ref: float = T_REF, P_ref: float = P_REF) -> float:
    """
    Ideal-gas entropy at (T, P) relative to reference state [J/mol·K]:
        ΔS_ig = ∫[T_ref→T] Cp/T dT − R·ln(P/P_ref)
    """
    return (A * np.log(T/T_ref)
            + B*(T - T_ref)
            + C/2*(T**2 - T_ref**2)
            + D/3*(T**3 - T_ref**3)
            - R * np.log(P/P_ref))


def total_H(T: float, P: float, Z: float,
            A: float, B: float, C: float, D: float,
            Tc: float, Pc: float, omega: float) -> float:
    return h_ideal(T, A, B, C, D) + h_departure(T, P, Z, Tc, Pc, omega)


def total_S(T: float, P: float, Z: float,
            A: float, B: float, C: float, D: float,
            Tc: float, Pc: float, omega: float) -> float:
    return s_ideal(T, P, A, B, C, D) + s_departure(T, P, Z, Tc, Pc, omega)


# ─────────────────────────────────────────────────────────────────────────────
# ISENTROPIC COMPRESSION  (Newton-Raphson on T₂ₛ)
# ─────────────────────────────────────────────────────────────────────────────

def find_T2s(S1: float, P_cond: float,
             Tc: float, Pc: float, omega: float,
             A: float, B: float, C: float, D: float,
             T_init: float, tol: float = 1e-7,
             max_iter: int = 100) -> Optional[float]:
    """
    Newton-Raphson: find T₂ₛ such that S(T₂ₛ, P_cond) = S₁.
    Returns T₂ₛ in K, or None if not converged.
    """
    T2s = T_init
    dT  = 0.05    # finite-difference step for dS/dT

    def S_func(T):
        a, b, _ = pr_params(T, Tc, Pc, omega)
        A_d = a * P_cond / (R*T)**2
        B_d = b * P_cond / (R*T)
        ZV, _ = solve_cubic_Z(A_d, B_d)
        if ZV is None:
            return np.nan
        return total_S(T, P_cond, ZV, A, B, C, D, Tc, Pc, omega)

    for i in range(max_iter):
        S_val  = S_func(T2s)
        S_high = S_func(T2s + dT)
        S_low  = S_func(T2s - dT)
        if not (np.isfinite(S_val) and np.isfinite(S_high) and np.isfinite(S_low)):
            return None
        dSdT = (S_high - S_low) / (2.0*dT)
        if abs(dSdT) < 1e-12:
            return None
        step = -(S_val - S1) / dSdT
        step = np.clip(step, -10.0, 10.0)    # guard large jumps
        T2s += step
        if abs(S_func(T2s) - S1) < tol:
            return T2s

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FLUID CYCLE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CycleResult:
    fluid     : str
    COP       : float = np.nan
    Q_evap    : float = np.nan   # J/mol
    W_comp    : float = np.nan   # J/mol
    Q_vol     : float = np.nan   # J/m³ — volumetric capacity
    P_evap    : float = np.nan   # Pa
    P_cond    : float = np.nan   # Pa
    PR_ratio  : float = np.nan   # P_cond/P_evap
    T2s_K     : float = np.nan   # compressor outlet temp [K]
    H1        : float = np.nan
    H2        : float = np.nan
    H3        : float = np.nan
    error     : str   = ''


def run_cycle(fluid: str,
              Tc: float, Pc: float, omega: float, MW: float,
              A: float, B: float, C: float, D: float,
              cond: CycleConditions = DEFAULT_CONDITIONS) -> CycleResult:
    """
    Full vapor compression cycle via PR EOS.

    State points:
        1 : Evaporator outlet  (superheated vapor, T₁ = T_evap + ΔT_sh)
        2s: Compressor outlet  (isentropic,  S₂ₛ = S₁)
        2 : Same as 2s if η=1; H₂ = H₁ + (H₂ₛ−H₁)/η
        3 : Condenser outlet   (subcooled liquid, T₃ = T_cond − ΔT_sc)
        4 : Expansion valve    (isenthalpic, H₄ = H₃)

    COP     = Q_evap / W_comp = (H₁−H₄) / (H₂−H₁)
    Q_vol   = Q_evap × ρ_evap  [J/m³]  — determines compressor size
    """
    res = CycleResult(fluid=fluid)

    # Guard: operating range check
    if cond.T_evap >= Tc:
        res.error = 'T_evap >= Tc'; return res
    if cond.T_cond >= Tc:
        res.error = 'T_cond >= Tc'; return res

    # ── Saturation pressures ──────────────────────────────────────────────
    P_evap = psat_iter(cond.T_evap, Tc, Pc, omega)
    P_cond = psat_iter(cond.T_cond, Tc, Pc, omega)
    if P_evap is None: res.error = 'P_evap not converged'; return res
    if P_cond is None: res.error = 'P_cond not converged'; return res
    if P_evap >= P_cond: res.error = 'P_evap >= P_cond'; return res

    res.P_evap   = P_evap
    res.P_cond   = P_cond
    res.PR_ratio = P_cond / P_evap

    # ── State 1: superheated vapor ────────────────────────────────────────
    a1, b1, _ = pr_params(cond.T1, Tc, Pc, omega)
    A1 = a1 * P_evap / (R*cond.T1)**2
    B1 = b1 * P_evap / (R*cond.T1)
    Z1, _ = solve_cubic_Z(A1, B1)
    if Z1 is None: res.error = 'State1: no vapor Z'; return res

    H1 = total_H(cond.T1, P_evap, Z1, A, B, C, D, Tc, Pc, omega)
    S1 = total_S(cond.T1, P_evap, Z1, A, B, C, D, Tc, Pc, omega)

    # Molar volume at State 1 for Q_vol
    V1 = Z1 * R * cond.T1 / P_evap    # m³/mol

    # ── State 2s: isentropic compression ─────────────────────────────────
    T2s_init = cond.T1 + 20.0
    T2s = find_T2s(S1, P_cond, Tc, Pc, omega, A, B, C, D, T_init=T2s_init)
    if T2s is None or not np.isfinite(T2s):
        res.error = 'T2s NR not converged'; return res
    if T2s < cond.T1:
        res.error = f'T2s={T2s:.1f} < T1={cond.T1:.1f} (unphysical)'; return res

    a2, b2, _ = pr_params(T2s, Tc, Pc, omega)
    A2 = a2 * P_cond / (R*T2s)**2
    B2 = b2 * P_cond / (R*T2s)
    Z2, _ = solve_cubic_Z(A2, B2)
    if Z2 is None: res.error = 'State2: no vapor Z'; return res

    H2s = total_H(T2s, P_cond, Z2, A, B, C, D, Tc, Pc, omega)
    H2  = H1 + (H2s - H1) / cond.eta_is    # actual enthalpy

    # ── State 3: subcooled liquid ─────────────────────────────────────────
    a3, b3, _ = pr_params(cond.T3, Tc, Pc, omega)
    A3 = a3 * P_cond / (R*cond.T3)**2
    B3 = b3 * P_cond / (R*cond.T3)
    _, Z3 = solve_cubic_Z(A3, B3)
    if Z3 is None: res.error = 'State3: no liquid Z'; return res

    H3 = total_H(cond.T3, P_cond, Z3, A, B, C, D, Tc, Pc, omega)

    # ── State 4: isenthalpic expansion ───────────────────────────────────
    H4 = H3

    # ── Cycle performance ─────────────────────────────────────────────────
    Q_evap = H1 - H4
    W_comp = H2 - H1

    if W_comp <= 0:
        res.error = f'W_comp={W_comp:.2f} <= 0 (unphysical)'; return res
    if Q_evap <= 0:
        res.error = f'Q_evap={Q_evap:.2f} <= 0'; return res

    # Q_vol [J/m³] = Q_evap / V1 × (1 kg/mol / MW in kg/mol)
    # V1 in m³/mol → Q_vol in J/m³
    Q_vol = Q_evap / V1    # J/m³ (molar basis)

    # Fill result
    res.COP      = Q_evap / W_comp
    res.Q_evap   = Q_evap
    res.W_comp   = W_comp
    res.Q_vol    = Q_vol
    res.T2s_K    = T2s
    res.H1, res.H2, res.H3 = H1, H2, H3

    return res


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SCREENER
# ─────────────────────────────────────────────────────────────────────────────

def screen_refrigerants(csv_path: str,
                         cond: CycleConditions = DEFAULT_CONDITIONS,
                         validate_coolprop: bool = True) -> pd.DataFrame:
    """
    Run VCC cycle for all fluids in the CSV.

    CSV must contain: Fluid, Tc_K, Pc_Pa, Omega, MW_g_mol,
                      A_coeff, B_coeff, C_coeff, D_coeff, Cp_is_constant
    """
    df = pd.read_csv(csv_path)

    # Exclude noble gases / constant-Cp fluids (not viable refrigerants)
    df = df[~df['Cp_is_constant']].reset_index(drop=True)

    results = []
    for _, row in df.iterrows():
        res = run_cycle(
            fluid = row['Fluid'],
            Tc    = row['Tc_K'],
            Pc    = row['Pc_Pa'],
            omega = row['Omega'],
            MW    = row['MW_g_mol'],
            A     = row['A_coeff'],
            B     = row['B_coeff'],
            C     = row['C_coeff'],
            D     = row['D_coeff'],
            cond  = cond,
        )
        results.append(res)

    # Build output dataframe
    records = []
    for r in results:
        records.append({
            'Fluid'      : r.fluid,
            'COP'        : round(r.COP,    4) if np.isfinite(r.COP)    else np.nan,
            'Q_evap_J_mol': round(r.Q_evap, 1) if np.isfinite(r.Q_evap)  else np.nan,
            'W_comp_J_mol': round(r.W_comp, 1) if np.isfinite(r.W_comp)  else np.nan,
            'Q_vol_J_m3' : round(r.Q_vol,  0) if np.isfinite(r.Q_vol)   else np.nan,
            'P_evap_bar' : round(r.P_evap/1e5, 4) if np.isfinite(r.P_evap) else np.nan,
            'P_cond_bar' : round(r.P_cond/1e5, 4) if np.isfinite(r.P_cond) else np.nan,
            'PR_ratio'   : round(r.PR_ratio, 3) if np.isfinite(r.PR_ratio) else np.nan,
            'T2s_K'      : round(r.T2s_K,  2) if np.isfinite(r.T2s_K)   else np.nan,
            'error'      : r.error,
        })

    out_df = pd.DataFrame(records)
    ok  = out_df[out_df['error'] == ''].copy()
    bad = out_df[out_df['error'] != ''].copy()

    print(f"\n{'═'*60}")
    print(f"  SCREENING COMPLETE")
    print(f"{'═'*60}")
    print(f"  Converged : {len(ok)} / {len(out_df)}")
    print(f"  Failed    : {len(bad)}")
    if len(bad):
        print(bad[['Fluid','error']].to_string(index=False))

    if len(ok):
        print(f"\n  COP statistics:")
        print(f"    mean = {ok['COP'].mean():.4f}")
        print(f"    min  = {ok['COP'].min():.4f}  ({ok.loc[ok['COP'].idxmin(),'Fluid']})")
        print(f"    max  = {ok['COP'].max():.4f}  ({ok.loc[ok['COP'].idxmax(),'Fluid']})")

    # ── CoolProp validation ────────────────────────────────────────────────
    if validate_coolprop:
        _validate_vs_coolprop(ok, cond)

    return out_df.sort_values('COP', ascending=False).reset_index(drop=True)


def _validate_vs_coolprop(ok_df: pd.DataFrame, cond: CycleConditions):
    """
    Compare PR EOS COP against CoolProp reference for known fluids.
    Uses CoolProp PropsSI for exact thermodynamic properties.
    """
    try:
        import CoolProp.CoolProp as CP
    except ImportError:
        print("\n  CoolProp not available — skipping validation")
        return

    # Reference fluids from paper Table (Fig. 13)
    ref_fluids = {
        'R134a'      : 8.5197,
        'R22'        : 8.3827,
        'R32'        : 8.0456,
        'Ammonia'    : 8.4784,
        'R1234yf'    : 8.3605,
        'n-Propane'  : 8.3775,
    }
    # CoolProp name mapping
    cp_names = {
        'R134a': 'R134a', 'R22': 'R22', 'R32': 'R32',
        'Ammonia': 'Ammonia', 'R1234yf': 'R1234yf', 'n-Propane': 'n-Propane'
    }

    print(f"\n{'─'*60}")
    print(f"  COOLPROP VALIDATION  (T_evap={cond.T_evap_C}°C, T_cond={cond.T_cond_C}°C)")
    print(f"{'─'*60}")
    print(f"  {'Fluid':<15} {'COP_PR':>8} {'COP_CP':>8} {'Error%':>8}")
    print(f"  {'─'*46}")

    errors = []
    for fluid, cop_ref in ref_fluids.items():
        row = ok_df[ok_df['Fluid'] == fluid]
        if row.empty:
            print(f"  {fluid:<15} {'—':>8} {cop_ref:>8.4f} {'N/A':>8}")
            continue
        cop_pr = row['COP'].iloc[0]

        # Also compute CoolProp cycle directly for this set of conditions
        try:
            T1  = cond.T1; T3 = cond.T3
            P_ev = CP.PropsSI('P','T',cond.T_evap,'Q',1, cp_names[fluid])
            P_co = CP.PropsSI('P','T',cond.T_cond,'Q',0, cp_names[fluid])
            H1cp = CP.PropsSI('H','T',T1,'P',P_ev, cp_names[fluid])
            S1cp = CP.PropsSI('S','T',T1,'P',P_ev, cp_names[fluid])
            H2cp = CP.PropsSI('H','P',P_co,'S',S1cp, cp_names[fluid])
            H3cp = CP.PropsSI('H','T',T3,'P',P_co, cp_names[fluid])
            cop_cp = (H1cp - H3cp) / (H2cp - H1cp)
        except Exception:
            cop_cp = cop_ref   # fall back to paper reference

        err = abs(cop_pr - cop_cp) / cop_cp * 100
        errors.append(err)
        flag = '✓' if err < 2.0 else ('~' if err < 5.0 else '✗')
        print(f"  {fluid:<15} {cop_pr:>8.4f} {cop_cp:>8.4f} {err:>7.2f}% {flag}")

    if errors:
        print(f"  {'─'*46}")
        print(f"  {'Mean abs error':.<30} {np.mean(errors):>7.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    csv_path = 'gnn_training_data_Cp_poly.csv'
    out_dir  = Path('cycle_output')
    out_dir.mkdir(exist_ok=True)

    print("═" * 60)
    print("  PR EOS VAPOR COMPRESSION CYCLE SCREENER")
    print(f"  T_evap = {DEFAULT_CONDITIONS.T_evap_C}°C  |  "
          f"T_cond = {DEFAULT_CONDITIONS.T_cond_C}°C  |  "
          f"ΔT_sh = {DEFAULT_CONDITIONS.dT_sh} K  |  "
          f"ΔT_sc = {DEFAULT_CONDITIONS.dT_sc} K")
    print("═" * 60)

    results_df = screen_refrigerants(csv_path, validate_coolprop=True)

    # ── Top-20 ranking ────────────────────────────────────────────────────
    ok_df = results_df[results_df['error'] == ''].copy()
    print(f"\n{'─'*60}")
    print(f"  TOP 20 REFRIGERANTS BY COP")
    print(f"{'─'*60}")
    print(ok_df.head(20)[['Fluid','COP','Q_vol_J_m3','PR_ratio','P_evap_bar','T2s_K']].to_string(index=False))

    print(f"\n  BOTTOM 10 BY COP")
    print(ok_df.tail(10)[['Fluid','COP','Q_vol_J_m3','PR_ratio']].to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────────
    results_df.to_csv(out_dir / 'cop_screening_results.csv', index=False)
    ok_df.to_csv(out_dir / 'cop_converged.csv', index=False)
    print(f"\n  ✓ Saved to {out_dir}/")

    return results_df


if __name__ == '__main__':
    df = main()
