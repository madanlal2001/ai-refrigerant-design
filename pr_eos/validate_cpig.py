"""
=============================================================================
  Cp_ig VALIDATION SCRIPT
=============================================================================
  Purpose : Validate the polynomial coefficients [a,b,c,d] computed in
            cp_ig_pipeline.py against independent reference values.

  Three validation levels:
  ─────────────────────────────────────────────────────────────────────────
  Level 1 — POINT CHECK (Table)
            Compare our polynomial Cp at T = 300 K against:
            → CoolProp (HEOS_FIT) reference for CoolProp-group fluids
            → thermo library best available method for Joback-group fluids
            Reports absolute error and % error per fluid.

  Level 2 — CURVE CHECK (Plots)
            Plot our polynomial Cp(T) vs reference Cp(T) across 250–450 K
            for all CoolProp-group fluids (where reference is exact).
            Visual check that the polynomial fits well across the full range.

  Level 3 — PHYSICAL RULES CHECK (Table)
            For every fluid check:
            (a) Cp > 0 at all temperatures in 250–450 K
            (b) Cp increases with T (dCp/dT > 0) — true for most organics
            (c) Cp at 300 K is in physically reasonable range

  Outputs:
    → validation_point_check.csv   — Level 1 table
    → validation_curve_plots.pdf   — Level 2 plots (one subplot per fluid)
    → validation_physical_check.csv — Level 3 table
    → validation_summary.txt        — plain-text summary
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')

import numpy             as np
import pandas            as pd
import matplotlib
matplotlib.use('Agg')                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

from CoolProp.CoolProp   import PropsSI
from thermo              import Chemical


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

T_MIN       = 250.0     # K  — lower bound of validation range
T_MAX       = 450.0     # K  — upper bound of validation range
T_CHECK     = 300.0     # K  — single point for Level 1 comparison
N_PLOT_PTS  = 100       # number of T points for curve plots

# Acceptable % error thresholds
PASS_THRESHOLD_CP  = 2.0    # %  — good fit
WARN_THRESHOLD_CP  = 5.0    # %  — acceptable but note it
# Above 5% → FAIL


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def poly_cp(a, b, c, d, T):
    """Evaluate Cp polynomial at temperature T (K). Returns J/mol·K."""
    return a + b*T + c*T**2 + d*T**3


def poly_dcp_dT(a, b, c, d, T):
    """Derivative dCp/dT at temperature T. Returns J/mol·K²."""
    return b + 2*c*T + 3*d*T**2


def get_reference_cp(cp_fluid_name, smiles, source, T):
    """
    Get the best available reference Cp_ig at temperature T.

    Priority:
      1. CoolProp direct (for CoolProp-group fluids)
      2. thermo Chemical best method (for Joback-group fluids)
      3. None if unavailable
    """
    # ── CoolProp direct ──────────────────────────────────────────────────────
    if source == 'CoolProp' and cp_fluid_name:
        try:
            val = PropsSI('Cp0molar', 'T', T, 'P', 101325.0, cp_fluid_name)
            return float(val), 'CoolProp_direct'
        except Exception:
            pass

    # ── thermo Chemical (uses best available DB method) ──────────────────────
    if smiles and smiles != 'nan':
        try:
            chem = Chemical(smiles)
            val  = chem.HeatCapacityGas(T)
            meth = chem.HeatCapacityGas.method
            if val and val > 0:
                return float(val), meth
        except Exception:
            pass

    return None, 'unavailable'


def flag_error(err_pct):
    """Return a readable flag based on % error."""
    if err_pct is None:
        return 'NO_REF'
    if err_pct <= PASS_THRESHOLD_CP:
        return 'PASS'
    if err_pct <= WARN_THRESHOLD_CP:
        return 'WARN'
    return 'FAIL'


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — POINT CHECK AT T = 300 K
# ─────────────────────────────────────────────────────────────────────────────

def level1_point_check(df):
    """
    Compare polynomial Cp at T=300 K against best available reference.
    Returns a DataFrame with error analysis for every fluid.
    """
    print("\n" + "="*70)
    print("  LEVEL 1 — POINT CHECK AT T = 300 K")
    print("="*70)

    records = []

    for _, row in df.iterrows():
        name          = row['Name']
        source        = row['source']
        smiles        = str(row.get('SMILES', '')).strip()
        cp_fluid_name = str(row.get('cp_fluid_name', '')).strip()

        # Skip Missing group — no coefficients
        if source == 'Missing':
            records.append({
                'Name'       : name,
                'source'     : source,
                'Cp_poly'    : None,
                'Cp_ref'     : None,
                'ref_method' : 'unavailable',
                'abs_err'    : None,
                'pct_err'    : None,
                'flag'       : 'NO_REF',
            })
            continue

        a, b, c, d = row['a'], row['b'], row['c'], row['d']

        # Our polynomial value
        cp_poly = poly_cp(a, b, c, d, T_CHECK)

        # Best reference value
        cp_ref, ref_method = get_reference_cp(cp_fluid_name, smiles,
                                               source, T_CHECK)

        # Error calculation
        if cp_ref is not None:
            abs_err = abs(cp_poly - cp_ref)
            pct_err = abs_err / cp_ref * 100
        else:
            abs_err = None
            pct_err = None

        flag = flag_error(pct_err)

        records.append({
            'Name'       : name,
            'source'     : source,
            'Cp_poly'    : round(cp_poly, 3),
            'Cp_ref'     : round(cp_ref, 3) if cp_ref else None,
            'ref_method' : ref_method,
            'abs_err'    : round(abs_err, 3) if abs_err else None,
            'pct_err'    : round(pct_err, 2) if pct_err else None,
            'flag'       : flag,
        })

        # Print to console — show fluid, values, flag
        cp_ref_str = f"{cp_ref:8.3f}" if cp_ref else "       N/A"
        err_str    = f"{pct_err:6.2f}%" if pct_err else "    N/A"
        print(f"  {name:30s} [{source:8s}]  "
              f"poly={cp_poly:8.3f}  ref={cp_ref_str}  "
              f"err={err_str}  {flag}")

    result = pd.DataFrame(records)

    # Print counts
    print("\n  Summary:")
    for flag in ['PASS', 'WARN', 'FAIL', 'NO_REF']:
        n = (result['flag'] == flag).sum()
        print(f"    {flag:6s} : {n}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — CURVE PLOTS (CoolProp group only, reference is exact)
# ─────────────────────────────────────────────────────────────────────────────

def level2_curve_plots(df, output_path):
    """
    For each CoolProp-group fluid, plot:
      — Blue line  : Our polynomial Cp(T)
      — Red dots   : CoolProp exact Cp0molar(T)
    Saves all subplots into a single PDF.
    """
    print("\n" + "="*70)
    print("  LEVEL 2 — CURVE PLOTS (CoolProp group)")
    print("="*70)

    cp_group = df[df['source'] == 'CoolProp'].reset_index(drop=True)
    n        = len(cp_group)

    T_arr = np.linspace(T_MIN, T_MAX, N_PLOT_PTS)

    # Layout: 4 columns, enough rows
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 4*nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, (_, row) in enumerate(cp_group.iterrows()):
        ax            = axes[i]
        name          = row['Name']
        cp_fluid_name = str(row.get('cp_fluid_name', '')).strip()
        a, b, c, d    = row['a'], row['b'], row['c'], row['d']

        # Polynomial curve
        cp_poly_arr = [poly_cp(a, b, c, d, T) for T in T_arr]

        # CoolProp reference points (every 10 K for readability)
        T_ref_arr  = np.linspace(T_MIN, T_MAX, 20)
        cp_ref_arr = []
        for T in T_ref_arr:
            try:
                cp_ref_arr.append(PropsSI('Cp0molar','T',T,'P',101325,
                                          cp_fluid_name))
            except Exception:
                cp_ref_arr.append(np.nan)

        # Compute max error for title
        errs = [abs(p - r)/r*100
                for p, r in zip(cp_poly_arr,
                                [PropsSI('Cp0molar','T',T,'P',101325,
                                         cp_fluid_name)
                                 for T in T_arr])
                if r > 0]
        max_err = max(errs) if errs else 0

        # Plot
        ax.plot(T_arr, cp_poly_arr, 'b-',  linewidth=2,
                label='Polynomial fit')
        ax.plot(T_ref_arr, cp_ref_arr, 'ro', markersize=5,
                label='CoolProp exact')

        ax.set_title(f"{name}\n(max err={max_err:.2f}%)", fontsize=9)
        ax.set_xlabel("T (K)",         fontsize=8)
        ax.set_ylabel("Cp_ig (J/mol·K)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        print(f"  Plotted: {name:30s}  max_err={max_err:.3f}%")

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Level 2 — Cp_ig Polynomial vs CoolProp Reference\n"
                 "Blue=Polynomial  Red=CoolProp exact",
                 fontsize=12, fontweight='bold')

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved curve plots → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3 — PHYSICAL RULES CHECK
# ─────────────────────────────────────────────────────────────────────────────

def level3_physical_check(df):
    """
    For every fluid with valid coefficients, verify:
      Rule A: Cp > 0 for all T in [250, 450] K
      Rule B: dCp/dT > 0 at T = 300 K  (most organics)
      Rule C: Cp(300 K) in range [10, 500] J/mol·K  (physically sane)
    """
    print("\n" + "="*70)
    print("  LEVEL 3 — PHYSICAL RULES CHECK")
    print("="*70)

    T_check_arr = np.linspace(T_MIN, T_MAX, 50)
    records     = []

    for _, row in df.iterrows():
        name   = row['Name']
        source = row['source']

        if source == 'Missing':
            records.append({'Name': name, 'source': source,
                            'rule_A': 'SKIP', 'rule_B': 'SKIP',
                            'rule_C': 'SKIP', 'overall': 'SKIP'})
            continue

        a, b, c, d = row['a'], row['b'], row['c'], row['d']

        # Rule A: Cp > 0 everywhere
        cp_values = [poly_cp(a, b, c, d, T) for T in T_check_arr]
        rule_A    = 'PASS' if all(v > 0 for v in cp_values) else 'FAIL'

        # Rule B: dCp/dT > 0 at 300 K
        dcp_300   = poly_dcp_dT(a, b, c, d, 300.0)
        rule_B    = 'PASS' if dcp_300 > 0 else 'WARN'

        # Rule C: Cp(300 K) in [10, 500] J/mol·K
        cp_300    = poly_cp(a, b, c, d, 300.0)
        rule_C    = 'PASS' if 10 <= cp_300 <= 500 else 'FAIL'

        # Overall
        if 'FAIL' in [rule_A, rule_B, rule_C]:
            overall = 'FAIL'
        elif 'WARN' in [rule_A, rule_B, rule_C]:
            overall = 'WARN'
        else:
            overall = 'PASS'

        print(f"  {name:30s} [{source:8s}]  "
              f"A:{rule_A}  B:{rule_B}  C:{rule_C}  → {overall}")

        records.append({
            'Name'    : name,
            'source'  : source,
            'Cp_300K' : round(cp_300, 3),
            'dCp_300K': round(dcp_300, 6),
            'rule_A'  : rule_A,
            'rule_B'  : rule_B,
            'rule_C'  : rule_C,
            'overall' : overall,
        })

    result = pd.DataFrame(records)

    print("\n  Summary:")
    for flag in ['PASS', 'WARN', 'FAIL', 'SKIP']:
        n = (result['overall'] == flag).sum()
        print(f"    {flag:4s} : {n}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — WRITE PLAIN-TEXT SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(l1_df, l3_df, output_path):
    """Write a plain-text summary report combining all validation levels."""

    lines = []
    lines.append("=" * 70)
    lines.append("  Cp_ig VALIDATION SUMMARY REPORT")
    lines.append("=" * 70)

    # Level 1 summary
    lines.append("\nLEVEL 1 — Point check at T=300 K")
    lines.append(f"  {'Flag':<8} {'Count':>6}")
    lines.append("  " + "-" * 20)
    for flag in ['PASS', 'WARN', 'FAIL', 'NO_REF']:
        n = (l1_df['flag'] == flag).sum()
        lines.append(f"  {flag:<8} {n:>6}")

    # Level 1 worst offenders
    fail_warn = l1_df[l1_df['flag'].isin(['FAIL', 'WARN'])].copy()
    fail_warn = fail_warn.dropna(subset=['pct_err']).sort_values('pct_err',
                                                                  ascending=False)
    if not fail_warn.empty:
        lines.append("\n  Worst errors (FAIL/WARN):")
        for _, r in fail_warn.head(10).iterrows():
            lines.append(f"    {r['Name']:30s}  "
                         f"{r['pct_err']:6.2f}%  [{r['source']}]")

    # Level 3 summary
    lines.append("\nLEVEL 3 — Physical rules check")
    lines.append(f"  {'Flag':<8} {'Count':>6}")
    lines.append("  " + "-" * 20)
    for flag in ['PASS', 'WARN', 'FAIL', 'SKIP']:
        n = (l3_df['overall'] == flag).sum()
        lines.append(f"  {flag:<8} {n:>6}")

    # Physical fails
    phys_fail = l3_df[l3_df['overall'].isin(['FAIL', 'WARN'])]
    if not phys_fail.empty:
        lines.append("\n  Physical rule violations:")
        for _, r in phys_fail.iterrows():
            lines.append(f"    {r['Name']:30s}  "
                         f"A:{r['rule_A']}  B:{r['rule_B']}  "
                         f"C:{r['rule_C']}  [{r['source']}]")

    lines.append("\n" + "=" * 70)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    # Also print to console
    print('\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    INPUT_CSV    = "/mnt/user-data/outputs/refrigerants_cpig_coefficients.csv"
    OUT_L1_CSV   = "/mnt/user-data/outputs/validation_level1_point_check.csv"
    OUT_L2_PDF   = "/mnt/user-data/outputs/validation_level2_curve_plots.pdf"
    OUT_L3_CSV   = "/mnt/user-data/outputs/validation_level3_physical_check.csv"
    OUT_SUMMARY  = "/mnt/user-data/outputs/validation_summary.txt"

    # Load coefficients computed by cp_ig_pipeline.py
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} fluids from {INPUT_CSV}")

    # ── Run all three validation levels ──────────────────────────────────────
    l1_df = level1_point_check(df)
    level2_curve_plots(df, OUT_L2_PDF)
    l3_df = level3_physical_check(df)

    # ── Save outputs ──────────────────────────────────────────────────────────
    l1_df.to_csv(OUT_L1_CSV,  index=False)
    l3_df.to_csv(OUT_L3_CSV,  index=False)
    write_summary(l1_df, l3_df, OUT_SUMMARY)

    print(f"\n  Saved: {OUT_L1_CSV}")
    print(f"  Saved: {OUT_L2_PDF}")
    print(f"  Saved: {OUT_L3_CSV}")
    print(f"  Saved: {OUT_SUMMARY}")
    print("\n  VALIDATION COMPLETE\n")
