"""
================================================================================
 PR-EOS vs CoolProp — Binary Mixture COP Validation
================================================================================

Quantifies the error the PR-EOS model (binary_mixture_cop_sweep.py) introduces
into the ideal vapour-compression-cycle COP, relative to CoolProp's HEOS
mixture model, across every unique binary pair of "usable" CoolProp pure
fluids and a fixed composition grid.

This script does NOT reimplement any PR-EOS thermodynamics. It imports
binary_mixture_cop_sweep.py as a module and calls its functions directly
(mix_params, compute_cycle_mixture, dew_point_P, bubble_point_P, ...). The
only thing this script changes on that module are its CONFIG constants
(DT_SH, DT_SC, ETA_IS) — exactly the block that module's own docstring
invites you to edit — so both sides of the comparison run the same cycle:

    T_evap = 10 degC        (dew pressure of the mixture, Bell-mode)
    T_cond = 40 degC        (bubble pressure of the mixture, Bell-mode)
    superheat = 0 K
    subcooling = 0 K
    isentropic efficiency = 1.0   (ideal/reversible compressor: H2 = H2s)

(eta_is=1.0 was confirmed with the user: the request said "isentropic
efficiency = 0", which divides by zero in the COP formula and would fail
every single calculation. eta=1.0 is the ideal-compressor reading of that
requirement and isolates the EOS's effect on COP from compressor losses.)

--------------------------------------------------------------------------------
 Usage
--------------------------------------------------------------------------------
    python validate_pr_binary_vs_coolprop.py                  # full production run
    python validate_pr_binary_vs_coolprop.py --max-fluids 6   # smoke test
    python validate_pr_binary_vs_coolprop.py --fresh          # ignore checkpoint
    python validate_pr_binary_vs_coolprop.py --workers 4

Interrupting (Ctrl-C) saves a checkpoint; re-running the same command resumes
from the last completed pair automatically.
--------------------------------------------------------------------------------
"""

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

import binary_mixture_cop_sweep as bms  # noqa: E402  (the validated PR-EOS module)
import CoolProp.CoolProp as CP  # noqa: E402

# ==============================================================================
#  CONFIG
# ==============================================================================
T_EVAP_C = 10.0
T_COND_C = 40.0
DT_SH = 0.0
DT_SC = 0.0
ETA_IS = 1.0
KIJ = 0.0

X1_LIST = [round(0.1 * k, 2) for k in range(1, 10)]  # [0.1, 0.2, ..., 0.9]

MIN_COMPONENT_TC_K = T_COND_C + 273.15  # 313.15 K — exclude pure fluids at/under this Tc

CP_FIT_LO_CAP = 200.0     # lower cap on the Cp0 polynomial fit window [K]
CP_FIT_HI_CAP = 600.0     # upper cap on the Cp0 polynomial fit window [K]
CP_FIT_MIN_SPAN = 80.0    # minimum usable fit window width [K]
CP_FIT_POINTS = 40        # samples for the degree-4 polyfit

# ---- Output files -------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(OUT_DIR, "validation_results.csv")
FAILED_CSV = os.path.join(OUT_DIR, "validation_failed_pairs.csv")
CHECKPOINT_JSON = os.path.join(OUT_DIR, "validation_checkpoint.json")
LOG_FILE = os.path.join(OUT_DIR, "validation_run.log")
SCATTER_PNG = os.path.join(OUT_DIR, "cop_scatter.png")
HIST_PNG = os.path.join(OUT_DIR, "error_histogram.png")
COMPOSITION_PNG = os.path.join(OUT_DIR, "error_vs_composition.png")

CHECKPOINT_EVERY_PAIRS = 25
CHECKPOINT_EVERY_SECONDS = 120

RESULTS_COLUMNS = [
    "Fluid1", "Fluid2", "x1", "x2",
    "Tc1", "Pc1", "omega1", "Tc2", "Pc2", "omega2",
    "COP_CoolProp", "COP_PR",
    "P_evap_CP", "P_cond_CP", "P_evap_PR", "P_cond_PR",
    "Absolute_Error", "Percent_Error",
]
FAILED_COLUMNS = ["Fluid1", "Fluid2", "x1", "x2", "Failure_Source", "Failure_Reason"]

# ==============================================================================
#  Apply the cycle-assumption overrides to the imported PR-EOS module.
#  This runs unconditionally at import time (not inside __main__), so every
#  spawned worker process re-imports this file, re-imports bms, and gets the
#  same overrides automatically — no PR-EOS code is modified, only its
#  editable CONFIG constants.
# ==============================================================================
bms.T_EVAP_C = T_EVAP_C
bms.T_COND_C = T_COND_C
bms.DT_SH = DT_SH
bms.DT_SC = DT_SC
bms.ETA_IS = ETA_IS
bms.KIJ = KIJ
bms.T_EVAP = T_EVAP_C + 273.15
bms.T_COND = T_COND_C + 273.15
bms.T_REF = bms.T_EVAP

T_EVAP = bms.T_EVAP
T_COND = bms.T_COND

# CoolProp's HEOS backend refuses to evaluate a binary pair it has no stored
# interaction/departure data for ("Could not match the binary pair..."), and
# for the handful of pairs it DOES have tuned data for, that data would give
# CoolProp an advantage the PR-EOS side (kij=0, per its own CONFIG comment)
# never gets. Forcing a generic "linear" (no-interaction) combining rule on
# every pair keeps both sides on the same footing: zero fitted binary
# interaction data, exactly matching KIJ=0 used throughout the PR-EOS model.
CP.set_config_bool(CP.OVERWRITE_BINARY_INTERACTION, True)

# ==============================================================================
#  Logging
# ==============================================================================
def setup_logging():
    logger = logging.getLogger("validate")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = setup_logging()

# ==============================================================================
#  CoolProp fluid enumeration & filtering
# ==============================================================================
def is_predefined_blend(name):
    """
    CoolProp pseudo-pure / predefined blends (Air, R404A, R407C, R410A, R507A,
    SES36, ...) carry a synthetic CAS of the form '<NAME>.PPF' instead of a
    real CAS registry number. This is a stable, general way to detect them
    without hardcoding a fluid-name list.
    """
    try:
        cas = CP.get_fluid_param_string(name, "CAS")
    except Exception:
        return True  # can't confirm it's a real pure fluid -> exclude
    return cas.strip().lower().endswith(".ppf")


def fit_cp_ig(name, Tmin, Tmax):
    """
    Fit Cp_ig(T) = a + b*T + c*T^2 + d*T^3 + e*T^4  [J/mol/K] to CoolProp's
    CP0MOLAR ideal-gas heat capacity, sampled over a window clipped to the
    fluid's valid EOS range. Returns (a,b,c,d,e) or None if the fluid's valid
    range doesn't usefully cover the cycle's temperatures.
    """
    lo = max(Tmin + 1.0, CP_FIT_LO_CAP)
    hi = min(Tmax - 1.0, CP_FIT_HI_CAP)
    if hi - lo < CP_FIT_MIN_SPAN or lo > T_EVAP:
        return None
    Ts = np.linspace(lo, hi, CP_FIT_POINTS)
    cps = np.empty(CP_FIT_POINTS)
    for i, T in enumerate(Ts):
        cps[i] = CP.PropsSI("CP0MOLAR", "T", float(T), "P", 101325.0, name)
    coeff_hi_first = np.polyfit(Ts, cps, 4)
    a, b, c, d, e = coeff_hi_first[::-1]
    return float(a), float(b), float(c), float(d), float(e)


def build_usable_fluids():
    """
    Query every CoolProp pure fluid, apply all exclusion rules, and return
    the final usable-molecule list plus the raw/usable counts for reporting.
    """
    raw_fluids = sorted(set(CP.get_global_param_string("FluidsList").split(",")))
    total_found = len(raw_fluids)

    excluded = {"predefined_blend": 0, "low_tc_or_missing": 0, "bad_cp_fit_range": 0}
    mols = []

    for name in raw_fluids:
        if is_predefined_blend(name):
            excluded["predefined_blend"] += 1
            log.debug(f"exclude {name}: predefined/pseudo-pure blend (CAS ends .PPF)")
            continue

        try:
            Tc = CP.PropsSI("Tcrit", name)
            Pc = CP.PropsSI("Pcrit", name)
            omega = CP.PropsSI("acentric", name)
            Tmin = CP.PropsSI("Tmin", name)
            Tmax = CP.PropsSI("Tmax", name)
        except Exception as e:
            excluded["low_tc_or_missing"] += 1
            log.debug(f"exclude {name}: missing Tc/Pc/omega ({e})")
            continue

        if not all(np.isfinite(v) for v in (Tc, Pc, omega)):
            excluded["low_tc_or_missing"] += 1
            log.debug(f"exclude {name}: non-finite Tc/Pc/omega")
            continue

        if Tc <= MIN_COMPONENT_TC_K:
            excluded["low_tc_or_missing"] += 1
            log.debug(f"exclude {name}: Tc={Tc:.2f}K <= {MIN_COMPONENT_TC_K:.2f}K")
            continue

        cp_coeffs = fit_cp_ig(name, Tmin, Tmax)
        if cp_coeffs is None:
            excluded["bad_cp_fit_range"] += 1
            log.debug(f"exclude {name}: insufficient Cp0 fit range (Tmin={Tmin:.1f}, Tmax={Tmax:.1f})")
            continue

        try:
            cas = CP.get_fluid_param_string(name, "CAS")
        except Exception as e:
            excluded["low_tc_or_missing"] += 1
            log.debug(f"exclude {name}: missing CAS ({e})")
            continue

        mols.append({
            "name": name, "Tc": float(Tc), "Pc": float(Pc), "omega": float(omega),
            "cp": cp_coeffs, "CAS": cas,
        })

    log.info(f"Fluid filtering breakdown: {excluded}")
    return mols, total_found


# ==============================================================================
#  CoolProp-side cycle (same Bell-mode convention, own instrumented failures)
# ==============================================================================
def coolprop_binary_cycle(name1, name2, x1, x2):
    """
    Ideal vapour-compression cycle evaluated with CoolProp HEOS mixture
    properties, using EXACTLY the cycle definition applied to the PR-EOS
    side (dew@T_evap, bubble@T_cond, DT_SH/DT_SC/ETA_IS from bms module).

    Returns (state_dict, "OK") or (None, "<stage>: <reason>").
    """
    s = f"{name1}[{x1}]&{name2}[{x2}]"

    try:
        P_evap = CP.PropsSI("P", "T", T_EVAP, "Q", 1, s)
    except Exception as e:
        return None, f"dew@evap: {e}"

    try:
        P_cond = CP.PropsSI("P", "T", T_COND, "Q", 0, s)
    except Exception as e:
        return None, f"bubble@cond: {e}"

    if not (np.isfinite(P_evap) and np.isfinite(P_cond)) or P_evap <= 0 or P_cond <= 0:
        return None, "pressure: non-finite or non-positive"
    if P_cond <= P_evap:
        return None, "pressure: P_cond<=P_evap"

    T1 = T_EVAP + bms.DT_SH
    try:
        H1 = CP.PropsSI("H", "T", T1, "P", P_evap, s)
        S1 = CP.PropsSI("S", "T", T1, "P", P_evap, s)
    except Exception as e:
        return None, f"state1: {e}"

    def resid(T):
        return CP.PropsSI("S", "T", T, "P", P_cond, s) - S1

    T_lo, T_hi = T_COND, T_COND + 250.0
    try:
        f_lo, f_hi = resid(T_lo), resid(T_hi)
        tries = 0
        while f_lo * f_hi > 0 and tries < 10:
            T_hi += 250.0
            f_hi = resid(T_hi)
            tries += 1
        if f_lo * f_hi > 0:
            return None, "T2s_bracket: could not bracket isentropic root"
        T2s = brentq(resid, T_lo, T_hi, xtol=1e-6, rtol=1e-10, maxiter=200)
        H2s = CP.PropsSI("H", "T", T2s, "P", P_cond, s)
    except Exception as e:
        return None, f"T2s_solve: {e}"

    H2 = H1 + (H2s - H1) / bms.ETA_IS

    T3 = T_COND - bms.DT_SC
    try:
        H3 = CP.PropsSI("H", "T", T3, "P", P_cond, s)
    except Exception as e:
        return None, f"state3: {e}"

    W = H2 - H1
    Q = H1 - H3
    if not np.isfinite(W) or W <= 0:
        return None, f"W<=0: W={W}"
    cop = Q / W
    if not np.isfinite(cop) or cop < bms.COP_MIN or cop > bms.COP_MAX:
        # Same physical COP window bms.compute_cycle_mixture already enforces
        # on the PR-EOS side (COP_MIN..COP_MAX) — applied here too so neither
        # side can contribute a non-physical outlier (e.g. near-zero
        # compressor work at these T_evap/T_cond) that would dominate RMSE/R^2.
        return None, f"COP_invalid: cop={cop} outside [{bms.COP_MIN},{bms.COP_MAX}]"

    return {"COP": cop, "P_evap_bar": P_evap / 1e5, "P_cond_bar": P_cond / 1e5, "T2s_K": T2s}, "OK"


# ==============================================================================
#  Per-pair worker
# ==============================================================================
def process_pair(pair):
    """
    pair = (molA, molB) dicts. Runs both PR-EOS and CoolProp for all 9
    compositions. Every composition is independently exception-safe; a
    pair-level try/except is the last-resort catch-all so no exception can
    silently drop data.
    """
    molA, molB = pair
    success_rows, failure_rows = [], []
    try:
        props = [(molA["Tc"], molA["Pc"], molA["omega"]), (molB["Tc"], molB["Pc"], molB["omega"])]
        cps = [molA["cp"], molB["cp"]]

        try:
            CP.apply_simple_mixing_rule(molA["CAS"], molB["CAS"], "linear")
        except Exception as e:
            failure_rows = [{
                "Fluid1": molA["name"], "Fluid2": molB["name"], "x1": x1, "x2": round(1.0 - x1, 2),
                "Failure_Source": "CoolProp", "Failure_Reason": f"apply_simple_mixing_rule: {e}",
            } for x1 in X1_LIST]
            return molA["name"], molB["name"], success_rows, failure_rows

        for x1 in X1_LIST:
            x2 = round(1.0 - x1, 2)

            try:
                pr_state, pr_status = bms.compute_cycle_mixture(cps, [x1, x2], props, bms.KIJ)
            except Exception as e:
                pr_state, pr_status = None, f"exception:{type(e).__name__}:{e}"

            if pr_state is None:
                failure_rows.append({
                    "Fluid1": molA["name"], "Fluid2": molB["name"], "x1": x1, "x2": x2,
                    "Failure_Source": "PR-EOS", "Failure_Reason": pr_status,
                })
                continue

            try:
                cp_state, cp_status = coolprop_binary_cycle(molA["name"], molB["name"], x1, x2)
            except Exception as e:
                cp_state, cp_status = None, f"exception:{type(e).__name__}:{e}"

            if cp_state is None:
                failure_rows.append({
                    "Fluid1": molA["name"], "Fluid2": molB["name"], "x1": x1, "x2": x2,
                    "Failure_Source": "CoolProp", "Failure_Reason": cp_status,
                })
                continue

            cop_pr, cop_cp = pr_state["COP"], cp_state["COP"]
            abs_err = abs(cop_pr - cop_cp)
            pct_err = 100.0 * (cop_pr - cop_cp) / cop_cp

            success_rows.append({
                "Fluid1": molA["name"], "Fluid2": molB["name"], "x1": x1, "x2": x2,
                "Tc1": molA["Tc"], "Pc1": molA["Pc"], "omega1": molA["omega"],
                "Tc2": molB["Tc"], "Pc2": molB["Pc"], "omega2": molB["omega"],
                "COP_CoolProp": cop_cp, "COP_PR": cop_pr,
                "P_evap_CP": cp_state["P_evap_bar"], "P_cond_CP": cp_state["P_cond_bar"],
                "P_evap_PR": pr_state["P_evap_bar"], "P_cond_PR": pr_state["P_cond_bar"],
                "Absolute_Error": abs_err, "Percent_Error": pct_err,
            })
    except Exception as e:
        tb = traceback.format_exc()
        for x1 in X1_LIST:
            failure_rows.append({
                "Fluid1": molA["name"], "Fluid2": molB["name"], "x1": x1, "x2": round(1.0 - x1, 2),
                "Failure_Source": "Unhandled", "Failure_Reason": f"{type(e).__name__}:{e}",
            })
        log.debug(f"Unhandled exception on pair ({molA['name']}, {molB['name']}):\n{tb}")

    return molA["name"], molB["name"], success_rows, failure_rows


# ==============================================================================
#  Checkpoint I/O
# ==============================================================================
def load_checkpoint():
    if not (os.path.exists(CHECKPOINT_JSON) and os.path.exists(RESULTS_CSV) and os.path.exists(FAILED_CSV)):
        return set(), [], []
    try:
        with open(CHECKPOINT_JSON) as f:
            ck = json.load(f)
        completed = {tuple(p) for p in ck.get("completed_pairs", [])}
        success_rows = pd.read_csv(RESULTS_CSV).to_dict("records")
        failure_rows = pd.read_csv(FAILED_CSV).to_dict("records")
        return completed, success_rows, failure_rows
    except Exception as e:
        log.warning(f"Could not load checkpoint ({e}); starting fresh.")
        return set(), [], []


def save_checkpoint(completed_pairs, success_rows, failure_rows):
    pd.DataFrame(success_rows, columns=RESULTS_COLUMNS).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(failure_rows, columns=FAILED_COLUMNS).to_csv(FAILED_CSV, index=False)
    with open(CHECKPOINT_JSON, "w") as f:
        json.dump({
            "completed_pairs": [list(p) for p in completed_pairs],
            "n_success": len(success_rows),
            "n_failed": len(failure_rows),
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)


# ==============================================================================
#  Statistics
# ==============================================================================
def compute_stats(df):
    err_pct = df["Percent_Error"].to_numpy()
    abs_err_pct = np.abs(err_pct)
    cp = df["COP_CoolProp"].to_numpy()
    pr = df["COP_PR"].to_numpy()

    ss_res = np.sum((pr - cp) ** 2)
    ss_tot = np.sum((cp - cp.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((pr - cp) ** 2)))

    return {
        "mean_error_pct": float(np.mean(err_pct)),
        "median_error_pct": float(np.median(err_pct)),
        "mape_pct": float(np.mean(abs_err_pct)),
        "rmse": rmse,
        "r2": float(r2),
        "min_error_pct": float(np.min(abs_err_pct)),
        "max_error_pct": float(np.max(abs_err_pct)),
        "p95_error_pct": float(np.percentile(abs_err_pct, 95)),
    }


# ==============================================================================
#  Plots  (styling per the dataviz skill: single accent hue, thin marks,
#  recessive gridlines, direct annotations, legend only when >=2 series)
# ==============================================================================
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def plot_scatter(df, stats):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=150, facecolor=SURFACE)
    _style_axes(ax)

    cp, pr = df["COP_CoolProp"].to_numpy(), df["COP_PR"].to_numpy()
    lims = [min(cp.min(), pr.min()) * 0.95, max(cp.max(), pr.max()) * 1.05]

    ax.plot(lims, lims, "--", color=BASELINE, linewidth=1.5, zorder=1, label="y = x")
    ax.scatter(cp, pr, s=10, color=SERIES_1, alpha=0.35, linewidths=0, zorder=2)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("COP (CoolProp HEOS)")
    ax.set_ylabel("COP (PR-EOS)")
    ax.set_title("PR-EOS vs CoolProp — Binary Mixture COP")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left")

    text = (f"R² = {stats['r2']:.4f}\n"
            f"MAPE = {stats['mape_pct']:.2f}%\n"
            f"N = {len(df):,}")
    ax.text(0.98, 0.03, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=INK_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID))

    fig.tight_layout()
    fig.savefig(SCATTER_PNG, facecolor=SURFACE)
    plt.close(fig)


def plot_histogram(df, stats):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150, facecolor=SURFACE)
    _style_axes(ax)

    err = df["Percent_Error"].to_numpy()
    ax.hist(err, bins=60, color=SERIES_1, alpha=0.85, edgecolor=SURFACE, linewidth=0.3, zorder=2)
    ax.axvline(0.0, color=BASELINE, linewidth=1.5, linestyle="--", zorder=1)
    ax.axvline(stats["mean_error_pct"], color=SERIES_2, linewidth=1.8, zorder=3,
               label=f"mean = {stats['mean_error_pct']:.2f}%")

    ax.set_xlabel("Percent Error  (COP_PR − COP_CoolProp) / COP_CoolProp × 100")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of PR-EOS COP Error vs CoolProp")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper right")

    fig.tight_layout()
    fig.savefig(HIST_PNG, facecolor=SURFACE)
    plt.close(fig)


def plot_error_vs_composition(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150, facecolor=SURFACE)
    _style_axes(ax)

    ax.scatter(df["x1"], df["Percent_Error"], s=8, color=SERIES_1, alpha=0.12,
               linewidths=0, zorder=2, label="Pairs")

    med = df.groupby("x1")["Percent_Error"].median().reindex(X1_LIST)
    ax.plot(med.index, med.values, color=SERIES_2, linewidth=2.2, marker="o",
            markersize=5, zorder=3, label="Median per composition")

    ax.axhline(0.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlabel("x1 (mole fraction of Fluid1)")
    ax.set_ylabel("Percent Error (%)")
    ax.set_title("PR-EOS COP Error vs Composition")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="best")

    fig.tight_layout()
    fig.savefig(COMPOSITION_PNG, facecolor=SURFACE)
    plt.close(fig)


# ==============================================================================
#  Main
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Validate PR-EOS binary-mixture COP against CoolProp HEOS.")
    p.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: all cores).")
    p.add_argument("--max-fluids", type=int, default=None, help="Debug: cap the usable-fluid list to N fluids.")
    p.add_argument("--fresh", action="store_true", help="Ignore any existing checkpoint and start over.")
    p.add_argument("--checkpoint-every-pairs", type=int, default=CHECKPOINT_EVERY_PAIRS)
    p.add_argument("--checkpoint-every-seconds", type=float, default=CHECKPOINT_EVERY_SECONDS)
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 78)
    log.info("  PR-EOS vs CoolProp — Binary Mixture COP Validation")
    log.info(f"  T_evap={T_EVAP_C}C  T_cond={T_COND_C}C  DT_sh={DT_SH}K  DT_sc={DT_SC}K  eta_is={ETA_IS}")
    log.info(f"  Compositions x1 = {X1_LIST}")
    log.info("=" * 78)

    mols, total_found = build_usable_fluids()
    if args.max_fluids:
        mols = mols[:args.max_fluids]

    total_usable = len(mols)
    log.info(f"Total CoolProp fluids found : {total_found}")
    log.info(f"Total usable fluids         : {total_usable}")

    all_pairs = list(itertools.combinations(mols, 2))
    total_pairs = len(all_pairs)
    total_calcs = total_pairs * len(X1_LIST)
    log.info(f"Total unique pairs          : {total_pairs}")
    log.info(f"Total calculations          : {total_pairs} x {len(X1_LIST)} = {total_calcs}")

    if args.fresh:
        for fpath in (RESULTS_CSV, FAILED_CSV, CHECKPOINT_JSON):
            if os.path.exists(fpath):
                os.remove(fpath)
        completed, success_rows, failure_rows = set(), [], []
    else:
        completed, success_rows, failure_rows = load_checkpoint()
        if completed:
            log.info(f"Resuming: {len(completed)}/{total_pairs} pairs already completed "
                      f"({len(success_rows)} rows succeeded, {len(failure_rows)} failed previously).")

    remaining_pairs = [pr for pr in all_pairs if (pr[0]["name"], pr[1]["name"]) not in completed]
    n_workers = args.workers or os.cpu_count() or 1
    log.info(f"Remaining pairs to process  : {len(remaining_pairs)}  (workers={n_workers})")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    pairs_since_checkpoint = 0
    last_checkpoint_time = time.time()
    t0 = time.time()

    try:
        if remaining_pairs:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                with tqdm(total=total_pairs, initial=len(completed), desc="Pairs", unit="pair") as pbar:
                    for nameA, nameB, srows, frows in pool.imap_unordered(process_pair, remaining_pairs):
                        success_rows.extend(srows)
                        failure_rows.extend(frows)
                        completed.add((nameA, nameB))
                        pbar.update(1)
                        pairs_since_checkpoint += 1

                        if frows:
                            for r in frows:
                                log.debug(f"FAIL {r['Fluid1']}/{r['Fluid2']} x1={r['x1']} "
                                          f"[{r['Failure_Source']}] {r['Failure_Reason']}")

                        now = time.time()
                        if (pairs_since_checkpoint >= args.checkpoint_every_pairs
                                or now - last_checkpoint_time >= args.checkpoint_every_seconds):
                            save_checkpoint(completed, success_rows, failure_rows)
                            log.info(f"[checkpoint] {len(completed)}/{total_pairs} pairs saved "
                                     f"({len(success_rows)} ok, {len(failure_rows)} failed)")
                            pairs_since_checkpoint = 0
                            last_checkpoint_time = now

        save_checkpoint(completed, success_rows, failure_rows)

    except KeyboardInterrupt:
        log.warning("Interrupted by user — saving checkpoint before exit.")
        save_checkpoint(completed, success_rows, failure_rows)
        log.warning("Progress saved. Re-run the same command to resume.")
        sys.exit(1)
    except Exception as e:
        log.critical(f"Unhandled top-level exception: {e}\n{traceback.format_exc()}")
        save_checkpoint(completed, success_rows, failure_rows)
        log.critical("Progress saved despite the error. Re-run the same command to resume.")
        raise

    elapsed = time.time() - t0
    log.info(f"Sweep finished in {elapsed/60:.1f} min")

    # --- Final outputs -----------------------------------------------------
    df = pd.DataFrame(success_rows, columns=RESULTS_COLUMNS)
    fdf = pd.DataFrame(failure_rows, columns=FAILED_COLUMNS)
    df.to_csv(RESULTS_CSV, index=False)
    fdf.to_csv(FAILED_CSV, index=False)

    n_attempted = total_calcs
    n_success = len(df)
    n_failed = len(fdf)
    success_rate = 100.0 * n_success / n_attempted if n_attempted else float("nan")

    if n_success == 0:
        log.critical("No successful calculations — skipping statistics and plots.")
        return

    stats = compute_stats(df)

    plot_scatter(df, stats)
    plot_histogram(df, stats)
    plot_error_vs_composition(df)

    log.info("=" * 78)
    log.info("  SUMMARY")
    log.info("=" * 78)
    log.info(f"  Total fluids            : {total_usable}")
    log.info(f"  Total pairs             : {total_pairs}")
    log.info(f"  Total calculations      : {n_attempted}")
    log.info(f"  Total attempted         : {n_attempted}")
    log.info(f"  Total successful        : {n_success}")
    log.info(f"  Total failed            : {n_failed}")
    log.info(f"  Success rate            : {success_rate:.2f}%")
    log.info(f"  Mean Error (%)          : {stats['mean_error_pct']:.4f}")
    log.info(f"  Median Error (%)        : {stats['median_error_pct']:.4f}")
    log.info(f"  MAPE (%)                : {stats['mape_pct']:.4f}")
    log.info(f"  RMSE (COP units)        : {stats['rmse']:.4f}")
    log.info(f"  R²                     : {stats['r2']:.4f}")
    log.info(f"  Minimum |Error| (%)     : {stats['min_error_pct']:.4f}")
    log.info(f"  Maximum |Error| (%)     : {stats['max_error_pct']:.4f}")
    log.info(f"  95th Percentile |Error| : {stats['p95_error_pct']:.4f}")
    log.info("-" * 78)
    log.info(f"  validation_results.csv      -> {RESULTS_CSV}")
    log.info(f"  validation_failed_pairs.csv -> {FAILED_CSV}")
    log.info(f"  cop_scatter.png             -> {SCATTER_PNG}")
    log.info(f"  error_histogram.png         -> {HIST_PNG}")
    log.info(f"  error_vs_composition.png    -> {COMPOSITION_PNG}")
    log.info(f"  validation_run.log          -> {LOG_FILE}")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
