"""
================================================================================
 Binary Refrigerant Mixture Dataset Generator (PR-EOS)
================================================================================

PURPOSE
-------
Computational screening / dataset generation for binary refrigerant mixtures.
For every unique pair of molecules in the input database and every composition
on a configurable grid, this program runs an idealised vapour-compression-cycle
calculation and records COP, volumetric capacity (Qvol), blended GWP100 and
compressor operating conditions. The output is a large, structured dataset
intended for downstream machine-learning / multi-objective-optimisation /
inverse-design work — that downstream modelling is explicitly OUT OF SCOPE for
this script.

This file is a refactor/unification of two prior scripts:
    - binary_mixture_cop_sweep.py   (validated PR-EOS thermodynamic engine)
    - generate_binary_dataset.py    (dataset-generation infrastructure)
Both are preserved unmodified in legacy_backup/. The scientific engine below
(PR-EOS, mixing rule, VLE solvers, departure functions, cycle convention) is
carried over UNCHANGED from binary_mixture_cop_sweep.py — see "SCIENTIFIC
METHOD" below for exactly what "unchanged" covers, and what got added purely
as infrastructure (caching, categorised failures, dual output, checkpoints).

SCIENTIFIC METHOD
------------------
Equation of state
    Peng-Robinson cubic EOS with the standard alpha/kappa formulation
    (kappa = 0.37464 + 1.54226*omega - 0.26992*omega^2) and van der Waals
    one-fluid mixing rules:
        a_mix   = sum_i sum_j x_i x_j sqrt(a_i a_j) (1 - k_ij)
        b_mix   = sum_i x_i b_i
    k_ij defaults to 0.0 (no binary-interaction data available for this
    molecule set). The cubic-Z equation is solved with numpy.roots and the
    largest/smallest real roots > B are taken as the vapour/liquid Z.

Vapour-liquid equilibrium
    Bubble-point and dew-point pressures are found by successive-substitution
    K-value iteration using fugacity coefficients from the PR-EOS (Wilson
    correlation initial guess). Saturation temperatures at a target pressure
    are found by bisecting the corresponding bubble/dew pressure solver.

Ideal-gas heat capacity / departure functions
    Cp_ig(T) = a + b*T + c*T^2 + d*T^3 + e*T^4  [J/mol/K], integrated
    analytically for enthalpy and entropy; PR-EOS departure enthalpy/entropy
    are added on top (standard PR-EOS departure-function expressions).

Cycle convention (Bell et al. 2019 "simplified cycle", preserved exactly)
    P_evap = DEW-point pressure of the feed composition z at T_evap
    P_cond = BUBBLE-point pressure of the feed composition z at T_cond
    State 1 (compressor suction)   : T_evap + superheat, P_evap
    State 2s (isentropic outlet)   : S2s = S1, at P_cond (superheated OR
                                      wet -- both regimes handled explicitly)
    State 2 (real outlet)          : H2 = H1 + (H2s - H1) / eta_is
    State 3 (condenser outlet)     : T_cond - subcooling, P_cond
    State 4 (evaporator inlet)     : H4 = H3   (isenthalpic throttling)
    W   = H2 - H1          (compressor work)
    Q   = H1 - H4           (refrigeration effect)
    COP = Q / W
    No pressure drop is modelled (documented limitation carried over as-is).
    All state enthalpies use the fixed OVERALL feed composition z (sealed
    single-loop cycle -- no separation of vapour/liquid phases between
    states).

Qvol (volumetric refrigeration capacity), Bell et al. (2019) definition
    Qvol = |h1 - h4| * rho_suction   where rho_suction = 1/v1 at state 1
    (evaporator outlet / compressor suction), v1 = Z1*R*T1/P_evap.

GWP_mix definition (explicit -- preserved from generate_binary_dataset.py)
    GWP_mix = x1*GWP_1 + x2*GWP_2
    This is a simple MOLE-FRACTION-WEIGHTED LINEAR BLEND of GWP100 values.
    It is a convenient screening metric, NOT an experimentally validated
    mixture-GWP model (real blend GWP depends on atmospheric lifetime,
    radiative efficiency and composition-dependent decomposition chemistry
    of the actual blend, none of which is modelled here).

Composition convention
    x_1 = mole fraction of molecule_1, x_2 = 1 - x_1, x_1 + x_2 = 1 exactly
    (grid built with Decimal arithmetic so no float noise like
    0.30000000000000004 reaches the output).

Pure-component COP_1_ref / COP_2_ref / Qvol_1_ref / Qvol_2_ref -- IMPORTANT CAVEAT
    mclinden_predicted_Tc_Pc_omega.csv already ships pure-component COP and
    Qvol_MJ_m3 columns. These were checked against this repository's own
    PR-EOS engine (bms.pure_cop-equivalent) for several molecules and differ
    by ~25% -- they are LITERATURE / EXTERNAL REFERENCE values, not values
    produced by this PR-EOS engine (almost certainly computed with a
    higher-fidelity EOS / REFPROP in the original McLinden-style screening
    study). Per explicit user decision (2026-08-13) this script uses those
    literature values directly, exposed in the detailed output as
    COP_1_ref/COP_2_ref/Qvol_1_ref/Qvol_2_ref -- the "_ref" suffix is
    deliberate and MUST be preserved: it marks these as externally-sourced
    reference data, not PR-EOS output, so they are never mistaken for the
    pure-fluid limit of COP_mix/Qvol_mix. CONSEQUENCE: COP_mix / Qvol_mix
    (always computed by THIS PR-EOS engine) do NOT asymptote to
    COP_1_ref/COP_2_ref or Qvol_1_ref/Qvol_2_ref as x1 -> 1 or x1 -> 0.
    These _ref columns are carried through for convenience/context only and
    are deliberately EXCLUDED from the ML dataset (see ML_COLUMNS) so they
    are never mistaken for a training target -- the intended regression
    targets are COP_mix, Qvol_mix and GWP_mix, all computed by this engine.
    Unit note: the input file reports Qvol in MJ/m3; this engine's Qvol_mix
    is natively in kJ/m3. Qvol_1_ref/Qvol_2_ref are converted (x1000) to
    kJ/m3 in the output so all three Qvol columns share one unit -- this is
    a unit conversion of the literature numbers, not a re-derivation of them.

Checkpoint / resume
    Checkpointing is PAIR-LEVEL: a pair is only marked complete once every
    composition in the grid has been processed for it, so resuming never
    reprocesses a partially-done pair and never skips a pair with missing
    compositions. The checkpoint records file hashes, the molecule-set hash,
    a hash of the scientific configuration, and the composition-grid
    definition; if any of these change incompatibly the checkpoint is
    rejected and generation starts fresh (with a clear log message).

CPU / GPU execution
    This engine's root-finding (numpy.roots for the cubic EOS, scipy.optimize
    .brentq for the isentropic-outlet search, successive-substitution VLE
    iteration) is inherently scalar and per-(pair, composition) -- it is not
    safely vectorisable onto a GPU without rewriting the numerical method,
    which this project deliberately avoids to protect scientific correctness.
    BACKEND="auto" therefore detects GPU hardware (for honest reporting) but
    always executes the thermodynamic solve on CPU multiprocessing. This is
    not a placeholder -- it is the documented, correctness-driven choice for
    this class of numerical problem.

Output datasets
    binary_mixture_detailed_part_NNNN.csv  -- full audit/traceability dataset
    binary_mixture_ml_part_NNNN.csv        -- compact ML/screening subset
                                               (derived from the same in-memory
                                               row as the detailed dataset,
                                               never recomputed separately)
    failed_cases.csv                       -- every failed calculation, with
                                               a normalised failure_stage
    checkpoint.json, dataset_summary.json, dataset_manifest.json,
    generation_run.log
"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from decimal import Decimal
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

__version__ = "1.0.0"

# ==============================================================================
#  USER CONFIGURATION
# ==============================================================================
#  Everything a researcher may reasonably want to change lives here. Nothing
#  scientifically or operationally important is scattered elsewhere.
# ==============================================================================

# ---------- INPUT FILES --------------------------------------------------------
INPUT_PROPERTY_FILE = "mclinden_predicted_Tc_Pc_omega.csv"   # Tc, Pc, omega, GWP, SMILES, pure COP/Qvol
INPUT_CP_FILE        = "mclinden_cp_values.csv"                # ideal-gas Cp polynomial coefficients

# ---------- INPUT COLUMNS ------------------------------------------------------
COL_NAME   = "IUPAC_Name"          # merge key, present in both files
COL_TC     = "Tc_K"                # critical temperature [K]  (NOT "Tc_predicted (K)")
COL_PC     = "Pc_bar"              # critical pressure column
PC_UNIT    = "bar"                 # unit of COL_PC: "bar" or "Pa"
COL_OMEGA  = "omega"               # acentric factor
COL_GWP    = "GWP100"              # 100-year GWP, taken from INPUT_PROPERTY_FILE
COL_SMILES = "SMILES"              # canonical SMILES, taken from INPUT_PROPERTY_FILE
COL_COP_PURE  = "COP"              # pure-component literature COP (see caveat in module docstring)
COL_QVOL_PURE = "Qvol_MJ_m3"       # pure-component literature Qvol [MJ/m3] (converted to kJ/m3 on output)
CP_COLS    = ["a", "b", "c", "d", "e"]   # Cp_ig(T) = a + b T + c T^2 + d T^3 + e T^4  [J/mol/K], from INPUT_CP_FILE

# ---------- CYCLE ----------------------------------------------------------------
T_EVAP_C = 10.0     # evaporator temperature [C]
T_COND_C = 40.0     # condenser temperature  [C]
DT_SH    = 0.0      # evaporator superheat   [K]
DT_SC    = 0.0      # condenser subcooling   [K]
ETA_IS   = 1.0      # isentropic compressor efficiency
KIJ      = 0.0      # binary interaction parameter (no data available -> 0)

# ---------- COMPOSITION ------------------------------------------------------------
X1_START = 0.10     # mole fraction of molecule_1, grid start (inclusive)
X1_END   = 0.90     # grid end (inclusive)
X1_STEP  = 0.01

# ---------- PHYSICAL FILTERS / GUARDS --------------------------------------------
# Pure components with Tc <= this can't be condensed at T_cond as a pure fluid;
# pairing them produces non-physical cycles, so they are dropped before pairing.
MIN_COMPONENT_TC = T_COND_C + 273.15     # = 313.15 K by default; 0.0 keeps all
DROP_BLENDS      = True                  # drop rows whose name marks a blend/mixture
T2S_MAX_K        = 550.0                 # reject cycles with discharge temp above this
P_RATIO_MAX      = 25.0                  # reject P_cond/P_evap above this (spurious VLE)
COP_MIN, COP_MAX = 1.0, 25.0             # physical COP window at these cycle temps

# ---------- OUTPUT -----------------------------------------------------------------
OUTPUT_DIR              = "binary_dataset_output"
DETAILED_PREFIX         = "binary_mixture_detailed_part"
ML_PREFIX                = "binary_mixture_ml_part"
FAILED_CASES_FILE       = "failed_cases.csv"
CHECKPOINT_FILE         = "checkpoint.json"
SUMMARY_FILE            = "dataset_summary.json"
MANIFEST_FILE           = "dataset_manifest.json"
LOG_FILE                = "generation_run.log"

# ---------- PERFORMANCE -------------------------------------------------------------
BACKEND          = "auto"     # "auto" | "cpu" | "gpu"  (see module docstring: GPU is
                               # detected/reported but the solve always runs on CPU)
N_WORKERS        = None       # None -> os.cpu_count()
CHUNK_SIZE_ROWS  = 1_000_000  # rows per output part file (detailed AND ml, independently)

# ---------- CHECKPOINT ---------------------------------------------------------------
ENABLE_CHECKPOINTS       = True
CHECKPOINT_EVERY_PAIRS   = 25
CHECKPOINT_EVERY_SECONDS = 120.0
OVERWRITE_EXISTING       = False   # True -> ignore/clear existing output + checkpoint

# ---------- FAILURE LOGGING -----------------------------------------------------------
SAVE_FAILED_CASES = True

# ---------- PROGRESS ------------------------------------------------------------------
PROGRESS_EVERY_SECONDS = 15.0

# ---------- TESTING / DEBUG -----------------------------------------------------------
DEBUG_MAX_MOLECULES = None    # cap molecule list to N (debug runs); None = no cap
DEBUG_MAX_PAIRS     = None    # cap number of pairs processed; None = no cap


# ==============================================================================
#  Derived constants (do not edit below this line for normal use)
# ==============================================================================
T_EVAP = T_EVAP_C + 273.15
T_COND = T_COND_C + 273.15
R      = 8.314462618
SQRT2  = np.sqrt(2.0)
P_REF  = 101325.0
T_REF  = T_EVAP                     # any consistent reference cancels in the cycle


def _decimals_for_step(step):
    s = f"{step:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def build_x1_grid(start, end, step):
    """
    Decimal-safe composition grid: x1 = start, start+step, ..., end (inclusive).
    Avoids float accumulation artifacts (e.g. 0.30000000000000004) by doing the
    stepping in Decimal and converting to float only once per grid point.
    Returns (list_of_(x1, x2)_floats, decimals_for_rounding).
    """
    start_d, end_d, step_d = Decimal(str(start)), Decimal(str(end)), Decimal(str(step))
    n_steps = int(((end_d - start_d) / step_d).to_integral_value())
    decimals = _decimals_for_step(step)
    grid = []
    for i in range(n_steps + 1):
        x1_d = start_d + i * step_d
        x1 = float(x1_d)
        x2 = float(Decimal("1") - x1_d)
        grid.append((round(x1, decimals), round(x2, decimals)))
    return grid, decimals


X1_GRID, X1_DECIMALS = build_x1_grid(X1_START, X1_END, X1_STEP)


# ==============================================================================
#  PR-EOS thermodynamic engine
#  -- carried over unchanged from binary_mixture_cop_sweep.py --
#  Only change from the original: pr_pure_params is memoised (functools.lru_cache)
#  since within one binary pair's composition sweep it is repeatedly evaluated at
#  the SAME handful of temperatures (T_evap, T_cond, T1, T3, and the temperatures
#  explored by the root solvers). Caching is a pure performance optimisation on a
#  deterministic pure function of (T, Tc, Pc, omega) -- it does not change any
#  computed value.
# ==============================================================================
@lru_cache(maxsize=200_000)
def pr_pure_params(T, Tc, Pc, omega):
    """Return a(T), b, da/dT for one pure component (PR-EOS, standard alpha/kappa)."""
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    Tr    = T / Tc
    sq    = 1.0 + kappa * (1.0 - np.sqrt(Tr))
    alpha = sq * sq
    ac    = 0.45724 * R ** 2 * Tc ** 2 / Pc
    a     = ac * alpha
    b     = 0.07780 * R * Tc / Pc
    dadT  = ac * (-kappa * np.sqrt(alpha / (T * Tc)))
    return a, b, dadT


def mix_params(T, x, props, kij=0.0):
    """van der Waals one-fluid mixing. props = list of (Tc, Pc, omega)."""
    n = len(x)
    a_i = np.empty(n); b_i = np.empty(n); da_i = np.empty(n)
    for k, (Tc, Pc, om) in enumerate(props):
        a_i[k], b_i[k], da_i[k] = pr_pure_params(T, Tc, Pc, om)
    a_mix = 0.0; dadT_mix = 0.0
    for i in range(n):
        for j in range(n):
            aij = np.sqrt(a_i[i] * a_i[j]) * (1.0 - kij)
            a_mix += x[i] * x[j] * aij
            if a_i[i] > 0 and a_i[j] > 0:
                daij = (1.0 - kij) * 0.5 * (da_i[i] * np.sqrt(a_i[j] / a_i[i]) +
                                            da_i[j] * np.sqrt(a_i[i] / a_i[j]))
            else:
                daij = 0.0
            dadT_mix += x[i] * x[j] * daij
    b_mix = float(np.dot(x, b_i))
    return a_mix, b_mix, dadT_mix, a_i, b_i


def compute_AB(T, P, a, b):
    return a * P / (R ** 2 * T ** 2), b * P / (R * T)


def solve_cubic_Z(A, B):
    """Return (Z_vapour, Z_liquid) real roots > B, or None."""
    coeffs = [1.0, -(1.0 - B), (A - 3.0 * B ** 2 - 2.0 * B), -(A * B - B ** 2 - B ** 3)]
    roots = np.roots(coeffs)
    real = sorted([r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B], reverse=True)
    if len(real) >= 2:
        return real[0], real[-1]
    if len(real) == 1:
        return real[0], real[0]
    return None


def log_term(Z, B):
    num = Z + (1.0 + SQRT2) * B
    den = Z + (1.0 - SQRT2) * B
    if num <= 0 or den <= 0:
        return 0.0
    return np.log(num / den)


def ln_phi_all(Z, A, B, x, a_i, b_i, a_mix, b_mix, kij=0.0):
    n = len(x)
    lt = log_term(Z, B)
    out = []
    for i in range(n):
        sum_xj_aij = sum(x[j] * np.sqrt(a_i[i] * a_i[j]) * (1.0 - kij) for j in range(n))
        out.append(b_i[i] / b_mix * (Z - 1.0) - np.log(max(Z - B, 1e-15))
                   - A / (2.0 * SQRT2 * B) * (2.0 * sum_xj_aij / a_mix - b_i[i] / b_mix) * lt)
    return out


def _wilson_K(T, props):
    return [max((Pc / 1e6) * np.exp(5.373 * (1.0 + om) * (1.0 - Tc / T)), 1e-12)
            for (Tc, Pc, om) in props]


def bubble_point_P(T, z, props, kij=0.0, tol=1e-7, max_iter=300):
    """Bubble pressure and equilibrium vapour composition for liquid z at T."""
    n = len(z); x = list(z)
    K = _wilson_K(T, props)
    P = max(1.0 / sum(x[i] / K[i] for i in range(n)) * 1e6, 1e4)
    for _ in range(max_iter):
        y = [K[i] * x[i] for i in range(n)]; S = sum(y); y = [v / S for v in y]
        aL, bL, _, ai, bi = mix_params(T, x, props, kij)
        aV, bV, _, _, _   = mix_params(T, y, props, kij)
        AL, BL = compute_AB(T, P, aL, bL); AV, BV = compute_AB(T, P, aV, bV)
        rL = solve_cubic_Z(AL, BL); rV = solve_cubic_Z(AV, BV)
        if rL is None or rV is None:
            return None, None
        lpL = ln_phi_all(rL[-1], AL, BL, x, ai, bi, aL, bL, kij)
        lpV = ln_phi_all(rV[0],  AV, BV, y, ai, bi, aV, bV, kij)
        Kn = [np.exp(lpL[i] - lpV[i]) for i in range(n)]
        Sn = sum(Kn[i] * x[i] for i in range(n)); Pn = P * Sn
        if abs(Sn - 1.0) < tol and all(abs(Kn[i] - K[i]) < tol for i in range(n)):
            return Pn, [Kn[i] * x[i] / Sn for i in range(n)]
        K = Kn; P = Pn
        if P <= 0 or not np.isfinite(P) or P > 1e9:
            return None, None
    return None, None


_PSAT_CACHE = {}
def _pure_psat(T, prop, kij=0.0):
    """Memoized pure-component saturation pressure (bubble solver, n=1)."""
    key = (round(T, 4), round(prop[0], 5), round(prop[1], 2), round(prop[2], 6))
    v = _PSAT_CACHE.get(key)
    if v is None:
        P, _ = bubble_point_P(T, [1.0], [prop], kij)
        _PSAT_CACHE[key] = v = P
    return v


def dew_point_P(T, y, props, kij=0.0, tol=1e-8, max_iter=120):
    """
    Dew-point pressure for vapour composition y at T -- fast successive
    substitution with a Raoult-law initial guess and physical clamping.

    The clamp to [0.6*min(Psat), 1.4*max(Psat)] keeps the iteration from running
    off to a spurious high-pressure root (the failure mode of the naive scheme at
    low T), while a good initial guess gives convergence in a handful of steps.
    Returns (P_dew, x_liquid) or (None, None).
    """
    n = len(y)
    psats = [_pure_psat(T, props[i], kij) for i in range(n)]
    if any(p is None or p <= 0 for p in psats):
        return None, None
    P_lo, P_hi = 0.6 * min(psats), 1.4 * max(psats)

    P = 1.0 / sum(y[i] / psats[i] for i in range(n))
    P = min(max(P, P_lo), P_hi)
    K = [psats[i] / P for i in range(n)]

    for _ in range(max_iter):
        x = [y[i] / K[i] for i in range(n)]; S = sum(x); x = [v / S for v in x]
        aL, bL, _, ai, bi = mix_params(T, x, props, kij)
        aV, bV, _, _, _   = mix_params(T, y, props, kij)
        AL, BL = compute_AB(T, P, aL, bL); AV, BV = compute_AB(T, P, aV, bV)
        rL = solve_cubic_Z(AL, BL); rV = solve_cubic_Z(AV, BV)
        if rL is None or rV is None:
            return None, None
        lpL = ln_phi_all(rL[-1], AL, BL, x, ai, bi, aL, bL, kij)
        lpV = ln_phi_all(rV[0],  AV, BV, y, ai, bi, aV, bV, kij)
        Kn = [np.exp(lpL[i] - lpV[i]) for i in range(n)]
        Sn = sum(y[i] / Kn[i] for i in range(n))
        Pn = min(max(P / Sn, P_lo), P_hi)
        if abs(Sn - 1.0) < tol and all(abs(Kn[i] - K[i]) < 1e-8 for i in range(n)):
            xl = [y[i] / Kn[i] for i in range(n)]; s = sum(xl)
            return Pn, [v / s for v in xl]
        K = Kn; P = Pn
    if abs(Sn - 1.0) < 1e-4:
        xl = [y[i] / K[i] for i in range(n)]; s = sum(xl)
        return P, [v / s for v in xl]
    return None, None


def _sat_T_at_P(kind, w, P_target, props, kij=0.0):
    """Temperature at which composition w has bubble/dew pressure = P_target."""
    solver = bubble_point_P if kind == "bubble" else dew_point_P
    def f(T):
        P, _ = solver(T, w, props, kij)
        return None if P is None else P - P_target

    def bisect(a, fa, b, fb):
        for _ in range(60):
            mmid = 0.5 * (a + b)
            fm = f(mmid)
            if fm is None:
                mmid = 0.5 * (a + mmid); fm = f(mmid)
                if fm is None:
                    return None
            if abs(b - a) < 1e-3:
                return mmid
            if fa * fm <= 0:
                b, fb = mmid, fm
            else:
                a, fa = mmid, fm
        return 0.5 * (a + b)

    T_hi = 1.2 * max(p[0] for p in props)
    T_lo = 150.0
    f_lo = f(T_lo); f_hi = f(T_hi)
    if f_lo is not None and f_hi is not None and f_lo * f_hi <= 0:
        return bisect(T_lo, f_lo, T_hi, f_hi)
    prev_T, prev_f = None, None
    for T in np.linspace(160.0, T_hi, 60):
        fv = f(T)
        if fv is None:
            prev_T, prev_f = None, None; continue
        if prev_f is not None and prev_f * fv <= 0:
            return bisect(prev_T, prev_f, T, fv)
        prev_T, prev_f = T, fv
    return None


def cp0_integ_H(cp, T, Tref):
    a, b, c, d, e = cp
    return (a * (T - Tref) + b / 2 * (T ** 2 - Tref ** 2) + c / 3 * (T ** 3 - Tref ** 3)
            + d / 4 * (T ** 4 - Tref ** 4) + e / 5 * (T ** 5 - Tref ** 5))


def cp0_integ_S(cp, T, Tref):
    a, b, c, d, e = cp
    return (a * np.log(T / Tref) + b * (T - Tref) + c / 2 * (T ** 2 - Tref ** 2)
            + d / 3 * (T ** 3 - Tref ** 3) + e / 4 * (T ** 4 - Tref ** 4))


def H_dep(T, P, Z, a_mix, b_mix, dadT_mix):
    _, B = compute_AB(T, P, a_mix, b_mix)
    return R * T * (Z - 1.0) + (T * dadT_mix - a_mix) / (2.0 * SQRT2 * b_mix) * log_term(Z, B)


def S_dep(T, P, Z, a_mix, b_mix, dadT_mix):
    _, B = compute_AB(T, P, a_mix, b_mix)
    return R * np.log(max(Z - B, 1e-15)) + dadT_mix / (2.0 * SQRT2 * b_mix) * log_term(Z, B)


def total_H(cps, x, T, P, Z, props, kij=0.0):
    a_mix, b_mix, dadT_mix, *_ = mix_params(T, x, props, kij)
    H_ig = sum(x[i] * cp0_integ_H(cps[i], T, T_REF) for i in range(len(x)))
    return H_ig + H_dep(T, P, Z, a_mix, b_mix, dadT_mix)


def total_S(cps, x, T, P, Z, props, kij=0.0):
    a_mix, b_mix, dadT_mix, *_ = mix_params(T, x, props, kij)
    S = 0.0
    for i in range(len(x)):
        s_i = cp0_integ_S(cps[i], T, T_REF) - R * np.log(P / P_REF)
        s_i += -R * np.log(x[i]) if x[i] > 0 else 0.0
        S += x[i] * s_i
    return S + S_dep(T, P, Z, a_mix, b_mix, dadT_mix)


def find_T2s_mix(cps, S_target, w, P2, props, kij=0.0):
    """
    Isentropic outlet at pressure P2, vapour composition w, entropy S_target.
    Returns (T2s, H2s, regime).  regime in {'superheated','wet'}.
    """
    T_dew = _sat_T_at_P("dew", w, P2, props, kij)

    def S_vap_at(T):
        a_m, b_m, _, *_ = mix_params(T, w, props, kij)
        A, B = compute_AB(T, P2, a_m, b_m); r = solve_cubic_Z(A, B)
        if r is None:
            return None
        return total_S(cps, w, T, P2, r[0], props, kij)

    def H_vap_at(T):
        a_m, b_m, _, *_ = mix_params(T, w, props, kij)
        A, B = compute_AB(T, P2, a_m, b_m); r = solve_cubic_Z(A, B)
        return total_H(cps, w, T, P2, r[0], props, kij)

    if T_dew is not None:
        S_dew = S_vap_at(T_dew)
    else:
        S_dew = None

    if S_dew is not None and S_target >= S_dew - 1e-9:
        Tc_max = max(p[0] for p in props)
        def resid(T):
            s = S_vap_at(T)
            return 1e9 if s is None else s - S_target
        T_lo = T_dew
        T_hi = max(T_dew + 10.0, T_COND + 10.0)
        f_lo = S_dew - S_target
        f_hi = resid(T_hi)
        n = 0
        while f_lo * f_hi > 0 and n < 400:
            T_hi += 10.0
            if T_hi > 6.0 * Tc_max:
                break
            f_hi = resid(T_hi); n += 1
        if f_lo * f_hi <= 0:
            T2s = brentq(resid, T_lo, T_hi, xtol=1e-5, rtol=1e-10, maxiter=300)
            return T2s, H_vap_at(T2s), "superheated"
        return None

    if T_dew is None:
        return None
    T_bub = _sat_T_at_P("bubble", w, P2, props, kij)
    if T_bub is None:
        return None
    a_v, b_v, _, *_ = mix_params(T_dew, w, props, kij)
    Av, Bv = compute_AB(T_dew, P2, a_v, b_v); rv = solve_cubic_Z(Av, Bv)
    a_l, b_l, _, *_ = mix_params(T_bub, w, props, kij)
    Al, Bl = compute_AB(T_bub, P2, a_l, b_l); rl = solve_cubic_Z(Al, Bl)
    if rv is None or rl is None:
        return None
    S_vap = total_S(cps, w, T_dew, P2, rv[0], props, kij)
    H_vap = total_H(cps, w, T_dew, P2, rv[0], props, kij)
    S_liq = total_S(cps, w, T_bub, P2, rl[-1], props, kij)
    H_liq = total_H(cps, w, T_bub, P2, rl[-1], props, kij)
    denom = S_vap - S_liq
    if abs(denom) < 1e-12:
        return None
    x = (S_target - S_liq) / denom
    x = min(max(x, 0.0), 1.0)
    H2s = H_liq + x * (H_vap - H_liq)
    T2s = T_bub + x * (T_dew - T_bub)
    return T2s, H2s, "wet"


def compute_cycle_mixture(cps, z, props, kij=0.0):
    """
    Ideal vapour-compression cycle for feed composition z, Bell et al. (2019)
    'simplified cycle' convention:
        p_evap = dew-point pressure of z at T_evap   (evaporator dew = 10 C)
        p_cond = bubble-point pressure of z at T_cond (condenser bubble = 40 C)
    All state enthalpies use the fixed overall (feed) composition z, as in a
    sealed vapour-compression loop. Pressure drop is NOT modelled.
    Returns (state_dict_or_None, status_string).
    """
    P_evap, _ = dew_point_P(T_EVAP, z, props, kij)
    if P_evap is None:
        return None, "dew@evap failed"
    P_cond, _ = bubble_point_P(T_COND, z, props, kij)
    if P_cond is None:
        return None, "bubble@cond failed"
    if P_evap <= 0 or P_cond <= 0 or P_cond <= P_evap:
        return None, "P_cond<=P_evap"
    if P_RATIO_MAX and (P_cond / P_evap) > P_RATIO_MAX:
        return None, "pressure ratio too high (spurious VLE)"

    T1 = T_EVAP + DT_SH
    a1, b1, _, *_ = mix_params(T1, z, props, kij)
    A1, B1 = compute_AB(T1, P_evap, a1, b1); r1 = solve_cubic_Z(A1, B1)
    if r1 is None:
        return None, "state1 cubic"
    H1 = total_H(cps, z, T1, P_evap, r1[0], props, kij)
    S1 = total_S(cps, z, T1, P_evap, r1[0], props, kij)

    out = find_T2s_mix(cps, S1, z, P_cond, props, kij)
    if out is None:
        return None, "T2s failed"
    T2s, H2s, regime = out
    if T2S_MAX_K and T2s > T2S_MAX_K:
        return None, "T2s too high (unphysical)"
    H2 = H1 + (H2s - H1) / ETA_IS

    T3 = T_COND - DT_SC
    a3, b3, _, *_ = mix_params(T3, z, props, kij)
    A3, B3 = compute_AB(T3, P_cond, a3, b3); r3 = solve_cubic_Z(A3, B3)
    if r3 is None:
        return None, "state3 cubic"
    H3 = total_H(cps, z, T3, P_cond, r3[-1], props, kij)

    H4 = H3
    W  = H2 - H1
    Q  = H1 - H4
    if W <= 0:
        return None, "W<=0"
    COP = Q / W
    if not np.isfinite(COP) or COP < COP_MIN or COP > COP_MAX:
        return None, "COP out of physical range"
    return {"COP": COP, "P_evap_bar": P_evap / 1e5, "P_cond_bar": P_cond / 1e5,
            "T2s_K": T2s, "regime": regime}, "OK"


def compute_qvol_mix(cps, z, props, kij, P_evap_bar, P_cond_bar):
    """
    Qvol = |h1 - h4| * rho_suction   [kJ/m3] -- Bell et al. (2019).
    h1          : evaporator-outlet / compressor-suction enthalpy (state 1)
    h4          : expansion-valve-outlet enthalpy = h3 (isenthalpic throttling,
                  same assumption compute_cycle_mixture makes internally)
    rho_suction : density of the vapour entering the compressor (state 1)
    """
    P_evap = P_evap_bar * 1e5
    P_cond = P_cond_bar * 1e5

    T1 = T_EVAP + DT_SH
    a1, b1, _, *_ = mix_params(T1, z, props, kij)
    A1, B1 = compute_AB(T1, P_evap, a1, b1)
    r1 = solve_cubic_Z(A1, B1)
    if r1 is None:
        return None
    Z1 = r1[0]
    h1 = total_H(cps, z, T1, P_evap, Z1, props, kij)
    v1 = Z1 * R * T1 / P_evap
    if v1 <= 0 or not np.isfinite(v1):
        return None
    rho_suction = 1.0 / v1

    T3 = T_COND - DT_SC
    a3, b3, _, *_ = mix_params(T3, z, props, kij)
    A3, B3 = compute_AB(T3, P_cond, a3, b3)
    r3 = solve_cubic_Z(A3, B3)
    if r3 is None:
        return None
    Z3 = r3[-1]
    h3 = total_H(cps, z, T3, P_cond, Z3, props, kij)
    h4 = h3

    qvol = abs(h1 - h4) * rho_suction
    return qvol / 1000.0    # kJ/m3


# ==============================================================================
#  Hardware / execution-backend detection
# ==============================================================================
def detect_hardware():
    """
    Detect CPU/GPU hardware. GPU detection is purely informational here: see
    module docstring for why this engine's solve always runs on CPU.
    """
    info = {
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_present": False,
        "gpu_name": None,
        "gpu_detect_method": None,
    }
    nv = shutil.which("nvidia-smi")
    if nv:
        try:
            result = subprocess.run(
                [nv, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                info["gpu_present"] = True
                info["gpu_name"] = result.stdout.strip().splitlines()[0]
                info["gpu_detect_method"] = "nvidia-smi"
        except Exception:
            pass
    if not info["gpu_present"]:
        try:
            import cupy  # noqa: F401
            info["gpu_present"] = True
            info["gpu_detect_method"] = "cupy"
        except ImportError:
            pass
    return info


def resolve_backend(requested, hw_info, log):
    """
    Decide CPU vs GPU execution for this run. See module docstring
    "CPU / GPU execution" for the scientific rationale: the PR-EOS root
    solvers are not safely GPU-vectorisable without rewriting the numerics,
    so this never happens automatically. GPU presence is only reported.
    """
    requested = (requested or "auto").lower()
    if requested == "cpu":
        chosen = "CPU"
    elif requested == "gpu":
        if hw_info["gpu_present"]:
            log.warning(
                f"GPU detected ({hw_info['gpu_name'] or hw_info['gpu_detect_method']}) but "
                "this PR-EOS engine's root-finding (numpy.roots / scipy.optimize.brentq / "
                "successive-substitution VLE) is not validated for GPU execution -> "
                "falling back to CPU multiprocessing."
            )
        else:
            log.warning("--backend gpu requested but no GPU was detected -> falling back to CPU.")
        chosen = "CPU"
    else:  # auto
        chosen = "CPU"
        if hw_info["gpu_present"]:
            log.info(
                f"GPU detected ({hw_info['gpu_name'] or hw_info['gpu_detect_method']}) but is not "
                "used: PR-EOS root-finding is scalar/iterative per (pair, composition) and is not "
                "safely vectorisable onto GPU without rewriting the numerical method (explicitly "
                "avoided to protect scientific correctness). Using CPU multiprocessing."
            )
    log.info(f"Execution backend: {chosen}")
    return chosen


# ==============================================================================
#  Logging
# ==============================================================================
def setup_logging(out_dir):
    logger = logging.getLogger("generate_binary_mixture_dataset")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(out_dir, LOG_FILE), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ==============================================================================
#  SMILES sanity check (opportunistic rdkit, else a syntax-only fallback)
# ==============================================================================
try:
    from rdkit import Chem as _RDChem
    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except ImportError:
    _HAVE_RDKIT = False

_SMILES_CHARSET_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\%.]+$")


def is_valid_smiles(s):
    """
    Best-effort SMILES validity check. Uses rdkit if available (real chemical
    parse); otherwise falls back to a character-set syntax check only (does
    NOT verify chemical validity). Never fabricates or alters SMILES.
    """
    if not isinstance(s, str) or not s.strip():
        return False
    s = s.strip()
    if _HAVE_RDKIT:
        return _RDChem.MolFromSmiles(s) is not None
    return bool(_SMILES_CHARSET_RE.match(s))


# ==============================================================================
#  Input loading + validation + quality-control report
# ==============================================================================
def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_molecules(log):
    """
    Merge INPUT_PROPERTY_FILE and INPUT_CP_FILE on COL_NAME, validate every
    field the calculation depends on, and print an INPUT QUALITY CONTROL
    report. Never assumes a column exists -- every column referenced here was
    confirmed present in the actual Mclinden_Analysis input files.
    """
    for path in (INPUT_PROPERTY_FILE, INPUT_CP_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file not found: {path}")

    props = pd.read_csv(INPUT_PROPERTY_FILE)
    cp = pd.read_csv(INPUT_CP_FILE)

    required_props_cols = [COL_NAME, COL_TC, COL_PC, COL_OMEGA, COL_GWP, COL_SMILES,
                            COL_COP_PURE, COL_QVOL_PURE]
    required_cp_cols = [COL_NAME] + CP_COLS
    missing_props = [c for c in required_props_cols if c not in props.columns]
    missing_cp = [c for c in required_cp_cols if c not in cp.columns]
    if missing_props:
        raise KeyError(f"{INPUT_PROPERTY_FILE} is missing required columns: {missing_props}")
    if missing_cp:
        raise KeyError(f"{INPUT_CP_FILE} is missing required columns: {missing_cp}")

    for d in (props, cp):
        d[COL_NAME] = d[COL_NAME].astype(str).str.strip()

    n_props_raw, n_cp_raw = len(props), len(cp)
    dup_props = int(props[COL_NAME].duplicated().sum())
    dup_cp = int(cp[COL_NAME].duplicated().sum())
    props = props.drop_duplicates(subset=COL_NAME, keep="first")
    cp = cp.drop_duplicates(subset=COL_NAME, keep="first")

    merged = pd.merge(
        props[[COL_NAME, COL_TC, COL_PC, COL_OMEGA, COL_GWP, COL_SMILES,
               COL_COP_PURE, COL_QVOL_PURE]],
        cp[[COL_NAME] + CP_COLS],
        on=COL_NAME, how="inner",
    )
    n_merged = len(merged)

    numeric_cols = [COL_TC, COL_PC, COL_OMEGA, COL_GWP, COL_COP_PURE, COL_QVOL_PURE] + CP_COLS
    missing_breakdown = {c: int(merged[c].isna().sum()) for c in numeric_cols}
    inf_breakdown = {
        c: int(np.isinf(pd.to_numeric(merged[c], errors="coerce")).sum())
        for c in numeric_cols
    }

    clean = merged.dropna(subset=numeric_cols).copy()
    for c in numeric_cols:
        clean = clean[np.isfinite(pd.to_numeric(clean[c], errors="coerce"))]

    n_after_nan_inf = len(clean)

    # --- Physical-plausibility filters (reject, don't silently coerce) --------
    n_invalid_tc = int((clean[COL_TC] <= 0).sum())
    clean = clean[clean[COL_TC] > 0]
    n_invalid_pc = int((clean[COL_PC] <= 0).sum())
    clean = clean[clean[COL_PC] > 0]
    n_invalid_gwp = int((clean[COL_GWP] < 0).sum())
    clean = clean[clean[COL_GWP] >= 0]

    # Unit-consistency sanity check on Pc: bar-scale should be O(1-200); if the
    # data is actually already in Pa this would show up as O(1e5-1e7).
    pc_scale = 1e5 if PC_UNIT.lower() == "bar" else 1.0
    pc_median = float(clean[COL_PC].median()) if len(clean) else float("nan")
    unit_sane = (1.0 <= pc_median <= 300.0) if PC_UNIT.lower() == "bar" else (1e4 <= pc_median <= 1e8)
    if not unit_sane:
        log.warning(
            f"UNIT-CONSISTENCY WARNING: PC_UNIT='{PC_UNIT}' but median {COL_PC} = "
            f"{pc_median:.4g} looks implausible for that unit. Check COL_PC/PC_UNIT config."
        )

    dropped_blends = 0
    if DROP_BLENDS:
        mask = clean[COL_NAME].str.contains("mixture|blend", case=False, na=False)
        dropped_blends = int(mask.sum())
        clean = clean[~mask]

    dropped_lowtc = 0
    if MIN_COMPONENT_TC and MIN_COMPONENT_TC > 0:
        mask = clean[COL_TC].astype(float) <= MIN_COMPONENT_TC
        dropped_lowtc = int(mask.sum())
        clean = clean[~mask]

    smiles_valid = clean[COL_SMILES].apply(is_valid_smiles)
    n_bad_smiles = int((~smiles_valid).sum())

    mols = []
    for _, r in clean.iterrows():
        mols.append({
            "name": r[COL_NAME],
            "Tc": float(r[COL_TC]),
            "Pc": float(r[COL_PC]) * pc_scale,
            "Pc_bar": float(r[COL_PC]) if PC_UNIT.lower() == "bar" else float(r[COL_PC]) / 1e5,
            "omega": float(r[COL_OMEGA]),
            "GWP": float(r[COL_GWP]),
            "cp": tuple(float(r[c]) for c in CP_COLS),
            "smiles": str(r[COL_SMILES]).strip(),
            "smiles_valid": bool(smiles_valid.loc[r.name]),
            "COP_lit": float(r[COL_COP_PURE]),
            "Qvol_lit_kJ_m3": float(r[COL_QVOL_PURE]) * 1000.0,   # MJ/m3 -> kJ/m3
        })

    log.info("=" * 78)
    log.info("  INPUT QUALITY CONTROL")
    log.info("=" * 78)
    log.info(f"  Rows in property file ({INPUT_PROPERTY_FILE:35s}) : {n_props_raw}")
    log.info(f"  Rows in Cp file       ({INPUT_CP_FILE:35s}) : {n_cp_raw}")
    log.info(f"  Duplicate names in property file               : {dup_props}")
    log.info(f"  Duplicate names in Cp file                     : {dup_cp}")
    log.info(f"  Rows after merge (inner join on {COL_NAME})    : {n_merged}")
    log.info("  Missing-field breakdown (of merged rows):")
    for c, n in missing_breakdown.items():
        log.info(f"      {c:14s}: {n}  (inf: {inf_breakdown[c]})")
    log.info(f"  Removed (NaN/inf in required fields)           : {n_merged - n_after_nan_inf}")
    log.info(f"  Removed (Tc <= 0)                              : {n_invalid_tc}")
    log.info(f"  Removed (Pc <= 0)                               : {n_invalid_pc}")
    log.info(f"  Removed (GWP < 0)                                : {n_invalid_gwp}")
    log.info(f"  Removed (blend/mixture name, DROP_BLENDS=True)  : {dropped_blends}")
    log.info(f"  Removed (Tc <= MIN_COMPONENT_TC={MIN_COMPONENT_TC:.2f} K)     : {dropped_lowtc}")
    log.info(f"  SMILES that fail validation ({'rdkit' if _HAVE_RDKIT else 'syntax-only'} check)"
              f"   : {n_bad_smiles}  (kept, flagged via smiles_valid)")
    log.info(f"  Usable molecules                                 : {len(mols)}")
    log.info("=" * 78)

    if len(mols) < 2:
        raise ValueError("Fewer than 2 usable molecules after validation -- cannot generate pairs.")

    return mols


def molecule_set_hash(mols):
    names = sorted(m["name"] for m in mols)
    return hashlib.md5("|".join(names).encode("utf-8")).hexdigest()


def config_hash():
    """Hash of the scientific configuration that affects computed results."""
    payload = {
        "T_EVAP_C": T_EVAP_C, "T_COND_C": T_COND_C, "DT_SH": DT_SH, "DT_SC": DT_SC,
        "ETA_IS": ETA_IS, "KIJ": KIJ, "MIN_COMPONENT_TC": MIN_COMPONENT_TC,
        "DROP_BLENDS": DROP_BLENDS, "T2S_MAX_K": T2S_MAX_K, "P_RATIO_MAX": P_RATIO_MAX,
        "COP_MIN": COP_MIN, "COP_MAX": COP_MAX,
        "X1_START": X1_START, "X1_END": X1_END, "X1_STEP": X1_STEP,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ==============================================================================
#  Failure categorisation
# ==============================================================================
_FAILURE_STAGE_MAP = {
    "dew@evap failed": "dew_point_failed",
    "bubble@cond failed": "bubble_point_failed",
    "P_cond<=P_evap": "pressure_ratio_guard",
    "pressure ratio too high (spurious VLE)": "pressure_ratio_guard",
    "state1 cubic": "state1_failed",
    "T2s failed": "isentropic_outlet_failed",
    "T2s too high (unphysical)": "T2s_guard",
    "state3 cubic": "state3_failed",
    "W<=0": "work_guard",
    "COP out of physical range": "invalid_cop",
    "Qvol calculation failed": "qvol_failed",
}


def classify_failure(reason):
    if reason in _FAILURE_STAGE_MAP:
        return _FAILURE_STAGE_MAP[reason]
    if reason and (reason.startswith("exception:") or reason.startswith("unhandled:")):
        return "numerical_exception"
    return "unknown"


def extract_exception_type(reason):
    if reason and (reason.startswith("exception:") or reason.startswith("unhandled:")):
        parts = reason.split(":", 2)
        if len(parts) >= 2:
            return parts[1]
    return ""


# ==============================================================================
#  Per-pair worker (top-level, picklable under the 'spawn' multiprocessing context)
# ==============================================================================
DETAILED_COLUMNS = [
    "molecule_1", "molecule_2", "smile_1", "smile_2", "x_1", "x_2",
    "COP_1_ref", "COP_2_ref", "Qvol_1_ref", "Qvol_2_ref", "GWP_1", "GWP_2",
    "Tc_1", "Tc_2", "Pc_1", "Pc_2", "omega_1", "omega_2",
    "COP_mix", "Qvol_mix", "GWP_mix",
    "P_evap_bar", "P_cond_bar", "pressure_ratio", "T2s_K", "compressor_regime",
    "kij", "T_evap_C", "T_cond_C", "DT_SH_K", "DT_SC_K", "eta_is", "status",
]
ML_COLUMNS = [
    "molecule_1", "molecule_2", "smile_1", "smile_2", "x_1", "x_2",
    "COP_mix", "Qvol_mix", "GWP_mix",
    "P_evap_bar", "P_cond_bar", "pressure_ratio", "T2s_K", "compressor_regime",
]
FAILED_COLUMNS = [
    "molecule_1", "molecule_2", "x_1", "x_2",
    "failure_stage", "failure_reason", "exception_type",
]


def _process_pair(args):
    """
    Process every composition for one binary pair. Returns:
        (pair_idx, name_A, name_B, detailed_rows, ml_rows, failure_rows)
    ml_rows are derived from the SAME in-memory detailed row (never
    recomputed), so the ML dataset is always exactly reproducible from the
    detailed dataset.
    """
    pair_idx, molA, molB, x1_grid, kij = args
    props = [(molA["Tc"], molA["Pc"], molA["omega"]), (molB["Tc"], molB["Pc"], molB["omega"])]
    cps = [molA["cp"], molB["cp"]]
    detailed_rows, ml_rows, failure_rows = [], [], []

    def _fail(x1, x2, stage_reason):
        stage = classify_failure(stage_reason)
        failure_rows.append({
            "molecule_1": molA["name"], "molecule_2": molB["name"],
            "x_1": x1, "x_2": x2,
            "failure_stage": stage, "failure_reason": stage_reason,
            "exception_type": extract_exception_type(stage_reason),
        })

    try:
        for x1, x2 in x1_grid:
            try:
                state, status = compute_cycle_mixture(cps, [x1, x2], props, kij)
            except Exception as e:
                state, status = None, f"exception:{type(e).__name__}:{e}"

            if state is None:
                if SAVE_FAILED_CASES:
                    _fail(x1, x2, status)
                continue

            try:
                qvol = compute_qvol_mix(cps, [x1, x2], props, kij,
                                         state["P_evap_bar"], state["P_cond_bar"])
            except Exception as e:
                qvol = None

            if qvol is None:
                if SAVE_FAILED_CASES:
                    _fail(x1, x2, "Qvol calculation failed")
                continue

            gwp_mix = x1 * molA["GWP"] + x2 * molB["GWP"]
            pressure_ratio = state["P_cond_bar"] / state["P_evap_bar"]

            row = {
                "molecule_1": molA["name"], "molecule_2": molB["name"],
                "smile_1": molA["smiles"], "smile_2": molB["smiles"],
                "x_1": x1, "x_2": x2,
                # "_ref" suffix is deliberate: these are literature/external reference
                # values loaded from the input file, NOT computed by this PR-EOS engine.
                # See module docstring "Pure-component COP_1_ref / ... -- IMPORTANT CAVEAT".
                "COP_1_ref": round(molA["COP_lit"], 5), "COP_2_ref": round(molB["COP_lit"], 5),
                "Qvol_1_ref": round(molA["Qvol_lit_kJ_m3"], 4), "Qvol_2_ref": round(molB["Qvol_lit_kJ_m3"], 4),
                "GWP_1": molA["GWP"], "GWP_2": molB["GWP"],
                "Tc_1": round(molA["Tc"], 3), "Tc_2": round(molB["Tc"], 3),
                "Pc_1": round(molA["Pc_bar"], 5), "Pc_2": round(molB["Pc_bar"], 5),
                "omega_1": round(molA["omega"], 6), "omega_2": round(molB["omega"], 6),
                "COP_mix": round(state["COP"], 6),
                "Qvol_mix": round(qvol, 4),
                "GWP_mix": round(gwp_mix, 4),
                "P_evap_bar": round(state["P_evap_bar"], 4),
                "P_cond_bar": round(state["P_cond_bar"], 4),
                "pressure_ratio": round(pressure_ratio, 4),
                "T2s_K": round(state["T2s_K"], 3),
                "compressor_regime": state["regime"],
                "kij": kij,
                "T_evap_C": T_EVAP_C, "T_cond_C": T_COND_C,
                "DT_SH_K": DT_SH, "DT_SC_K": DT_SC, "eta_is": ETA_IS,
                "status": "OK",
            }
            detailed_rows.append(row)
            ml_rows.append({k: row[k] for k in ML_COLUMNS})

    except Exception as e:
        if SAVE_FAILED_CASES:
            for x1, x2 in x1_grid:
                _fail(x1, x2, f"unhandled:{type(e).__name__}:{e}")

    return pair_idx, molA["name"], molB["name"], detailed_rows, ml_rows, failure_rows


# ==============================================================================
#  Streaming, chunk-rotating CSV writer -- never holds the full dataset in RAM
# ==============================================================================
class ChunkedCSVWriter:
    def __init__(self, out_dir, prefix, columns, chunk_size,
                 start_part_index=1, start_rows_in_file=0, start_total_rows=0):
        self.out_dir = out_dir
        self.prefix = prefix
        self.columns = columns
        self.chunk_size = chunk_size
        self.part_index = start_part_index
        self.rows_in_file = start_rows_in_file
        self.total_rows = start_total_rows
        self._open_current(append=start_rows_in_file > 0)

    def _path(self, idx):
        return os.path.join(self.out_dir, f"{self.prefix}_{idx:04d}.csv")

    def _open_current(self, append):
        path = self._path(self.part_index)
        write_header = not (append and os.path.exists(path))
        self.fh = open(path, "a" if append else "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=self.columns)
        if write_header:
            self.writer.writeheader()
        self.current_path = path

    def write_rows(self, rows):
        for row in rows:
            if self.rows_in_file >= self.chunk_size:
                self.fh.close()
                self.part_index += 1
                self.rows_in_file = 0
                self._open_current(append=False)
            self.writer.writerow(row)
            self.rows_in_file += 1
            self.total_rows += 1

    def all_part_files(self):
        return sorted(
            f for f in os.listdir(self.out_dir)
            if f.startswith(self.prefix + "_") and f.endswith(".csv")
        )

    def flush(self):
        self.fh.flush()

    def close(self):
        self.fh.close()


class SimpleCSVWriter:
    """Single-file streaming writer (used for failed_cases.csv -- not chunked)."""
    def __init__(self, path, columns, append=False):
        write_header = not (append and os.path.exists(path))
        self.fh = open(path, "a" if append else "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=columns)
        if write_header:
            self.writer.writeheader()
        self.total_rows = 0

    def write_rows(self, rows):
        for row in rows:
            self.writer.writerow(row)
            self.total_rows += 1

    def flush(self):
        self.fh.flush()

    def close(self):
        self.fh.close()


# ==============================================================================
#  Checkpoint manager
# ==============================================================================
def checkpoint_path(out_dir):
    return os.path.join(out_dir, CHECKPOINT_FILE)


def _count_data_rows(path):
    """Count data rows (excluding header) in a CSV file. 0 if it doesn't exist."""
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def reconcile_chunked_writer(out_dir, prefix, checkpointed_part_index, checkpointed_rows_in_file,
                              chunk_size, log):
    """
    Align on-disk part files with what the checkpoint last confirmed, before
    resuming. This matters because rows are written to a part file as soon as
    a pair finishes (immediately in the main process loop), but a pair is
    only added to the checkpoint's completed set periodically (every
    CHECKPOINT_EVERY_PAIRS / CHECKPOINT_EVERY_SECONDS). Python's buffered
    file writer can flush those not-yet-checkpointed rows to disk on its own
    (buffer full) ahead of the next scheduled checkpoint. If the process is
    then killed, those rows are on disk but their pair was never marked
    complete -- resuming would recompute that pair and duplicate its rows.
    This function discards any such unconfirmed rows so resume is exact-once.
    Returns the verified (part_index, rows_in_file, total_rows).
    """
    existing = sorted(
        f for f in os.listdir(out_dir) if f.startswith(prefix + "_") and f.endswith(".csv")
    )
    removed = 0
    for f in existing:
        try:
            idx = int(f[len(prefix) + 1: len(prefix) + 5])
        except ValueError:
            continue
        if idx > checkpointed_part_index:
            os.remove(os.path.join(out_dir, f))
            removed += 1
    if removed:
        log.warning(f"Reconcile '{prefix}': removed {removed} part file(s) written after the last "
                    "confirmed checkpoint (rotated post-crash) -- their pairs will be recomputed.")

    current_path = os.path.join(out_dir, f"{prefix}_{checkpointed_part_index:04d}.csv")
    actual_rows = _count_data_rows(current_path)
    if actual_rows > checkpointed_rows_in_file:
        with open(current_path, "r", newline="") as f:
            header = f.readline()
            kept = [f.readline() for _ in range(checkpointed_rows_in_file)]
        with open(current_path, "w", newline="") as f:
            f.write(header)
            f.writelines(kept)
        log.warning(f"Reconcile '{prefix}': truncated {os.path.basename(current_path)} from "
                    f"{actual_rows} to {checkpointed_rows_in_file} rows (discarding unconfirmed "
                    "post-checkpoint writes to prevent duplicate rows on resume).")
        rows_in_file = checkpointed_rows_in_file
    elif actual_rows < checkpointed_rows_in_file:
        log.warning(f"Reconcile '{prefix}': {os.path.basename(current_path)} has fewer rows "
                    f"({actual_rows}) than the checkpoint expects ({checkpointed_rows_in_file}) -- "
                    "trusting the file on disk.")
        rows_in_file = actual_rows
    else:
        rows_in_file = actual_rows

    total_rows = (checkpointed_part_index - 1) * chunk_size + rows_in_file
    return checkpointed_part_index, rows_in_file, total_rows


def reconcile_simple_writer(path, checkpointed_rows, log, label):
    """Same reconciliation as reconcile_chunked_writer, for a single unchunked file
    (failed_cases.csv). Returns the verified row count."""
    actual_rows = _count_data_rows(path)
    if actual_rows > checkpointed_rows:
        with open(path, "r", newline="") as f:
            header = f.readline()
            kept = [f.readline() for _ in range(checkpointed_rows)]
        with open(path, "w", newline="") as f:
            f.write(header)
            f.writelines(kept)
        log.warning(f"Reconcile '{label}': truncated {os.path.basename(path)} from {actual_rows} "
                    f"to {checkpointed_rows} rows (discarding unconfirmed post-checkpoint writes).")
        return checkpointed_rows
    if actual_rows < checkpointed_rows:
        log.warning(f"Reconcile '{label}': {os.path.basename(path)} has fewer rows ({actual_rows}) "
                    f"than the checkpoint expects ({checkpointed_rows}) -- trusting the file on disk.")
    return actual_rows


def load_checkpoint(log, out_dir, expected_mol_hash, n_molecules, expected_cfg_hash, input_hashes):
    if not ENABLE_CHECKPOINTS or OVERWRITE_EXISTING:
        return None
    path = checkpoint_path(out_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            ck = json.load(f)
    except Exception as e:
        log.warning(f"Could not read checkpoint ({e}); starting fresh.")
        return None

    checks = [
        (ck.get("molecule_set_hash") == expected_mol_hash, "molecule set changed"),
        (ck.get("n_molecules") == n_molecules, "molecule count changed"),
        (ck.get("config_hash") == expected_cfg_hash, "scientific configuration changed"),
        (ck.get("composition_grid") == {
            "start": X1_START, "end": X1_END, "step": X1_STEP, "n_points": len(X1_GRID),
        }, "composition grid changed"),
        (ck.get("input_files") == input_hashes, "input file contents changed"),
    ]
    for ok, why in checks:
        if not ok:
            log.warning(f"Checkpoint incompatible with current run ({why}) -- ignoring it and "
                        "starting fresh. Set OVERWRITE_EXISTING=True / --fresh to silence this.")
            return None
    return ck


def save_checkpoint(out_dir, mol_hash, n_molecules, cfg_hash, input_hashes,
                     completed_indices, total_pairs, detailed_writer, ml_writer,
                     n_success, n_failed):
    ck = {
        "generator": "generate_binary_mixture_dataset.py",
        "generator_version": __version__,
        "molecule_set_hash": mol_hash,
        "n_molecules": n_molecules,
        "config_hash": cfg_hash,
        "input_files": input_hashes,
        "composition_grid": {
            "start": X1_START, "end": X1_END, "step": X1_STEP, "n_points": len(X1_GRID),
        },
        "total_pairs": total_pairs,
        "completed_pair_indices": sorted(completed_indices),
        "detailed_output_file": os.path.basename(detailed_writer.current_path),
        "detailed_part_index": detailed_writer.part_index,
        "detailed_rows_in_current_file": detailed_writer.rows_in_file,
        "detailed_rows_written": detailed_writer.total_rows,
        "ml_output_file": os.path.basename(ml_writer.current_path),
        "ml_part_index": ml_writer.part_index,
        "ml_rows_in_current_file": ml_writer.rows_in_file,
        "ml_rows_written": ml_writer.total_rows,
        "n_success": n_success,
        "n_failed": n_failed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = checkpoint_path(out_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f, indent=2)
    os.replace(tmp, checkpoint_path(out_dir))


# ==============================================================================
#  Progress reporting
# ==============================================================================
def format_eta(seconds):
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def print_progress(log, t0, total_calcs, n_success, n_failed, current_file, pairs_done, total_pairs):
    elapsed = time.time() - t0
    n_done = n_success + n_failed
    rate_rows = n_success / elapsed if elapsed > 0 else 0.0
    rate_calcs = n_done / elapsed if elapsed > 0 else 0.0
    remaining_calcs = max(total_calcs - n_done, 0)
    eta = remaining_calcs / rate_calcs if rate_calcs > 0 else float("inf")
    success_rate = 100.0 * n_success / n_done if n_done else 0.0
    failure_rate = 100.0 * n_failed / n_done if n_done else 0.0

    log.info(
        f"  [{pairs_done}/{total_pairs} pairs]  rows={n_success:,}  "
        f"success={n_success:,} ({success_rate:.1f}%)  fail={n_failed:,} ({failure_rate:.1f}%)  "
        f"rows/sec={rate_rows:.1f}  ETA={format_eta(eta)}  file={current_file}"
    )


# ==============================================================================
#  Deterministic pairing (lazy) + GWP mixing
# ==============================================================================
def sorted_molecules(mols):
    """Canonical ordering: molecule_1 < molecule_2 is always alphabetical on
    `name`, so (A,B) and (B,A) never both appear as separate pairs."""
    return sorted(mols, key=lambda m: m["name"])


def n_pairs_for(n_molecules):
    return n_molecules * (n_molecules - 1) // 2


def pair_index_generator(n_molecules):
    """Lazy generator over (i, j) index pairs, i < j -- never materialises a
    full pair list, so this scales to arbitrarily large molecule counts."""
    for i in range(n_molecules):
        for j in range(i + 1, n_molecules):
            yield i, j


# GWP_mix = x1*GWP_1 + x2*GWP_2 -- see module docstring "GWP_mix definition"
# for the explicit mole-fraction-weighted-linear-blend caveat. Implemented
# inline in _process_pair; documented here as the single source of truth.
GWP_MIX_DEFINITION = "GWP_mix = x1 * GWP_1 + x2 * GWP_2"
GWP_MIX_CAVEAT = (
    "GWP_mix is a mole-fraction-weighted LINEAR BLEND of GWP100 values "
    "(GWP_mix = x1*GWP_1 + x2*GWP_2). It is a computational screening metric only, "
    "NOT an experimentally validated mixture-GWP model -- real blend GWP depends on "
    "atmospheric lifetime, radiative efficiency and composition-dependent decomposition "
    "chemistry of the actual blend, none of which is modelled here."
)


# ==============================================================================
#  CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate the binary-mixture PR-EOS dataset (detailed + ML + failed-cases)."
    )
    p.add_argument("--fresh", action="store_true",
                    help="Ignore any existing checkpoint/output and start over.")
    p.add_argument("--backend", choices=["auto", "cpu", "gpu"], default=None,
                    help="Execution backend (default: BACKEND in the config section).")
    p.add_argument("--workers", type=int, default=None,
                    help="Worker processes (default: N_WORKERS config / all cores).")
    p.add_argument("--max-molecules", type=int, default=None,
                    help="Debug: cap the molecule list to N.")
    p.add_argument("--max-pairs", type=int, default=None,
                    help="Debug: cap the number of pairs processed.")
    return p.parse_args()


# ==============================================================================
#  Main
# ==============================================================================
def main():
    args = parse_args()
    global OVERWRITE_EXISTING
    if args.fresh:
        OVERWRITE_EXISTING = True

    backend_request = args.backend or BACKEND
    max_molecules = args.max_molecules or DEBUG_MAX_MOLECULES
    max_pairs = args.max_pairs or DEBUG_MAX_PAIRS

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log = setup_logging(OUTPUT_DIR)

    log.info("=" * 78)
    log.info("  Binary Refrigerant Mixture Dataset Generator (PR-EOS)  v" + __version__)
    log.info(f"  T_evap={T_EVAP_C}C  T_cond={T_COND_C}C  dTsh={DT_SH}K  dTsc={DT_SC}K"
              f"  eta={ETA_IS}  kij={KIJ}")
    log.info(f"  Compositions: x1 = {X1_GRID[0][0]}..{X1_GRID[-1][0]}  step={X1_STEP}"
              f"  ({len(X1_GRID)} points)")
    log.info("=" * 78)

    hw_info = detect_hardware()
    log.info(f"  CPU cores detected : {hw_info['cpu_count']}")
    log.info(f"  GPU detected        : {hw_info['gpu_present']}"
              + (f"  ({hw_info['gpu_name'] or hw_info['gpu_detect_method']})" if hw_info["gpu_present"] else ""))
    backend = resolve_backend(backend_request, hw_info, log)

    mols = load_and_validate_molecules(log)
    if max_molecules:
        mols = mols[:max_molecules]
    mols = sorted_molecules(mols)
    n_molecules = len(mols)
    if n_molecules < 2:
        log.critical("Fewer than 2 usable molecules -- nothing to do.")
        return

    mol_hash = molecule_set_hash(mols)
    cfg_hash = config_hash()
    input_hashes = {
        "property_file": {"path": INPUT_PROPERTY_FILE, "sha256": _file_sha256(INPUT_PROPERTY_FILE)},
        "cp_file": {"path": INPUT_CP_FILE, "sha256": _file_sha256(INPUT_CP_FILE)},
    }

    total_pairs_full = n_pairs_for(n_molecules)
    total_pairs = min(total_pairs_full, max_pairs) if max_pairs else total_pairs_full
    total_compositions = len(X1_GRID)
    total_calcs = total_pairs * total_compositions

    log.info(f"  Usable molecules    : {n_molecules}")
    log.info(f"  Unique pairs        : {total_pairs}" +
              (f"  (capped from {total_pairs_full} via --max-pairs)" if max_pairs else "  (N choose 2)"))
    log.info(f"  Compositions/pair   : {total_compositions}")
    log.info(f"  Total calculations  : {total_pairs} x {total_compositions} = {total_calcs}")

    if OVERWRITE_EXISTING:
        for f in os.listdir(OUTPUT_DIR):
            if (f.startswith(DETAILED_PREFIX) or f.startswith(ML_PREFIX)
                    or f in (FAILED_CASES_FILE, CHECKPOINT_FILE, SUMMARY_FILE, MANIFEST_FILE)):
                os.remove(os.path.join(OUTPUT_DIR, f))
        log.info("  OVERWRITE_EXISTING=True -> cleared previous output/checkpoint.")

    ck = load_checkpoint(log, OUTPUT_DIR, mol_hash, n_molecules, cfg_hash, input_hashes)
    if ck:
        completed = set(ck["completed_pair_indices"])
        n_success = ck["n_success"]
        n_failed = ck["n_failed"]

        # Reconcile on-disk files with the checkpoint before resuming: rows
        # can reach disk (buffered-writer flush) for pairs processed after
        # the last checkpoint save but never marked complete. Without this,
        # those pairs would be recomputed and their rows duplicated. See
        # reconcile_chunked_writer() docstring.
        d_idx, d_rows, d_total = reconcile_chunked_writer(
            OUTPUT_DIR, DETAILED_PREFIX, ck["detailed_part_index"],
            ck["detailed_rows_in_current_file"], CHUNK_SIZE_ROWS, log)
        m_idx, m_rows, m_total = reconcile_chunked_writer(
            OUTPUT_DIR, ML_PREFIX, ck["ml_part_index"],
            ck["ml_rows_in_current_file"], CHUNK_SIZE_ROWS, log)

        detailed_writer = ChunkedCSVWriter(
            OUTPUT_DIR, DETAILED_PREFIX, DETAILED_COLUMNS, CHUNK_SIZE_ROWS,
            start_part_index=d_idx, start_rows_in_file=d_rows, start_total_rows=d_total,
        )
        ml_writer = ChunkedCSVWriter(
            OUTPUT_DIR, ML_PREFIX, ML_COLUMNS, CHUNK_SIZE_ROWS,
            start_part_index=m_idx, start_rows_in_file=m_rows, start_total_rows=m_total,
        )
        log.info(f"  Resuming from checkpoint: {len(completed)}/{total_pairs} pairs already done "
                  f"({n_success:,} rows, {n_failed:,} failed).")
    else:
        completed = set()
        n_success = 0
        n_failed = 0
        detailed_writer = ChunkedCSVWriter(OUTPUT_DIR, DETAILED_PREFIX, DETAILED_COLUMNS, CHUNK_SIZE_ROWS)
        ml_writer = ChunkedCSVWriter(OUTPUT_DIR, ML_PREFIX, ML_COLUMNS, CHUNK_SIZE_ROWS)

    failed_writer = None
    if SAVE_FAILED_CASES:
        failed_path = os.path.join(OUTPUT_DIR, FAILED_CASES_FILE)
        if ck:
            reconcile_simple_writer(failed_path, n_failed, log, FAILED_CASES_FILE)
        failed_writer = SimpleCSVWriter(failed_path, FAILED_COLUMNS, append=bool(ck))

    def remaining_work_items():
        n = 0
        for idx, (i, j) in enumerate(pair_index_generator(n_molecules)):
            if max_pairs and idx >= total_pairs_full:
                break
            if idx in completed:
                continue
            if max_pairs and n >= max_pairs:
                break
            n += 1
            yield (idx, mols[i], mols[j], X1_GRID, KIJ)

    n_workers = args.workers or N_WORKERS or os.cpu_count() or 1
    n_remaining = max(total_pairs - len(completed), 0)
    log.info(f"  Remaining pairs     : {n_remaining}  (workers={n_workers})")
    log.info("=" * 78)

    t0 = time.time()
    pairs_since_ckpt = 0
    last_ckpt_time = time.time()
    last_progress_time = time.time()

    def _checkpoint_now():
        detailed_writer.flush()
        ml_writer.flush()
        if failed_writer is not None:
            failed_writer.flush()
        if ENABLE_CHECKPOINTS:
            save_checkpoint(OUTPUT_DIR, mol_hash, n_molecules, cfg_hash, input_hashes,
                             completed, total_pairs, detailed_writer, ml_writer,
                             n_success, n_failed)

    try:
        work_items = remaining_work_items()
        has_work = n_remaining > 0
        if has_work:
            if n_workers == 1:
                results_iter = (_process_pair(w) for w in work_items)
            else:
                ctx = mp.get_context("spawn")
                pool = ctx.Pool(processes=n_workers)
                results_iter = pool.imap_unordered(_process_pair, work_items)

            for pair_idx, nameA, nameB, drows, mlrows, frows in results_iter:
                detailed_writer.write_rows(drows)
                ml_writer.write_rows(mlrows)
                if failed_writer is not None and frows:
                    failed_writer.write_rows(frows)

                n_success += len(drows)
                n_failed += len(frows)
                completed.add(pair_idx)
                pairs_since_ckpt += 1

                now = time.time()
                if now - last_progress_time >= PROGRESS_EVERY_SECONDS:
                    print_progress(log, t0, total_calcs, n_success, n_failed,
                                   os.path.basename(detailed_writer.current_path),
                                   len(completed), total_pairs)
                    last_progress_time = now

                if ENABLE_CHECKPOINTS and (
                    pairs_since_ckpt >= CHECKPOINT_EVERY_PAIRS
                    or now - last_ckpt_time >= CHECKPOINT_EVERY_SECONDS
                ):
                    _checkpoint_now()
                    pairs_since_ckpt = 0
                    last_ckpt_time = now

            if n_workers != 1:
                pool.close()
                pool.join()

        detailed_writer.close()
        ml_writer.close()
        if failed_writer is not None:
            failed_writer.close()

    except KeyboardInterrupt:
        log.warning("Interrupted by user -- saving checkpoint before exit.")
        _checkpoint_now()
        detailed_writer.close()
        ml_writer.close()
        if failed_writer is not None:
            failed_writer.close()
        log.warning("Progress saved. Re-run the same command to resume.")
        sys.exit(1)
    except Exception as e:
        log.critical(f"Unhandled top-level exception: {e}\n{traceback.format_exc()}")
        _checkpoint_now()
        detailed_writer.close()
        ml_writer.close()
        if failed_writer is not None:
            failed_writer.close()
        log.critical("Progress saved despite the error. Re-run the same command to resume.")
        raise

    elapsed = time.time() - t0

    if ENABLE_CHECKPOINTS:
        save_checkpoint(OUTPUT_DIR, mol_hash, n_molecules, cfg_hash, input_hashes,
                         completed, total_pairs, detailed_writer, ml_writer,
                         n_success, n_failed)

    # --- Manifest ---------------------------------------------------------------
    manifest = {
        "detailed_dataset": detailed_writer.all_part_files(),
        "ml_dataset": ml_writer.all_part_files(),
        "failed_cases": FAILED_CASES_FILE if SAVE_FAILED_CASES else None,
    }
    with open(os.path.join(OUTPUT_DIR, MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Failure breakdown --------------------------------------------------------
    failure_by_category = {}
    failed_path = os.path.join(OUTPUT_DIR, FAILED_CASES_FILE)
    if SAVE_FAILED_CASES and os.path.exists(failed_path):
        try:
            fdf = pd.read_csv(failed_path)
            if "failure_stage" in fdf.columns:
                failure_by_category = fdf["failure_stage"].value_counts().to_dict()
        except Exception:
            pass

    success_rate = 100.0 * n_success / total_calcs if total_calcs else 0.0
    failure_rate = 100.0 * n_failed / total_calcs if total_calcs else 0.0

    summary = {
        "software_version": __version__,
        "date_created": datetime.now().isoformat(timespec="seconds"),
        "input_files": input_hashes,
        "n_molecules": n_molecules,
        "n_pairs": total_pairs,
        "n_compositions": total_compositions,
        "total_calculations": total_calcs,
        "successful_rows": n_success,
        "failed_rows": n_failed,
        "success_rate_pct": success_rate,
        "failure_rate_pct": failure_rate,
        "failure_reasons_by_category": failure_by_category,
        "cycle_conditions": {
            "T_evap_C": T_EVAP_C, "T_cond_C": T_COND_C, "DT_SH_K": DT_SH, "DT_SC_K": DT_SC,
            "eta_is": ETA_IS, "kij": KIJ,
        },
        "eos_configuration": {
            "eos": "Peng-Robinson", "mixing_rule": "van der Waals one-fluid",
            "min_component_Tc_K": MIN_COMPONENT_TC, "T2s_max_K": T2S_MAX_K,
            "P_ratio_max": P_RATIO_MAX, "COP_min": COP_MIN, "COP_max": COP_MAX,
        },
        "gwp_mix_definition": GWP_MIX_DEFINITION,
        "gwp_mix_caveat": GWP_MIX_CAVEAT,
        "pure_component_reference_columns_caveat": (
            "COP_1_ref/COP_2_ref/Qvol_1_ref/Qvol_2_ref in the detailed dataset are literature/"
            "external reference values loaded from the input file, NOT computed by this PR-EOS "
            "engine (they differ from this engine's own pure-fluid COP/Qvol by roughly 25% where "
            "checked). COP_mix/Qvol_mix do not asymptote to them at x1->0 or x1->1. They are "
            "excluded from the ML dataset for this reason."
        ),
        "backend": backend,
        "hardware": hw_info,
        "worker_count": n_workers,
        "chunk_size_rows": CHUNK_SIZE_ROWS,
        "execution_time_seconds": elapsed,
        "execution_time_hours": elapsed / 3600.0,
        "output_files": manifest,
        "molecule_set_hash": mol_hash,
        "config_hash": cfg_hash,
    }
    with open(os.path.join(OUTPUT_DIR, SUMMARY_FILE), "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 78)
    log.info("  DONE")
    log.info("=" * 78)
    log.info(f"  Elapsed              : {elapsed/60:.1f} min")
    log.info(f"  Detailed rows (OK)   : {n_success:,}")
    log.info(f"  Failed rows          : {n_failed:,}")
    log.info(f"  Success rate         : {success_rate:.2f}%")
    log.info(f"  Failure rate         : {failure_rate:.2f}%")
    if failure_by_category:
        log.info("  Failure reasons by category:")
        for cat, n in sorted(failure_by_category.items(), key=lambda kv: -kv[1]):
            log.info(f"      {n:>8}  {cat}")
    log.info(f"  Output dir           : {OUTPUT_DIR}")
    log.info(f"  Summary              : {os.path.join(OUTPUT_DIR, SUMMARY_FILE)}")
    log.info(f"  Manifest             : {os.path.join(OUTPUT_DIR, MANIFEST_FILE)}")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
