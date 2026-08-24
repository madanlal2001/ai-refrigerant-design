"""Tests for the thermodynamic core.

Two kinds live here:
  * unit tests -- internal consistency, no external truth needed
  * validation tests -- against reference fluids with literature properties

The validation set is deliberately thin in V1 (R134a, propane). It should grow.
Reference Psat values are from NIST/REFPROP; the 1% tolerance reflects what
Peng-Robinson can actually deliver, not what we wish it delivered.
"""

import numpy as np
import pytest

from refrig import (Component, CycleSpec, CycleStatus, Fluid, Poly4Cp,
                    PengRobinson, Source, psat, run_cycle)
from refrig.core.constants import R

# np.trapezoid is numpy>=2.0; np.trapz is deprecated in numpy>=2.0.
# This box runs numpy 1.26 (torch 2.2.1 predates numpy 2.0 support).
_trapz = getattr(np, "trapezoid", None) or np.trapz


def make_fluid(Tc, Pc_bar, omega, cp, name="test"):
    c = Component(canonical_smiles="X", name=name)
    c.set("Tc", Tc, Source.LITERATURE)
    c.set("Pc", Pc_bar * 1e5, Source.LITERATURE)
    c.set("omega", omega, Source.LITERATURE)
    c.cp = cp
    return Fluid.pure(c)


R134A = dict(Tc=374.21, Pc_bar=40.593, omega=0.32684,
             cp=Poly4Cp(19.4, 0.2585, -1.29e-4, 2.05e-8, 0.0))
PROPANE = dict(Tc=369.89, Pc_bar=42.512, omega=0.1521,
               cp=Poly4Cp(-4.224, 0.3063, -1.586e-4, 3.215e-8, 0.0))


# ---------------------------------------------------------------- units ----

def test_property_override_keeps_audit_chain():
    c = Component(canonical_smiles="CC")
    c.set("Tc", 374.0, Source.PREDICTED, model_id="dmpnn@abc")
    c.override("Tc", 378.0)
    p = c.get("Tc")
    assert p.value == 378.0
    assert p.source is Source.USER_OVERRIDE
    assert p.was_overridden
    assert p.overrides.value == 374.0
    assert p.overrides.model_id == "dmpnn@abc"


def test_poly4_integrals_match_numerical_quadrature():
    cp = Poly4Cp(19.4, 0.2585, -1.29e-4, 2.05e-8, 1e-12)
    T_ref, T = 233.15, 350.0
    grid = np.linspace(T_ref, T, 200001)
    h_num = _trapz(cp.cp_ig(grid), grid)
    assert cp.h_ig(T, T_ref) == pytest.approx(h_num, rel=1e-8)
    s_num = _trapz(cp.cp_ig(grid) / grid, grid)
    s_analytic = cp.s_ig(T, 1e5, T_ref, 1e5)  # P == P_ref kills the -R ln term
    assert s_analytic == pytest.approx(s_num, rel=1e-8)


def test_fluid_composition_defaults_and_validates():
    c = Component(canonical_smiles="CC")
    f = Fluid.pure(c)
    assert f.is_pure and f.n_components == 1
    assert f.composition.tolist() == [1.0]
    assert f.kij.shape == (1, 1)


# ------------------------------------------------------------------ eos ----

def test_critical_point_is_reproduced():
    """At (Tc, Pc) the PR cubic must have a triple root at Z = 0.3074."""
    f = make_fluid(**R134A)
    eos = PengRobinson(f)
    Zv, Zl = eos.Z_roots(eos.Tc, eos.Pc)
    assert Zv == pytest.approx(0.30740, abs=1e-4)
    assert Zl == pytest.approx(0.30740, abs=1e-4)


def test_dadT_matches_finite_difference():
    f = make_fluid(**R134A)
    eos = PengRobinson(f)
    T, dT = 300.0, 1e-4
    fd = (eos.a_of_T(T + dT) - eos.a_of_T(T - dT)) / (2 * dT)
    assert eos.dadT(T) == pytest.approx(fd, rel=1e-6)


def test_ideal_gas_limit_has_zero_departure():
    """At very low P the vapour must approach Z -> 1 and zero departures."""
    f = make_fluid(**R134A)
    eos = PengRobinson(f)
    T, P = 400.0, 1.0  # 1 Pa
    Zv, _ = eos.Z_roots(T, P)
    A, B = eos.AB(T, P)
    assert Zv == pytest.approx(1.0, abs=1e-4)
    assert eos.h_departure(T, Zv, B) == pytest.approx(0.0, abs=1.0)
    assert eos.s_departure(T, Zv, B) == pytest.approx(0.0, abs=1e-2)


# ------------------------------------------------------------------ vle ----

def test_psat_fugacities_are_equal_at_convergence():
    f = make_fluid(**R134A)
    eos = PengRobinson(f)
    r = psat(eos, 283.15)
    assert r.converged
    A, B = eos.AB(283.15, r.P)
    assert eos.ln_phi(r.Z_vapor, A, B) == pytest.approx(
        eos.ln_phi(r.Z_liquid, A, B), abs=1e-7)


def test_psat_at_critical_temperature_equals_pc():
    f = make_fluid(**R134A)
    eos = PengRobinson(f)
    r = psat(eos, 0.999 * eos.Tc)
    assert r.P == pytest.approx(eos.Pc, rel=0.02)


@pytest.mark.parametrize("fluid_kw, T, P_ref_bar", [
    (R134A, 283.15, 4.1461),
    (R134A, 313.15, 10.166),
    (PROPANE, 283.15, 6.3660),
    (PROPANE, 313.15, 13.694),
])
def test_psat_against_reference_data(fluid_kw, T, P_ref_bar):
    """VALIDATION: PR-predicted Psat vs NIST, using literature Tc/Pc/omega."""
    eos = PengRobinson(make_fluid(**fluid_kw))
    r = psat(eos, T)
    assert r.P * 1e-5 == pytest.approx(P_ref_bar, rel=0.01)


def test_psat_above_tc_is_infeasible_not_a_crash():
    from refrig.core.errors import InfeasibleCycleError
    eos = PengRobinson(make_fluid(**R134A))
    with pytest.raises(InfeasibleCycleError):
        psat(eos, 400.0)


# ---------------------------------------------------------------- cycle ----

def test_cycle_energy_balance_closes():
    f = make_fluid(**R134A)
    r = run_cycle(f, CycleSpec(283.15, 313.15, superheat=5, subcool=5,
                               eta_isen=0.7))
    assert r.ok
    assert r.Q_cond == pytest.approx(r.Q_evap + r.W_comp, rel=1e-10)


def test_isentropic_compression_conserves_entropy():
    f = make_fluid(**R134A)
    r = run_cycle(f, CycleSpec(283.15, 313.15, superheat=5))
    assert r.ok
    assert r.state("2s").state.s == pytest.approx(r.state("1").state.s, rel=1e-8)


def test_throttle_is_isenthalpic():
    f = make_fluid(**R134A)
    r = run_cycle(f, CycleSpec(283.15, 313.15, subcool=5))
    assert r.ok
    assert r.state("4").state.h == pytest.approx(r.state("3").state.h, rel=1e-12)


def test_lower_efficiency_lowers_cop():
    f = make_fluid(**R134A)
    hi = run_cycle(f, CycleSpec(283.15, 313.15, superheat=5, eta_isen=1.0))
    lo = run_cycle(f, CycleSpec(283.15, 313.15, superheat=5, eta_isen=0.7))
    assert lo.COP < hi.COP
    assert lo.COP == pytest.approx(hi.COP * 0.7, rel=0.02)


def test_cop_is_below_carnot():
    """A hard physical bound. If this ever fails, the EOS is wrong."""
    f = make_fluid(**R134A)
    spec = CycleSpec(283.15, 313.15)
    r = run_cycle(f, spec)
    carnot = spec.T_evap / (spec.T_cond - spec.T_evap)
    assert r.ok and 0 < r.COP < carnot


def test_reference_state_does_not_affect_cop():
    """The H/S datum is arbitrary; COP must not depend on it."""
    f = make_fluid(**R134A)
    a = run_cycle(f, CycleSpec(283.15, 313.15, T_ref=233.15, P_ref=101325.0))
    b = run_cycle(f, CycleSpec(283.15, 313.15, T_ref=298.15, P_ref=1e5))
    assert a.COP == pytest.approx(b.COP, rel=1e-9)


def test_transcritical_is_a_result_not_an_exception():
    f = make_fluid(**PROPANE)
    r = run_cycle(f, CycleSpec(283.15, 380.0))
    assert r.status is CycleStatus.TRANSCRITICAL
    assert r.COP is None
    assert "Tc" in r.message


def test_invalid_spec_is_reported_not_raised():
    f = make_fluid(**R134A)
    r = run_cycle(f, CycleSpec(T_evap=313.15, T_cond=283.15))
    assert r.status is CycleStatus.INFEASIBLE


def test_missing_property_is_reported_not_raised():
    c = Component(canonical_smiles="CC", name="incomplete")
    c.set("Tc", 300.0, Source.PREDICTED)
    c.cp = Poly4Cp(30.0)
    r = run_cycle(Fluid.pure(c), CycleSpec())
    assert r.status is CycleStatus.INFEASIBLE
    assert "missing Pc" in r.message


def test_run_cycle_is_pure_and_repeatable():
    f = make_fluid(**R134A)
    spec = CycleSpec(283.15, 313.15, superheat=3, eta_isen=0.8)
    assert run_cycle(f, spec).COP == run_cycle(f, spec).COP


def test_cop_sensitivity_to_tc_is_computable():
    """The thesis question, expressed as a test: perturb a Property, re-run.

    This is only three lines because run_cycle is a pure function of (fluid,
    spec). It is the payoff for the whole design.
    """
    f = make_fluid(**R134A)
    base = run_cycle(f, CycleSpec(283.15, 313.15)).COP
    f.component.override("Tc", R134A["Tc"] * 1.02)
    perturbed = run_cycle(f, CycleSpec(283.15, 313.15)).COP
    assert base != perturbed
    assert abs(perturbed - base) / base < 0.5   # sanity, not a physics claim
