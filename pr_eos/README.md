# pr_eos/ — Peng-Robinson equation of state

The core PR-EOS implementation lives at **`cop/refrig/eos/peng_robinson.py`**, as part of the integrated `cop/` package, not duplicated here. This is a deliberate structural choice, not an oversight: `refrig`'s own design rationale (see `cop/PLATFORM_README.md` → "Why `run_cycle` is a pure function") is that the EOS and the cycle solver are tightly coupled by construction — `run_cycle(fluid, spec)` needs both to stay a single pure function, which is what makes sensitivity/uncertainty propagation "free" (see that section for the concrete example). Splitting the EOS out into a standalone `pr_eos` package would break that.

This folder holds the **validation and standalone-utility scripts** that don't belong inside the `cop/refrig/` package proper:

- `validation/validate_pr_binary_vs_coolprop.py` — validates the PR-EOS binary-mixture implementation against CoolProp reference values. (An identical duplicate of this script existed in three places in the original working tree — `Binary_Data_Generation/`, `PR_EOS_Validation/Binary_Data_Generation/`, and one more nested copy; only one canonical copy is kept here.)
- `pr_eos_cycle.py` — a standalone PR-EOS + cycle script (predates the `refrig` package's `cycle/vcc.py`; kept for reference/comparison).
- `cp_ig_pipeline.py`, `cp_joback_validation.py`, `validate_cpig.py` — ideal-gas heat capacity Cp(T) pipeline and validation against the Joback group-contribution method (feeds `refrig/core/cp.py`'s `CpProvider` interface — see `cop/PLATFORM_README.md` roadmap, "V2: ML Cp(T)").

## Peng-Robinson constants — a documented precision fix

`cop/refrig/eos/peng_robinson.py` uses full-double-precision PR constants (`OMEGA_A = 0.4572355289213822`, not the textbook-rounded `0.45724`) because the rounded values displace the critical compressibility factor `Zc` enough to fail the critical-point test — see the docstring in that file and "Notes on the port from `COP.ipynb`" in `cop/PLATFORM_README.md` for the full derivation. If you're comparing against a script that still uses the rounded constants (e.g. `pr_eos_cycle.py` above may), expect ~0.002% COP differences away from the critical point and a possible pass/fail flip exactly at it.
