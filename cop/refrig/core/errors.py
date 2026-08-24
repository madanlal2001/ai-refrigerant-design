"""Exception hierarchy.

Deliberately split so the UI (and later, batch screening) can tell the
difference between "this fluid cannot do this cycle" (a *result*) and
"the solver broke" (a *bug or a bad guess*).
"""


class RefrigError(Exception):
    """Base for everything raised by this package."""


class InfeasibleCycleError(RefrigError):
    """The requested cycle is physically impossible for this fluid.

    Example: T_cond >= Tc (transcritical). This is a legitimate screening
    outcome, not a failure. Callers should report it, not crash.
    """


class ConvergenceError(RefrigError):
    """An iterative solver failed to converge within its budget."""


class EOSRootError(RefrigError):
    """The PR cubic yielded no physically admissible root (Z > B)."""


class PropertyError(RefrigError):
    """A required property is missing, non-physical, or inconsistent."""
