"""refrig -- molecular refrigerant screening core.

Pure Python. Imports no UI framework and no ML framework at module scope.
Streamlit imports this; this never imports Streamlit.
"""

from .core.component import Component, Fluid
from .core.constants import BAR_TO_PA, PA_TO_BAR, R
from .core.cp import ConstantCp, CpProvider, Poly4Cp
from .core.property import Property, Source
from .cycle.vcc import CycleResult, CycleSpec, CycleState, CycleStatus, run_cycle
from .eos.peng_robinson import EOSSettings, EOSState, PengRobinson, Phase
from .equilibrium.psat import SaturationResult, psat

__version__ = "1.0.0"

__all__ = [
    "Component", "Fluid", "Property", "Source",
    "CpProvider", "Poly4Cp", "ConstantCp",
    "PengRobinson", "EOSSettings", "EOSState", "Phase",
    "psat", "SaturationResult",
    "CycleSpec", "CycleState", "CycleResult", "CycleStatus", "run_cycle",
    "R", "BAR_TO_PA", "PA_TO_BAR", "__version__",
]
