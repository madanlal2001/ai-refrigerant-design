"""Physical constants. SI only. Single source of truth."""

# Universal gas constant [J/(mol*K)] -- CODATA 2018 exact value.
# NOTE: COP.ipynb used 8.314. The difference is ~0.005%, negligible for COP but
# free to get right.
R = 8.314462618

BAR_TO_PA = 1.0e5
PA_TO_BAR = 1.0e-5
