"""Operational modal analysis results, Bridge Walker Road processing report (2026-09-23),
Section 6: SSI-COV over 17,027 hourly blocks, 2023-01-26 to 2025-07-26."""
import numpy as np

MODES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M9", "M8", "M10"]
FREQ = np.array([3.718, 4.263, 8.258, 9.024, 13.450, 13.610, 14.246, 15.831, 15.927, 28.843])
DAMP = np.array([1.05, 2.20, 1.75, 2.75, 1.67, 1.59, 1.16, 1.41, 1.38, 1.60]) / 100
PRESENT = np.array([100, 99, 97, 24, 100, 100, 93, 97, 96, 16]) / 100
MAC_STAB = np.array([0.998, 0.997, 0.999, 0.999, 0.909, 0.994, 0.991, 0.971, 0.938, 0.945])
TEMP_SENS = np.array([-0.176, -0.127, -0.161, -0.267, -0.175, -0.188, -0.163, -0.208, -0.241, -0.315])  # %/degC

# Mode families (m longitudinal half-waves, n transverse order) used for pairing with the
# grillage. M1/M2 are unambiguous from Fig. 5; M5/M6 are the close antisymmetric pair
# (2,0)/(2,1); M3 is the first transverse-bending mode (1,2). The FE ratio
# f(2,2)/f(2,0) = 1.16 matches M9/M5 or M8/M5 (1.18) rather than M7/M5 (1.06), so M7 is not
# paired; M4, M7, M8, M9 and M10 are compared after calibration only.
PAIR = {"M1": (1, 0), "M2": (1, 1), "M3": (1, 2), "M5": (2, 0), "M6": (2, 1)}
UNPAIRED = ["M4", "M7", "M9", "M8", "M10"]


def freq_of(name: str) -> float:
    return float(FREQ[MODES.index(name)])
