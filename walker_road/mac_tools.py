"""Sensor layout (total-station survey) in the grillage frame, FE mode shapes at the
accelerometers, MAC-based pairing and measured mode-shape loading."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import walker_grillage as wg

SENSOR_FILE = os.path.join(os.path.dirname(__file__), "..", "sensors.parquet")
# midspan girder gauges used to register the survey frame (SG02 sits 0.45 m off midspan)
MIDSPAN_GAUGES = {"SG01": 0, "SG04": 2, "SG08": 3, "SG12": 4, "SG14": 5, "SG15": 6}


def load_sensors(path: str = SENSOR_FILE) -> pd.DataFrame:
    """Survey coordinates (x along girders, y across, mm) mapped into the grillage frame.
    Survey frame (confirmed by the monitoring team): y = 0 at the west deck edge, x = 0 at the
    north end, so G1 is the west girder and x runs north -> south.

    y: offset only, so the seven girder-gauge lines average onto the model girder lines.
    x: shift so the midspan gauges fall on their girders' midspans (model skew, 29.6 deg).
    Every sensor is then snapped to its nearest girder line (all gauges and accelerometers
    are mounted on girders) and given its fraction of span xi on that girder."""
    df = pd.read_parquet(path).copy()
    df["x_s"] = df.x_mm / 1000.0
    df["y_s"] = df.y_mm / 1000.0
    sg = df[df.role == "girder_strain"]
    lines = np.sort(sg.y_s.round(3).unique())
    if len(lines) != wg.N_GIRDER:
        raise ValueError(f"expected {wg.N_GIRDER} girder-gauge lines, found {len(lines)}")
    dy = float(np.mean(wg.Y_GIRDER - lines))
    shifts = [float(sg.loc[sg.sensor == s, "x_s"].iloc[0]) - wg.X0_GIRDER[j] - wg.L_SPAN / 2
              for s, j in MIDSPAN_GAUGES.items()]
    dx = float(np.median(shifts))
    df["x"] = df.x_s - dx
    df["y"] = df.y_s + dy
    df["girder"] = [int(np.argmin(np.abs(wg.Y_GIRDER - y))) for y in df.y]
    df["offset_from_girder"] = df.y - wg.Y_GIRDER[df.girder.values]
    df["xi"] = (df.x - wg.X0_GIRDER[df.girder.values]) / wg.L_SPAN
    df.attrs.update(dx=dx, dy=dy, shift_spread=float(np.ptp(shifts)))
    return df


def accelerometers(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_sensors() if df is None else df
    return df[df.role == "acceleration"].sort_values("sensor").reset_index(drop=True)


def fe_at_sensors(shapes: np.ndarray, sens: pd.DataFrame) -> np.ndarray:
    """Vertical FE shapes (n_modes, 7, N_SEG+1) sampled at the surveyed sensor positions ->
    (n_modes, n_sensors): linear along the two girder lines either side of the sensor (each at
    its own skewed station), lever rule between them, linear extrapolation outside G1/G7."""
    st = np.arange(wg.N_SEG + 1) / wg.N_SEG
    out = np.zeros((shapes.shape[0], len(sens)))
    for k, (x, y) in enumerate(zip(sens.x.values, sens.y.values)):
        j = int(np.clip(np.searchsorted(wg.Y_GIRDER, y) - 1, 0, wg.N_GIRDER - 2))
        t = (y - wg.Y_GIRDER[j]) / wg.S_GIRDER
        for jj, w in ((j, 1 - t), (j + 1, t)):
            xi = (x - wg.X0_GIRDER[jj]) / wg.L_SPAN
            out[:, k] += w * np.array([np.interp(xi, st, shapes[m, jj], left=0.0, right=0.0)
                                       for m in range(shapes.shape[0])])
    return out


def mac_matrix(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """MAC between rows of phi_a (n_a, n_s) and rows of phi_b (n_b, n_s), real or complex."""
    num = np.abs(phi_a.conj() @ phi_b.T) ** 2
    den = np.outer(np.real((phi_a.conj() * phi_a).sum(1)), np.real((phi_b.conj() * phi_b).sum(1)))
    return num / den


def realise(phi: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate a complex shape to its best real approximation; returns (real shape, MPC)."""
    phi = np.asarray(phi)
    if not np.iscomplexobj(phi):
        return phi.astype(float), 1.0
    sxx = np.real(phi) @ np.real(phi)
    syy = np.imag(phi) @ np.imag(phi)
    sxy = np.real(phi) @ np.imag(phi)
    eta = (syy - sxx) / (2 * sxy) if sxy != 0 else 0.0
    lam = (sxx + syy) / 2 + abs(sxy) * np.sqrt(eta ** 2 + 1)
    mpc = ((2 * lam - sxx - syy) / (sxx + syy)) ** 2 if (sxx + syy) > 0 else 1.0
    ang = 0.5 * np.angle(np.sum(phi ** 2))
    return np.real(phi * np.exp(-1j * ang)), float(mpc)


def load_measured_shapes(path: str, sensor_order) -> tuple[list, np.ndarray, np.ndarray]:
    """Measured mode shapes. Accepts parquet/CSV in long form (mode, sensor, value | re, im |
    abs, phase_deg) or wide form (sensor column + one column per mode, or mode column + one
    column per sensor). Returns (mode names, shapes (n_modes, n_sensors) real, MPC per mode)."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "mode" in cols and "sensor" in cols:
        m, s = cols["mode"], cols["sensor"]
        if "re" in cols and "im" in cols:
            val = df[cols["re"]] + 1j * df[cols["im"]]
        elif "abs" in cols and ("phase_deg" in cols or "phase" in cols):
            ph = df[cols.get("phase_deg", cols.get("phase"))]
            val = df[cols["abs"]] * np.exp(1j * np.radians(ph))
        else:
            val = df[cols.get("value", cols.get("phi"))]
        wide = pd.DataFrame({"mode": df[m], "sensor": df[s], "v": val}).pivot(index="mode", columns="sensor", values="v")
    elif "sensor" in cols:
        wide = df.set_index(cols["sensor"]).T
    else:
        wide = df.set_index(df.columns[0])
    wide = wide[list(sensor_order)]
    names, shapes, mpc = [], [], []
    for name, row in wide.iterrows():
        r, q = realise(row.values.astype(complex) if np.iscomplexobj(row.values) else row.values.astype(float))
        names.append(str(name))
        shapes.append(r / np.abs(r).max())
        mpc.append(q)
    return names, np.array(shapes), np.array(mpc)


def pair_by_mac(f_meas, phi_meas, f_fe, phi_fe, mac_min=0.6, df_max=0.30, w_freq=1.0):
    """Hungarian assignment on (1 - MAC) + w_freq |df/f|, rejecting pairs below mac_min or with
    a frequency gap above df_max. Returns list of (i_meas, k_fe, MAC, df/f)."""
    mac = mac_matrix(phi_meas, phi_fe)
    dfr = np.abs(f_fe[None, :] - f_meas[:, None]) / f_meas[:, None]
    cost = (1 - mac) + w_freq * dfr
    cost[(mac < mac_min) | (dfr > df_max)] = 1e3
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(k), float(mac[i, k]), float((f_fe[k] - f_meas[i]) / f_meas[i]))
            for i, k in zip(r, c) if cost[i, k] < 1e3]


if __name__ == "__main__":
    df = load_sensors()
    pd.set_option("display.width", 200)
    print("frame shift dx = %.3f m, dy = %.3f m, midspan-gauge spread %.3f m" %
          (df.attrs["dx"], df.attrs["dy"], df.attrs["shift_spread"]))
    print(df[["sensor", "role", "x", "y", "girder", "offset_from_girder", "xi"]].round(3).to_string())
    df.to_csv("results/sensors_model_frame.csv", index=False)
