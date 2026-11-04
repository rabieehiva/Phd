"""Frequency + mode-shape (MAC) calibration using the surveyed accelerometer layout.

Usage
  py -V:3.12 run_calibration_mac.py <shapes file>   # measured shapes (parquet or csv)
  py -V:3.12 run_calibration_mac.py --synthetic     # recovery test with a known "truth"

Shapes file: long form (mode, sensor, value | re, im | abs, phase_deg) or wide form; sensor
ids AC01..AC09 as in sensors.parquet; mode names M1..M10 as in the OMA report (an optional
`freq` column overrides the report frequencies)."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import mac_tools as mt
import measured as ms
import walker_grillage as wg

N_EIG = 30
SIGMA_F = 0.01          # 1 % per frequency
SIGMA_MAC = 0.05        # a MAC deficit of 0.05 weighs as much as a 1 % frequency error
MAC_USE = 0.70          # a measured mode enters the objective if its initial MAC reaches this

# name: (transform, lower, upper, prior mean in search space, prior sd)
BASE = {
    "Eg": ("log", 0.8, 2.2, np.log10(1.3), 0.15),
    "Ed": ("log", 0.6, 2.2, np.log10(1.3), 0.15),
    "barrier": ("lin", 0.0, 1.0, 1.0, 0.5),
    "kv": ("log", 0.5, 5.0, 0.0, 0.3),
    "tors": ("log", 0.5, 3.0, 0.0, 0.3),
    "diaph": ("log", 0.3, 3.0, 0.0, 0.3),
}
# asymmetric set: side-specific barrier and abutment-specific pad stiffness replace the common
# factors (only mode shapes can separate the sides: frequencies are mirror-blind).
# barrier_G1 = west barrier, barrier_G7 = east (median-side) barrier; kv_N / kv_S = pads at the
# north / south abutment (survey: y = 0 west edge, x = 0 north end).
EXTENDED = {
    "Eg": BASE["Eg"], "Ed": BASE["Ed"], "tors": BASE["tors"], "diaph": BASE["diaph"],
    "barrier_G1": ("lin", 0.0, 1.2, 1.0, 0.5),
    "barrier_G7": ("lin", 0.0, 1.2, 1.0, 0.5),
    "kv_N": ("log", 0.5, 5.0, 0.0, 0.3),
    "kv_S": ("log", 0.5, 5.0, 0.0, 0.3),
}


def make_params(vals: dict) -> dict:
    p = wg.default_params()
    for k in ("Eg", "Ed", "barrier", "tors", "diaph", "trans", "mass"):
        if k in vals:
            p[k] = float(vals[k])
    if "barrier_G1" in vals or "barrier_G7" in vals:           # side-specific, absolute
        p["barrier"] = 1.0
        p["barrier_side"] = np.array([vals.get("barrier_G1", vals.get("barrier", 1.0)),
                                      vals.get("barrier_G7", vals.get("barrier", 1.0))])
    kv = float(vals.get("kv", 1.0))
    p["kv"] = np.concatenate([np.full(7, vals.get("kv_N", kv)), np.full(7, vals.get("kv_S", kv))])
    for k in ("kh", "kt", "kr"):
        p[k] = np.full(14, float(vals.get(k, 1.0)))
    return p


def fe_modes(vals, acc):
    m = wg.build(make_params(vals))
    f, sh, v = wg.modal(m, N_EIG)
    fd, shd, _ = wg.deck_modes(f, sh, v)
    return fd, mt.fe_at_sensors(shd, acc), shd


class Problem:
    def __init__(self, names, f_meas, phi_meas, spec, fixed, use=None):
        self.names, self.f, self.phi = names, np.asarray(f_meas), np.asarray(phi_meas)
        self.spec, self.fixed = spec, fixed
        self.keys = list(spec)
        self.use = np.ones(len(names), bool) if use is None else np.asarray(use)
        self.lo = np.array([s[1] if s[0] == "lin" else np.log10(s[1]) for s in spec.values()])
        self.hi = np.array([s[2] if s[0] == "lin" else np.log10(s[2]) for s in spec.values()])
        self.pm = np.array([s[3] for s in spec.values()])
        self.ps = np.array([s[4] for s in spec.values()])

    def vals(self, x):
        v = dict(self.fixed)
        for k, xi in zip(self.keys, x):
            v[k] = xi if self.spec[k][0] == "lin" else 10 ** xi
        return v

    def x_of(self, vals):
        return np.array([vals[k] if self.spec[k][0] == "lin" else np.log10(vals[k]) for k in self.keys])

    def evaluate(self, x, mac_min=0.3):
        fd, phi_fe, _ = fe_modes(self.vals(x), ACC)
        pairs = mt.pair_by_mac(self.f, self.phi, fd, phi_fe, mac_min=mac_min, df_max=0.35)
        got = {i: (k, mac, dfr) for i, k, mac, dfr in pairs}
        return fd, phi_fe, got

    def residuals(self, x, use_mac=True):
        fd, phi_fe, got = self.evaluate(x)
        r = []
        for i in np.nonzero(self.use)[0]:
            if i in got:
                k, mac, dfr = got[i]
                r.append(dfr / SIGMA_F)
                if use_mac:
                    r.append((1 - mac) / SIGMA_MAC)
            else:                                   # lost its partner: constant penalty
                r += [0.35 / SIGMA_F] + ([0.7 / SIGMA_MAC] if use_mac else [])
        return np.concatenate([r, (x - self.pm) / self.ps])

    def solve(self, x0, use_mac=True):
        res = least_squares(lambda x: self.residuals(x, use_mac), np.clip(x0, self.lo + 1e-9, self.hi - 1e-9),
                            bounds=(self.lo, self.hi), diff_step=0.01, x_scale="jac", xtol=1e-8, ftol=1e-8)
        cov = np.linalg.pinv(res.jac.T @ res.jac)
        return res, np.sqrt(np.diag(cov)), cov


def report_pairs(prob, x, label):
    fd, phi_fe, got = prob.evaluate(x, mac_min=0.0)
    rows = []
    for i, n in enumerate(prob.names):
        if i in got:
            k, mac, dfr = got[i]
            fam = wg.classify(fe_modes(prob.vals(x), ACC)[2][k])[:2] if False else None
            rows.append(dict(mode=n, f_meas=float(prob.f[i]), f_fe=float(fd[k]), df_pct=100 * dfr, mac=mac,
                             fe_index=int(k), used=bool(prob.use[i])))
        else:
            rows.append(dict(mode=n, f_meas=float(prob.f[i]), f_fe=None, df_pct=None, mac=None, fe_index=None,
                             used=bool(prob.use[i])))
    print(label)
    for r in rows:
        print("  %-4s f=%7.3f  FE=%s  df=%s  MAC=%s  %s" % (
            r["mode"], r["f_meas"], "-" if r["f_fe"] is None else "%7.3f" % r["f_fe"],
            "-" if r["df_pct"] is None else "%+6.2f%%" % r["df_pct"],
            "-" if r["mac"] is None else "%.3f" % r["mac"], "used" if r["used"] else ""))
    return rows


def start_values():
    d = json.load(open("results/calibration.json"))["calibrated"]["values"]
    return {k: float(v) for k, v in d.items()}


def synthetic_data(rng):
    """Truth = frequency-calibrated model with an asymmetric twist the frequencies cannot see."""
    truth = start_values()
    truth.update(barrier_G1=1.0, barrier_G7=0.75, kv_N=truth["kv"], kv_S=1.2 * truth["kv"])
    truth["Eg"] *= 1.03
    fd, phi, shd = fe_modes(truth, ACC)
    sel = list(range(8))
    f = fd[sel] * (1 + 0.003 * rng.standard_normal(len(sel)))
    ph = phi[sel] / np.abs(phi[sel]).max(1, keepdims=True)
    ph = ph + 0.03 * rng.standard_normal(ph.shape)
    names = [f"S{k + 1}" for k in sel]
    return truth, names, f, ph


def main(argv):
    global ACC
    ACC = mt.accelerometers()
    out = {"sensors": ACC[["sensor", "x", "y", "xi"]].to_dict("records"),
           "sigma_f": SIGMA_F, "sigma_mac": SIGMA_MAC}
    rng = np.random.default_rng(5)
    if argv and argv[0] == "--synthetic":
        truth, names, f, ph = synthetic_data(rng)
        out["truth"] = truth
        tag = "synthetic"
    else:
        names, ph, mpc = mt.load_measured_shapes(argv[0], ACC.sensor.tolist())
        raw = pd.read_parquet(argv[0]) if argv[0].endswith(".parquet") else pd.read_csv(argv[0])
        if "freq" in [c.lower() for c in raw.columns]:
            fc = [c for c in raw.columns if c.lower() == "freq"][0]
            mc = [c for c in raw.columns if c.lower() == "mode"][0]
            fmap = raw.groupby(mc)[fc].first().to_dict()
            f = np.array([fmap[n] for n in names])
        else:
            f = np.array([ms.freq_of(n) for n in names])
        out["mpc"] = dict(zip(names, mpc.tolist()))
        tag = "measured"
    start = start_values()
    start_ext = {**start, "barrier_G1": start["barrier"], "barrier_G7": start["barrier"],
                 "kv_N": start["kv"], "kv_S": start["kv"]}
    spec = BASE
    fixed = {k: v for k, v in start.items() if k not in spec}
    prob = Problem(names, f, ph, spec, fixed)
    x0 = prob.x_of(start)
    # orientation check: the survey frame could be mirrored relative to the model
    if tag == "measured":
        orient = {}
        for lab, fx, fy in (("as surveyed", False, False), ("x mirrored", True, False),
                            ("y mirrored", False, True), ("rotated 180 deg", True, True)):
            a2 = ACC.copy()
            if fx:
                a2["x"] = wg.X0_GIRDER.mean() * 2 + wg.L_SPAN - a2.x
            if fy:
                a2["y"] = -a2.y
            fdo, phio, _ = fe_modes(prob.vals(x0), a2)
            mm = mt.mac_matrix(ph, phio[:12])
            orient[lab] = float(np.mean(mm.max(1)[:3]))
        out["orientation_mean_mac_M1_M3"] = orient
        print("orientation check (mean best MAC, first three modes):", {k: round(v, 3) for k, v in orient.items()})
    # initial MAC matrix and which modes are usable
    fd0, phi0, _ = fe_modes(prob.vals(x0), ACC)
    out["mac_initial"] = mt.mac_matrix(ph, phi0[:12]).tolist()
    out["fe_freq_initial"] = fd0[:12].tolist()
    rows0 = report_pairs(prob, x0, "initial (frequency-calibrated model):")
    prob.use = np.array([r["mac"] is not None and r["mac"] >= MAC_USE for r in rows0])
    out["initial"] = rows0
    results = {}
    for label, use_mac, sp in (("frequency + MAC, base set", True, BASE),
                               ("frequency only, asymmetric set", False, EXTENDED),
                               ("frequency + MAC, asymmetric set", True, EXTENDED)):
        p2 = Problem(names, f, ph, sp, {k: v for k, v in start.items() if k not in sp and k not in ("barrier", "kv")}
                     if sp is EXTENDED else {k: v for k, v in start.items() if k not in sp}, prob.use)
        x0s = p2.x_of(start_ext if sp is EXTENDED else start)
        res, sd, cov = p2.solve(x0s, use_mac)
        prob_l = p2
        vals = prob_l.vals(res.x)
        rows = report_pairs(prob_l, res.x, label + ":")
        results[label] = dict(values={k: float(vals[k]) for k in sp}, sd_search=sd.tolist(), keys=prob_l.keys,
                              cost=float(res.cost), pairs=rows,
                              corr=(cov / np.outer(sd, sd)).tolist())
        print("  params", {k: round(float(vals[k]), 3) for k in sp})
        if tag == "synthetic":
            print("  truth ", {k: round(float(out["truth"].get(k, 1.0)), 3) for k in sp})
        fd1, phi1, _ = fe_modes(vals, ACC)
        results[label]["mac_final"] = mt.mac_matrix(ph, phi1[:12]).tolist()
        results[label]["fe_freq_final"] = fd1[:12].tolist()
    out["results"] = results
    out["names"] = names
    out["f_meas"] = f.tolist()
    with open(f"results/calibration_mac_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])
