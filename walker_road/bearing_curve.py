"""Load-displacement (compression) curve of one elastomeric pad as modelled in limit_states.py:
tangent stiffness k_v(R) = k_v0 (1 - eps_DL)^2 / (1 - eps)^2, eps = R / (k_v0 h_rt), integrated
to delta(R) = int dR / k_v(R). Compared with linear springs (Goodco schedule and calibrated)."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import walker_grillage as wg
import limit_states as ls

p0 = wg.calibrated_params()
kv0 = wg.BRG_KV * float(np.mean(p0["kv"]))           # kN/m, calibrated
d = json.load(open("results/limit_states.json"))
r_dl = float(np.mean(d["stage1"]["bearing_R_dead"]))
eps_dl = r_dl / (kv0 * ls.H_RT)

R = np.linspace(0, 1600, 801)                          # kN
eps = R / (kv0 * ls.H_RT)
k_t = kv0 * (1 - eps_dl) ** 2 / (1 - eps) ** 2          # tangent stiffness, kN/m
delta = np.concatenate([[0], np.cumsum(np.diff(R) / (0.5 * (k_t[1:] + k_t[:-1])))]) * 1e3   # mm

marks = {"dead load": r_dl}
for lam in ("1.0", "4.0", "7.5"):
    if lam in d["stage1"]["iteration"]:
        marks[f"λ = {float(lam):g} (most loaded pad)"] = max(d["stage1"]["iteration"][lam]["bearing_R"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(delta, R, "k-", lw=2, label="model rule (stiffening)")
a1.plot(R / (wg.BRG_KV) * 1e3, R, "--", color="C0", label=f"linear, Goodco schedule {wg.BRG_KV / 1e3:.1f} kN/mm")
a1.plot(R / kv0 * 1e3, R, ":", color="C1", label=f"linear, calibrated {kv0 / 1e3:.0f} kN/mm")
for (lab, r), c in zip(marks.items(), ("C2", "C3", "C4", "C5")):
    a1.plot(np.interp(r, R, delta), r, "o", color=c, label=f"{lab}: {r:.0f} kN")
a1.set_xlabel("vertical compression of the pad (mm)")
a1.set_ylabel("vertical load on the pad (kN)")
a1.set_title("Pad load-displacement (600 x 250 x 80, 70 mm rubber)", fontsize=9)
a1.grid(alpha=0.3)
a1.legend(fontsize=7)
a2.plot(R, k_t / kv0 * 100, "k-", lw=2)
for (lab, r), c in zip(marks.items(), ("C2", "C3", "C4", "C5")):
    a2.plot(r, np.interp(r, R, k_t / kv0 * 100), "o", color=c)
a2.set_xlabel("vertical load on the pad (kN)")
a2.set_ylabel("stiffness (slope) / calibrated value (%)")
a2.set_title("Slope of the curve = spring used in the modal analysis", fontsize=9)
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/bearing_load_displacement.png", dpi=150)
for lab, r in marks.items():
    print(f"{lab:28s} R = {r:6.0f} kN  delta = {np.interp(r, R, delta):5.2f} mm  k/k_cal = {np.interp(r, R, k_t / kv0):.3f}")
