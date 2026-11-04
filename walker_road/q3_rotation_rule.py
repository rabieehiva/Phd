"""Q3 step 3b -- bilinear rotational-stiffness rule at the girder ends.

Supervisor's suggestion: the support rotational stiffness is k1 up to a threshold load and
k2 = 0.75 k1 above it (the bearing stress shifts to the front edge as the girder end rotates, the
back edge unloads and the support becomes more flexible).

Threshold: a physical event, not a chosen number.
  * "pad back edge unloads": end rotation x pad half-length >= pad compression (checked at every
    load level by limit_states.py; for Walker Road it is never reached), and
  * "first girder cracking" (lambda = first cracking from limit_states.py) -- used for the plots,
    since it is the first physical event at the girder ends' load level.
k1 cases: the ends as they are (pad rotational stiffness only, practically pinned), both ends
half-fixed, north end half-fixed (fixity factor 0.5, as in the supervisor's paper).

For each case the bridge state at every load level (girder/deck cracking, stiffness only) is
analysed with and without the rule; OMA response = frequencies and MAC at the 9 accelerometers
relative to the same supports at zero load.

Run:  python q3_rotation_rule.py   ->  results/Q3/q3_rotation_rule.json (+ figure via q3_run.py)
"""
from __future__ import annotations

import json
import os

import numpy as np

import bc_identification as bc
import limit_states as ls
import mac_tools as mt
import walker_grillage as wg

K2_RATIO = 0.75
CASES = {"ends as they are (pad only)": (0.0, 0.0),
         "both ends half-fixed": (0.5, 0.5),
         "north end half-fixed": (0.5, 0.0)}
N_MODES = 5


def main():
    p0 = wg.calibrated_params()
    ls.LOSS = ls.s6_losses(verbose=False)[0]
    gp = ls.girder_properties(p0)
    m1, _, m_ref, _, _ = ls.static_response(p0, 1.0)
    r_dl = np.array([gp[j]["r_dead"] for j in range(7)] * 2)
    lam_crack = min(float(np.min(np.where(m1[j] > 0, gp[j]["m_cr_ll"] / m1[j], np.inf))) for j in range(7))
    acc = mt.accelerometers()
    lams = sorted(set(np.round(np.r_[np.arange(0.25, 4.0 + 1e-9, 0.25), lam_crack - 1e-3, lam_crack + 1e-3,
                                     list(ls.LIMIT_STATES.values())], 4)))
    states = {}
    edge_unloads = {}
    for lam in lams:
        st = ls.converge_state(p0, gp, lam, m_ref, r_dl)
        states[lam] = {**p0, "I_el": st["f_el"], "trans_el": st["trans"]}
        edge_unloads[lam] = bool(np.min(st["contact"]) < 1.0)
    lam_edge = next((l for l in lams if edge_unloads[l]), None)
    print(f"first girder cracking lambda = {lam_crack:.2f};  pad back edge unloads at: "
          f"{'never (up to 4 trucks/lane)' if lam_edge is None else f'{lam_edge:.2f}'}")

    out = dict(k2_ratio=K2_RATIO, threshold="first girder cracking", lam_threshold=lam_crack,
               lam_pad_edge_unloads=lam_edge, limit_states=ls.LIMIT_STATES, lams=[float(l) for l in lams], cases={})
    for name, (fn, fs) in CASES.items():
        base = bc.with_bc(p0, fn, fs, 1.0, 1.0)
        f0, ph0 = bc.oma(base, acc)
        f0, ph0 = f0[:N_MODES], ph0[:N_MODES]
        res = {}
        for rule in (False, True):
            df, mac = [], []
            for lam in lams:
                p = bc.with_bc(states[lam], fn, fs, 1.0, 1.0)
                if rule and lam > lam_crack:
                    p["kr"] = np.asarray(p["kr"]) * K2_RATIO          # k2 = 0.75 k1 at both ends
                f, ph = bc.oma(p, acc)
                pairs = {i: (m_, d_) for i, k, m_, d_ in mt.pair_by_mac(f0, ph0, f, ph, mac_min=0.0, df_max=1.0)}
                df.append([100 * pairs[i][1] for i in range(N_MODES)])
                mac.append([pairs[i][0] for i in range(N_MODES)])
            res["with rule" if rule else "without rule"] = dict(df_pct=df, mac=mac)
        out["cases"][name] = res
        k_jump = [a[0] - b[0] for a, b in zip(res["with rule"]["df_pct"], res["without rule"]["df_pct"])]
        print(f"{name:30s}: extra M1 change from the rule above the threshold: {min(k_jump):+.2f} %")
    os.makedirs("results/Q3", exist_ok=True)
    with open("results/Q3/q3_rotation_rule.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
