"""Q3 step 4 -- removing the effect of the test load from OMA results at higher limit states.

At a load level lambda the bridge is measured WITH the test load on it (vehicles / blocks):
its modes reflect the damaged stiffness AND the added mass. The goal is to recover the modes
of the damaged bridge WITHOUT the added mass ("target" = the jacked case of limit_states.py),
using only things an engineer has: the measured loaded modes, the known test load (mass and
position), and the calibrated (undamaged) model.

Frequency corrections
  F1 Rayleigh / mass-normalised shapes:  f_corr = f_loaded * sqrt(1 + phi^T dM phi), phi = mass-
     normalised mode shape of the calibrated model at the load positions (supervisor's idea).
  F2 model ratio:  f_corr = f_loaded * f_0 / f_mass, with f_0, f_mass = undamaged model without /
     with the test-load mass (the model predicts how much the mass alone shifts each mode).
Mode-shape corrections (at the 9 accelerometers)
  S1 first-order perturbation with the model's mass-normalised modes:
     d phi_i = sum_{j != i} [ -w_i^2 (phi_j^T dM phi_i) / (w_i^2 - w_j^2) ] phi_j - 1/2 (phi_i^T dM phi_i) phi_i,
     phi_corr = phi_loaded - d phi_i.
  S2 model subtraction:  phi_corr = phi_loaded - (phi_mass - phi_0)  (model, mass-normalised, aligned).
Quality = error of the corrected frequency vs the target, and MAC(corrected, target).

Run:  python q3_load_removal.py   ->  results/Q3/q3_load_removal.json (figure + summary via q3_run.py)
"""
from __future__ import annotations

import json
import os

import numpy as np
import openseespy.opensees as ops
from scipy.optimize import linear_sum_assignment

import limit_states as ls
import mac_tools as mt
import walker_grillage as wg

N_EIG = 30
N_TRACK = 5


def modes(p):
    """All modes (vertical shapes at the 7 x 31 girder nodes, mass-normalised by OpenSees)."""
    m = wg.build(p)
    f, sh, v = wg.modal(m, N_EIG)
    return f, sh, v


class Follower:
    """Continuation tracking in lambda (MAC to the previous step + small frequency term)."""
    def __init__(self, f0, sh0):
        self.f, self.sh = f0.copy(), sh0.copy()

    def __call__(self, f, sh):
        macm = np.array([[wg.mac(a, b) for b in sh] for a in self.sh])
        cost = (1 - macm) + 0.5 * np.abs(f[None, :] - self.f[:, None]) / self.f[:, None]
        r, c = linear_sum_assignment(cost)
        idx = c[np.argsort(r)]
        self.f, self.sh = f[idx], sh[idx]
        return idx


def mu_matrix(sh, dm_nodes):
    """phi_a^T dM phi_b for all mode pairs (vertical translation of the added point masses)."""
    J = np.array([j for (j, i) in dm_nodes])
    I = np.array([i for (j, i) in dm_nodes])
    w = np.array(list(dm_nodes.values()))
    v = sh[:, J, I]                                   # (n_modes, n_points)
    return (v * w) @ v.T


def align(a, ref):
    return a if np.sum(a * ref) >= 0 else -a


def main():
    p0 = wg.calibrated_params()
    ls.LOSS = ls.s6_losses(verbose=False)[0]
    gp = ls.girder_properties(p0)
    _, _, m_ref, _, _ = ls.static_response(p0, 1.0)
    r_dl = np.array([gp[j]["r_dead"] for j in range(7)] * 2)
    acc = mt.accelerometers()

    f0a, sh0a, v0a = modes(p0)                       # undamaged model, all modes
    deck = np.where(v0a >= 0.6)[0]
    track0 = deck[:N_TRACK]
    f0, sh0 = f0a[track0], sh0a[track0]
    w0a = (2 * np.pi * f0a) ** 2
    fol = {c: Follower(f0, sh0) for c in ("target", "loaded", "mass")}

    lams = sorted(set(np.round(np.r_[np.arange(0.25, 4.0 + 1e-9, 0.25), list(ls.LIMIT_STATES.values())], 4)))
    rows = []
    for lam in lams:
        st = ls.converge_state(p0, gp, lam, m_ref, r_dl)
        p_dam = {**p0, "I_el": st["f_el"], "trans_el": st["trans"]}
        dm = ls.truck_mass(lam)
        res = {}
        for case, p in (("target", p_dam), ("loaded", {**p_dam, "extra_mass": dm}),
                        ("mass", {**p0, "extra_mass": dm})):
            f, sh, _ = modes(p)
            idx = fol[case](f, sh)
            res[case] = (f[idx], np.array([align(sh[k], sh0[n]) for n, k in enumerate(idx)]))
        ft, sht = res["target"]
        fl, shl = res["loaded"]
        fm, shm = res["mass"]
        # --- frequency corrections
        mu_all = mu_matrix(sh0a, dm)
        mu_ii = np.array([mu_all[k, k] for k in track0])
        f_F1 = fl * np.sqrt(1 + mu_ii)
        f_F2 = fl * f0 / fm
        # --- shape corrections (vertical shapes on the grid, then at the 9 accelerometers)
        sh_S1, sh_S2 = [], []
        for n, i in enumerate(track0):
            d = -0.5 * mu_all[i, i] * sh0a[i]
            for j in range(len(f0a)):
                if j != i and abs(w0a[i] - w0a[j]) > 1e-6 * w0a[i]:
                    d = d + (-w0a[i] * mu_all[j, i] / (w0a[i] - w0a[j])) * sh0a[j]
            sh_S1.append(shl[n] - d)
            sh_S2.append(shl[n] - (shm[n] - sh0[n]))
        at = lambda s: mt.fe_at_sensors(np.asarray(s), acc)
        A_t, A_l, A_1, A_2 = at(sht), at(shl), at(sh_S1), at(sh_S2)
        mac = lambda a, b: [float(mt.mac_matrix(a[k:k + 1], b[k:k + 1])[0, 0]) for k in range(N_TRACK)]
        pct = lambda f: (100 * (f - f0) / f0).tolist()
        rows.append(dict(lam=float(lam), df_target=pct(ft), df_loaded=pct(fl), df_F1=pct(f_F1), df_F2=pct(f_F2),
                         err_F1=(100 * (f_F1 - ft) / ft).tolist(), err_F2=(100 * (f_F2 - ft) / ft).tolist(),
                         err_loaded=(100 * (fl - ft) / ft).tolist(),
                         mac_loaded=mac(A_l, A_t), mac_S1=mac(A_1, A_t), mac_S2=mac(A_2, A_t),
                         mac_target_vs_base=mac(A_t, at(sh0)), mu_ii=mu_ii.tolist()))
        r = rows[-1]
        print(f"lam {lam:4.2f}  M1: target {r['df_target'][0]:+6.2f}%  loaded {r['df_loaded'][0]:+6.2f}%  "
              f"F1 err {r['err_F1'][0]:+5.2f}%  F2 err {r['err_F2'][0]:+5.2f}%   "
              f"MAC to target M1-M3: loaded {min(r['mac_loaded'][:3]):.3f}  S1 {min(r['mac_S1'][:3]):.3f}  "
              f"S2 {min(r['mac_S2'][:3]):.3f}")
    names = ["M1", "M2", "M3", "M5", "M6"]
    os.makedirs("results/Q3", exist_ok=True)
    with open("results/Q3/q3_load_removal.json", "w") as fh:
        json.dump(dict(names=names, f0=f0.tolist(), limit_states=ls.LIMIT_STATES, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
