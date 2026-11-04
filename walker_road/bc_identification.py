"""Q3 step 3 -- can OMA tell what the supports are doing at higher limit states?

For each limit state (load level lambda, structural state from limit_states.py: girder
cracking, deck cracking, bearings; stiffness only, i.e. test-load mass removed) and each
boundary-condition scenario (support fixity / pad stiffness, north and south ends), a "true"
bridge is built and its OMA data are simulated at the 9 surveyed accelerometers (first 6 deck
modes, noise added). The model is then calibrated to those data with the usual unknowns --
girder modulus Eg, rotational fixity at the north and south ends, pad stiffness kv at the
north and south ends -- exactly as one would update a model after an OMA campaign. Question:
does the calibration recover the support state, and where does it put the girder cracking?

Rotational fixity factor gamma (0 = pin, 1 = fully fixed), the usual measure in boundary-
condition updating: k_rot = 3 E I gamma / (L (1 - gamma)) per girder end.

Run:  python bc_identification.py   ->  results/bc_identification.json (figure: q3_run.py)
"""
from __future__ import annotations

import json
import math
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

import limit_states as ls
import mac_tools as mt
import walker_grillage as wg

N_MODES = 6
SIGMA_F = 0.005           # 0.5 % frequency noise / weight
SIGMA_MAC = 0.02          # MAC weight
SHAPE_NOISE = 0.02        # relative noise added to simulated shapes
K_ROT_PAD = wg.BRG_KV * wg.BRG_A_LONG ** 2 / 12.0

LIMIT_STATES = ls.LIMIT_STATES          # SLS, 75 % ULS, ULS (= 1.7 x 1.25 x 0.9 trucks/lane), beyond
# support scenarios: fixity (north, south) and kv factor (north, south) on top of the calibrated state
SCENARIOS = {
    "supports unchanged": dict(fix=(0.0, 0.0), kv=(1.0, 1.0)),
    "north half-fixed": dict(fix=(0.5, 0.0), kv=(1.0, 1.0)),
    "both half-fixed": dict(fix=(0.5, 0.5), kv=(1.0, 1.0)),
    "north ~fixed, south pin": dict(fix=(0.9, 0.0), kv=(1.0, 1.0)),
    "south pads stiffer x2": dict(fix=(0.0, 0.0), kv=(1.0, 2.0)),
}


def kr_factors(p, fix_n, fix_s):
    """Bearing rotational-spring factors (14) giving the requested end fixity per girder."""
    f = np.asarray(p["kr"], float).copy()
    eg = wg.E_GIRDER_S6 * p["Eg"]
    for end, gam in ((0, fix_n), (1, fix_s)):
        if gam <= 0:
            continue
        for j in range(wg.N_GIRDER):
            iy = wg.girder_section(j in (0, wg.N_GIRDER - 1), p, side=0 if j == 0 else 1)["Iy"]
            k = 3 * eg * iy * gam / (wg.L_SPAN * (1 - gam))
            f[end * wg.N_GIRDER + j] = max(f[end * wg.N_GIRDER + j], k / K_ROT_PAD)
    return f


def with_bc(p, fix_n, fix_s, kv_n, kv_s):
    q = dict(p)
    kv = np.asarray(p["kv"], float).copy()
    kv[:7] *= kv_n
    kv[7:] *= kv_s
    q["kv"] = kv
    q["kr"] = kr_factors(q, fix_n, fix_s)
    return q


def oma(p, acc):
    m = wg.build(p)
    f, sh, v = wg.modal(m, 30)
    fd, shd, _ = wg.deck_modes(f, sh, v)
    return fd, mt.fe_at_sensors(shd, acc)


def calibrate(job):
    """Calibrate Eg, fixity N/S, kv N/S of the (undamaged-stiffness) model to the data."""
    name, f_meas, phi_meas, p0, acc = job
    # search space: log10 Eg factor, fixity N, fixity S, log10 kv N, log10 kv S
    lo = np.array([-0.3, 0.0, 0.0, -0.5, -0.5])
    hi = np.array([0.1, 0.95, 0.95, 0.7, 0.7])

    def model(x):
        q = dict(p0)
        q["Eg"] = p0["Eg"] * 10 ** x[0]
        q = with_bc(q, x[1], x[2], 10 ** x[3], 10 ** x[4])
        return oma(q, acc)

    def resid(x):
        fd, ph = model(x)
        pairs = mt.pair_by_mac(f_meas, phi_meas, fd, ph, mac_min=0.3, df_max=0.4)
        got = {i: (mac, dfr) for i, k, mac, dfr in pairs}
        r = []
        for i in range(len(f_meas)):
            mac, dfr = got.get(i, (0.0, 0.4))
            r += [dfr / SIGMA_F, (1 - mac) / SIGMA_MAC]
        return np.array(r)

    best = None
    for x0 in ([0, 0.05, 0.05, 0, 0], [0, 0.5, 0.5, 0, 0], [-0.1, 0.3, 0.0, 0.2, 0.2]):
        r = least_squares(resid, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), diff_step=0.02,
                          x_scale="jac", xtol=1e-6, ftol=1e-6, max_nfev=200)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    fd, ph = model(x)
    pairs = mt.pair_by_mac(f_meas, phi_meas, fd, ph, mac_min=0.0, df_max=1.0)
    return name, dict(Eg_change_pct=100 * (10 ** x[0] - 1), fix_N=x[1], fix_S=x[2], kv_N=10 ** x[3],
                      kv_S=10 ** x[4], cost=float(best.cost),
                      df_resid_pct=[100 * d for _, _, _, d in pairs], mac_resid=[m_ for _, _, m_, _ in pairs])


def main():
    p0 = wg.calibrated_params()
    ls.LOSS = ls.s6_losses(verbose=False)[0]
    gp = ls.girder_properties(p0)
    _, _, m_ref, _, _ = ls.static_response(p0, 1.0)
    r_dl = np.array([gp[j]["r_dead"] for j in range(7)] * 2)
    acc = mt.accelerometers()
    rng = np.random.default_rng(3)

    f0, ph0 = oma(p0, acc)
    f0, ph0 = f0[:N_MODES], ph0[:N_MODES]
    jobs, truth = [], {}
    for ls_name, lam in LIMIT_STATES.items():
        st = ls.converge_state(p0, gp, lam, m_ref, r_dl)
        p_ls = {**p0, "I_el": st["f_el"], "trans_el": st["trans"],
                "kv": np.asarray(p0["kv"]) * st["f_kv"], "kr": np.asarray(p0["kr"]) * st["f_kr"]}
        for sc_name, sc in SCENARIOS.items():
            p_true = with_bc(p_ls, sc["fix"][0], sc["fix"][1], sc["kv"][0], sc["kv"][1])
            ft, pht = oma(p_true, acc)
            ft, pht = ft[:N_MODES], pht[:N_MODES]
            # what OMA would show vs the baseline (before noise)
            pairs = mt.pair_by_mac(f0, ph0, ft, pht, mac_min=0.0, df_max=1.0)
            change = dict(df_pct=[100 * d for _, _, _, d in pairs], mac=[m_ for _, _, m_, _ in pairs])
            f_meas = ft * (1 + 0.002 * rng.standard_normal(N_MODES))
            ph_meas = pht / np.abs(pht).max(1, keepdims=True)
            ph_meas = ph_meas + SHAPE_NOISE * rng.standard_normal(ph_meas.shape)
            name = f"{ls_name} | {sc_name}"
            truth[name] = dict(lam=lam, limit_state=ls_name, scenario=sc_name, fix_N=sc["fix"][0],
                               fix_S=sc["fix"][1], kv_N=sc["kv"][0], kv_S=sc["kv"][1],
                               cracked_girder_el=int(st["cracked"].sum()), oma_change=change)
            jobs.append((name, f_meas, ph_meas, p0, acc))
    with Pool(4) as pool:
        res = dict(pool.map(calibrate, jobs))
    out = {k: {**truth[k], "identified": res[k]} for k in truth}
    with open("results/bc_identification.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"{'case':42s} {'cracked':>7s} {'M1 df%':>7s} {'minMAC':>6s} | {'true fixN/S':>11s} {'ident fixN/S':>12s}"
          f" {'true kvN/S':>10s} {'ident kvN/S':>11s} {'ident dEg%':>10s}")
    for k, v in out.items():
        i = v["identified"]
        print(f"{k:42s} {v['cracked_girder_el']:7d} {v['oma_change']['df_pct'][0]:+7.2f} {min(v['oma_change']['mac']):6.3f} |"
              f" {v['fix_N']:4.2f}/{v['fix_S']:4.2f}   {i['fix_N']:4.2f}/{i['fix_S']:4.2f}    "
              f"{v['kv_N']:3.1f}/{v['kv_S']:3.1f}    {i['kv_N']:4.2f}/{i['kv_S']:4.2f}   {i['Eg_change_pct']:+8.1f}")


if __name__ == "__main__":
    main()
