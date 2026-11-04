"""Sensitivity screening and frequency-based calibration of the Walker Road grillage."""
import json
import time

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

import measured as ms
import walker_grillage as wg

N_EIG = 28
FAMILIES = list(ms.PAIR.values())
F_MEAS = np.array([ms.freq_of(k) for k in ms.PAIR])

# candidate updating parameters: name -> (transform, lower, upper) in the search space
CANDIDATES = {
    "Eg": ("log", 0.8, 2.0),
    "Ed": ("log", 0.6, 2.0),
    "barrier": ("lin", 0.0, 1.0),
    "kh": ("log", 1.0, 1e4),
    "kv": ("log", 0.5, 5.0),
    "trans": ("log", 0.4, 2.0),
    "diaph": ("log", 0.3, 3.0),
    "tors": ("log", 0.5, 3.0),
    "mass": ("lin", 0.5, 1.5),
}
SELECTED = ["Eg", "Ed", "barrier", "kv", "tors", "diaph"]
# weak engineering priors (search-space mean, sd): in-situ + dynamic modulus of 30-year-old
# concrete ~1.3 x S6 Ec; barrier composite (as in the CSiBridge model); others nominal.
PRIOR = {"Eg": (np.log10(1.3), 0.15), "Ed": (np.log10(1.3), 0.15), "barrier": (1.0, 0.5),
         "kv": (0.0, 0.3), "tors": (0.0, 0.3), "diaph": (0.0, 0.3)}
SIGMA_F = 0.01                      # 1 % model + identification error per frequency


def make_params(values: dict) -> dict:
    p = wg.default_params()
    for k, v in values.items():
        if k in ("kh", "kv", "kt", "kr"):
            p[k] = np.full(2 * wg.N_GIRDER, float(v))
        else:
            p[k] = float(v)
    return p


def solve(p: dict, n: int = N_EIG):
    m = wg.build(p)
    f, sh, v = wg.modal(m, n)
    return wg.deck_modes(f, sh, v)


_REF = {}


def reference_shapes():
    """Pure-family reference shapes from the nominal model (used for MAC tracking)."""
    if not _REF:
        fd, shd, _ = solve(wg.default_params())
        for k in range(len(fd)):
            fam = wg.classify(shd[k])[:2]
            if fam in FAMILIES and fam not in _REF:
                _REF[fam] = shd[k]
    return _REF


def paired(fd, shd):
    """Assign FE deck modes to the measured families by maximising MAC."""
    ref = reference_shapes()
    macm = np.array([[wg.mac(ref[fam], shd[k]) for k in range(len(fd))] for fam in FAMILIES])
    rows, cols = linear_sum_assignment(-macm)
    order = np.argsort(rows)
    idx = cols[order]
    return fd[idx], macm[rows[order], idx], idx


def to_x(values: dict, names) -> np.ndarray:
    x = []
    for k in names:
        t = CANDIDATES[k][0]
        x.append(np.log10(values[k]) if t == "log" else values[k])
    return np.array(x)


def from_x(x, names) -> dict:
    return {k: (10 ** xi if CANDIDATES[k][0] == "log" else xi) for k, xi in zip(names, x)}


def bounds(names):
    lo, hi = [], []
    for k in names:
        t, a, b = CANDIDATES[k]
        lo.append(np.log10(a) if t == "log" else a)
        hi.append(np.log10(b) if t == "log" else b)
    return np.array(lo), np.array(hi)


def sensitivity_matrix(base: dict, names, rel: float = 0.05):
    """S_ij = d ln f_i / d ln theta_j (barrier, mass: per unit change, scaled by 1)."""
    f0, _, _ = paired(*solve(make_params(base))[:2])
    s = np.zeros((len(f0), len(names)))
    for j, k in enumerate(names):
        up = dict(base)
        if CANDIDATES[k][0] == "log":
            up[k] = base[k] * (1 + rel)
            d = np.log(1 + rel)
        else:
            step = 0.1 if base[k] < 0.95 else -0.1
            up[k] = base[k] + step
            d = step / max(base[k], 1e-9)
        f1, _, _ = paired(*solve(make_params(up))[:2])
        s[:, j] = (np.log(f1) - np.log(f0)) / d
    return f0, s


def calibrate(names, x0s, sigma=SIGMA_F, prior=PRIOR):
    lo, hi = bounds(names)
    fixed = {k: 1.0 for k in CANDIDATES if k not in names}
    pm = np.array([prior[k][0] for k in names]) if prior else None
    ps = np.array([prior[k][1] for k in names]) if prior else None

    def resid(x):
        vals = {**fixed, **from_x(x, names)}
        fe, macs, _ = paired(*solve(make_params(vals))[:2])
        r = (fe - F_MEAS) / F_MEAS / sigma
        return np.concatenate([r, (x - pm) / ps]) if prior else r

    best = None
    for x0 in x0s:
        r = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                          diff_step=0.01, x_scale="jac", xtol=1e-10, ftol=1e-10)
        print("  start", np.round(x0, 3), "->", np.round(r.x, 4), "cost", round(r.cost, 4))
        if best is None or r.cost < best.cost:
            best = r
    return best, fixed


def main():
    t0 = time.time()
    out = {}
    # 1. nominal model
    p0 = wg.default_params()
    m = wg.build(p0)
    f, sh, v = wg.modal(m, N_EIG)
    fd, shd, keep = wg.deck_modes(f, sh, v)
    out["nominal_all"] = [dict(f=float(f[k]), vert=float(v[k])) for k in range(N_EIG)]
    out["nominal_deck"] = [dict(f=float(fd[k]), fam=wg.classify(shd[k])) for k in range(len(fd))]
    fe0, mac0, _ = paired(fd, shd)
    out["nominal_paired"] = fe0.tolist()
    print("nominal paired", np.round(fe0, 3), "meas", F_MEAS)

    # 2. sensitivity screening at nominal and at an intermediate state
    names = list(CANDIDATES)
    for tag, base in (("nominal", {k: 1.0 for k in names}),
                      ("intermediate", {**{k: 1.0 for k in names}, "Eg": 1.5, "Ed": 1.5, "kh": 10.0,
                                        "barrier": 0.9, "kv": 2.0, "tors": 1.5})):
        fb, s = sensitivity_matrix(base, names)
        out[f"sens_{tag}"] = dict(names=names, f=fb.tolist(), S=s.tolist())
        print(tag, "sensitivities d ln f / d ln theta")
        print("      " + " ".join(f"{k:>8s}" for k in names))
        for i, fam in enumerate(FAMILIES):
            print(f"{str(fam):6s}" + " ".join(f"{x:8.3f}" for x in s[i]))
        sel = [names.index(k) for k in SELECTED]
        u, sv, vt = np.linalg.svd(s[:, sel], full_matrices=False)
        print("  singular values (selected)", np.round(sv, 4), "cond", round(sv[0] / sv[-1], 1))
        out[f"sens_{tag}"]["sv_selected"] = sv.tolist()

    # 3. calibration (MAP with weak priors) and an unregularised fit for comparison
    rng = np.random.default_rng(1)
    lo, hi = bounds(SELECTED)
    starts = [to_x(dict(Eg=1.3, Ed=1.3, barrier=1.0, kv=1.0, tors=1.0, diaph=1.0), SELECTED),
              to_x(dict(Eg=1.6, Ed=1.2, barrier=0.6, kv=2.0, tors=1.5, diaph=0.7), SELECTED),
              to_x(dict(Eg=1.2, Ed=1.7, barrier=0.9, kv=3.0, tors=2.0, diaph=1.5), SELECTED)]
    starts += [lo + (hi - lo) * rng.random(len(SELECTED)) for _ in range(5)]
    print("unregularised fit:")
    free, _ = calibrate(SELECTED, starts[:4], prior=None)
    out["unregularised"] = dict(values=from_x(free.x, SELECTED), cost=float(free.cost),
                                resid_pct=(free.fun * SIGMA_F * 100).tolist())
    print("MAP fit:")
    best, fixed = calibrate(SELECTED, starts)
    cal = {**fixed, **from_x(best.x, SELECTED)}
    print("calibrated", cal, "cost", best.cost)

    # Laplace approximation of the posterior (search space)
    jac = best.jac
    cov = np.linalg.pinv(jac.T @ jac)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    jd = jac[: len(F_MEAS)]
    out["data_information"] = np.linalg.svd(jd, compute_uv=False).tolist()
    out["prior"] = {k: list(v) for k, v in PRIOR.items()}
    fd, shd, keep = solve(make_params(cal), 40)
    fe, macs, idx = paired(fd, shd)
    out["calibrated"] = dict(values=cal, cost=float(best.cost), x=best.x.tolist(),
                             x_sd=sd.tolist(), corr=corr.tolist(), names=SELECTED,
                             f_fe=fe.tolist(), f_meas=F_MEAS.tolist(), mac_to_ref=macs.tolist(),
                             deck=[dict(f=float(fd[k]), fam=wg.classify(shd[k])) for k in range(len(fd))])
    for fam, a, b, mm in zip(FAMILIES, fe, F_MEAS, macs):
        print(fam, round(a, 3), b, f"{100 * (a - b) / b:+.2f}%", "MAC", round(mm, 3))
    print("all calibrated deck modes:")
    for k in range(len(fd)):
        print("  ", round(fd[k], 3), wg.classify(shd[k]))
    with open("results/calibration.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
