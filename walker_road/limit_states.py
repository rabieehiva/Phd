"""Q3 -- modal parameters at higher limit states (Walker Road, West structure L930).

Stage 1: structural state as a function of the test-load factor lambda
         (lambda = CL-625 trucks per lane, both lanes loaded), CSA S6-19 throughout:
           * girder cracking: f_r = 0.4 sqrt(f'c) (8.4.1.8.1) with the as-built prestress (Dwg 11/18),
           * cracked stiffness: I_e = I_cr + (I_g - I_cr)(M_cr/M_a)^3 <= I_g (effective moment of
             inertia, S6-19 deflection clause / A23.3-19 9.8.2.3), total moments, element by element,
             with I_cr from a cracked transformed-section analysis of the composite girder,
           * secant iteration at every lambda: static solve -> crack check -> new I_e -> static
             solve ... until the cracked stiffness stops changing (load redistribution between girders),
           * local transverse deck cracking under the wheels (cracked transformed slab section),
           * bearing pressures / lift-off at the 14 pads (stiffness unchanged for now),
           * flexural resistance M_r (8.8.4, unfactored) -> upper limit of lambda.
Stage 2: modal analysis of each converged state, mode tracking by MAC, detectability as the
         equivalent temperature change of the measured frequencies. Cases:
           "mass only"        - test-load mass on the deck, undamaged stiffness,
           "under load"       - test-load mass + cracked secant stiffness (cracks open),
           "jacked"           - same force without mass (jacks on ground anchors): stiffness only,
           "jacked (tangent)" - I_cr over the whole cracked length (lower bound),
           "after unload"     - no load, cracks closed by the prestress, residual stiffness only.

Run:  python limit_states.py            (needs results/calibration.json from run_calibration.py)
Out:  results/limit_states.json, results/limit_states.png
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

import measured as ms
import walker_grillage as wg
import openseespy.opensees as ops

G = wg.G_ACC

# ----------------------------------------------------------------------------------------
# As-built prestress (Dwg 11/18, sections D-D / E-E, notes 3-4)
# ----------------------------------------------------------------------------------------
A_STRAND = 98.7e-6              # m^2, 12.7 mm 7-wire strand
FPU = 1862e3                    # kPa
EP = 195e6                      # kPa, strand modulus (S6-19 8.4.3)
P_JACK = 138.0                  # kN per strand (0.75 fpu A)
N_STRAIGHT, N_HARPED = 40, 10
Z_STRAIGHT = 0.150              # centroid of 4 rows x 10 at 75/125/175/225 mm
Z_HARP_END = 1.300              # 5 levels x 2 at 100..300 mm below the top at the ends
Z_HARP_MID = 0.175              # 5 levels x 2 at 75..275 mm between hold-downs
X_HOLD = (13.5, 19.5)           # hold-down points from the north bearing (m)
A_PS = (N_STRAIGHT + N_HARPED) * A_STRAND

# Assumptions (flagged; change here)
LOSS = 0.20                     # total long-term prestress loss (fraction of jacking)
FR = 0.4 * math.sqrt(wg.FC_GIRDER) * 1e3   # S6-19 8.4.1.8.1, kPa
FR_DECK = 0.4 * math.sqrt(wg.FC_DECK) * 1e3
EC_INSITU = 1.3                 # in-situ / S6 Ec ratio used for the modular ratios of cracked sections
N_P = EP / (EC_INSITU * wg.E_GIRDER_S6)
RES_AT_ULT = 0.84               # residual dynamic stiffness after loading near ultimate (Zhou et al. 2019)
DETECT_PCT = 0.5                # detection threshold on |df| after temperature compensation (%)
BEARING_EXP = 0.0               # kv ~ (sigma / sigma_DL)^exp ; 0 = pressure-independent (no data yet)
ITER_TOL, ITER_MAX = 1e-3, 30   # secant iteration: max change of any I_e/I_g, max iterations

# CL-625 truck (S6-19 Fig. 3.4): axle loads (kN) and distances from the front axle (m)
AXLE_P = np.array([50.0, 125.0, 125.0, 175.0, 150.0])
AXLE_X = np.array([0.0, 3.6, 4.8, 11.4, 18.0])
GAUGE = 1.8
# Lanes (Dwg 1/18, West structure, eastbound = southbound): from the west edge, barrier 0.555,
# shoulder 3.0, two 3.7 m lanes, shoulder 2.0, barrier 0.555. Model y: G4 at 0.
Y_WEST_EDGE = wg.Y_GIRDER[0] - wg.OVERHANG
LANE_C = [Y_WEST_EDGE + 0.555 + 3.0 + 1.85, Y_WEST_EDGE + 0.555 + 3.0 + 3.7 + 1.85]
X_NODES = np.linspace(0, wg.L_SPAN, wg.N_SEG + 1)


# ----------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------
def z_ps(x):
    """Prestress centroid height above the girder soffit at distance x from the bearing."""
    x = np.asarray(x, float)
    a, b = X_HOLD
    zh = np.where(x < a, Z_HARP_END + (Z_HARP_MID - Z_HARP_END) * x / a,
                  np.where(x > b, Z_HARP_MID + (Z_HARP_END - Z_HARP_MID) * (x - b) / (wg.L_SPAN - b), Z_HARP_MID))
    return (N_STRAIGHT * Z_STRAIGHT + N_HARPED * zh) / (N_STRAIGHT + N_HARPED)


def composite_parts(j, p):
    """Composite cross-section of girder line j as (polygon, n) in girder-concrete units,
    identical to walker_grillage.girder_section (girder, haunch, slab, barrier)."""
    ext = j in (0, wg.N_GIRDER - 1)
    n_d = wg.E_DECK_S6 * p["Ed"] / (wg.E_GIRDER_S6 * p["Eg"])
    z_h0, z_s0 = wg.H_GIRDER, wg.H_GIRDER + wg.HAUNCH
    z_s1 = z_s0 + wg.T_SLAB
    b_in, b_out = wg.S_GIRDER / 2, (wg.OVERHANG if ext else wg.S_GIRDER / 2)
    parts = [(wg.girder_polygon(), 1.0),
             (wg.rect(-wg.W_TOPFL / 2, wg.W_TOPFL / 2, z_h0, z_s0), n_d),
             (wg.rect(-b_in, b_out, z_s0, z_s1), n_d)]
    fb = p["barrier"] * np.asarray(p.get("barrier_side", np.ones(2)))[0 if j == 0 else 1]
    if ext and fb > 0:
        bp = wg.barrier_polygon().copy()
        bp[:, 0] = wg.OVERHANG - bp[:, 0]
        bp[:, 1] = bp[:, 1] + z_s1
        parts.append((bp[::-1], n_d * fb))
    return parts, b_in + b_out


def poly_width(poly, z):
    """Width of a simple polygon cut by the horizontal line at height z."""
    ys = []
    n = len(poly)
    for k in range(n):
        (ya, za), (yb, zb) = poly[k], poly[(k + 1) % n]
        if (za > z) != (zb > z):
            ys.append(ya + (z - za) * (yb - ya) / (zb - za))
    ys.sort()
    return sum(ys[i + 1] - ys[i] for i in range(0, len(ys) - 1, 2))


def cracked_section(parts, z_p, dz=0.002):
    """Cracked transformed section (concrete in tension ignored, strands n_p A_ps at z_p):
    neutral axis from first-moment equilibrium, returns (I_cr, z_na) in girder units."""
    z_top = max(pl[:, 1].max() for pl, _ in parts)
    zs = np.arange(dz / 2, z_top, dz)
    w = np.array([sum(n * poly_width(pl, z) for pl, n in parts) for z in zs])
    nas = n_p_as = N_P * A_PS

    def first_moment(zna):
        c = zs > zna
        return np.sum(w[c] * (zs[c] - zna)) * dz - n_p_as * (zna - z_p)

    lo, hi = z_p + 1e-4, z_top - 1e-4
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if first_moment(mid) > 0:
            lo = mid
        else:
            hi = mid
    zna = 0.5 * (lo + hi)
    c = zs > zna
    icr = np.sum(w[c] * (zs[c] - zna) ** 2) * dz + nas * (zna - z_p) ** 2
    return icr, zna


def flexural_resistance(b_slab):
    """Unfactored M_r at midspan, S6-19 8.8.4 (bonded strand, rectangular block in the slab)."""
    fcs = wg.FC_DECK
    a1, b1 = 0.85 - 0.0015 * fcs, 0.97 - 0.0025 * fcs
    d_p = wg.H_GIRDER + wg.HAUNCH + wg.T_SLAB - z_ps(wg.L_SPAN / 2)
    c = A_PS * FPU / (a1 * fcs * 1e3 * b1 * b_slab + 0.28 * A_PS * FPU / d_p)
    fps = FPU * (1 - 0.28 * c / d_p)
    return float(A_PS * fps * (d_p - b1 * c / 2)), float(c)


def girder_dead_moments(p):
    """Per-girder non-composite (girder alone) and superimposed (composite) dead-load moments
    at the 31 nodes (kN m) and end reactions (kN). Superimposed load shared equally."""
    a_g = wg.poly_props(wg.girder_polygon())[0]
    a_b = wg.poly_props(wg.barrier_polygon())[0]
    x, L = X_NODES, wg.L_SPAN
    w_sdl = (2 * (wg.UNIT_WT_CONC * a_b + wg.RAIL_MASS * G)
             + p["mass"] * wg.UNIT_WT_ASPH * wg.T_ASPHALT * (wg.DECK_W - 2 * wg.BARRIER_W)) / wg.N_GIRDER
    l_skew = wg.S_GIRDER / math.cos(math.radians(wg.SKEW_DEG))
    p_d_bay = wg.UNIT_WT_CONC * wg.DIAPH_T * wg.DIAPH_H * (l_skew - 0.15)
    out = {}
    for j in range(wg.N_GIRDER):
        ext = j in (0, wg.N_GIRDER - 1)
        b = wg.S_GIRDER / 2 + (wg.OVERHANG if ext else wg.S_GIRDER / 2)
        w_nc = wg.UNIT_WT_CONC * (a_g + wg.W_TOPFL * wg.HAUNCH + b * wg.T_SLAB)
        m_nc = w_nc * x * (L - x) / 2
        pd = p_d_bay * (0.5 if ext else 1.0)
        for xd in (L / 3, 2 * L / 3):
            m_nc = m_nc + np.where(x <= xd, pd * x * (L - xd) / L, pd * xd * (L - x) / L)
        out[j] = dict(m_nc=m_nc, m_sdl=w_sdl * x * (L - x) / 2, r=w_nc * L / 2 + pd + w_sdl * L / 2)
    return out


def girder_properties(p):
    """Everything the crack check needs, per girder and node:
    sig_dp  bottom-fibre stress from prestress + dead load (kPa, tension +),
    k_b     bottom-fibre stress per kN m of live-load (composite) moment,
    m_d     total dead-load moment, m_cr_ll live-load moment at cracking,
    icr     I_cr / I_g (cracked transformed composite section), M_r."""
    a_g, _, zc_g, i_g, _ = wg.poly_props(wg.girder_polygon())
    dead = girder_dead_moments(p)
    pe = (1 - LOSS) * P_JACK * (N_STRAIGHT + N_HARPED)
    zp = z_ps(X_NODES)
    out = {}
    icr_cache = {}
    for j in range(wg.N_GIRDER):
        ext = j in (0, wg.N_GIRDER - 1)
        s = wg.girder_section(ext, p, side=0 if j == 0 else 1)
        parts, b_slab = composite_parts(j, p)
        d = dead[j]
        sig_dp = (-pe / a_g - pe * (zc_g - zp) * zc_g / i_g + d["m_nc"] * zc_g / i_g
                  + d["m_sdl"] * s["zc"] / s["Iy"])
        k_b = s["zc"] / s["Iy"]
        key = (round(s["Iy"], 6), round(b_slab, 3))
        if key not in icr_cache:          # I_cr only varies with z_p: evaluate on a few points
            zq = np.linspace(Z_STRAIGHT + 0.0, zp.max(), 6)
            icr_cache[key] = (zq, np.array([cracked_section(parts, z)[0] for z in zq]))
        zq, iq = icr_cache[key]
        icr = np.interp(zp, zq, iq) / s["Iy"]
        mr, c = flexural_resistance(b_slab)
        out[j] = dict(sig_dp=sig_dp, k_b=k_b, m_d=d["m_nc"] + d["m_sdl"], m_cr_ll=(FR - sig_dp) / k_b,
                      icr=icr, Mr=mr, c=c, r_dead=d["r"], I_g=s["Iy"])
    return out


# ----------------------------------------------------------------------------------------
# Truck loads on the grillage
# ----------------------------------------------------------------------------------------
def wheel_points(scale=1.0):
    """Wheel (x, y, P) for one CL-625 per lane, placed for maximum midspan moment."""
    a_ref = 0.5 * (np.sum(AXLE_P * AXLE_X) / AXLE_P.sum() + AXLE_X[3])
    pts = []
    for yc in LANE_C:
        x_mid = (yc - wg.Y_GIRDER[0]) * wg.TAN_SKEW + wg.L_SPAN / 2
        for pa, xa in zip(AXLE_P, AXLE_X):
            x = x_mid + (a_ref - xa)             # front axle leading, travelling south (+x)
            for dy in (-GAUGE / 2, GAUGE / 2):
                pts.append((x, yc + dy, scale * pa / 2))
    return pts


def distribute(x, y):
    """Statically equivalent weights of a point at (x, y) on the grillage nodes {(j, i): w}."""
    Y = wg.Y_GIRDER
    if y <= Y[0]:
        j0, t = 0, (y - Y[0]) / wg.S_GIRDER
    elif y >= Y[-1]:
        j0, t = wg.N_GIRDER - 2, 1 + (y - Y[-1]) / wg.S_GIRDER
    else:
        j0 = min(int((y - Y[0]) // wg.S_GIRDER), wg.N_GIRDER - 2)
        t = (y - Y[j0]) / wg.S_GIRDER
    out = {}
    for j, wy in ((j0, 1 - t), (j0 + 1, t)):
        xi = (x - wg.X0_GIRDER[j]) / wg.DX
        if xi < 0 or xi > wg.N_SEG:
            continue
        i0 = min(int(xi), wg.N_SEG - 1)
        u = xi - i0
        for i, wx in ((i0, 1 - u), (i0 + 1, u)):
            out[(j, i)] = out.get((j, i), 0.0) + wy * wx
    return out


def truck_mass(scale):
    """Test-load mass lumped at grillage nodes (positive weights only)."""
    mass = {}
    for x, y, pw in wheel_points(scale):
        w = {k: max(v, 0.0) for k, v in distribute(x, y).items()}
        s = sum(w.values())
        for k, v in w.items():
            mass[k] = mass.get(k, 0.0) + pw / G * v / s
    return mass


def static_response(p, lam):
    """Static response to lam x (one CL-625 per lane): girder moments (7, 31), bearing reactions."""
    m = wg.build(p)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for x, y, pw in wheel_points(lam):
        for (j, i), w in distribute(x, y).items():
            ops.load(int(m.node[j, i]), 0, 0, -pw * w, 0, 0, 0)
    wg.setup_static()
    ops.analyze(1)
    mom = np.array([[wg.girder_moment_at(m, j, i) for i in range(wg.N_SEG + 1)] for j in range(wg.N_GIRDER)])
    return mom, wg.bearing_reactions(m), m


# ----------------------------------------------------------------------------------------
# Stage 1: CSA effective stiffness with secant iteration
# ----------------------------------------------------------------------------------------
def ie_ratio(gp, m_ll):
    """Node-wise I_e / I_g (S6 / A23.3 form, total moments) and cracked flags for the live
    moments m_ll (7, 31)."""
    r = np.ones_like(m_ll)
    cracked = np.zeros(m_ll.shape, bool)
    for j in range(wg.N_GIRDER):
        g = gp[j]
        m_a = g["m_d"] + m_ll[j]
        m_cr = g["m_d"] + g["m_cr_ll"]
        c = m_ll[j] > g["m_cr_ll"]
        ratio = np.where(c, (m_cr / np.maximum(m_a, 1e-9)) ** 3, 1.0)
        r[j] = np.minimum(g["icr"] + (1 - g["icr"]) * ratio, 1.0)
        cracked[j] = c
    return r, cracked


def to_elements(node_vals, how="mean"):
    a, b = node_vals[:, :-1], node_vals[:, 1:]
    return 0.5 * (a + b) if how == "mean" else np.maximum(a, b)


def deck_local_moment():
    """Local transverse slab moment per m under one wheel of the 175 kN axle (AASHTO Std. Spec.
    3.24.3.1 form, continuity 0.8 -- NOT a CSA expression) and dead-load moment w S^2/10."""
    m_ll = 0.8 * (wg.S_GIRDER + 0.61) / 9.74 * (AXLE_P.max() / 2)
    w = wg.UNIT_WT_CONC * wg.T_SLAB + wg.UNIT_WT_ASPH * wg.T_ASPHALT
    return m_ll, w * wg.S_GIRDER ** 2 / 10


def deck_factors(m, lam):
    """Transverse slab members under the wheel lines: I_e = I_cr + (I_g - I_cr)(M_cr/M_a)^3,
    M_cr = f_r t^2/6, I_cr of the cracked transformed slab (15M @ 300, Dwg 16/18)."""
    m_ll, m_dl = deck_local_moment()
    m_cr = FR_DECK * wg.T_SLAB ** 2 / 6
    f = np.ones(len(m.trans_ele))
    m_a = lam * m_ll + m_dl
    if m_a <= m_cr:
        return f
    n = 200e6 / (wg.E_DECK_S6 * EC_INSITU)
    a_s, d = 200e-6 / 0.300, wg.T_SLAB - 0.030 - 0.008
    nr = n * a_s / d
    k = math.sqrt(2 * nr + nr ** 2) - nr
    icr = ((k * d) ** 3 / 3 + n * a_s * (d - k * d) ** 2) / (wg.T_SLAB ** 3 / 12)
    ie = icr + (1 - icr) * (m_cr / m_a) ** 3
    loaded = {jk for x, y, _ in wheel_points() for jk in distribute(x, y)}
    for k_, (_, ja, ia, jb, ib) in enumerate(m.trans_ele):
        if (ja, ia) in loaded or (jb, ib) in loaded:
            f[k_] = ie
    return f


def converge_state(p0, gp, lam, m_ref):
    """Secant iteration at load level lam. Returns the converged element I_e factors, cracked
    element mask, live moments, bearing reactions and the number of iterations."""
    trans = deck_factors(m_ref, lam)
    f_el = np.ones((wg.N_GIRDER, wg.N_SEG))
    hist = []
    for it in range(1, ITER_MAX + 1):
        m_ll, reac, _ = static_response({**p0, "I_el": f_el, "trans_el": trans}, lam)
        r_node, c_node = ie_ratio(gp, m_ll)
        f_new = to_elements(r_node)
        delta = float(np.abs(f_new - f_el).max())
        hist.append(delta)
        f_el = 0.5 * (f_el + f_new) if it > 3 else f_new      # mild relaxation against oscillation
        if delta < ITER_TOL:
            break
    cracked_el = to_elements(c_node.astype(float), "max") > 0
    icr_el = to_elements(np.array([gp[j]["icr"] for j in range(wg.N_GIRDER)]))
    return dict(f_el=f_new, cracked=cracked_el, icr_el=icr_el, m_ll=m_ll, reac=reac, trans=trans,
                iterations=it, history=hist)


def residual_factors(st, gp, lam, lam_cr, lam_u):
    """After unloading: cracks closed by prestress, residual factor on the cracked elements
    interpolated from 1 at first cracking to RES_AT_ULT at the resistance (literature-based)."""
    f = np.ones((wg.N_GIRDER, wg.N_SEG))
    for j in range(wg.N_GIRDER):
        t = np.clip((lam - lam_cr[j]) / max(lam_u[j] - lam_cr[j], 1e-9), 0, 1)
        f[j, st["cracked"][j]] = 1 - (1 - RES_AT_ULT) * t
    return f


# ----------------------------------------------------------------------------------------
# Stage 2: modal analysis and tracking
# ----------------------------------------------------------------------------------------
N_TRACK = 8
MAC_TRACK = 0.8                 # points with a weaker MAC match are not plotted (tracking unreliable)


def deck_modes(p, n=30):
    m = wg.build(p)
    f, sh, v = wg.modal(m, n)
    return wg.deck_modes(f, sh, v)


def track(ref_sh, fd, shd):
    macm = np.array([[wg.mac(r, s) for s in shd] for r in ref_sh])
    rows, cols = linear_sum_assignment(-macm)
    return fd[cols], macm[rows, cols]


def main():
    p0 = wg.calibrated_params()
    out = {"assumptions": dict(LOSS=LOSS, FR_kPa=FR, EC_INSITU=EC_INSITU, N_P=N_P, RES_AT_ULT=RES_AT_ULT,
                               DETECT_PCT=DETECT_PCT, BEARING_EXP=BEARING_EXP, lanes_y=LANE_C,
                               ITER_TOL=ITER_TOL)}

    # --- Stage 1 ---------------------------------------------------------------------
    gp = girder_properties(p0)
    m1, r1, m_ref = static_response(p0, 1.0)          # uncracked, lambda = 1
    lam_cr = [float(np.min(np.where(m1[j] > 0, gp[j]["m_cr_ll"] / m1[j], np.inf))) for j in range(7)]
    lam_0 = [float(np.min(np.where(m1[j] > 0, (gp[j]["m_cr_ll"] - FR / gp[j]["k_b"]) / m1[j], np.inf)))
             for j in range(7)]
    i_mid = wg.N_SEG // 2
    lam_u_lin = [float((gp[j]["Mr"] - gp[j]["m_d"][i_mid]) / m1[j].max()) for j in range(7)]
    r_dl = np.array([gp[j]["r_dead"] for j in range(7)] * 2)
    a_pad = wg.BRG_A_LONG * wg.BRG_A_TRAN
    m_ll_deck, m_dl_deck = deck_local_moment()
    m_cr_deck = FR_DECK * wg.T_SLAB ** 2 / 6
    lam_deck = (m_cr_deck - m_dl_deck) / m_ll_deck
    print(f"effective prestress {(1 - LOSS) * P_JACK * 50:.0f} kN/girder, f_r = {FR / 1e3:.2f} MPa, n_p = {N_P:.2f}")
    print("girder  M_LL,mid(l=1)  sig_bot(D+P)  lam_decomp  lam_crack  lam_ult(lin)  Icr/Ig(mid)")
    for j in range(7):
        print(f"  G{j + 1}   {m1[j].max():7.0f}      {gp[j]['sig_dp'][i_mid] / 1e3:7.2f} MPa   {lam_0[j]:6.2f}"
              f"     {lam_cr[j]:6.2f}     {lam_u_lin[j]:6.2f}       {gp[j]['icr'][i_mid]:.3f}")
    print(f"deck: local M_LL {m_ll_deck:.1f} kNm/m per truck, M_DL {m_dl_deck:.1f}, M_cr {m_cr_deck:.1f}"
          f" -> lam_deck_crack = {lam_deck:.2f}")
    print("bearing pressure under dead load (MPa):", np.round(r_dl / a_pad / 1e3, 2))

    # secant iteration at each load level
    lam_grid = np.arange(0.5, 12.0 + 1e-9, 0.5)
    lams, states = [], {}
    print("\nsecant iteration (girder I_e redistributes the live load):")
    print("  lam   iters  cracked el   G1..G7 midspan M_LL (kN m)                     max M/M_r  min R (kN)")
    for lam in sorted(set(np.round(np.r_[lam_grid, min(lam_cr)], 3))):
        st = converge_state(p0, gp, lam, m_ref)
        util = max((gp[j]["m_d"] + st["m_ll"][j]).max() / gp[j]["Mr"] for j in range(7))
        rmin = float((r_dl + st["reac"]).min())
        print(f"  {lam:5.2f}  {st['iterations']:3d}    {int(st['cracked'].sum()):4d}      " +
              " ".join(f"{st['m_ll'][j].max():6.0f}" for j in range(7)) + f"     {util:6.3f}   {rmin:7.1f}")
        if util >= 1.0:
            print(f"  -> flexural resistance reached at lambda = {lam:.2f}; stopping")
            break
        st["util"] = util
        lams.append(float(lam))
        states[float(lam)] = st
    lam_u = [lam_u_lin[j] for j in range(7)]
    out["stage1"] = dict(
        lam_decomp=lam_0, lam_crack=lam_cr, lam_ult_linear=lam_u_lin, Mr=[gp[j]["Mr"] for j in range(7)],
        icr_ratio_mid=[float(gp[j]["icr"][i_mid]) for j in range(7)],
        sig_bot_DP_mid_MPa=[float(gp[j]["sig_dp"][i_mid] / 1e3) for j in range(7)],
        deck=dict(M_ll=m_ll_deck, M_dl=m_dl_deck, M_cr=m_cr_deck, lam_crack=lam_deck),
        bearing_R_dead=r_dl.tolist(), bearing_R_per_lambda_uncracked=r1.tolist(),
        iteration={str(l): dict(iterations=s["iterations"], history=s["history"], cracked_el=int(s["cracked"].sum()),
                                M_ll_mid=[float(s["m_ll"][j].max()) for j in range(7)], util=s["util"],
                                bearing_R=(r_dl + s["reac"]).tolist())
                   for l, s in states.items()})

    # --- Stage 2 ---------------------------------------------------------------------
    f_ref, sh_ref, _ = deck_modes(p0)
    sh_ref, f_ref = sh_ref[:N_TRACK], f_ref[:N_TRACK]
    fams = [wg.classify(s)[:2] for s in sh_ref]
    inv = {v: k for k, v in ms.PAIR.items()}
    names = [inv.get(f, f"FE{f}") for f in fams]
    tsens = {n: abs(ms.TEMP_SENS[ms.MODES.index(n)]) for n in names if n in ms.MODES}
    print("\ntracked baseline modes:", list(zip(names, np.round(f_ref, 3))))

    rows = []
    for lam in lams:
        st = states[lam]

        def stiff(p, key):
            if key == "under":
                p["I_el"] = st["f_el"]
            elif key == "tangent":
                p["I_el"] = np.where(st["cracked"], st["icr_el"], 1.0)
            else:
                p["I_el"] = residual_factors(st, gp, lam, lam_cr, lam_u)
                return p
            p["trans_el"] = st["trans"]
            if BEARING_EXP:
                p["kv"] = p0["kv"] * (np.maximum(r_dl + st["reac"], 1e-6) / r_dl) ** BEARING_EXP
            return p

        fd_m, sh_m, _ = deck_modes({**p0, "extra_mass": truck_mass(lam)})
        f_m, mac_m = track(sh_ref, fd_m, sh_m)
        sh_m_t = [sh_m[np.argmax([wg.mac(r, s) for s in sh_m])] for r in sh_ref]
        cases = {"mass only": (f_m, mac_m)}
        fd, shd, _ = deck_modes(stiff({**p0, "extra_mass": truck_mass(lam)}, "under"))
        cases["under load"] = track(sh_m_t, fd, shd)
        for case, key in (("jacked", "under"), ("jacked (tangent)", "tangent"), ("after unload", "after")):
            fd, shd, _ = deck_modes(stiff(dict(p0), key))
            cases[case] = track(sh_ref, fd, shd)
        for case, (ft, macs) in cases.items():
            df = 100 * (ft - f_ref) / f_ref
            row = dict(lam=lam, case=case, f=ft.tolist(), df_pct=df.tolist(), mac=macs.tolist(),
                       n_cracked_girder_el=int(st["cracked"].sum()), n_cracked_deck_el=int((st["trans"] < 0.999).sum()),
                       dT_equiv=[float(d / tsens[n]) if n in tsens else None for d, n in zip(df, names)])
            if case == "under load":
                row["df_stiffness_pct"] = (100 * (ft - f_m) / f_m).tolist()
            rows.append(row)
            print(f"lam {lam:5.2f} {case:18s} df% " + " ".join(f"{d:+6.2f}" for d in df) +
                  "  minMAC %.3f" % macs.min())
    out["stage2"] = dict(names=names, families=[list(f) for f in fams], f_ref=f_ref.tolist(), rows=rows)
    with open("results/limit_states.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = out["stage2"]
    names = st["names"][:5]
    fig, axs = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.4), sharex=True)
    top = [("jacked", "df_pct", "-"), ("jacked (tangent)", "df_pct", ":"),
           ("under load", "df_stiffness_pct", "--"), ("after unload", "df_pct", "-.")]
    bot = [("mass only", "df_pct", "-"), ("under load", "df_pct", "--")]
    for k, n in enumerate(names):
        for row_ax, series in ((axs[0, k], top), (axs[1, k], bot)):
            for c, fld, ls in series:
                rr = [r for r in st["rows"] if r["case"] == c and r["mac"][k] >= MAC_TRACK]
                lab = c + (" / mass only" if fld == "df_stiffness_pct" else "")
                row_ax.plot([r["lam"] for r in rr], [r[fld][k] for r in rr], ls, marker="o", ms=3, label=lab)
            row_ax.axhspan(-DETECT_PCT, DETECT_PCT, color="0.85", zorder=0)
            row_ax.axvline(min(out["stage1"]["lam_crack"]), color="k", lw=0.6)
            row_ax.axvline(out["stage1"]["deck"]["lam_crack"], color="0.5", lw=0.6, ls="--")
            row_ax.grid(alpha=0.3)
        axs[0, k].set_title(f"{n} ({st['f_ref'][k]:.2f} Hz)")
        axs[1, k].set_xlabel("λ (CL-625 trucks per lane)")
    axs[0, 0].set_ylabel("Δf (%) - stiffness change only")
    axs[1, 0].set_ylabel("Δf (%) - with test-load mass")
    axs[0, 0].legend(fontsize=6)
    axs[1, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig("results/limit_states.png", dpi=150)


if __name__ == "__main__":
    main()
