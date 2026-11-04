"""Q3 -- modal parameters at higher limit states (Walker Road, West structure L930).

Stage 1: physically based structural states as a function of the test-load factor lambda
         (lambda = number of CL-625 truck loads per lane, two lanes loaded):
           * girder flexural cracking from the as-built prestress (Dwg 11/18) and S6-19 f_r,
           * effective stiffness of the cracked length (under load) and residual stiffness
             after unloading (cracks closed by prestress),
           * local transverse deck cracking under the wheels,
           * bearing pressure / lift-off at the 14 pads,
           * nominal flexural resistance -> upper limit of lambda.
Stage 2: modal analysis of each state starting from the frequency-calibrated grillage, mode
         tracking by MAC, and detectability in terms of the measured temperature sensitivity.
         Three cases separate the effect of the test load from structural change:
           "mass only"   - load mass on the deck, undamaged stiffness,
           "under load"  - load mass + cracked stiffness (cracks open),
           "after unload"- no load mass, residual stiffness only.

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
EP = 195e6                      # kPa, strand modulus
P_JACK = 138.0                  # kN per strand (0.75 fpu A)
N_STRAIGHT, N_HARPED = 40, 10
Z_STRAIGHT = 0.150              # centroid of 4 rows x 10 at 75/125/175/225 mm
Z_HARP_END = 1.300              # 5 levels x 2 at 100..300 mm below the top at the ends
Z_HARP_MID = 0.175              # 5 levels x 2 at 75..275 mm between hold-downs
X_HOLD = (13.5, 19.5)           # hold-down points from the north bearing (m)
A_PS = (N_STRAIGHT + N_HARPED) * A_STRAND

# Assumptions (flagged; change here)
LOSS = 0.20                     # total long-term prestress loss (fraction of jacking), ~30-year-old girder
FR = 0.4 * math.sqrt(wg.FC_GIRDER) * 1e3   # S6-19 8.4.1.8.1 cracking strength, kPa
FR_DECK = 0.4 * math.sqrt(wg.FC_DECK) * 1e3
N_P_REF = EP / (1.3 * wg.E_GIRDER_S6)      # modular ratio for the cracked section (in-situ Ec ~1.3 S6)
RES_AT_ULT = 0.84               # residual dynamic stiffness after loading near ultimate (Zhou et al. 2019)
DETECT_PCT = 0.5                # detection threshold on |df| after temperature compensation (%)
BEARING_EXP = 0.0               # kv ~ (sigma / sigma_DL)^exp ; 0 = pressure-independent (no data yet)

# CL-625 truck (S6-19 Fig. 3.4): axle loads (kN) and distances from the front axle (m)
AXLE_P = np.array([50.0, 125.0, 125.0, 175.0, 150.0])
AXLE_X = np.array([0.0, 3.6, 4.8, 11.4, 18.0])
GAUGE = 1.8
# Lanes (Dwg 1/18, West structure, eastbound = southbound): from the west edge, barrier 0.555,
# shoulder 3.0, two 3.7 m lanes, shoulder 2.0, barrier 0.555. Model y: G4 at 0, west edge -6.755.
Y_WEST_EDGE = wg.Y_GIRDER[0] - wg.OVERHANG
LANE_C = [Y_WEST_EDGE + 0.555 + 3.0 + 1.85, Y_WEST_EDGE + 0.555 + 3.0 + 3.7 + 1.85]


# ----------------------------------------------------------------------------------------
# Section and load quantities
# ----------------------------------------------------------------------------------------
def z_ps(x):
    """Prestress centroid height above the girder soffit at distance x from the bearing."""
    x = np.asarray(x, float)
    a, b = X_HOLD
    zh = np.where(x < a, Z_HARP_END + (Z_HARP_MID - Z_HARP_END) * x / a,
                  np.where(x > b, Z_HARP_MID + (Z_HARP_END - Z_HARP_MID) * (x - b) / (wg.L_SPAN - b), Z_HARP_MID))
    return (N_STRAIGHT * Z_STRAIGHT + N_HARPED * zh) / (N_STRAIGHT + N_HARPED)


def girder_dead_moments(p):
    """Per-girder non-composite (girder carries) and superimposed (composite) dead-load
    moments at the 31 nodes, kN m. Superimposed load shared equally by the 7 girders."""
    a_g = wg.poly_props(wg.girder_polygon())[0]
    a_b = wg.poly_props(wg.barrier_polygon())[0]
    x = np.linspace(0, wg.L_SPAN, wg.N_SEG + 1)
    L = wg.L_SPAN
    w_sdl_tot = 2 * (wg.UNIT_WT_CONC * a_b + wg.RAIL_MASS * G) \
        + p["mass"] * wg.UNIT_WT_ASPH * wg.T_ASPHALT * (wg.DECK_W - 2 * wg.BARRIER_W)
    w_sdl = w_sdl_tot / wg.N_GIRDER
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
            m_nc += np.where(x <= xd, pd * x * (L - xd) / L, pd * xd * (L - x) / L)
        out[j] = (m_nc, w_sdl * x * (L - x) / 2, w_nc * L / 2 + pd, w_sdl * L / 2)
    return out


def composite_props(p):
    """Per girder: girder-alone and composite (transformed to girder E) section properties."""
    a_g, _, zc_g, i_g, _ = wg.poly_props(wg.girder_polygon())
    out = {}
    for j in range(wg.N_GIRDER):
        ext = j in (0, wg.N_GIRDER - 1)
        s = wg.girder_section(ext, p, side=0 if j == 0 else 1)
        b = wg.S_GIRDER / 2 + (wg.OVERHANG if ext else wg.S_GIRDER / 2)
        out[j] = dict(A_g=a_g, zc_g=zc_g, I_g=i_g, I_c=s["Iy"], zc_c=s["zc"], b_slab=b,
                      n_d=wg.E_DECK_S6 * p["Ed"] / (wg.E_GIRDER_S6 * p["Eg"]))
    return out


def cracked_I(sec, x):
    """Cracked composite I (girder-concrete units), bonded prestressed approximation
    I_cr = n_p A_ps d_p^2 (1 - 1.6 sqrt(n_p rho_p))  (ACI 318 / PCI design handbook),
    returned as a ratio of the gross composite I_c."""
    d_p = wg.H_GIRDER + wg.HAUNCH + wg.T_SLAB - z_ps(x)
    b = sec["b_slab"] * sec["n_d"]
    rho = A_PS / (b * d_p)
    icr = N_P_REF * A_PS * d_p ** 2 * (1 - 1.6 * np.sqrt(N_P_REF * rho))
    return np.clip(icr / sec["I_c"], 0.05, 1.0)


def nominal_resistance(sec):
    """Unfactored flexural resistance at midspan (S6-19 8.8.4, bonded low-relaxation strand)."""
    fcs = wg.FC_DECK
    a1, b1 = 0.85 - 0.0015 * fcs, 0.97 - 0.0025 * fcs
    d_p = wg.H_GIRDER + wg.HAUNCH + wg.T_SLAB - z_ps(wg.L_SPAN / 2)
    b = sec["b_slab"]
    c = A_PS * FPU / (a1 * fcs * 1e3 * b1 * b + 0.28 * A_PS * FPU / d_p)
    fps = FPU * (1 - 0.28 * c / d_p)
    return A_PS * fps * (d_p - b1 * c / 2), c


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
            continue                             # wheel off the span on this line
        i0 = min(int(xi), wg.N_SEG - 1)
        u = xi - i0
        for i, wx in ((i0, 1 - u), (i0 + 1, u)):
            out[(j, i)] = out.get((j, i), 0.0) + wy * wx
    return out


def truck_mass(scale):
    """Test-vehicle mass lumped at grillage nodes (positive weights only)."""
    mass = {}
    for x, y, pw in wheel_points(scale):
        w = {k: max(v, 0.0) for k, v in distribute(x, y).items()}
        s = sum(w.values())
        for k, v in w.items():
            mass[k] = mass.get(k, 0.0) + pw / G * v / s
    return mass


def live_load_response(p):
    """Static response to lambda = 1 (one CL-625 per lane): girder moments (7, 31), bearing
    reactions (14), transverse member moments."""
    m = wg.build(p)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for x, y, pw in wheel_points():
        for (j, i), w in distribute(x, y).items():
            ops.load(int(m.node[j, i]), 0, 0, -pw * w, 0, 0, 0)
    wg.setup_static()
    ops.analyze(1)
    mom = np.array([[wg.girder_moment_at(m, j, i) for i in range(wg.N_SEG + 1)] for j in range(wg.N_GIRDER)])
    reac = wg.bearing_reactions(m)
    tmom = np.array([max(abs(ops.eleResponse(t[0], "localForce")[4]), abs(ops.eleResponse(t[0], "localForce")[10]))
                     for t in m.trans_ele])
    return mom, reac, tmom, m


# ----------------------------------------------------------------------------------------
# Stage 1: states
# ----------------------------------------------------------------------------------------
def girder_state(p, m_ll):
    """Bottom-fibre stress terms and the lambda at decompression / cracking / ultimate."""
    x = np.linspace(0, wg.L_SPAN, wg.N_SEG + 1)
    dead = girder_dead_moments(p)
    secs = composite_props(p)
    pe = (1 - LOSS) * P_JACK * (N_STRAIGHT + N_HARPED)
    res = {}
    for j in range(wg.N_GIRDER):
        s = secs[j]
        m_nc, m_sdl, _, _ = dead[j]
        e = s["zc_g"] - z_ps(x)
        sig_dp = -pe / s["A_g"] - pe * e * s["zc_g"] / s["I_g"] + m_nc * s["zc_g"] / s["I_g"] \
            + m_sdl * s["zc_c"] / s["I_c"]                                   # kPa, tension +
        k_ll = m_ll[j] * s["zc_c"] / s["I_c"]                                 # stress per lambda
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_cr_x = np.where(k_ll > 0, (FR - sig_dp) / k_ll, np.inf)
            lam_0_x = np.where(k_ll > 0, -sig_dp / k_ll, np.inf)
        mn, c = nominal_resistance(s)
        i_mid = wg.N_SEG // 2
        m_dead_mid = m_nc[i_mid] + m_sdl[i_mid]
        lam_u = (mn - m_dead_mid) / m_ll[j].max()
        res[j] = dict(sig_dp=sig_dp, k_ll=k_ll, lam_cr_x=lam_cr_x, lam_cr=float(lam_cr_x.min()),
                      lam_0=float(lam_0_x.min()), lam_u=float(lam_u), Mn=float(mn), c=float(c),
                      icr_ratio=cracked_I(s, x), sig_bot_dead_mid=float(sig_dp[i_mid]),
                      m_cr_ll=np.where(k_ll > 0, (FR - sig_dp) / np.maximum(k_ll, 1e-9) * m_ll[j], np.inf))
    return res


def element_factors(gs, m_ll, lam, case):
    """Iy factor per longitudinal element (7, 30) for a peak load lam.
    case 'under': cracks open, secant Branson I_e on cracked elements (PCI / ACI 318 form);
    case 'tangent': cracks open, I_cr on cracked elements (lower bound);
    case 'after': cracks closed by prestress, residual factor interpolated between 1 at first
    cracking and RES_AT_ULT at ultimate."""
    f = np.ones((wg.N_GIRDER, wg.N_SEG))
    for j in range(wg.N_GIRDER):
        g = gs[j]
        for i in range(wg.N_SEG):
            lam_cr = min(g["lam_cr_x"][i], g["lam_cr_x"][i + 1])
            if lam <= lam_cr:
                continue
            icr = 0.5 * (g["icr_ratio"][i] + g["icr_ratio"][i + 1])
            if case == "under":
                r = (lam_cr / lam) ** 3
                f[j, i] = r + (1 - r) * icr
            elif case == "tangent":
                f[j, i] = icr
            else:
                t = np.clip((lam - g["lam_cr"]) / max(g["lam_u"] - g["lam_cr"], 1e-9), 0, 1)
                f[j, i] = 1 - (1 - RES_AT_ULT) * t
    return f


def deck_local_moment():
    """Local transverse slab moment per m under one CL-625 wheel of the 175 kN axle,
    continuous slab on girders (AASHTO Std. Spec. 3.24.3.1 form, continuity 0.8), kN m/m,
    plus the dead-load moment w S^2 / 10."""
    s_clear = wg.S_GIRDER
    m_ll = 0.8 * (s_clear + 0.61) / 9.74 * (AXLE_P.max() / 2)
    w = wg.UNIT_WT_CONC * wg.T_SLAB + wg.UNIT_WT_ASPH * wg.T_ASPHALT
    return m_ll, w * s_clear ** 2 / 10


def deck_factors(m, lam):
    """Transverse slab members under the wheel lines crack when the local moment reaches
    f_r t^2 / 6; cracked Iy ratio from 15M @ 300 (Dwg 16/18) singly reinforced section."""
    m_ll, m_dl = deck_local_moment()
    m_cr = FR_DECK * wg.T_SLAB ** 2 / 6
    f = np.ones(len(m.trans_ele))
    if lam * m_ll + m_dl <= m_cr:
        return f, m_cr, m_ll, m_dl
    n = 200e6 / (wg.E_DECK_S6 * 1.3)
    a_s, d = 200e-6 / 0.300, wg.T_SLAB - 0.030 - 0.008
    nr = n * a_s / d
    k = math.sqrt(2 * nr + nr ** 2) - nr
    icr = (k * d) ** 3 / 3 + n * a_s * (d - k * d) ** 2
    ratio = icr / (wg.T_SLAB ** 3 / 12)
    lam_cr = (m_cr - m_dl) / m_ll
    r = (lam_cr / lam) ** 3
    ie = r + (1 - r) * ratio
    loaded = set()
    for x, y, _ in wheel_points():
        for (j, i) in distribute(x, y):
            loaded.add((j, i))
    for k_, (_, ja, ia, jb, ib) in enumerate(m.trans_ele):
        if (ja, ia) in loaded or (jb, ib) in loaded:
            f[k_] = ie
    return f, m_cr, m_ll, m_dl


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
    out = {"assumptions": dict(LOSS=LOSS, FR_kPa=FR, N_P=N_P_REF, RES_AT_ULT=RES_AT_ULT,
                               DETECT_PCT=DETECT_PCT, BEARING_EXP=BEARING_EXP, lanes_y=LANE_C)}

    # --- Stage 1 ---------------------------------------------------------------------
    m_ll, r_ll, t_ll, m = live_load_response(p0)
    gs = girder_state(p0, m_ll)
    dead = girder_dead_moments(p0)
    r_dl = np.array([dead[j][2] + dead[j][3] for j in range(wg.N_GIRDER)] * 2)
    print(f"effective prestress {(1 - LOSS) * P_JACK * 50:.0f} kN/girder, f_r = {FR / 1e3:.2f} MPa")
    print("girder  M_LL,mid  sig_bot(D+P)  lam_decomp  lam_crack  lam_ult   Icr/Ic")
    for j in range(wg.N_GIRDER):
        g = gs[j]
        print(f"  G{j + 1}   {m_ll[j].max():7.0f}   {g['sig_bot_dead_mid'] / 1e3:8.2f} MPa"
              f"   {g['lam_0']:6.2f}     {g['lam_cr']:6.2f}   {g['lam_u']:6.2f}   {g['icr_ratio'][15]:.3f}")
    lam_cr_all = min(g["lam_cr"] for g in gs.values())
    lam_u_all = min(g["lam_u"] for g in gs.values())
    m_ll_deck, m_dl_deck = deck_local_moment()
    m_cr_deck = FR_DECK * wg.T_SLAB ** 2 / 6
    lam_deck = (m_cr_deck - m_dl_deck) / m_ll_deck
    a_pad = wg.BRG_A_LONG * wg.BRG_A_TRAN
    print(f"deck: local M_LL {m_ll_deck:.1f} kNm/m per truck, M_DL {m_dl_deck:.1f}, M_cr {m_cr_deck:.1f}"
          f" -> lam_deck_crack = {lam_deck:.2f}")
    print("bearing pressure under dead load (MPa):", np.round(r_dl / a_pad / 1e3, 2))
    print("bearing reaction per lambda (kN):", np.round(r_ll, 1))
    with np.errstate(divide="ignore"):
        lam_lift = np.where(r_ll < 0, -r_dl / r_ll, np.inf)
    print("lambda at lift-off (min over pads):", np.round(lam_lift.min(), 2))
    out["stage1"] = dict(
        M_ll_mid=[float(m_ll[j].max()) for j in range(7)], lam_decomp=[gs[j]["lam_0"] for j in range(7)],
        lam_crack=[gs[j]["lam_cr"] for j in range(7)], lam_ult=[gs[j]["lam_u"] for j in range(7)],
        Mn=[gs[j]["Mn"] for j in range(7)], icr_ratio_mid=[float(gs[j]["icr_ratio"][15]) for j in range(7)],
        sig_bot_DP_mid_MPa=[gs[j]["sig_bot_dead_mid"] / 1e3 for j in range(7)],
        deck=dict(M_ll=m_ll_deck, M_dl=m_dl_deck, M_cr=m_cr_deck, lam_crack=lam_deck),
        bearing_R_dead=r_dl.tolist(), bearing_R_per_lambda=r_ll.tolist(), lam_liftoff=float(lam_lift.min()))

    # --- Stage 2 ---------------------------------------------------------------------
    f_ref, sh_ref, _ = deck_modes(p0)
    sh_ref = sh_ref[:N_TRACK]
    f_ref = f_ref[:N_TRACK]
    fams = [wg.classify(s)[:2] for s in sh_ref]
    inv = {v: k for k, v in ms.PAIR.items()}
    names = [inv.get(f, f"FE{f}") for f in fams]
    tsens = {n: abs(ms.TEMP_SENS[ms.MODES.index(n)]) for n in names if n in ms.MODES}
    print("tracked baseline modes:", list(zip(names, np.round(f_ref, 3))))

    lams = sorted(set(np.round(np.r_[np.arange(0.5, lam_u_all - 1e-6, 0.5), lam_cr_all], 3)))
    rows = []

    def stiffness(p, lam, key):
        p["I_el"] = element_factors(gs, m_ll, lam, key)
        if key != "after":
            p["trans_el"] = deck_factors(m, lam)[0]
            if BEARING_EXP:
                p["kv"] = p0["kv"] * (np.maximum(r_dl + lam * r_ll, 1e-6) / r_dl) ** BEARING_EXP
        return p

    for lam in lams:
        # mass of the test load only (vehicles / blocks), undamaged stiffness
        fd_m, sh_m, _ = deck_modes({**p0, "extra_mass": truck_mass(lam)})
        f_m, mac_m = track(sh_ref, fd_m, sh_m)
        sh_m_t = [sh_m[np.argmax([wg.mac(r, s) for s in sh_m])] for r in sh_ref]
        cases = {"mass only": (f_m, mac_m)}
        # gravity load: mass + cracked stiffness; tracked against the mass-only shapes
        fd, shd, _ = deck_modes(stiffness({**p0, "extra_mass": truck_mass(lam)}, lam, "under"))
        cases["under load"] = track(sh_m_t, fd, shd)
        # same force applied without mass (jacks against ground anchors / reaction frame)
        for case, key in (("jacked", "under"), ("jacked (tangent)", "tangent"), ("after unload", "after")):
            fd, shd, _ = deck_modes(stiffness(dict(p0), lam, key))
            cases[case] = track(sh_ref, fd, shd)
        n_cr = int((element_factors(gs, m_ll, lam, "under") < 0.999).sum())
        n_deck = int((deck_factors(m, lam)[0] < 0.999).sum())
        for case, (ft, macs) in cases.items():
            df = 100 * (ft - f_ref) / f_ref
            row = dict(lam=float(lam), case=case, f=ft.tolist(), df_pct=df.tolist(), mac=macs.tolist(),
                       n_cracked_girder_el=n_cr, n_cracked_deck_el=n_deck,
                       dT_equiv=[float(d / tsens[n]) if n in tsens else None for d, n in zip(df, names)])
            if case == "under load":
                row["df_stiffness_pct"] = (100 * (ft - f_m) / f_m).tolist()
            rows.append(row)
            print(f"lam {lam:5.2f} {case:18s} girder el {n_cr:3d} deck el {n_deck:3d}  df% " +
                  " ".join(f"{d:+6.2f}" for d in df) + "  minMAC %.3f" % macs.min())
        print(f"{'':10s} under/mass-only (stiffness part)  df% " +
              " ".join(f"{d:+6.2f}" for d in rows[-4]["df_stiffness_pct"]))
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
