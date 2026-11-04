"""
Walker Road (Lower Walker Road Rte. 2 Overpass, West structure / L930) -- grillage model.

Grillage analogy per CSA S6-19 Section 5 (refined methods of analysis) and its commentary
(CSA S6.1-19 C5.9), following Hambly / Jaeger & Bakht for slab-on-girder decks:
  * one longitudinal member per girder line (composite girder + haunch + tributary slab,
    exterior members also carry the overhang and the concrete barrier wall),
  * transverse members orthogonal to the girders (skew > 20 deg) representing the slab
    (I = b t^3/12, J = b t^3/6 per tributary width b),
  * skewed members along the abutment and intermediate diaphragm lines,
  * elastomeric bearings as 6-DOF springs, offset below the grillage plane by a rigid link
    so the horizontal bearing stiffness acts at the bearing level (eccentric restraint).

Units: kN, m, s, tonne (t)  ->  stress in kPa, mass density in t/m^3.
Geometry from as-built drawings, contract 93-4013 (ADI Ltd, 1992), West structure
Dwg 1, 5, 11, 12, 13, 16 of 18.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import openseespy.opensees as ops
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ----------------------------------------------------------------------------------------
# Geometry (drawings)
# ----------------------------------------------------------------------------------------
L_SPAN = 33.000                       # c/c bearings along girder (Dwg 11/18, 12/18)
SKEW_DEG = 29 + 36 / 60 + 22 / 3600   # 29 deg 36' 22"
TAN_SKEW = math.tan(math.radians(SKEW_DEG))
N_GIRDER = 7
S_GIRDER = 1.900                      # girder spacing (perpendicular)
OVERHANG = 1.055                      # exterior girder CL to deck edge
DECK_W = 13.510
T_SLAB = 0.200
HAUNCH = 0.065                        # at CL bearings (Dwg 12/18); also used by CSiBridge model
H_GIRDER = 1.500                      # NBDOT Type 1
W_TOPFL = 0.550
T_ASPHALT = 0.065
BARRIER_W = 0.555
N_SEG = 30                            # elements per girder -> dx = 1.10 m
DIAPH_IDX = (0, 10, 20, 30)           # abutment, 1/3, 2/3, abutment diaphragm lines (along skew)
DX = L_SPAN / N_SEG
Y_GIRDER = np.array([(j - (N_GIRDER - 1) / 2) * S_GIRDER for j in range(N_GIRDER)])
X0_GIRDER = (Y_GIRDER - Y_GIRDER[0]) * TAN_SKEW   # skew offset of the north bearing line
# Orientation (sensor survey: y = 0 at the west deck edge, x = 0 at the north end): G1 is the west
# girder, G7 the east (median-side) girder, and x runs north -> south with eastbound traffic.
# Obtuse deck corners: G7 north (north-east) and G1 south (south-west).
SIDE_NAME = {0: "west", N_GIRDER - 1: "east"}

# Diaphragms (Dwg 12/18, 16/18): 300 mm cast-in-place, ~1.10 m below the slab soffit
DIAPH_T = 0.300
DIAPH_H = 1.100

# Elastomeric bearings (Dwg 5/18): Goodco EL 600 x 250 x 80, 14 required
BRG_KV = 257.8e3                      # kN/m   (257.8 kN/mm compressive stiffness)
BRG_KH = 1.74e3                       # kN/m   (1.74 kN/mm shear stiffness)
BRG_A_LONG = 0.250                    # pad dimension along girder
BRG_A_TRAN = 0.600                    # pad dimension across girder
BRG_T = 0.080

# ----------------------------------------------------------------------------------------
# Materials (CSA S6-19)
# ----------------------------------------------------------------------------------------
FC_GIRDER = 40.0                      # MPa, P/S concrete at 28 d (Dwg 1/18 design criteria)
FC_DECK = 30.0                        # MPa, cast-in-place concrete
GAMMA_C_KG = 2400.0                   # kg/m^3 for the modulus expression
UNIT_WT_CONC = 24.0                   # kN/m^3, reinforced / prestressed concrete
UNIT_WT_ASPH = 23.5                   # kN/m^3, bituminous wearing surface
G_ACC = 9.81
RHO_CONC = UNIT_WT_CONC / G_ACC       # t/m^3
RHO_ASPH = UNIT_WT_ASPH / G_ACC
NU_CONC = 0.2
RAIL_MASS = 0.015                     # t/m, aluminium rail + posts per side (approx.)


def ec_s6(fc_mpa: float, gamma_kg: float = GAMMA_C_KG) -> float:
    """CSA S6-19 Cl. 8.4.1.7: Ec = (3000 sqrt(f'c) + 6900)(gamma_c/2300)^1.5, returned in kPa."""
    return (3000.0 * math.sqrt(fc_mpa) + 6900.0) * (gamma_kg / 2300.0) ** 1.5 * 1e3


E_GIRDER_S6 = ec_s6(FC_GIRDER)
E_DECK_S6 = ec_s6(FC_DECK)


# ----------------------------------------------------------------------------------------
# Cross-section geometry and properties
# ----------------------------------------------------------------------------------------
def girder_polygon() -> np.ndarray:
    """NBDOT Type 1 girder (Dwg 11/18 section A-A), (y, z) in m, origin at bottom centre."""
    pts_mm = [(-330, 0), (330, 0), (330, 200), (75, 455), (75, 1180), (275, 1300),
              (275, 1500), (-275, 1500), (-275, 1300), (-75, 1180), (-75, 455), (-330, 200)]
    return np.array(pts_mm, float) / 1000.0


def barrier_polygon() -> np.ndarray:
    """Concrete barrier wall incl. granite curb (Dwg 13/18 detail A), (y, z) in m.
    y measured from the outer deck edge towards the roadway, z from top of slab."""
    pts_mm = [(0, 0), (555, 0), (555, 140), (380, 406), (380, 776), (40, 776), (0, 400)]
    return np.array(pts_mm, float) / 1000.0


def rect(y0, y1, z0, z1) -> np.ndarray:
    return np.array([(y0, z0), (y1, z0), (y1, z1), (y0, z1)], float)


def poly_props(p: np.ndarray):
    """Area, centroid and centroidal second moments of a simple polygon (CCW)."""
    y, z = p[:, 0], p[:, 1]
    y1, z1 = np.roll(y, -1), np.roll(z, -1)
    cr = y * z1 - y1 * z
    a = 0.5 * cr.sum()
    yc = ((y + y1) * cr).sum() / (6 * a)
    zc = ((z + z1) * cr).sum() / (6 * a)
    iyy0 = ((z * z + z * z1 + z1 * z1) * cr).sum() / 12.0     # about y axis through origin
    izz0 = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12.0
    return a, yc, zc, iyy0 - a * zc ** 2, izz0 - a * yc ** 2


def _point_in_poly(py, pz, poly):
    inside = np.zeros(py.shape, bool)
    n = len(poly)
    for k in range(n):
        ya, za = poly[k]
        yb, zb = poly[(k + 1) % n]
        cond = (za > pz) != (zb > pz)
        with np.errstate(divide="ignore", invalid="ignore"):
            yint = ya + (pz - za) * (yb - ya) / (zb - za)
        inside ^= cond & (py < yint)
    return inside


def saint_venant_j(poly: np.ndarray, h: float = 0.004) -> float:
    """St-Venant torsion constant by finite differences on the Prandtl stress function
    (lap(phi) = -2, phi = 0 on the boundary, J = 2 int phi dA), Richardson-extrapolated."""
    def solve(hh):
        ymin, zmin = poly.min(axis=0) - hh
        ymax, zmax = poly.max(axis=0) + hh
        ny, nz = int(round((ymax - ymin) / hh)) + 1, int(round((zmax - zmin) / hh)) + 1
        gy, gz = np.meshgrid(ymin + hh * np.arange(ny), zmin + hh * np.arange(nz), indexing="ij")
        mask = _point_in_poly(gy, gz, poly)
        idx = -np.ones(mask.shape, int)
        idx[mask] = np.arange(mask.sum())
        n = int(mask.sum())
        a = lil_matrix((n, n))
        b = np.full(n, -2.0 * hh * hh)
        for (i, k), m in np.ndenumerate(idx):
            if m < 0:
                continue
            a[m, m] = -4.0
            for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = idx[i + di, k + dk]
                if q >= 0:
                    a[m, q] = 1.0
        phi = spsolve(a.tocsr(), b)
        return 2.0 * phi.sum() * hh * hh

    j1, j2 = solve(2 * h), solve(h)
    return j2 + (j2 - j1) / 3.0          # O(h^2) Richardson


_J_CACHE: dict = {}


def girder_slab_polygon() -> np.ndarray:
    """Girder + haunch + the slab strip directly over the top flange, as one solid: the
    torsion constant of this block is much larger than the sum of its parts (J ~ t^3)."""
    g = girder_polygon()[:6].tolist()                    # bottom flange .. top taper (right)
    z_s1 = H_GIRDER + HAUNCH + T_SLAB
    top = [(W_TOPFL / 2, z_s1), (-W_TOPFL / 2, z_s1)]
    left = girder_polygon()[8:].tolist()                 # top taper (left) .. bottom flange
    return np.array(g + top + left)


def barrier_slab_polygon() -> np.ndarray:
    """Barrier wall with the slab under its base (solid block for torsion)."""
    b = barrier_polygon().tolist()
    return np.array([(0.0, -T_SLAB), (BARRIER_W, -T_SLAB)] + b[1:] + [(0.0, 0.0)])


def j_girder() -> float:
    """St-Venant J of the girder + haunch + slab strip over the top flange (one block)."""
    if "g" not in _J_CACHE:
        _J_CACHE["g"] = saint_venant_j(girder_slab_polygon(), 0.005)
    return _J_CACHE["g"]


def j_girder_alone() -> float:
    if "g0" not in _J_CACHE:
        _J_CACHE["g0"] = saint_venant_j(girder_polygon(), 0.005)
    return _J_CACHE["g0"]


def j_barrier() -> float:
    """Increase in J when the barrier is added to the slab strip under it."""
    if "b" not in _J_CACHE:
        _J_CACHE["b"] = saint_venant_j(barrier_slab_polygon(), 0.005) - BARRIER_W * T_SLAB ** 3 / 3
    return _J_CACHE["b"]


def composite(parts):
    """parts: list of (polygon, n) with n = E_part / E_ref. Returns transformed properties."""
    at = sy = sz = 0.0
    props = []
    for poly, n in parts:
        a, yc, zc, iyy, izz = poly_props(poly)
        props.append((a, yc, zc, iyy, izz, n))
        at += n * a
        sy += n * a * yc
        sz += n * a * zc
    yc_t, zc_t = sy / at, sz / at
    iyy_t = sum(n * (iyy + a * (zc - zc_t) ** 2) for a, yc, zc, iyy, izz, n in props)
    izz_t = sum(n * (izz + a * (yc - yc_t) ** 2) for a, yc, zc, iyy, izz, n in props)
    return at, yc_t, zc_t, iyy_t, izz_t


# ----------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------
def default_params() -> dict:
    """Design (nominal, CSA S6) parameter set. Multipliers are relative to nominal values."""
    return dict(
        Eg=1.0,            # girder concrete modulus factor (on S6 Ec, f'c = 40 MPa)
        Ed=1.0,            # deck / diaphragm / barrier concrete modulus factor (f'c = 30 MPa)
        barrier=1.0,       # barrier composite effectiveness (0 = mass only, 1 = fully composite)
        barrier_side=np.ones(2),         # extra barrier factors for the G1 side and the G7 side
        trans=1.0,         # transverse slab stiffness factor (cracking / effective slab)
        diaph=1.0,         # diaphragm stiffness factor
        tors=1.0,          # St-Venant torsion factor of the longitudinal (girder) members
        mass=1.0,          # superimposed dead mass factor (asphalt, rails)
        kv=np.ones(2 * N_GIRDER),        # bearing vertical stiffness factors (14 bearings)
        kh=np.ones(2 * N_GIRDER),        # bearing longitudinal (along girder) stiffness factors
        kt=np.ones(2 * N_GIRDER),        # bearing transverse stiffness factors
        kr=np.ones(2 * N_GIRDER),        # bearing rotational stiffness factors
        E_mult=np.ones(N_GIRDER),        # girder-by-girder E multiplier (sensitivity)
        I_mult=np.ones(N_GIRDER),        # girder-by-girder I multiplier (sensitivity)
        I_el=None,         # optional (N_GIRDER, N_SEG) element-by-element Iy factor (cracking)
        J_el=None,         # optional (N_GIRDER, N_SEG) element-by-element J factor
        trans_el=None,     # optional factor per transverse slab member (order of Model.trans_ele)
        extra_mass=None,   # optional {(j, i): translational mass (t)} at grillage nodes (test load)
    )


def calibrated_params(path: str = "results/calibration.json") -> dict:
    """Default parameters updated with the frequency-calibrated (MAP) values."""
    import json
    with open(path) as fh:
        vals = json.load(fh)["calibrated"]["values"]
    p = default_params()
    for k, v in vals.items():
        if k in ("kh", "kv", "kt", "kr"):
            p[k] = np.full(2 * N_GIRDER, float(v))
        else:
            p[k] = float(v)
    return p


def girder_section(ext: bool, p: dict, side: int = 1) -> dict:
    """Composite longitudinal member properties, referenced to the girder modulus.
    side: 0 = G1 edge, 1 = G7 edge (selects the per-side barrier factor)."""
    eg = E_GIRDER_S6 * p["Eg"]
    ed = E_DECK_S6 * p["Ed"]
    n_d = ed / eg
    z_h0, z_s0 = H_GIRDER, H_GIRDER + HAUNCH
    z_s1 = z_s0 + T_SLAB
    if ext:
        b_in, b_out = S_GIRDER / 2, OVERHANG
    else:
        b_in = b_out = S_GIRDER / 2
    parts = [(girder_polygon(), 1.0),
             (rect(-W_TOPFL / 2, W_TOPFL / 2, z_h0, z_s0), n_d),
             (rect(-b_in, b_out, z_s0, z_s1), n_d)]
    # girder+haunch+slab-over-flange block, plus the remaining slab at b t^3/6 (the other half
    # of the plate torsion is carried by the transverse members)
    jt = j_girder() + n_d * (b_in + b_out - W_TOPFL) * T_SLAB ** 3 / 6.0
    fb = p["barrier"] * np.asarray(p.get("barrier_side", np.ones(2)))[side]
    if ext and fb > 0:
        bp = barrier_polygon().copy()
        # outer deck edge at y = +OVERHANG, barrier extends towards the roadway (negative y)
        bp[:, 0] = OVERHANG - bp[:, 0]
        bp[:, 1] = bp[:, 1] + z_s1
        bp = bp[::-1]                     # keep CCW after the mirror
        parts.append((bp, n_d * fb))
        jt += n_d * fb * j_barrier()
    a, yc, zc, iyy, izz = composite(parts)
    return dict(E=eg, G=eg / (2 * (1 + NU_CONC)), A=a, Iy=iyy, Iz=izz, J=jt * p["tors"], zc=zc, yc=yc)


def girder_mass(ext: bool, p: dict) -> tuple[float, float, float]:
    """Mass per unit length lumped on the girder line, and (mass, lever arm) of the part
    outboard of an exterior girder line (overhang slab + barrier + curb + rail + asphalt)."""
    a_g = poly_props(girder_polygon())[0]
    m_line = RHO_CONC * (a_g + W_TOPFL * HAUNCH + S_GIRDER / 2 * T_SLAB)
    m_line += p["mass"] * RHO_ASPH * T_ASPHALT * S_GIRDER / 2
    if not ext:
        m_line += RHO_CONC * S_GIRDER / 2 * T_SLAB + p["mass"] * RHO_ASPH * T_ASPHALT * S_GIRDER / 2
        return m_line, 0.0, 0.0
    a_b, yc_b = poly_props(barrier_polygon())[:2]
    comps = [(RHO_CONC * OVERHANG * T_SLAB, OVERHANG / 2),
             (RHO_CONC * a_b, OVERHANG - yc_b),
             (p["mass"] * RHO_ASPH * T_ASPHALT * (OVERHANG - BARRIER_W), (OVERHANG - BARRIER_W) / 2),
             (p["mass"] * RAIL_MASS, OVERHANG - 0.19)]
    m_out = sum(m for m, _ in comps)
    d_out = sum(m * d for m, d in comps) / m_out
    return m_line, m_out, d_out


def slab_section(b: float, p: dict) -> dict:
    ed = E_DECK_S6 * p["Ed"]
    t = T_SLAB
    return dict(E=ed, G=ed / (2 * (1 + NU_CONC)), A=b * t, Iy=p["trans"] * b * t ** 3 / 12,
                Iz=t * b ** 3 / 12, J=p["trans"] * b * t ** 3 / 6)


def diaphragm_section(p: dict) -> dict:
    """Cast-in-place diaphragm acting with an effective slab flange of one grid spacing."""
    ed = E_DECK_S6 * p["Ed"]
    web = rect(-DIAPH_T / 2, DIAPH_T / 2, 0.0, DIAPH_H)
    fl = rect(-DX / 2, DX / 2, DIAPH_H, DIAPH_H + T_SLAB)
    a, yc, zc, iyy, izz = composite([(web, 1.0), (fl, 1.0)])
    hw = DIAPH_H + T_SLAB
    beta = 1 / 3 - 0.21 * (DIAPH_T / hw) * (1 - (DIAPH_T / hw) ** 4 / 12)
    j = beta * hw * DIAPH_T ** 3 + DX * T_SLAB ** 3 / 6
    f = p["diaph"]
    return dict(E=ed, G=ed / (2 * (1 + NU_CONC)), A=a, Iy=f * iyy, Iz=izz, J=f * j)


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
@dataclass
class Model:
    node: np.ndarray                    # (N_GIRDER, N_SEG+1) grillage node tags
    xy: np.ndarray                      # (N_GIRDER, N_SEG+1, 2) plan coordinates
    long_ele: np.ndarray                # (N_GIRDER, N_SEG) longitudinal element tags
    brg_ele: list                       # 14 zeroLength bearing elements (north 0..6, south 0..6)
    brg_ground: list                    # 14 fixed ground nodes
    oh_nodes: list = field(default_factory=list)
    mass_lumped: dict = field(default_factory=dict)   # node -> translational mass
    sections: dict = field(default_factory=dict)
    trans_ele: list = field(default_factory=list)       # (tag, ja, ia, jb, ib) slab members


def build(p: dict | None = None) -> Model:
    p = default_params() if p is None else p
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    node = np.zeros((N_GIRDER, N_SEG + 1), int)
    xy = np.zeros((N_GIRDER, N_SEG + 1, 2))
    tag = 1
    for j in range(N_GIRDER):
        for i in range(N_SEG + 1):
            x, y = X0_GIRDER[j] + i * DX, Y_GIRDER[j]
            ops.node(tag, x, y, 0.0)
            node[j, i], xy[j, i] = tag, (x, y)
            tag += 1

    ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ele = 1
    mass_lumped: dict[int, float] = {}

    def add_mass(nd, m):
        mass_lumped[nd] = mass_lumped.get(nd, 0.0) + m

    # --- longitudinal composite girder members
    long_ele = np.zeros((N_GIRDER, N_SEG), int)
    secs = {}
    for j in range(N_GIRDER):
        ext = j in (0, N_GIRDER - 1)
        s = girder_section(ext, p, side=0 if j == 0 else 1)
        secs[j] = s
        m_line, m_out, d_out = girder_mass(ext, p)
        e = s["E"] * p["E_mult"][j]
        g = s["G"] * p["E_mult"][j]
        iy = s["Iy"] * p["I_mult"][j]
        for i in range(N_SEG):
            fi = 1.0 if p.get("I_el") is None else float(p["I_el"][j][i])
            fj = 1.0 if p.get("J_el") is None else float(p["J_el"][j][i])
            ops.element("elasticBeamColumn", ele, int(node[j, i]), int(node[j, i + 1]),
                        s["A"], e, g, s["J"] * fj, iy * fi, s["Iz"], 1, "-mass", m_line)
            long_ele[j, i] = ele
            add_mass(int(node[j, i]), m_line * DX / 2)
            add_mass(int(node[j, i + 1]), m_line * DX / 2)
            ele += 1
        secs[f"m_line_{j}"] = m_line
        secs[f"m_out_{j}"] = (m_out, d_out)

    # --- outboard mass of exterior girders on rigid transverse arms (exact eccentric mass)
    oh_nodes = []
    for j, sgn in ((0, -1.0), (N_GIRDER - 1, 1.0)):
        m_out, d_out = secs[f"m_out_{j}"]
        for i in range(N_SEG + 1):
            trib = DX / 2 if i in (0, N_SEG) else DX
            x, y = xy[j, i]
            ops.node(tag, x, y + sgn * d_out, 0.0)
            ops.mass(tag, m_out * trib, m_out * trib, m_out * trib, 0.0, 0.0, 0.0)
            ops.rigidLink("beam", int(node[j, i]), tag)
            add_mass(tag, m_out * trib)
            oh_nodes.append(tag)
            tag += 1

    # --- transverse slab members. "orthogonal" (default, skew > 20 deg): (j, i+1) -> (j+1, i),
    # within 0.6 deg of square to the girders. "skew": parallel to the supports, (j, i) -> (j+1, i),
    # slab width measured square to the member = DX cos(skew); none on the diaphragm lines.
    if p.get("mesh", "orthogonal") == "skew":
        sl = slab_section(DX * math.cos(math.radians(SKEW_DEG)), p)
        pairs = [(j, i, j + 1, i) for j in range(N_GIRDER - 1) for i in range(N_SEG + 1) if i not in DIAPH_IDX]
    else:
        sl = slab_section(DX, p)
        pairs = [(j, i + 1, j + 1, i) for j in range(N_GIRDER - 1) for i in range(N_SEG)]
    trans_ele = []
    for k, (ja, ia, jb, ib) in enumerate(pairs):
        ft = 1.0 if p.get("trans_el") is None else float(p["trans_el"][k])
        ops.element("elasticBeamColumn", ele, int(node[ja, ia]), int(node[jb, ib]),
                    sl["A"], sl["E"], sl["G"], sl["J"] * ft, sl["Iy"] * ft, sl["Iz"], 1)
        trans_ele.append((ele, ja, ia, jb, ib))
        ele += 1

    # --- diaphragm lines along the skew: (j, i) -> (j+1, i), i in DIAPH_IDX
    ds = diaphragm_section(p)
    a_web = DIAPH_T * DIAPH_H
    l_skew = S_GIRDER / math.cos(math.radians(SKEW_DEG))
    m_diaph: dict[int, float] = {}
    for i in DIAPH_IDX:
        for j in range(N_GIRDER - 1):
            ops.element("elasticBeamColumn", ele, int(node[j, i]), int(node[j + 1, i]),
                        ds["A"], ds["E"], ds["G"], ds["J"], ds["Iy"], ds["Iz"], 1)
            ele += 1
            md = RHO_CONC * a_web * (l_skew - 0.15)       # web of diaphragm between girder webs
            for nd in (int(node[j, i]), int(node[j + 1, i])):
                m_diaph[nd] = m_diaph.get(nd, 0.0) + md / 2
    # optional test-load mass (vehicles / blocks parked on the deck during OMA); ops.mass
    # overwrites, so it is merged with the diaphragm nodal mass before assignment
    for (j, i), mx in (p.get("extra_mass") or {}).items():
        nd = int(node[j, i])
        m_diaph[nd] = m_diaph.get(nd, 0.0) + mx
    for nd, md in m_diaph.items():
        ops.mass(nd, md, md, md, 0.0, 0.0, 0.0)
        add_mass(nd, md)
    secs["slab"], secs["diaph"] = sl, ds

    # --- bearings: rigid offset to the bearing level, 6-DOF springs to ground
    brg_ele, brg_ground = [], []
    mat = 1
    k_rot_long = BRG_KV * BRG_A_LONG ** 2 / 12.0     # rotation in the girder plane (about Y)
    k_rot_tran = BRG_KV * BRG_A_TRAN ** 2 / 12.0     # rotation about the girder axis (about X)
    k_rot_z = BRG_KH * (BRG_A_LONG ** 2 + BRG_A_TRAN ** 2) / 12.0
    for end, i in enumerate((0, N_SEG)):
        for j in range(N_GIRDER):
            b = end * N_GIRDER + j
            x, y = xy[j, i]
            ecc = secs[j]["zc"] + BRG_T / 2
            ops.node(tag, x, y, -ecc)
            ops.rigidLink("beam", int(node[j, i]), tag)
            ops.node(tag + 1, x, y, -ecc)
            ops.fix(tag + 1, 1, 1, 1, 1, 1, 1)
            ks = [BRG_KH * p["kh"][b], BRG_KH * p["kt"][b], BRG_KV * p["kv"][b],
                  k_rot_tran * p["kr"][b], k_rot_long * p["kr"][b], k_rot_z]
            mats = []
            for k in ks:
                ops.uniaxialMaterial("Elastic", mat, k)
                mats.append(mat)
                mat += 1
            ops.element("zeroLength", ele, tag + 1, tag, "-mat", *mats, "-dir", 1, 2, 3, 4, 5, 6)
            brg_ele.append(ele)
            brg_ground.append(tag + 1)
            ele += 1
            tag += 2

    return Model(node=node, xy=xy, long_ele=long_ele, brg_ele=brg_ele, brg_ground=brg_ground,
                 oh_nodes=oh_nodes, mass_lumped=mass_lumped, sections=secs, trans_ele=trans_ele)


# ----------------------------------------------------------------------------------------
# Modal analysis
# ----------------------------------------------------------------------------------------
def modal(m: Model, n_modes: int = 40, solver: str = "-genBandArpack"):
    """Returns frequencies (Hz), vertical shapes at girder nodes (n, 7, 31), and the fraction
    of modal kinetic energy in vertical translation (to separate deck modes from sway)."""
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    lam = np.array(ops.eigen(solver, n_modes))
    freqs = np.sqrt(np.abs(lam)) / (2 * np.pi)
    shapes = np.zeros((n_modes, N_GIRDER, N_SEG + 1))
    vert = np.zeros(n_modes)
    nodes = list(m.mass_lumped.keys())
    mass = np.array([m.mass_lumped[n] for n in nodes])
    for k in range(n_modes):
        for j in range(N_GIRDER):
            for i in range(N_SEG + 1):
                shapes[k, j, i] = ops.nodeEigenvector(int(m.node[j, i]), k + 1, 3)
        ev = np.array([ops.nodeEigenvector(int(n), k + 1)[:3] for n in nodes])
        e = mass[:, None] * ev ** 2
        vert[k] = e[:, 2].sum() / e.sum()
    return freqs, shapes, vert


_XI = np.linspace(0, 1, N_SEG + 1)
_ETA = np.linspace(-1, 1, N_GIRDER)
_LEG = [np.ones_like(_ETA), _ETA, 0.5 * (3 * _ETA ** 2 - 1), 0.5 * (5 * _ETA ** 3 - 3 * _ETA),
        (35 * _ETA ** 4 - 30 * _ETA ** 2 + 3) / 8]


def classify(shape: np.ndarray, m_max: int = 4, n_max: int = 4):
    """Dominant (m, n) family: m longitudinal half-waves (sin), n transverse order (Legendre)."""
    best, tot, out = None, (shape ** 2).sum(), np.zeros((m_max, n_max + 1))
    for mm in range(1, m_max + 1):
        sx = np.sin(mm * np.pi * _XI)
        for nn in range(n_max + 1):
            basis = np.outer(_LEG[nn], sx)
            c = (shape * basis).sum() / (basis ** 2).sum()
            out[mm - 1, nn] = c ** 2 * (basis ** 2).sum() / tot
    mm, nn = np.unravel_index(np.argmax(out), out.shape)
    return int(mm + 1), int(nn), float(out[mm, nn])


def mac(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))


def deck_modes(freqs, shapes, vert, vmin: float = 0.6):
    keep = [k for k in range(len(freqs)) if vert[k] >= vmin]
    return freqs[keep], shapes[keep], np.array(keep)


# ----------------------------------------------------------------------------------------
# Static analysis helpers (influence surfaces)
# ----------------------------------------------------------------------------------------
def setup_static():
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-10, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def girder_moment_at(m: Model, j: int, i: int) -> float:
    """Vertical bending moment (kN m, sagging +) in girder j at node i (mean of both sides)."""
    vals = []
    if i > 0:
        f = ops.eleResponse(int(m.long_ele[j, i - 1]), "localForce")
        vals.append(-f[10])              # My at end j
    if i < N_SEG:
        f = ops.eleResponse(int(m.long_ele[j, i]), "localForce")
        vals.append(f[4])                # My at end i
    return float(np.mean(vals))


def girder_shear_at(m: Model, j: int, i: int) -> float:
    f = ops.eleResponse(int(m.long_ele[j, i]), "localForce")
    return float(f[2])                   # Vz at end i


def bearing_reactions(m: Model) -> np.ndarray:
    """Upward vertical reaction at the 14 bearings (kN)."""
    return np.array([ops.eleResponse(e, "force")[2] for e in m.brg_ele])


if __name__ == "__main__":
    p = default_params()
    for ext in (False, True):
        s = girder_section(ext, p)
        print("ext" if ext else "int", {k: round(v, 5) for k, v in s.items()})
    print("J girder (m^4)", j_girder(), " J barrier", j_barrier())
    print("girder mass", girder_mass(False, p), girder_mass(True, p))
    print("Ec girder/deck (MPa)", E_GIRDER_S6 / 1e3, E_DECK_S6 / 1e3)
    m = build(p)
    f, sh, v = modal(m, 30)
    for k in range(30):
        print(k + 1, round(f[k], 3), round(v[k], 2), classify(sh[k]))
