"""Figures for the Q3 report (called by q3_run.py). One style throughout:
colour = mode (M1 blue, M2 orange, M3 green), line style = case, dashed vertical lines =
limit states, and panels that show the same quantity share their vertical scale."""
from __future__ import annotations

import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 150, "savefig.bbox": "tight"})

MODE_COL = ["#1f5f8b", "#d07a1e", "#3a8a5a"]
LS_COL = {"SLS": "#7a8590", "75% ULS": "#d07a1e", "ULS": "#b3261e", "beyond ULS": "#222222"}
LAM = "λ = CL-625 trucks per lane"


def ls_lines(ax, lsd, label=False):
    for n, lam in lsd.items():
        ax.axvline(lam, color=LS_COL[n], lw=0.9, ls="--", zorder=0)


def ls_handles(lsd):
    return [Line2D([], [], color=LS_COL[n], lw=0.9, ls="--", label=f"{n} (λ = {lam:.2f})") for n, lam in lsd.items()]


def legend_below(fig, handles, ncol=4):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.02))


def mode_handles(names):
    return [Line2D([], [], color=MODE_COL[k], lw=2.5, label=n) for k, n in enumerate(names)]


def style_handles(items):
    return [Line2D([], [], color="0.25", ls=lsty, lw=1.6, label=lab) for lab, lsty in items]


def tracked(rows, case, k):
    return [r for r in rows if r["case"] == case and r["tracked"][k]]


# ---------------------------------------------------------------------------------------------
def step1(d, out):
    s1 = d["stage1"]
    it = s1["iteration"]
    keys = sorted(it, key=float)
    lam = np.array([float(k) for k in keys])
    g = lambda key, f=lambda v: v: np.array([f(it[k][key]) for k in keys])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    a = ax[0]
    a.plot(lam, 100 * g("cracked_el") / 210, "-", color="#b3261e", lw=2, label="girder length cracked (%)")
    a.plot(lam, 100 * g("min_Ie_ratio"), "--", color="#1f5f8b", lw=2, label="lowest stiffness I$_e$/I$_g$ (%)")
    a.set_ylim(0, 105)
    a.set_ylabel("%")
    a.set_title("Girders (as-built prestress, CSA S6-19)")
    a.legend(loc="center left")
    b = ax[1]
    b.plot(lam, g("bearing_contact", min), "-", color="#1f5f8b", lw=2, label="pad contact (1 = no uplift)")
    b.plot(lam, g("slip_ratio", max), "--", color="#d07a1e", lw=2, label="H / μR (below 1 = no slip)")
    b.axhline(1, color="k", lw=0.8)
    b.set_ylim(0, 1.15)
    b.set_title("Bearings: uplift and slip")
    b.legend(loc="center right")
    c = ax[2]
    first = it[keys[0]]
    c.plot(lam, g("soffit_outward_mm"), "-", color="#1f5f8b", lw=2, label="girder soffit towards ballast wall")
    c.plot(lam, g("joint_opening_mm"), "--", color="#d07a1e", lw=2, label="deck joint opening")
    c.axhline(first["soffit_clearance_mm"], color="#1f5f8b", ls=":", label=f"clearance ≈ {first['soffit_clearance_mm']:.0f} mm")
    c.axhline(first["joint_gap_mm"], color="#d07a1e", ls=":", label=f"joint gap {first['joint_gap_mm']:.0f} mm")
    c.set_yscale("log")
    c.set_ylabel("movement (mm)")
    c.set_title("Bearings: can the girder ends lock?")
    c.legend(loc="lower right")
    for a_ in ax:
        ls_lines(a_, s1["limit_states"])
        a_.set_xlabel(LAM)
    legend_below(fig, ls_handles(s1["limit_states"]), ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{out}/Q3_step1_physical_changes.png")
    plt.close(fig)


def step2(d, out, n_modes=3):
    st = d["stage2"]
    names = st["names"][:n_modes]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k in range(n_modes):
        for c, lsty in (("jacked", "-"), ("after unload", "--")):
            rr = tracked(st["rows"], c, k)
            x = [r["lam"] for r in rr]
            ax[0].plot(x, [r["df_pct"][k] for r in rr], lsty, color=MODE_COL[k], lw=1.8)
            ax[1].semilogy(x, [max(1 - r["mac_acc"][k], 1e-6) for r in rr], lsty, color=MODE_COL[k], lw=1.8)
        n = names[k]
        if n in st["mac_stab_measured"]:
            ax[1].axhline(1 - st["mac_stab_measured"][n], color=MODE_COL[k], lw=0.8, ls=":")
    from limit_states import DETECT_PCT
    ax[0].axhspan(-DETECT_PCT, DETECT_PCT, color="0.88", zorder=0)
    ax[0].set_ylabel("frequency change (%)")
    ax[0].set_title(f"Frequencies (grey band = ±{DETECT_PCT} % detection limit)")
    ax[1].set_ylabel("1 − MAC at the 9 accelerometers")
    ax[1].set_ylim(1e-6, 0.2)
    ax[1].set_title("Mode shapes (dotted = measured MAC scatter of each mode)")
    for a_ in ax:
        ls_lines(a_, d["stage1"]["limit_states"])
        a_.set_xlabel(LAM)
    legend_below(fig, mode_handles(names) + style_handles([("load on (cracks open)", "-"),
                                                           ("after unloading", "--")])
                 + ls_handles(d["stage1"]["limit_states"]), ncol=5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{out}/Q3_step2_oma_response.png")
    plt.close(fig)


def step2b_detection_grid(d, out, n_modes=3):
    st = d["stage2"]
    from limit_states import DETECT_PCT
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.4))
    for ax, c in zip(axs, ("jacked", "after unload")):
        rr = [r for r in st["rows"] if r["case"] == c]
        lam = [r["lam"] for r in rr]
        labels, vals, flags = ["REAL CHANGE: girder bending\nat the strain gauges (%)"], [], []
        v = [max(r["dkappa_max_pct"][:3]) for r in rr]
        vals.append(v)
        flags.append([2 if x >= 1.0 else 0 for x in v])
        for k in range(n_modes):
            n = st["names"][k]
            labels.append(f"OMA frequency {n}: |Δf| (%)")
            v = [abs(r["df_pct"][k]) for r in rr]
            vals.append(v)
            flags.append([1 if x >= DETECT_PCT else 0 for x in v])
        for k in range(n_modes):
            n = st["names"][k]
            thr = 1 - st["mac_stab_measured"].get(n, 0.999)
            labels.append(f"OMA shape {n}: 1−MAC (×1000)")
            v = [1000 * (1 - r["mac_acc"][k]) for r in rr]
            vals.append(v)
            flags.append([1 if x >= 1000 * thr else 0 for x in v])
        ax.imshow(np.array(flags), cmap=ListedColormap(["#e9ecef", "#8fd19e", "#f6c38b"]), vmin=0, vmax=2,
                  aspect="auto")
        for i, row in enumerate(vals):
            for j, x in enumerate(row):
                ax.text(j, i, f"{abs(x):.1f}", ha="center", va="center", fontsize=6.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
        ax.set_xticks(range(len(lam)), [f"{x:g}" for x in lam], fontsize=7)
        ax.grid(False)
        ax.axhline(0.5, color="k", lw=1.2)
        j_cr = int(np.searchsorted(lam, min(d["stage1"]["lam_crack"])))
        ax.axvline(j_cr - 0.5, color="#b3261e", lw=1.2, ls="--")
        ax.set_title({"jacked": "Load on (cracks open, load mass removed)",
                      "after unload": "After unloading (cracks closed by the prestress)"}[c], loc="left")
    axs[1].set_xlabel(LAM + "   (red dashed line: first girder crack)")
    fig.text(0.01, -0.01, f"Orange = the structure has really changed (bending change ≥ 1 %).   Green = accelerometer "
             f"OMA can detect it (|Δf| ≥ {DETECT_PCT} %, or MAC below the measured scatter).   Grey = OMA cannot "
             "detect it.", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(f"{out}/Q3_step2b_detection_grid.png")
    plt.close(fig)


def step3(bc, out):
    lsn = list(dict.fromkeys(v["limit_state"] for v in bc.values()))
    scn = list(dict.fromkeys(v["scenario"] for v in bc.values()))
    lab = [textwrap.fill(s, 14) for s in scn]
    sc_col = ["#7a8590", "#1f5f8b", "#3a8a5a", "#b3261e", "#8a5fbf"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(scn))
    uls = [bc[f"ULS | {s}"] for s in scn]
    ax[0].bar(x - 0.2, [v["oma_change"]["df_pct"][0] for v in uls], 0.4, color="#1f5f8b", label="M1 frequency change (%)")
    ax[0].bar(x + 0.2, [100 * (1 - min(v["oma_change"]["mac"])) for v in uls], 0.4, color="#d07a1e",
              label="100 × (1 − lowest MAC)")
    ax[0].set_xticks(x, lab, fontsize=7.5)
    ax[0].set_title("What OMA shows at ULS")
    ax[0].legend()
    for s_i, s in enumerate(scn):
        for l in lsn:
            v = bc[f"{l} | {s}"]
            for end, mk in (("N", "o"), ("S", "s")):
                ax[1].scatter(v[f"fix_{end}"] + 0.012 * (s_i - 2), v["identified"][f"fix_{end}"], marker=mk,
                              color=sc_col[s_i], s=22, zorder=3)
    ax[1].plot([0, 1], [0, 1], "k-", lw=0.8)
    ax[1].set_xlim(-0.05, 1)
    ax[1].set_ylim(-0.05, 1)
    ax[1].set_xlabel("true end fixity")
    ax[1].set_ylabel("fixity found by calibration")
    ax[1].set_title("Does calibration find the supports?\n(circles = north end, squares = south end)")
    w = 0.8 / len(lsn)
    for l_i, l in enumerate(lsn):
        ax[2].bar(x + (l_i - (len(lsn) - 1) / 2) * w, [bc[f"{l} | {s}"]["identified"]["Eg_change_pct"] for s in scn],
                  w, color=LS_COL[l], label=l)
    ax[2].set_xticks(x, lab, fontsize=7.5)
    ax[2].set_ylabel("%")
    ax[2].set_title("Girder E change reported by calibration (true: 0)")
    ax[2].legend()
    legend_below(fig, [Line2D([], [], marker="o", ls="", color=sc_col[i], label=s) for i, s in enumerate(scn)], ncol=5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{out}/Q3_step3_boundary_conditions.png")
    plt.close(fig)


def step3b(rr, out):
    names = list(rr["cases"])
    lam = np.array(rr["lams"])
    fig, ax = plt.subplots(2, len(names), figsize=(14, 6.4), sharex=True, sharey="row")
    for c, n in enumerate(names):
        for rule, lsty in (("without rule", "--"), ("with rule", "-")):
            r = rr["cases"][n][rule]
            for k in range(3):
                ax[0, c].plot(lam, [x[k] for x in r["df_pct"]], lsty, color=MODE_COL[k], lw=1.6)
                ax[1, c].semilogy(lam, [max(1 - x[k], 1e-6) for x in r["mac"]], lsty, color=MODE_COL[k], lw=1.6)
        for a_ in ax[:, c]:
            a_.axvline(rr["lam_threshold"], color="#8a2be2", lw=1.4)
            ls_lines(a_, rr["limit_states"], label=False)
        ax[0, c].set_title(n)
        ax[1, c].set_xlabel(LAM)
    ax[0, 0].set_ylabel("frequency change (%)")
    ax[1, 0].set_ylabel("1 − MAC (9 accelerometers)")
    legend_below(fig, mode_handles(["M1", "M2", "M3"]) + style_handles(
        [(f"with rule (k2 = {rr['k2_ratio']} k1 above threshold)", "-"), ("without rule", "--")])
        + [Line2D([], [], color="#8a2be2", lw=1.4, label="threshold: first girder crack")]
        + ls_handles(rr["limit_states"]), ncol=5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{out}/Q3_step3b_rotation_rule.png")
    plt.close(fig)


def step3c(rr, out):
    rg = rr["range"]
    lsn = list(rr["limit_states"])
    k1n = list(rg["rows"])
    fig, ax = plt.subplots(2, len(k1n), figsize=(14, 6.4), sharey="row")
    x = np.arange(len(lsn))
    for c, k1 in enumerate(k1n):
        rows = rg["rows"][k1]
        for k in range(3):
            lo = [min(r["df"][k] for r in rows if r["limit_state"] == l) for l in lsn]
            hi = [max(r["df"][k] for r in rows if r["limit_state"] == l) for l in lsn]
            nr = [next(r["no_rule"]["df"][k] for r in rows if r["limit_state"] == l) for l in lsn]
            off = (k - 1) * 0.22
            ax[0, c].vlines(x + off, lo, hi, color=MODE_COL[k], lw=7, alpha=0.55)
            ax[0, c].plot(x + off, nr, "k_", ms=14, mew=2)
            m_lo = [max(1 - min(r["mac"][k] for r in rows if r["limit_state"] == l), 1e-6) for l in lsn]
            ax[1, c].semilogy(x + off, m_lo, "o", color=MODE_COL[k], ms=6)
        ax[0, c].set_title(f"starting supports: {k1}")
        for a_ in ax[:, c]:
            a_.set_xticks(x, lsn)
    ax[0, 0].set_ylabel("frequency change (%)")
    ax[1, 0].set_ylabel("largest 1 − MAC (9 accelerometers)")
    legend_below(fig, mode_handles(["M1", "M2", "M3"]) +
                 [Line2D([], [], color="k", marker="_", ls="", ms=14, mew=2, label="cracking only (no rule)"),
                  Line2D([], [], color="0.5", lw=7, alpha=0.55, label="range over all 36 rules")], ncol=5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{out}/Q3_step3c_rule_range.png")
    plt.close(fig)


def step4a_mass_vs_damage(d, out, n_modes=3):
    st = d["stage2"]
    names = st["names"][:n_modes]
    fig, ax = plt.subplots(1, n_modes, figsize=(14, 4), sharey=True)
    for k in range(n_modes):
        for c, lsty in (("jacked", "-"), ("mass only", ":"), ("under load", "--")):
            rr = tracked(st["rows"], c, k)
            ax[k].plot([r["lam"] for r in rr], [r["df_pct"][k] for r in rr], lsty, color=MODE_COL[k], lw=1.8)
        ax[k].set_title(f"{names[k]} ({st['f_ref'][k]:.2f} Hz)")
        ls_lines(ax[k], d["stage1"]["limit_states"])
        ax[k].set_xlabel(LAM)
    ax[0].set_ylabel("frequency change (%)")
    legend_below(fig, style_handles([("damage only (no load mass)", "-"), ("load mass only (no damage)", ":"),
                                     ("measured with the load on (mass + damage)", "--")])
                 + ls_handles(d["stage1"]["limit_states"]), ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{out}/Q3_step4a_mass_vs_damage.png")
    plt.close(fig)


def step4b_load_removal(lr, out):
    rows = lr["rows"]
    lam = [r["lam"] for r in rows]
    from limit_states import DETECT_PCT
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k in range(3):
        ax[0].plot(lam, [r["df_loaded"][k] for r in rows], ":", color=MODE_COL[k], lw=1.6)
        ax[0].plot(lam, [r["df_target"][k] for r in rows], "-", color=MODE_COL[k], lw=4, alpha=0.3)
        ax[0].plot(lam, [r["df_F2"][k] for r in rows], "--", color=MODE_COL[k], lw=1.6)
        ax[1].plot(lam, [abs(r["err_F1"][k]) for r in rows], "-.", color=MODE_COL[k], lw=1.6)
        ax[1].plot(lam, [abs(r["err_F2"][k]) for r in rows], "--", color=MODE_COL[k], lw=1.6)
        ax[2].semilogy(lam, [max(1 - r["mac_loaded"][k], 1e-6) for r in rows], ":", color=MODE_COL[k], lw=1.6)
        ax[2].semilogy(lam, [max(1 - r["mac_S1"][k], 1e-6) for r in rows], "-.", color=MODE_COL[k], lw=1.6)
        ax[2].semilogy(lam, [max(1 - r["mac_S2"][k], 1e-6) for r in rows], "--", color=MODE_COL[k], lw=1.6)
    ax[1].axhline(DETECT_PCT, color="k", lw=0.8, ls=":")
    ax[0].set_ylabel("frequency change (%)")
    ax[0].set_title("Frequencies: measured, true and corrected")
    ax[1].set_ylabel("|error after correction| (%)")
    ax[1].set_title(f"Frequency correction error (dotted = {DETECT_PCT} %)")
    ax[2].set_ylabel("1 − MAC to the true shape")
    ax[2].set_title("Mode shapes at the 9 accelerometers")
    for a_ in ax:
        ls_lines(a_, lr["limit_states"])
        a_.set_xlabel(LAM)
    legend_below(fig, mode_handles(["M1", "M2", "M3"]) + style_handles(
        [("measured with the load on / no correction", ":"), ("Rayleigh (freq.) or perturbation (shape)", "-.")
         , ("model ratio (freq.) or model subtraction (shape)", "--")])
        + [Line2D([], [], color="0.4", lw=4, alpha=0.3, label="true (no load mass)")]
        + ls_handles(lr["limit_states"]), ncol=4)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(f"{out}/Q3_step4b_load_removal.png")
    plt.close(fig)


def make_all(d, bc, rr, lr, out):
    step1(d, out)
    step2(d, out)
    step2b_detection_grid(d, out)
    step3(bc, out)
    if rr:
        step3b(rr, out)
        if "range" in rr:
            step3c(rr, out)
    step4a_mass_vs_damage(d, out)
    if lr:
        step4b_load_removal(lr, out)
