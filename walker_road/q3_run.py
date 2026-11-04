"""Q3 -- Can OMA (done at service level) say anything about the bridge at ULS?

One driver for the whole question, organised as the supervisor's chain:

  Step 1  Does Walker Road physically change at ULS?  (girder cracking from the as-built
          prestress + CSA S6-19; bearings: uplift, slip, locking checked from the drawings)
  Step 2  Do OMA frequencies and mode shapes (9 accelerometers) show those changes?
  Step 3  Can OMA tell what the supports are doing?  (model calibration on simulated OMA data,
          support fixity / pad stiffness scenarios at each limit state)
  Step 3b Bilinear rotational stiffness at the girder ends (k1 up to a physical threshold,
          k2 = 0.75 k1 above), the supervisor's rule.
  Step 4  (later) Removing the effect of the test load (mass, mass-normalised shapes).

Limit states: SLS (1 CL-625 truck per lane), 75 % ULS, ULS (1.70 x 1.25 x 0.90 = 1.91 trucks
per lane), beyond ULS (3 per lane). Changes are stiffness-only (test-load mass removed); the mass
effect is step 4.

Run:   python q3_run.py            (about 5 min; needs results/calibration.json)
       python q3_run.py --plots    (figures + summary from existing result files only)
Out:   results/Q3/Q3_step1_physical_changes.png
       results/Q3/Q3_step2_oma_response.png
       results/Q3/Q3_step3_boundary_conditions.png
       results/Q3/Q3_step3b_rotation_rule.png  (bilinear rotational stiffness k1 -> 0.75 k1)
       results/Q3/Q3_summary.md
"""
from __future__ import annotations

import json
import os
import textwrap
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import limit_states as ls
import measured as ms

OUT = "results/Q3"
LS_COL = {"SLS": "0.55", "75% ULS": "C1", "ULS": "C3", "beyond ULS": "k"}
MODES = 3                                   # M1, M2, M3 shown


def ls_lines(ax, lsd, labels=False):
    for n, lam in lsd.items():
        ax.axvline(lam, color=LS_COL[n], lw=1, ls="--")
        if labels:
            ax.text(lam, 1.0, " " + n, transform=ax.get_xaxis_transform(), rotation=90, va="top",
                    ha="right", fontsize=7, color=LS_COL[n])


def step1(d):
    s1 = d["stage1"]
    it = s1["iteration"]
    lam = np.array(sorted(float(k) for k in it))
    g = lambda key, f=lambda v: v: np.array([f(it[k][key]) for k in sorted(it, key=float)])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    a = ax[0]
    a.plot(lam, g("cracked_el"), "o-", ms=3, color="C3", label="cracked girder elements (of 210)")
    a.set_ylabel("cracked girder elements")
    a2 = a.twinx()
    a2.plot(lam, 100 * g("min_Ie_ratio"), "s--", ms=3, color="C0", label="lowest I_e / I_g (%)")
    a2.set_ylabel("lowest I_e / I_g (%)", color="C0")
    a2.set_ylim(0, 105)
    a.set_title("Girders: cracking (as-built prestress, CSA S6-19)", fontsize=9)
    a.legend(loc="center right", fontsize=7)
    a2.legend(loc="lower right", fontsize=7)
    b = ax[1]
    b.plot(lam, g("bearing_contact", min), "o-", ms=3, label="pad contact ratio (1 = no uplift)")
    b.plot(lam, g("slip_ratio", max), "s-", ms=3, label="H / (μR)  (< 1 = no slip)")
    b.axhline(1, color="k", lw=0.8)
    b.set_ylim(0, 1.15)
    b.set_title("Bearings: uplift and slip checks (14 pads)", fontsize=9)
    b.legend(fontsize=7)
    c = ax[2]
    c.plot(lam, g("soffit_outward_mm"), "o-", ms=3, label="girder soffit moves towards ballast wall")
    c.plot(lam, g("joint_opening_mm"), "s-", ms=3, label="deck joint OPENS by")
    c.axhline(it[next(iter(it))]["soffit_clearance_mm"], color="C0", ls=":", label="soffit clearance (≈225 mm)")
    c.axhline(it[next(iter(it))]["joint_gap_mm"], color="C1", ls=":", label="joint gap (38 mm)")
    c.set_yscale("log")
    c.set_ylabel("mm")
    c.set_title("Bearings: can the girder ends lock?", fontsize=9)
    c.legend(fontsize=7)
    for a_ in ax:
        ls_lines(a_, s1["limit_states"], labels=a_ is ax[0])
        a_.set_xlabel("λ = CL-625 trucks per lane")
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/Q3_step1_physical_changes.png", dpi=150)


def step2(d):
    st = d["stage2"]
    names = st["names"][:MODES]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, n in enumerate(names):
        for c, lsty in (("jacked", "-"), ("after unload", "--")):
            rr = [r for r in st["rows"] if r["case"] == c and r["tracked"][k]]
            lam = [r["lam"] for r in rr]
            lab = {"jacked": "load on", "after unload": "after unloading"}[c]
            ax[0].plot(lam, [r["df_pct"][k] for r in rr], lsty, color=f"C{k}", label=f"{n}, {lab}")
            ax[1].semilogy(lam, [max(1 - r["mac_acc"][k], 1e-6) for r in rr], lsty, color=f"C{k}",
                           label=f"{n}, {lab}")
        if n in st["mac_stab_measured"]:
            ax[1].axhline(1 - st["mac_stab_measured"][n], color=f"C{k}", lw=0.6, ls=":")
    ax[0].axhspan(-ls.DETECT_PCT, ls.DETECT_PCT, color="0.85", zorder=0)
    ax[0].set_ylabel("frequency change (%)")
    ax[0].set_title(f"Frequencies (grey band: ±{ls.DETECT_PCT} % detection limit)", fontsize=9)
    ax[1].set_ylabel("1 - MAC at the 9 accelerometers")
    ax[1].set_ylim(1e-6, 0.2)
    ax[1].set_title("Mode shapes (dotted: measured MAC scatter of each mode)", fontsize=9)
    for a_ in ax:
        ls_lines(a_, d["stage1"]["limit_states"], labels=a_ is ax[0])
        a_.set_xlabel("λ = CL-625 trucks per lane")
        a_.grid(alpha=0.3, which="both")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/Q3_step2_oma_response.png", dpi=150)


def step3(bc):
    lsn = list(dict.fromkeys(v["limit_state"] for v in bc.values()))
    scn = list(dict.fromkeys(v["scenario"] for v in bc.values()))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(scn))
    uls = [bc[f"ULS | {s}"] for s in scn]
    ax[0].bar(x - 0.2, [v["oma_change"]["df_pct"][0] for v in uls], 0.4, label="M1 frequency change (%)")
    ax[0].bar(x + 0.2, [100 * (1 - min(v["oma_change"]["mac"])) for v in uls], 0.4,
              label="100 x (1 - lowest MAC), 9 accelerometers")
    ax[0].set_xticks(x, [textwrap.fill(s, 12) for s in scn], fontsize=7)
    ax[0].set_title("What OMA shows at ULS for each support scenario", fontsize=9)
    ax[0].legend(fontsize=7)
    for s_i, s in enumerate(scn):
        for l in lsn:
            v = bc[f"{l} | {s}"]
            for end, mk in (("N", "o"), ("S", "s")):
                ax[1].scatter(v[f"fix_{end}"] + 0.012 * (s_i - 2), v["identified"][f"fix_{end}"], marker=mk,
                              color=f"C{s_i}", s=18, label=s if (l == lsn[0] and end == "N") else None)
    ax[1].plot([0, 1], [0, 1], "k-", lw=0.8)
    ax[1].set_xlabel("true end fixity (o north, s south)")
    ax[1].set_ylabel("fixity found by calibration")
    ax[1].set_title("Does calibration recover the supports? (all limit states)", fontsize=9)
    ax[1].legend(fontsize=6)
    w = 0.8 / len(lsn)
    for l_i, l in enumerate(lsn):
        ax[2].bar(x + (l_i - (len(lsn) - 1) / 2) * w, [bc[f"{l} | {s}"]["identified"]["Eg_change_pct"] for s in scn],
                  w, color=LS_COL[l], label=l)
    ax[2].set_xticks(x, [textwrap.fill(s, 12) for s in scn], fontsize=7)
    ax[2].set_ylabel("%")
    ax[2].set_title("Girder modulus change the calibration reports (true: 0)", fontsize=9)
    ax[2].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/Q3_step3_boundary_conditions.png", dpi=150)


def step3b(rr):
    names = list(rr["cases"])
    lam = np.array(rr["lams"])
    fig, ax = plt.subplots(2, len(names), figsize=(14, 6), sharex=True, sharey="row")
    for c, n in enumerate(names):
        for rule, lsty in (("without rule", "--"), ("with rule", "-")):
            r = rr["cases"][n][rule]
            for k, col in zip(range(3), ("C0", "C1", "C2")):
                ax[0, c].plot(lam, [x[k] for x in r["df_pct"]], lsty, color=col,
                              label=f"M{k + 1 if k < 2 else 3}, {rule}" if c == 0 else None)
                ax[1, c].semilogy(lam, [max(1 - x[k], 1e-6) for x in r["mac"]], lsty, color=col)
        for a_ in ax[:, c]:
            a_.axvline(rr["lam_threshold"], color="m", lw=1.2, label="threshold (first girder cracking)"
                       if (c == 0 and a_ is ax[0, 0]) else None)
            ls_lines(a_, rr["limit_states"])
            a_.grid(alpha=0.3, which="both")
        ax[0, c].axhspan(-ls.DETECT_PCT, ls.DETECT_PCT, color="0.85", zorder=0)
        ax[0, c].set_title(n, fontsize=9)
        ax[1, c].set_xlabel("λ = CL-625 trucks per lane")
    ax[0, 0].set_ylabel("frequency change (%)")
    ax[1, 0].set_ylabel("1 - MAC (9 accelerometers)")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle(f"Bilinear rotational stiffness at the girder ends: k1 up to the threshold, "
                 f"k2 = {rr['k2_ratio']} k1 above it", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/Q3_step3b_rotation_rule.png", dpi=150)


def summary(d, bc, rr=None):
    s1, st = d["stage1"], d["stage2"]
    it = s1["iteration"]
    lsd = s1["limit_states"]
    names = st["names"][:MODES]

    def at(lam):
        return it[min(it, key=lambda k: abs(float(k) - lam))]

    def row(case, lam):
        return min((r for r in st["rows"] if r["case"] == case), key=lambda r: abs(r["lam"] - lam))

    L = ["# Q3 – Can OMA at service level say anything about Walker Road at ULS?", "",
         "Generated by `q3_run.py`. Limit states in λ = CL-625 trucks per lane (both lanes): "
         + ", ".join(f"{k} = {v:.2f}" for k, v in lsd.items())
         + ". ULS = 1.70 × 1.25 × 0.90 (S6-19 load factor, dynamic allowance, two-lane factor).",
         f"Prestress losses (S6-19 8.7.4): {100 * d['assumptions']['LOSS']:.1f} %. Bearings: linear springs; "
         "uplift, slip and locking are checked, not assumed. Changes are stiffness-only (test-load mass "
         "removed, see step 4).", "",
         "## Step 1 – Does the bridge physically change at ULS?", "",
         "| Limit state | λ | Cracked girder elements | Lowest I_e/I_g | Pad contact (1 = no uplift) "
         "| Max H/(μR) (< 1 = no slip) | Soffit movement / clearance (mm) | Joint |",
         "|---|---|---|---|---|---|---|---|"]
    for n, lam in lsd.items():
        a = at(lam)
        L.append(f"| {n} | {lam:.2f} | {a['cracked_el']} | {100 * a['min_Ie_ratio']:.0f} % | "
                 f"{min(a['bearing_contact']):.2f} | {max(a['slip_ratio']):.2f} | "
                 f"{a['soffit_outward_mm']:.1f} / {a['soffit_clearance_mm']:.0f} | opens {a['joint_opening_mm']:.1f} mm |")
    L += ["", f"First girder decompression at λ = {min(s1['lam_decomp']):.2f}, first cracking at "
          f"λ = {min(s1['lam_crack']):.2f} (interior girders G4/G5).",
          "**Answer:** yes, but only the girders change: they start to crack right at ULS. The bearings do "
          "not lift off, slip or lock up to the flexural resistance, so the supports stay as they are.", "",
          "## Step 2 – Do OMA frequencies and mode shapes show it?", "",
          "| Limit state | " + " | ".join(f"Δf {n} load on" for n in names) + " | "
          + " | ".join(f"MAC {n} load on" for n in names) + " | Δf M1 after unloading | MAC M1 after unloading |",
          "|---|" + "---|" * (2 * MODES + 2)]
    for n, lam in lsd.items():
        j, u = row("jacked", lam), row("after unload", lam)
        L.append(f"| {n} | " + " | ".join(f"{j['df_pct'][k]:+.2f} %" for k in range(MODES)) + " | "
                 + " | ".join(f"{j['mac_acc'][k]:.3f}" for k in range(MODES))
                 + f" | {u['df_pct'][0]:+.2f} % | {u['mac_acc'][0]:.3f} |")
    stab = ", ".join(f"{n} {v:.3f}" for n, v in st["mac_stab_measured"].items() if n in names)
    L += ["", f"Detection limits: |Δf| ≥ {ls.DETECT_PCT} %; MAC below the measured scatter ({stab}).",
          "**Answer:** up to ULS, only M3 (first transverse-bending mode) changes detectably, by 1–2 %: that "
          "is local deck cracking under the wheels, which in reality has probably already happened under "
          "normal traffic. Girder cracking at ULS is too small to see (M1, M2 and all mode shapes stay "
          "within the detection limits). Beyond ULS, M1 drops about 5 % with the load on and the shapes "
          "change beyond the measured scatter; after unloading the prestress closes the cracks and the "
          "change is mostly gone.", "",
          "## Step 3 – Can OMA tell what the supports are doing?", "",
          "Simulated OMA (9 accelerometers, noise) of the bridge at each limit state with a support scenario; "
          "calibration of Eg, end fixity north/south and pad stiffness north/south.", "",
          "| Scenario at ULS | M1 Δf | Lowest MAC | True fixity N/S | Found fixity N/S | True kv N/S | Found kv N/S | Found Eg change |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in bc.items():
        if v["limit_state"] != "ULS":
            continue
        i = v["identified"]
        L.append(f"| {v['scenario']} | {v['oma_change']['df_pct'][0]:+.1f} % | {min(v['oma_change']['mac']):.3f} | "
                 f"{v['fix_N']:.2f}/{v['fix_S']:.2f} | {i['fix_N']:.2f}/{i['fix_S']:.2f} | "
                 f"{v['kv_N']:.1f}/{v['kv_S']:.1f} | {i['kv_N']:.2f}/{i['kv_S']:.2f} | {i['Eg_change_pct']:+.1f} % |")
    L += ["", "**Answer:** a change in support fixity is very visible to OMA (frequencies +20 to +40 %, MAC "
          "down to about 0.88), much more than girder cracking. Calibration finds which end changed, but not "
          "how much: it over-estimates the fixity and compensates with a girder-modulus drop that is not real. "
          "Pad vertical stiffness is hard to identify, and cracking beyond ULS partly ends up in the pad "
          "stiffness. Support and girder stiffness are therefore not cleanly separable with 9 accelerometers.", "",
          ] + (step3b_text(rr) if rr else []) + [
          "## Step 4 – Removing the load effect (next)", "",
          "Not done yet. For reference, the test-load mass alone lowers M1 by about "
          f"{-row('mass only', 1.0)['df_pct'][0]:.0f} % per truck pair (λ = 1), far more than the cracking at ULS.",
          "", "Figures: `Q3_step1_physical_changes.png`, `Q3_step2_oma_response.png`, "
          "`Q3_step3_boundary_conditions.png`, `Q3_step3b_rotation_rule.png`."]
    open(f"{OUT}/Q3_summary.md", "w").write("\n".join(L) + "\n")


def step3b_text(rr):
    lam = np.array(rr["lams"])
    j = int(np.argmax(lam > rr["lam_threshold"]))
    L = ["## Step 3b – Bilinear rotational stiffness at the supports (k1 → k2 = "
         f"{rr['k2_ratio']} k1)", "",
         f"Threshold = first girder cracking, λ = {rr['lam_threshold']:.2f}. The physical alternative, the pad's "
         "back edge unloading under end rotation, is "
         + ("never reached at Walker Road." if rr["lam_pad_edge_unloads"] is None
            else f"reached at λ = {rr['lam_pad_edge_unloads']:.2f}."), "",
         "| Supports (k1) | M1 Δf just above threshold, without rule | with rule | extra from the rule | lowest MAC with rule |",
         "|---|---|---|---|---|"]
    for n, r in rr["cases"].items():
        a, b = r["without rule"]["df_pct"][j][0], r["with rule"]["df_pct"][j][0]
        L.append(f"| {n} | {a:+.2f} % | {b:+.2f} % | {b - a:+.2f} % | {min(r['with rule']['mac'][j][:3]):.3f} |")
    L += ["", "**Answer:** with the ends as they are (practically pinned) the rule changes nothing (k1 is already "
          "tiny). If the ends were partly fixed, the 25 % drop in rotational stiffness lowers M1 by about 2–4 % "
          "in one step at the threshold – a detectable jump, larger than the girder cracking at ULS.", ""]
    return L


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--plots" not in sys.argv:
        import bc_identification
        import q3_rotation_rule
        ls.main()
        bc_identification.main()
        q3_rotation_rule.main()
    d = json.load(open("results/limit_states.json"))
    bc = json.load(open("results/bc_identification.json"))
    step1(d)
    step2(d)
    step3(bc)
    rr_path = f"{OUT}/q3_rotation_rule.json"
    rr = json.load(open(rr_path)) if os.path.exists(rr_path) else None
    if rr:
        step3b(rr)
    summary(d, bc, rr)
    print(open(f"{OUT}/Q3_summary.md").read())


if __name__ == "__main__":
    main()
