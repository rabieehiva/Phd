"""Q3 -- Can OMA (done at service level) say anything about the bridge at ULS?

One driver for the whole question, organised as the supervisor's chain:

  Step 1  Does Walker Road physically change at ULS?  (girder cracking from the as-built
          prestress + CSA S6-19; bearings: uplift, slip, locking checked from the drawings)
  Step 2  Do OMA frequencies and mode shapes (9 accelerometers) show those changes?
  Step 3  Can OMA tell what the supports are doing?  (model calibration on simulated OMA data,
          support fixity / pad stiffness scenarios at each limit state)
  Step 3b Bilinear rotational stiffness at the girder ends (k1 up to a physical threshold,
          k2 = 0.75 k1 above), the supervisor's rule.
  Step 3c Range of such rules (k2/k1, threshold, which end).
  Step 4  Removing the effect of the test load (mass, mass-normalised shapes).

Limit states: SLS (1 CL-625 truck per lane), 75 % ULS, ULS (1.70 x 1.25 x 0.90 = 1.91 trucks
per lane), beyond ULS (3 per lane). Changes are stiffness-only (test-load mass removed); the mass
effect is step 4.

Run:   python q3_run.py            (about 5 min; needs results/calibration.json)
       python q3_run.py --plots    (figures + summary from existing result files only)
Out:   results/Q3/  -- figures (drawn by q3_figures.py) and Q3_summary.md:
         Q3_step1_physical_changes, Q3_step2_oma_response, Q3_step2b_detection_grid,
         Q3_step3_boundary_conditions, Q3_step3b_rotation_rule, Q3_step3c_rule_range,
         Q3_step4a_mass_vs_damage, Q3_step4b_load_removal
"""
from __future__ import annotations

import json
import os
import textwrap
import sys

import numpy as np

import limit_states as ls
import q3_figures

OUT = "results/Q3"
MODES = 3                                   # M1, M2, M3 shown


def step3c_text(rr):
    rg = rr["range"]
    lsn = list(rr["limit_states"])
    L = ["## Step 3c – Range of support rules (sensitivity)", "",
         "All combinations of k2/k1 = " + ", ".join(str(a) for a in rg["alphas"])
         + ", threshold at SLS / 75 % ULS / ULS, and softening at both ends, the north end or the south end "
         "(" + str(len(rg["alphas"]) * 3 * 3) + " rules per k1), for three starting supports.", "",
         "| k1 (starting supports) | " + " | ".join(f"M1 Δf at {l}" for l in lsn) + " | lowest MAC (M1–M3), all |",
         "|---|" + "---|" * (len(lsn) + 1)]
    for k1, rows in rg["rows"].items():
        cells = []
        for l in lsn:
            v = [r["df"][0] for r in rows if r["limit_state"] == l]
            cells.append(f"{min(v):+.1f} … {max(v):+.1f} %")
        L.append(f"| {k1} | " + " | ".join(cells) + f" | {min(min(r['mac'][:3]) for r in rows):.3f} |")
    L += ["", "**Answer:** with the supports as built (pad only), no rule makes a difference – the ends are "
          "already practically pinned. If the ends carry some fixity (a quarter to a half), a softening rule "
          "can lower M1 by up to 7–9 % already at SLS/ULS (depending on α and which end), i.e. far more than "
          "girder cracking. The mode shapes at the 9 accelerometers change much less (lowest MAC 0.993 for "
          "quarter-fixed, 0.998 for half-fixed ends, up to ULS), which is only just beyond the measured MAC "
          "scatter of M1 (0.998). Support softening is therefore mainly a frequency signal, with a small "
          "mode-shape signal when one end softens more than the other.", ""]
    return L


def step4_text(lr):
    rows = lr["rows"]
    L = ["## Step 4 – Removing the effect of the test load", "",
         "OMA measured with the test load on the bridge sees the damage AND the added mass. Corrections use only "
         "the known load (mass, position) and the calibrated undamaged model.", "",
         "| Limit state | M1 true (no mass) | M1 measured with load | error after Rayleigh | error after model ratio "
         "| MAC M1–M3 no correction | MAC perturbation | MAC model subtraction |", "|---|---|---|---|---|---|---|---|"]
    for n, lam in lr["limit_states"].items():
        r = min(rows, key=lambda r: abs(r["lam"] - lam))
        L.append(f"| {n} | {r['df_target'][0]:+.2f} % | {r['df_loaded'][0]:+.1f} % | {r['err_F1'][0]:+.2f} % | "
                 f"{r['err_F2'][0]:+.2f} % | {min(r['mac_loaded'][:3]):.3f} | {min(r['mac_S1'][:3]):.3f} | "
                 f"{min(r['mac_S2'][:3]):.3f} |")
    L += ["", "**Answer:** the test-load mass hides the damage (M1 about −12 % per truck pair, against −0.05 % "
          "from cracking at ULS). Dividing out the mass effect predicted by the calibrated model recovers the true "
          "frequencies within 0.5 % and the true mode shapes (MAC ≥ 0.98) up to beyond ULS. The simpler Rayleigh "
          "correction with mass-normalised shapes works for light loads (error < 0.5 % up to about one truck "
          "per lane) but grows to 1–2.5 % for heavy loads, because the heavy mass also changes the shapes.", ""]
    return L


def summary(d, bc, rr=None, lr=None):
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
          ] + (step3b_text(rr) if rr else []) + (step3c_text(rr) if rr and "range" in rr else []) + [
          ] + (step4_text(lr) if lr else []) + [
          "Figures: `Q3_step1_physical_changes.png`, `Q3_step2_oma_response.png`, "
          "`Q3_step2b_detection_grid.png`, `Q3_step3_boundary_conditions.png`, `Q3_step3b_rotation_rule.png`, `Q3_step3c_rule_range.png`, `Q3_step4a_mass_vs_damage.png`, `Q3_step4b_load_removal.png`."]
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
        import q3_load_removal
        import q3_rotation_rule
        ls.main()
        bc_identification.main()
        q3_rotation_rule.main()
        q3_load_removal.main()
    d = json.load(open("results/limit_states.json"))
    bc = json.load(open("results/bc_identification.json"))
    rr = json.load(open(f"{OUT}/q3_rotation_rule.json")) if os.path.exists(f"{OUT}/q3_rotation_rule.json") else None
    lr = json.load(open(f"{OUT}/q3_load_removal.json")) if os.path.exists(f"{OUT}/q3_load_removal.json") else None
    q3_figures.make_all(d, bc, rr, lr, OUT)
    summary(d, bc, rr, lr)
    print(open(f"{OUT}/Q3_summary.md").read())


if __name__ == "__main__":
    main()
