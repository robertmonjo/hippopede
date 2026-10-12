"""Average over the light cone with equal weights and with a light weight of the sectors (Table light_weight).

For each weight of the sectors (sector_geometry.light_weight: equal, cos, atan, cos2, atan2, quad) the model of reading (iv)
(fit_observer_ball.py, fit_ball_unbinned.py) is fitted on a grid of (alpha_high, t0, theta_obs) and refined with
Nelder-Mead.  The projected axis points along the CatWISE excess where the count dipole of the model is positive and
opposite to it where it is negative (quasar_dipole_fit.dipole, x = 1.7, p(z) of CatWISE).  The rows are
  axis:          observer on the axis (theta_obs = 0), alpha_high and t0 fitted to the chronometers and supernovae
  cc_sn:         minimum of the chronometers and supernovae over (alpha_high, t0, theta_obs)
  joint_pos/neg: minimum of chronometers, supernovae and CatWISE amplitude with a positive / negative model dipole
Each row gives Delta chi2 of CC + SN relative to flat LCDM, Delta chi2 of CC + SN + CatWISE relative to flat LCDM
with the kinematic dipole only, the dipole D(z) at fixed redshift (x = 1.7, Gaussian p(z) of width 0.08), the dipole
of a radio sample with the count slope of NVSS above 15 mJy, x = 0.74 (Wagenveld et al. 2025, A&A 697, A112, Sect. 3.1),
and dN/dz ~ z^0.74 exp[-(z/0.71)^1.06] (Tiwari & Nusser 2016, JCAP 03, 062, Eq. 3.11), and the amplitudes predicted
for the three Quaia slices (quaia_zslice_model.py) with chi2 = sum (|D_i| - e_i)^2 / sigma_i^2, relative to no dipole.

Usage:
  HIPPOPEDE_THETA_MAX_DEG=90 python scripts/light_weight_comparison.py --weight equal|cos|atan|cos2|atan2|quad
  python scripts/light_weight_comparison.py --table      (after the four runs; writes the LaTeX table)
Writes json/light_weight_comparison_<weight>.json, json/light_weight_comparison.json, tables/light_weight_comparison.tex.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
JSON = ROOT / "json"
WEIGHTS = ("equal", "cos", "atan", "cos2", "atan2", "quad")


def options():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", choices=WEIGHTS)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not a.table and a.weight is None:
        ap.error("--weight or --table")
    return a


ARGS = options() if __name__ == "__main__" else None
if ARGS is not None and ARGS.weight:
    os.environ["HIPPOPEDE_LIGHT_WEIGHT"] = "0" if ARGS.weight == "equal" else ARGS.weight
    assert os.environ.get("HIPPOPEDE_THETA_MAX_DEG") == "90", "run with HIPPOPEDE_THETA_MAX_DEG=90 (whole lobe)"

sys.path.insert(0, str(SCRIPT_DIR))
import fit_ball_unbinned as F  # noqa: E402
import fit_observer_ball as OB  # noqa: E402
import pantheon_offaxis_test as T  # noqa: E402
import quasar_dipole_fit as Q  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

A_GRID = np.round(np.arange(0.17, 0.4301, 0.02), 3)
T_GRID = np.round(np.geomspace(0.7, 6.0, 16), 4)
TH_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 13.0, 17.0, 21.0,
           25.0, 30.0, 35.0, 40.0, 45.0]
THETA_OBS_MAX = 45.0
Z_SLICE = (0.3, 0.6, 1.0, 1.4, 1.8)
X_RADIO = 0.74
RADIO_OBS = {"NVSS": (0.0139, 0.0029, 0.0041), "RACS-low": (0.0143, 0.0024, 0.0041), "CatWISE": (0.0151, 0.0016, 0.0073)}
PZ_RADIO = Q.Z_Q**0.74 * np.exp(-(Q.Z_Q / 0.71) ** 1.06)
PZ_RADIO = PZ_RADIO / np.trapezoid(PZ_RADIO, Q.Z_Q)
_, D_GEO, _ = Q.excess_vector()
SIG = Q.SIG_GEO
CHI2_CW_LCDM = (D_GEO / SIG) ** 2
N_C, _ = F.PCA.catwise_axis_icrs()
QS = json.loads((JSON / "quaia_zslice_model.json").read_text())["slices"]
Q_AMP = np.array([s["excess_amp"] for s in QS])
Q_SIG = np.array([s["sigma"] for s in QS])
CHI2_Q_NONE = float(np.sum((Q_AMP / Q_SIG) ** 2))
C_LCDM = None


def _init(c):
    global C_LCDM
    C_LCDM = c


def table_of(ah, t0):
    T.set_alpha_high(ah)
    return T.e_table(t0)


def d_catwise(tab, th):
    return Q.dipole(tab, th, Q.X_SLOPE, Q.pz("gamma"))[0]


def dchi2_cc_sn(ah, t0, th, d):
    """CC + SN chi2 relative to flat LCDM, axis along the excess for d > 0 and opposite for d < 0."""
    pt = F.Point(ah, t0)
    c = sum(pt.chi2(th, N_C if d > 0 else -N_C))
    OB._T_CACHE.clear(); OB._W_CACHE.clear(); OB._M_CACHE.clear()
    return c - C_LCDM


def grid_task(args):
    ah, t0 = args
    out = []
    try:
        tab = table_of(ah, t0)
        pt = F.Point(ah, t0)   # built once per (alpha_high, t0): its sector tables serve every theta_obs
    except Exception:
        return out
    for th in TH_GRID:
        d = d_catwise(tab, th) if th > 0 else 0.0
        if not np.isfinite(d):
            continue
        c = sum(pt.chi2(th, N_C if d > 0 else -N_C)) - C_LCDM
        OB._W_CACHE.clear(); OB._M_CACHE.clear()
        if np.isfinite(c):
            out.append((ah, t0, th, c, d))
    OB._T_CACHE.clear()
    return out


def objective(p, mode):
    """mode: axis (p = alpha, ln t0), cc_sn, joint_pos, joint_neg (p = alpha, ln t0, theta)."""
    ah, t0 = p[0], float(np.exp(p[1]))
    th = 0.0 if mode == "axis" else abs(p[2])
    if not (A_GRID[0] - 0.05 <= ah <= A_GRID[-1] + 0.05 and 0.5 <= t0 <= 8.0 and th <= THETA_OBS_MAX):
        return 1e6
    try:
        tab = table_of(ah, t0)
        d = d_catwise(tab, th) if th > 0 else 0.0
        if not np.isfinite(d) or (mode == "joint_pos" and d <= 0) or (mode == "joint_neg" and d >= 0):
            return 1e6
        c = dchi2_cc_sn(ah, t0, th, d)
    except Exception:
        return 1e6
    if not np.isfinite(c):
        return 1e6
    return c + (((abs(d) - D_GEO) / SIG) ** 2 if mode.startswith("joint") else 0.0)


def refine(start, mode):
    if mode == "axis":
        x0 = np.array([start[0], np.log(start[1])])
        simplex = [x0, x0 + [0.01, 0.0], x0 + [0.0, 0.05]]
    else:
        x0 = np.array([start[0], np.log(start[1]), max(start[2], 0.01)])
        simplex = [x0, x0 + [0.01, 0.0, 0.0], x0 + [0.0, 0.05, 0.0], x0 + [0.0, 0.0, 0.3 * x0[2] + 0.02]]
    r = minimize(objective, x0, args=(mode,), method="Nelder-Mead",
                 options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 2e-3, "maxfev": 200})
    th = 0.0 if mode == "axis" else abs(float(r.x[2]))
    return float(r.x[0]), float(np.exp(r.x[1])), th, float(r.fun)


def describe(ah, t0, th):
    tab = table_of(ah, t0)
    d = d_catwise(tab, th) if th > 0 else 0.0
    sign = 1.0 if d >= 0 else -1.0   # amplitudes along the axis orientation of the row
    c = dchi2_cc_sn(ah, t0, th, d)
    chi2_cw = ((abs(d) - D_GEO) / SIG) ** 2
    if th > 0:
        dz = []
        for zc in Z_SLICE:
            p = np.exp(-0.5 * ((Q.Z_Q - zc) / 0.08) ** 2)
            dz.append(sign * Q.dipole(tab, th, Q.X_SLOPE, p / np.trapezoid(p, Q.Z_Q))[0])
        radio = sign * Q.dipole(tab, th, X_RADIO, PZ_RADIO)[0]
        dq = []
        for s in QS:
            p = ((Q.Z_Q >= s["z"][0]) & (Q.Z_Q <= s["z"][1])).astype(float)
            dq.append(sign * Q.dipole(tab, th, s["x"], p / np.trapezoid(p, Q.Z_Q))[0])
    else:
        dz, radio, dq = [0.0] * len(Z_SLICE), 0.0, [0.0] * len(QS)
    chi2_q = float(np.sum(((np.abs(dq) - Q_AMP) / Q_SIG) ** 2))
    return {"alpha_high": ah, "t0": t0, "theta_obs": th, "dipole_sign": int(np.sign(d)),
            "dchi2_cc_sn": c, "D_catwise": abs(d), "chi2_catwise": chi2_cw, "dchi2_cc_sn_catwise": c + chi2_cw - CHI2_CW_LCDM,
            "D_radio": radio, "radio_over_catwise": radio / abs(d) if d else None,
            "D_of_z": dict(zip([str(z) for z in Z_SLICE], dz)), "D_quaia": dq, "chi2_quaia": chi2_q,
            "dchi2_quaia": chi2_q - CHI2_Q_NONE, "on_theta_edge": bool(th > THETA_OBS_MAX - 0.5),
            "on_alpha_t0_edge": bool(ah < A_GRID[0] + 0.005 or ah > A_GRID[-1] - 0.005
                                     or t0 < T_GRID[0] * 1.02 or t0 > T_GRID[-1] * 0.98)}


def refine_task(args):
    starts, mode = args
    best = min((refine(s, mode) for s in starts), key=lambda r: r[3])
    return mode, describe(*best[:3])


def run(weight, workers):
    _, c_l = F.lcdm_reference()
    _init(c_l)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(c_l,)) as ex:
        rows = np.array([r for part in ex.map(grid_task, [(a, t) for a in A_GRID for t in T_GRID]) for r in part])
        fbu = json.loads((JSON / "fit_ball_unbinned.json").read_text())   # minima of equal weights, extra starts
        f_cc = fbu["fits"]["cc_sn"]; f_j = fbu["fits"]["cc_sn_catwise"]; f_0 = fbu["isotropic"]
        tot = rows[:, 3] + ((np.abs(rows[:, 4]) - D_GEO) / SIG) ** 2
        on = rows[:, 2] == 0.0
        starts = {"axis": [tuple(rows[on][np.argmin(rows[on, 3]), :2]), (f_0["alpha_high"], f_0["t0"])],
                  "cc_sn": [tuple(rows[np.argmin(rows[:, 3]), :3]), (f_cc["alpha_high"], f_cc["t0"], f_cc["theta_obs"])]}
        for mode, m in (("joint_pos", rows[:, 4] > 0), ("joint_neg", rows[:, 4] < 0)):
            starts[mode] = [tuple(rows[np.argmin(np.where(m, tot, np.inf)), :3])]
        starts["joint_neg"].append((f_j["alpha_high"], f_j["t0"], f_j["theta_obs"]))
        res = dict(ex.map(refine_task, [(s, m) for m, s in starts.items()]))
    if res["axis"]["dchi2_cc_sn"] < res["cc_sn"]["dchi2_cc_sn"]:
        res["cc_sn"] = dict(res["axis"])   # the minimum of CC + SN lies on the axis
    out = {"weight": weight, "D_geo": D_GEO, "sigma_geo": SIG, "chi2_catwise_lcdm": CHI2_CW_LCDM, "chi2_lcdm_cc_sn": c_l,
           "chi2_quaia_none": CHI2_Q_NONE, "x_radio": X_RADIO, "z_of_D": list(Z_SLICE),
           "grid": {"alpha_high": A_GRID.tolist(), "t0": T_GRID.tolist(), "theta_obs": TH_GRID},
           "rows": res, "grid_points": rows.tolist()}
    (JSON / f"light_weight_comparison_{weight}.json").write_text(json.dumps(out, indent=1))
    for k, r in res.items():
        print(f"{weight:5s} {k:9s} a={r['alpha_high']:.3f} t0={r['t0']:.3f} theta={r['theta_obs']:.2f} "
              f"dCC+SN={r['dchi2_cc_sn']:+.2f} dCC+SN+CW={r['dchi2_cc_sn_catwise']:+.2f} D_CW={r['D_catwise']:.4f} "
              f"radio/CW={r['radio_over_catwise']} dQuaia={r['dchi2_quaia']:+.1f} edges={r['on_theta_edge']},{r['on_alpha_t0_edge']}",
              flush=True)


HEADS = {"equal": r"Equal weights", "atan": r"$\mathcal{W}=(2/\pi)\arctan[(1-\zeta)/\zeta]$",
         "atan2": r"$\mathcal{W}=\{(2/\pi)\arctan[(1-\zeta)/\zeta]\}^2$", "cos": r"$\mathcal{W}=\cos(\pi\zeta/2)$",
         "cos2": r"$\mathcal{W}=\cos^2(\pi\zeta/2)$", "quad": r"$\mathcal{W}=(1-\zeta)^2$"}
ROWS = (("axis", "Axis"), ("cc_sn", "CC+SN"), ("joint_neg", r"+CatWISE, $\mathcal{D}<0$"),
        ("joint_pos", r"+CatWISE, $\mathcal{D}>0$"))
NCOL = 11


def fmt(v, nd, sign=False):
    if v is None:
        return "--"
    return f"${v:+.{nd}f}$" if sign else f"{v:.{nd}f}"


def table():
    res = {w: json.loads((JSON / f"light_weight_comparison_{w}.json").read_text()) for w in WEIGHTS}
    summary = {w: res[w]["rows"] for w in WEIGHTS}
    summary["reference"] = {k: res["equal"][k] for k in ("D_geo", "sigma_geo", "chi2_catwise_lcdm", "chi2_quaia_none",
                                                          "x_radio", "z_of_D")}
    a_const = float(np.sum(Q_AMP / Q_SIG**2) / np.sum(1.0 / Q_SIG**2))   # constant amplitude fitted to the slices
    summary["reference"]["quaia_constant_amplitude"] = a_const
    summary["reference"]["dchi2_quaia_constant"] = float(np.sum(((a_const - Q_AMP) / Q_SIG) ** 2)) - CHI2_Q_NONE
    # measured radio excess over the kinematic dipole relative to that of CatWISE, amplitudes of Wagenveld et al. 2025,
    # Table 1 (D, sigma, D_kin): (D_radio - D_kin) / (D_CatWISE - D_kin), with first-order error propagation
    cw, cw_s, cw_k = RADIO_OBS["CatWISE"]
    for name in ("NVSS", "RACS-low"):
        d, s_, k_ = RADIO_OBS[name]
        r = (d - k_) / (cw - cw_k)
        summary["reference"][f"radio_over_catwise_{name}"] = [r, float(np.hypot(s_ / (cw - cw_k), r * cw_s / (cw - cw_k)))]
    (JSON / "light_weight_comparison.json").write_text(json.dumps(summary, indent=1))
    lines = []
    for w in WEIGHTS:
        rows = res[w]["rows"]
        on_axis = rows["cc_sn"]["theta_obs"] == 0.0
        lines.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (NCOL, HEADS[w]))
        for k, label in ROWS:
            if k == "cc_sn" and on_axis:
                continue
            r = rows[k]
            if k.startswith("joint") and r["chi2_catwise"] > 1.0:
                continue   # no observer with this sign of the dipole reproduces the CatWISE excess
            lab = "Axis = CC+SN" if (k == "axis" and on_axis) else label
            th = r["theta_obs"]
            ths = (f"{th:.2f}" if 0 < th < 1 else f"{th:.1f}") + (r"$^{*}$" if r["on_theta_edge"] else "")
            dz = [r["D_of_z"][str(z)] for z in (0.3, 1.0, 1.8)]
            ratio = r["radio_over_catwise"]
            lines.append(" & ".join([lab, f"{r['alpha_high']:.3f}", f"{r['t0']:.2f}", ths,
                                     fmt(r["dchi2_cc_sn"], 2, True), fmt(r["dchi2_cc_sn_catwise"], 1, True),
                                     "--" if ratio is None else f"{ratio:.2f}",
                                     *(("--",) * 3 if th == 0 else tuple(f"${v:.4f}$" if v < 0 else f"{v:.4f}" for v in dz)),
                                     fmt(r["dchi2_quaia"], 1, True)]) + r" \\")
    head = (r"Observer & $\alpha_{\mathrm{high}}$ & $t_0$ & $\vartheta_{\mathrm{obs}}$ & \multicolumn{2}{c}{$\Delta\chi^2$} & "
            r"$\mathcal{D}_{\mathrm{radio}}/$ & \multicolumn{3}{c}{$\mathcal{D}(z)$} & $\Delta\chi^2$ \\" "\n"
            r" & & & [deg] & CC+SN & +CatWISE & $\mathcal{D}_{\mathrm{CatWISE}}$ & $z=0.3$ & $1.0$ & $1.8$ & Quaia \\")
    tex = "\n".join([r"\begin{tabular}{l" + "c" * (NCOL - 1) + "}", r"\hline", head, r"\hline", *lines, r"\hline",
                     r"\end{tabular}"]) + "\n"
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables" / "light_weight_comparison.tex").write_text(tex, encoding="utf-8")
    print(tex)


if __name__ == "__main__":
    if ARGS.table:
        table()
    else:
        run(ARGS.weight, ARGS.workers)
