"""Chi2 of the quasar dipoles as a function of the observer sector theta_obs (light-cone ball).

On the grid of the overview map (json/ball_zoom_unbinned_overview.json: chi2 of the 38 chronometers and the 1579
unbinned supernovae over alpha_high, theta_obs, t0) the CatWISE and Quaia chi2 of the dipole runs with suffix _wide
are added, and at every theta_obs the minimum over t0 and alpha_high is taken
  (a) of CC + SN + CatWISE + Quaia (the model), reporting the three contributions, and
  (b) of the dipoles alone, CatWISE + Quaia, which bounds what the geometry can reach.
theta_obs = 0 has no dipole beyond the kinematic one, as flat LCDM.  References: LCDM (no dipole) and LCDM with
one or two ad hoc amplitudes (json/compare_dipole_models.json).  Run with HIPPOPEDE_THETA_MAX_DEG=90, as the map.
Writes json/dipole_chi2_vs_theta.json and figures/hippopede_dipole_chi2_vs_theta.(png|pdf).

With --continuous the minimum at every theta_obs of THETA_CONTINUOUS is found by Nelder-Mead in (alpha_high, ln t0),
started from the best grid point at the nearest grid theta_obs, for (a) with the CC + SN chi2 recomputed at each
point (fit_ball_unbinned.Point, not interpolated) and for (b) with the dipoles alone; alpha_high is held in the range
of the dipole runs and t0 in that of the map.  The theta_obs values run in parallel (--workers).  Writes
json/dipole_chi2_vs_theta_continuous.json (rows as above plus the number of evaluations nfev) and draws the figure
from it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_ball_unbinned as F  # noqa: E402
import fit_observer_ball as OB  # noqa: E402
import plot_ball_zoom_unbinned as Z  # noqa: E402

JSON = ROOT / "json"   # numerical outputs; figures/ holds the png/pdf
FIG = ROOT / "figures"
DIPOLE_ALPHAS = [round(0.30 + 0.02 * k, 2) for k in range(21)]
THETA_CONTINUOUS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 45]
T0_RANGE = (1.4, 20.0)


def grid():
    """chi2 of CC + SN (minus LCDM), CatWISE and Quaia on the overview grid, and the grid axes."""
    F.set_dipole_runs(DIPOLE_ALPHAS, "_wide")
    d = json.loads((JSON / "ball_zoom_unbinned_overview.json").read_text())
    A, T, TH = (np.array(d[k], float) for k in ("alpha_grid", "t0", "theta_obs"))
    C = np.array(d["chi2_raw"], float) - d["lcdm_chi2"]
    CW, QW = np.full_like(C, np.nan), np.full_like(C, np.nan)
    for k, ah in enumerate(A):
        for i, th in enumerate(TH):
            for j, t0 in enumerate(T):
                if np.isfinite(C[k, i, j]):
                    CW[k, i, j], QW[k, i, j] = F.catwise_chi2(ah, t0, th), Z.quaia_chi2(ah, t0, th)
    return A, T, TH, C, CW, QW, d["lcdm_chi2"]


def grid_rows(A, T, TH, C, CW, QW):
    """Minimum over the grid of (a) the total and (b) the dipoles alone at every theta_obs."""
    DIP = CW + QW
    rows = []
    for i, th in enumerate(TH):
        m, t = DIP[:, i, :], (C + DIP)[:, i, :]
        if not np.isfinite(t).any():
            continue
        k, j = np.unravel_index(np.nanargmin(m), m.shape)
        k2, j2 = np.unravel_index(np.nanargmin(t), t.shape)
        rows.append({"theta_obs": float(th),
                     "with_cc_sn": {"total": float(t[k2, j2]), "dchi2_cc_sn": float(C[k2, i, j2]), "catwise": float(CW[k2, i, j2]),
                                    "quaia": float(QW[k2, i, j2]), "t0": float(T[j2]), "alpha_high": float(A[k2])},
                     "dipoles_only": {"chi2": float(m[k, j]), "catwise": float(CW[k, i, j]), "quaia": float(QW[k, i, j]),
                                      "t0": float(T[j]), "alpha_high": float(A[k]), "dchi2_cc_sn": float(C[k, i, j]),
                                      "alpha_at_grid_edge": bool(k in (0, len(A) - 1))}})
    return rows


def clip(x):
    """(alpha_high, t0) from (alpha_high, ln t0), held in the range of the dipole runs and of the map."""
    return (float(np.clip(x[0], DIPOLE_ALPHAS[0], DIPOLE_ALPHAS[-1])),
            float(np.exp(np.clip(x[1], np.log(T0_RANGE[0]), np.log(T0_RANGE[1])))))


def dipoles(ah, t0, th):
    return F.catwise_chi2(ah, t0, th), Z.quaia_chi2(ah, t0, th)


def cc_sn(ah, t0, th, c_l):
    c = sum(F.Point(ah, t0).chi2(th, F.axis_opposite_catwise()))
    OB._T_CACHE.clear(); OB._W_CACHE.clear(); OB._M_CACHE.clear()
    return c - c_l


def f_total(x, th, c_l):
    ah, t0 = clip(x)
    cw, qw = dipoles(ah, t0, th)
    if not (np.isfinite(cw) and np.isfinite(qw)):
        return 1e6
    c = cc_sn(ah, t0, th, c_l)
    return c + cw + qw if np.isfinite(c) else 1e6


def f_dipoles(x, th):
    cw, qw = dipoles(*clip(x), th)
    return cw + qw if np.isfinite(cw) and np.isfinite(qw) else 1e6


def nelder_mead(f, start, args):
    x0 = np.array([start["alpha_high"], np.log(start["t0"])])
    simplex = np.vstack([x0, x0 + [0.01, 0], x0 + [0, 0.05]])   # steps: alpha, ln t0
    return minimize(f, x0, args=args, method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "initial_simplex": simplex, "maxfev": 400})


def refine(task):
    """Continuous minimum at one theta_obs from the grid starting points (one worker)."""
    th, start, c_l = task
    F.set_dipole_runs(DIPOLE_ALPHAS, "_wide")
    r = nelder_mead(f_total, start["with_cc_sn"], (th, c_l))
    ah, t0 = clip(r.x)
    cw, qw = dipoles(ah, t0, th)
    w = {"total": float(r.fun), "dchi2_cc_sn": float(r.fun - cw - qw), "catwise": float(cw), "quaia": float(qw),
         "t0": t0, "alpha_high": ah, "nfev": int(r.nfev)}
    r = nelder_mead(f_dipoles, start["dipoles_only"], (th,))
    ah, t0 = clip(r.x)
    cw, qw = dipoles(ah, t0, th)
    o = {"chi2": float(r.fun), "catwise": float(cw), "quaia": float(qw), "t0": t0, "alpha_high": ah,
         "dchi2_cc_sn": float(cc_sn(ah, t0, th, c_l)), "alpha_at_range_edge": bool(min(ah - DIPOLE_ALPHAS[0], DIPOLE_ALPHAS[-1] - ah) < 1e-3),
         "nfev": int(r.nfev)}
    print(f"theta {th:5.2f} done", flush=True)
    return {"theta_obs": float(th), "with_cc_sn": w, "dipoles_only": o}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--continuous", action="store_true", help="Nelder-Mead in (alpha_high, ln t0) at every theta_obs")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    A, T, TH, C, CW, QW, c_l = grid()
    rows = grid_rows(A, T, TH, C, CW, QW)
    name = "dipole_chi2_vs_theta"
    if args.continuous:
        by_theta = {r["theta_obs"]: r for r in rows}
        tasks = [(float(th), by_theta[min(by_theta, key=lambda g: abs(g - th))], c_l) for th in THETA_CONTINUOUS]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(refine, tasks))
        name += "_continuous"
    ref = {r["model"]: r for r in json.loads((JSON / "compare_dipole_models.json").read_text())["rows"]}
    lcdm = ref["LCDM"]["total"]
    out = {"rows": rows, "references_total": {"LCDM": lcdm, "LCDM + common dipole": ref["LCDM + common dipole"]["total"],
                                              "LCDM + two dipoles": ref["LCDM + two dipoles"]["total"]}}
    (JSON / f"{name}.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        w = r["with_cc_sn"]
        print(f"theta {r['theta_obs']:5.2f}: total {w['total']:6.2f} (CC+SN {w['dchi2_cc_sn']:+.2f}, CatWISE {w['catwise']:.2f}, "
              f"Quaia {w['quaia']:.2f}) at t0 {w['t0']:.2f}, alpha {w['alpha_high']:.2f};  dipoles only {r['dipoles_only']['chi2']:.2f}")

    th = np.array([r["theta_obs"] for r in rows])
    g = lambda key, sub: np.array([r[key][sub] for r in rows])
    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    x = np.where(th > 0, th, 0.1)   # theta_obs = 0 drawn at the left edge of the log axis
    ax.plot(x, g("with_cc_sn", "total"), color="k", lw=2.0, label=r"total: CC + SN + CatWISE + Quaia")
    ax.plot(x, g("with_cc_sn", "quaia"), color="#E08A00", lw=1.6, label="Quaia slices")
    ax.plot(x, g("with_cc_sn", "catwise"), color="#1a9641", lw=1.6, label="CatWISE")
    ax.plot(x, g("dipoles_only", "chi2"), color="0.45", lw=1.2, ls="--", label="dipoles alone (lower bound of the geometry)")
    for name, ls in (("LCDM", "-"), ("LCDM + common dipole", "-."), ("LCDM + two dipoles", (0, (1, 1)))):
        ax.axhline(out["references_total"][name], color="#c00000", lw=0.9, ls=ls)
        ax.text(45, out["references_total"][name] + 0.8, {"LCDM": r"$\Lambda$CDM", "LCDM + common dipole": r"$\Lambda$CDM + 1 ad hoc amplitude",
                "LCDM + two dipoles": r"$\Lambda$CDM + 2 ad hoc amplitudes"}[name], color="#c00000", fontsize=8, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 45); ax.set_ylim(0, 62)
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 45]); ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20", "45"])
    ax.set_xlabel(r"Observer sector $\vartheta_{\rm obs}$ [deg] ($t_0$ and $\alpha_{\rm high}$ fitted at each value)")
    ax.set_ylabel(r"$\chi^2$")
    ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62), framealpha=0.9)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hippopede_dipole_chi2_vs_theta.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
