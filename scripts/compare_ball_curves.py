"""Which single expansion history of the light-cone ball model fits CC + binned Pantheon+ best?

Three curves, each with alpha_high and t0 fitted (Nelder-Mead), H0 and the SN amplitude minimised
analytically (likelihood.py), compared with flat LCDM (figures/fit_cc_pantheon.json):
  axis    the history of the sector theta = 0 alone (first row of fit_observer_ball.THETA);
  mean    the weighted mean over the ball, <E>(z) and <D_C>(z) (fit_observer_ball.chi2);
  median  the weighted median of E over the ball at every z, the central line of
          figures/hippopede_observer_ball_bands.png, with D_C = int dz / E_median.
The ball curves use the observer sector theta_obs = THETA_OBS (default 1 deg, on the CatWISE curve).
Writes figures/compare_ball_curves.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_observer_ball as OB  # noqa: E402
import likelihood as L  # noqa: E402

FIG = ROOT / "figures"
ZW = OB.ZW


def chi2_curve(e, dc):
    """CC + binned SN chi2 of one history E(z), D_C(z) on ZW (H0 and SN amplitude analytic)."""
    if e is None or not np.all(np.isfinite(e)) or np.any(e <= 0):
        return np.inf
    zc, hc, _ = L.CC
    ee = np.interp(zc, ZW, e)
    h0 = (ee @ L.CINV_CC @ hc) / (ee @ L.CINV_CC @ ee)
    r = hc - h0 * ee
    mm = np.interp(L.Z_SN, ZW, dc)
    amp = (mm @ L.CINV_SN @ L.D_SN) / (mm @ L.CINV_SN @ mm)
    s = L.D_SN - amp * mm
    return float(r @ L.CINV_CC @ r + s @ L.CINV_SN @ s)


def comoving(e):
    inv = 1.0 / e
    return np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(ZW))])


def curve(kind, ah, t0, theta_obs):
    if kind == "axis":
        E, _, _ = OB.sector_table(ah, t0)
        e = E[0]
        return (e, comoving(e)) if np.all(np.isfinite(e)) else (None, None)
    m = OB.model(ah, t0, theta_obs)
    if m is None:
        return None, None
    E, _, w, e_bar, dc_bar = m
    if kind == "mean":
        return e_bar, dc_bar
    e_med = np.array([OB._quantiles(E[:, j], w[:, j], [0.5])[0] for j in range(len(ZW))])
    return e_med, comoving(e_med)


def fit(args):
    kind, theta_obs, start = args
    f = lambda p: chi2_curve(*curve(kind, p[0], p[1], theta_obs))
    best = None
    for s in start:
        r = minimize(f, s, method="Nelder-Mead", options={"xatol": 2e-4, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    return kind, float(best.x[0]), float(best.x[1]), float(best.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-obs", type=float, default=1.0)
    a = ap.parse_args()
    lc = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]
    starts = [[0.335, 1.75], [0.36, 2.4], [0.40, 3.5]]
    jobs = [(k, a.theta_obs, starts) for k in ("axis", "mean", "median")]
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(fit, jobs))
    out = {"theta_obs": a.theta_obs, "lcdm_chi2": lc, "rows": []}
    for kind, ah, t0, c in res:
        out["rows"].append({"curve": kind, "alpha_high": ah, "t0": t0, "chi2": c, "dchi2_vs_lcdm": c - lc})
        print(f"{kind:7s}: alpha_high = {ah:.4f}, t0 = {t0:.3f}, chi2 = {c:.3f}, dchi2 vs LCDM = {c - lc:+.3f}", flush=True)
    (FIG / "compare_ball_curves.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
