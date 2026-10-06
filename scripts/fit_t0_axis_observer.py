"""Fit of t0 for an observer on the distinguished axis.

A source at redshift z lies at angular distance rhat(z) on the lobe, hence in the sector
theta(z) = rhat(z)/2 (sector_geometry.py).  All sectors share the time t0, so the rate of the
source sector is expressed in the observer's units by the factor H_c,theta(t0)/H_c,0(t0):
    E_obs(z; t0) = E^proj_{theta(z)}(z) * H_c,theta(z)(t0) / H_c,0(t0).
Likelihood: likelihood.py (CC compilation + binned Pantheon+ with full covariance; H0 and the SN
amplitude minimised).  Free parameter of the model: t0.  Output: chi2 profile, best t0, the
Delta chi2 = 1 and 4 intervals (one-sided when the profile is flat up to the upper edge of the
grid, where the history tends to the axial one), chi2_nu.  Writes json/fit_t0_axis_observer.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import hippopede_model as HM  # noqa: E402
import likelihood as L  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import sector_geometry as G  # noqa: E402

THETA_GRID = np.arange(0.0, 60.01, 0.5)
ZZ = np.linspace(0.0, 2.2, 1101)
T_MAX = 200.0
PLATEAU = 0.01


def e_obs(t0):
    run = lambda z: PH.alpha_sqrt(z, PH.ALPHA_LOW, PH.load_alpha_high())
    th = G.source_sector_deg(0.0, PH.rhat_of_z(ZZ, run), 1.0)
    h_ax = HM.h0_sector(0.0, t0)
    table = np.array([np.interp(ZZ, HM.Z_WORK, HM.projected_sector(a, t0)[0], left=np.nan, right=np.nan)
                      * HM.h0_sector(np.radians(a), t0) / h_ax for a in THETA_GRID])
    e = np.array([np.interp(th[j], THETA_GRID, table[:, j]) for j in range(len(ZZ))])
    return lambda z: np.interp(z, ZZ, e, left=np.nan, right=np.nan)


def chi2(t0):
    return sum(L.total(e_obs(t0))[:2])


def main():
    grid = np.geomspace(1.5, T_MAX, 40)
    c = np.array([chi2(t) for t in grid])
    i = int(np.nanargmin(c))
    if 0 < i < len(grid) - 1:
        r = minimize_scalar(chi2, bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-3})
        t_best, c_min = float(r.x), float(r.fun)
    else:
        t_best, c_min = float(grid[i]), float(c[i])
    interval = {}
    for name, lev in (("1sigma", 1.0), ("2sigma", 4.0)):
        f = lambda t: chi2(t) - c_min - lev
        lo = brentq(f, grid[0], t_best) if t_best > grid[0] and f(grid[0]) > 0 else None
        hi = brentq(f, t_best, grid[-1]) if t_best < grid[-1] and f(grid[-1]) > 0 else None
        interval[name] = [lo, hi]
    dof = L.N_DATA - 3
    # the profile is flat when chi2 at the largest t0 (the axial history) exceeds the minimum by
    # less than PLATEAU: the data then bound t0 from below only
    plateau = bool(c[-1] - c_min < PLATEAU)
    out = {"t0_best": t_best, "chi2_min": c_min, "dof": dof, "chi2_nu": c_min / dof, "interval": interval,
           "minimum_at_grid_edge": bool(i == len(grid) - 1), "plateau_to_axial_limit": plateau,
           "chi2_axial_limit": float(c[-1]), "profile": {"t0": grid.tolist(), "chi2": c.tolist()}}
    (ROOT / "json" / "fit_t0_axis_observer.json").write_text(json.dumps(out, indent=1))
    print(f"t0 best = {t_best:.3f}  chi2 = {c_min:.2f}  chi2_nu = {c_min / dof:.3f}  intervals: {interval}  "
          f"edge = {out['minimum_at_grid_edge']}  plateau = {plateau} (chi2 at t0={T_MAX:g}: {c[-1]:.4f})")


if __name__ == "__main__":
    main()
