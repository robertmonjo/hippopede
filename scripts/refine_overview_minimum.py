"""Profile of the CC + SN + CatWISE + Quaia chi2 along t0 and its continuous minimum (overview map).

Reads json/ball_zoom_unbinned_overview.json (chi2 of CC + unbinned SN on the grid of
plot_ball_zoom_unbinned.py) and adds the CatWISE and Quaia chi2 of the dipole runs with suffix _wide.
1. Profile: for every t0 of the grid, the minimum over theta_obs and alpha_high of the total.
2. Continuous minimum: Nelder-Mead in (alpha_high, ln t0, theta_obs) from the best grid point, with the
   CC + SN chi2 recomputed at each point (fit_ball_unbinned.Point), not interpolated.
Run with HIPPOPEDE_THETA_MAX_DEG=90, as the map.  Writes json/refine_overview_minimum.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_ball_unbinned as F  # noqa: E402
import fit_observer_ball as OB  # noqa: E402
import plot_ball_zoom_unbinned as Z  # noqa: E402

JSON = ROOT / "json"   # numerical outputs; figures/ holds the png/pdf
DIPOLE_ALPHAS = [round(0.30 + 0.02 * k, 2) for k in range(21)]


def dipoles(ah, t0, th):
    return F.catwise_chi2(ah, t0, th), Z.quaia_chi2(ah, t0, th)


def total(x, c_l):
    ah, t0, th = x[0], float(np.exp(x[1])), abs(x[2])
    cw, qw = dipoles(ah, t0, th)
    if not (np.isfinite(cw) and np.isfinite(qw)):
        return 1e6
    c = sum(F.Point(ah, t0).chi2(th, F.axis_opposite_catwise()))
    OB._T_CACHE.clear(); OB._W_CACHE.clear(); OB._M_CACHE.clear()
    return (c - c_l + cw + qw) if np.isfinite(c) else 1e6


def main():
    F.set_dipole_runs(DIPOLE_ALPHAS, "_wide")
    d = json.loads((JSON / "ball_zoom_unbinned_overview.json").read_text())
    A, T, TH = (np.array(d[k], float) for k in ("alpha_grid", "t0", "theta_obs"))
    C, c_l = np.array(d["chi2_raw"], float), d["lcdm_chi2"]
    tot = np.full_like(C, np.nan)
    for k, ah in enumerate(A):
        for i, th in enumerate(TH):
            for j, t0 in enumerate(T):
                if np.isfinite(C[k, i, j]):
                    cw, qw = dipoles(ah, t0, th)
                    tot[k, i, j] = C[k, i, j] - c_l + cw + qw
    prof = []
    for j, t0 in enumerate(T):
        m = tot[:, :, j]
        if np.isfinite(m).any():
            k, i = np.unravel_index(np.nanargmin(m), m.shape)
            cw, qw = dipoles(A[k], t0, TH[i])
            prof.append({"t0": float(t0), "total": float(m[k, i]), "theta_obs": float(TH[i]), "alpha_high": float(A[k]),
                         "dchi2_cc_sn": float(C[k, i, j] - c_l), "chi2_catwise": float(cw), "chi2_quaia": float(qw)})
            print(f"t0 {t0:6.3f}  total {m[k, i]:7.3f}  theta {TH[i]:5.2f}  alpha {A[k]:.2f}  "
                  f"CC+SN {C[k, i, j] - c_l:+.3f}  CatWISE {cw:.3f}  Quaia {qw:.3f}", flush=True)
    b = min(prof, key=lambda r: r["total"])
    x0 = np.array([b["alpha_high"], np.log(b["t0"]), b["theta_obs"]])
    simplex = np.vstack([x0, x0 + [0.01, 0, 0], x0 + [0, 0.05, 0], x0 + [0, 0, 1.0]])   # steps: alpha, ln t0, theta
    r = minimize(total, x0, args=(c_l,), method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-3, "initial_simplex": simplex})
    ah, t0, th = r.x[0], float(np.exp(r.x[1])), abs(r.x[2])
    cw, qw = dipoles(ah, t0, th)
    best = {"alpha_high": float(ah), "t0": t0, "theta_obs": float(th), "total": float(r.fun),
            "dchi2_cc_sn": float(r.fun - cw - qw), "chi2_catwise": float(cw), "chi2_quaia": float(qw), "nfev": int(r.nfev)}
    print("continuous minimum:", best, flush=True)
    (JSON / "refine_overview_minimum.json").write_text(json.dumps({"profile_t0": prof, "grid_best": b, "continuous_best": best}, indent=1))


if __name__ == "__main__":
    main()
