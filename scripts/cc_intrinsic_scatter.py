"""Intrinsic scatter of the cosmic chronometers about a smooth expansion history.

If the width of the reconstructed band reflected a physical spread of H(z) between directions, the
individual chronometers, which sample different fields, would scatter about a smooth curve by more
than their covariance allows.  The 38 chronometers (likelihood.py, full covariance) are fitted with
flat LCDM and with a cubic polynomial in z; the residuals of the subsets z < 0.5, z > 0.5 and z > 1
are then described by the covariance block plus sigma_int^2 on the diagonal, and sigma_int is found by
maximum likelihood with its 95 per cent upper limit (Delta(-2 ln L) = 3.84).
Writes json/cc_intrinsic_scatter.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as CHI2

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import likelihood as L  # noqa: E402

Z, H, _ = L.CC
COV = np.linalg.inv(L.CINV_CC)
SUBSETS = {"z<0.5": Z < 0.5, "z>0.5": Z > 0.5, "z>1": Z > 1.0}


def lcdm(p, z):
    return p[0] * np.sqrt(p[1] * (1 + z) ** 3 + 1 - p[1])


def cubic(p, z):
    return np.polyval(p, z)


def fit(model, p0):
    f = lambda p: (H - model(p, Z)) @ L.CINV_CC @ (H - model(p, Z))
    r = minimize(f, p0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 20000})
    return r.x, float(r.fun)


def scatter(resid, sel):
    """chi2 of the block, maximum-likelihood sigma_int and its 95% upper limit [km/s/Mpc]."""
    C = COV[np.ix_(sel, sel)]
    r = resid[sel]
    grid = np.linspace(0.0, 40.0, 4001)

    def m2lnl(s):
        Cx = C + s**2 * np.eye(len(r))
        return r @ np.linalg.solve(Cx, r) + np.linalg.slogdet(Cx)[1]

    v = np.array([m2lnl(s) for s in grid])
    k = int(np.argmin(v))
    upper = float(grid[np.where((v - v[k] <= 3.84) & (grid >= grid[k]))[0][-1]])
    c = float(r @ np.linalg.solve(C, r))
    return {"N": int(sel.sum()), "chi2": c, "p_value": float(CHI2.sf(c, sel.sum())), "sigma_int_ml": float(grid[k]),
            "sigma_int_95": upper, "dm2lnL_at_zero": float(v[0] - v[k]), "median_sigma_H": float(np.median(np.sqrt(np.diag(C))))}


def main():
    out = {}
    for name, model, p0, k in (("flat_lcdm", lcdm, [68.0, 0.3], 2), ("cubic", cubic, [0.0, 0.0, 60.0, 68.0], 4)):
        p, c = fit(model, p0)
        resid = H - model(p, Z)
        out[name] = {"params": p.tolist(), "chi2": c, "dof": len(Z) - k, "subsets": {s: scatter(resid, m) for s, m in SUBSETS.items()}}
        print(f"{name}: chi2 = {c:.2f} for {len(Z) - k} dof")
        for s, d in out[name]["subsets"].items():
            print(f"  {s:6s} N={d['N']:2d} chi2/N={d['chi2'] / d['N']:.2f} sigma_int={d['sigma_int_ml']:.2f} "
                  f"(<{d['sigma_int_95']:.2f} at 95%) Delta(-2lnL) at 0 = {d['dm2lnL_at_zero']:.2f}")
    out["sector_spread"] = sector_spread(out["flat_lcdm"]["params"][0])
    (ROOT / "json" / "cc_intrinsic_scatter.json").write_text(json.dumps(out, indent=1))


def sector_spread(h0, zs=(0.5, 0.8, 1.0, 1.3, 1.6, 1.9)):
    """1-sigma half-width of H(z) across the sectors 0 < theta < 70 deg, weighted by their volume on the
    lobe (t0_dispersion_match.py), at the reference t0 of the paper (json/t0_summary.json), in km/s/Mpc
    with H0 of flat LCDM fitted to the chronometers."""
    import t0_dispersion_match as DM
    t0 = float(json.loads((ROOT / "json" / "t0_summary.json").read_text())["figure_t0"])
    DM.Z_NODES = np.array(zs)
    E = DM.sector_values(t0)["E"]
    hw = [0.5 * (DM.weighted_quantile(E[:, j], DM.W, 0.84134) - DM.weighted_quantile(E[:, j], DM.W, 0.15866)) * h0
          for j in range(len(zs))]
    print(f"spread of H across sectors at t0 = {t0}: " + ", ".join(f"z={z}: {v:.1f}" for z, v in zip(zs, hw)) + " km/s/Mpc")
    return {"t0": t0, "H0": h0, "z": list(zs), "half_width_1sigma": hw}


if __name__ == "__main__":
    main()
