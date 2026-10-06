"""Fit of the asymptotic projection index alpha_high to CC + binned Pantheon+.

Model: the projected hyperconical history (the axial sector of the hippopede, independent of
t0) with alpha(z) = alpha_high - (alpha_high - alpha_low)/sqrt(1+z), alpha_low = 0.283
(Monjo 2018).  Likelihood: likelihood.py (H0 and SN amplitude minimised analytically).
Output: best alpha_high, 1-sigma interval (Delta chi2 = 1), chi2 and chi2_nu; the constant
index alpha = alpha_low is reported for comparison.  Writes json/fit_alpha_high.json,
which every other script reads through projected_hyperconical.load_alpha_high().
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

import likelihood as L  # noqa: E402
import projected_hyperconical as PH  # noqa: E402


def chi2(ah):
    c_cc, c_sn, _ = L.total(L.e_hyperconical(ah))
    return c_cc + c_sn


def main():
    grid = np.linspace(0.20, 0.80, 61)
    c = np.array([chi2(a) for a in grid])
    i = int(np.argmin(c))
    r = minimize_scalar(chi2, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]),
                        method="bounded", options={"xatol": 1e-5})
    ah, cmin = float(r.x), float(r.fun)
    lo = brentq(lambda a: chi2(a) - cmin - 1.0, grid[0], ah) if chi2(grid[0]) > cmin + 1 else None
    hi = brentq(lambda a: chi2(a) - cmin - 1.0, ah, grid[-1]) if chi2(grid[-1]) > cmin + 1 else None
    const = lambda z: PH.E_of_z(np.asarray(z, float), PH.alpha_const(PH.ALPHA_LOW))
    c_const = sum(L.total(const)[:2])
    dof = L.N_DATA - 3  # alpha_high, H0, SN amplitude
    out = {"alpha_low": PH.ALPHA_LOW, "alpha_high": ah, "alpha_high_1sigma": [lo, hi], "chi2": cmin,
           "dof": dof, "chi2_nu": cmin / dof, "chi2_constant_alpha_low": float(c_const),
           "N_CC": len(L.CC[0]), "N_SN_bins": len(L.Z_SN), "profile": {"alpha_high": grid.tolist(), "chi2": c.tolist()}}
    (ROOT / "json" / "fit_alpha_high.json").write_text(json.dumps(out, indent=1))
    print(f"alpha_high = {ah:.4f}  1sigma = [{lo}, {hi}]  chi2 = {cmin:.2f}  dof = {dof}  chi2_nu = {cmin / dof:.3f}  "
          f"(constant alpha = {PH.ALPHA_LOW}: chi2 = {c_const:.2f})")


if __name__ == "__main__":
    main()
