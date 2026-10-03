"""Check of the high-redshift evaluation of projected_hyperconical.py against 50-digit arithmetic.

The projected distance rhat(z) = 2 arctan[(gamma/2)(1 - gamma/gamma_U)^(-alpha)] is computed
with mpmath: x(z) from the line-of-sight integral xi_1(x) = ln(1+z) (integrand
sqrt(1-(1-b)^2)/[b(2b-1)], b = sqrt(1-x^2)), and H = (d rhat/dz)^(-1) by numerical
differentiation.  The ratio to the double-precision module is printed at z = 1e3, 3e8, 4e9,
for constant alpha = 1/2 and for the running index.  Also prints the local slope
n_eff = d ln H / d ln(1+z) for alpha = 1/2 between z = 1e3 and 4e9.
Requires mpmath.  Writes figures/verify_high_z.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import projected_hyperconical as PH  # noqa: E402

mp.mp.dps = 50
A_LOW, A_HIGH = PH.ALPHA_LOW, PH.load_alpha_high()  # alpha_high from fit_alpha_high.py
XM = mp.sqrt(3) / 2


def integrand(x):
    b = mp.sqrt(1 - x**2)
    return mp.sqrt(1 - (1 - b) ** 2) / (b * (2 * b - 1))


X1 = mp.mpf("0.8")
I1 = mp.quad(integrand, [0, 0.5, X1])


def xi(d):
    if d < mp.mpf("1e-3"):
        return I1 + mp.quad(integrand, [X1, XM - mp.mpf("1e-3"), XM - d])
    return I1 + mp.quad(integrand, [X1, XM - d])


def x_of_z(z):
    L = mp.log(1 + mp.mpf(z))
    ld = mp.findroot(lambda t: xi(mp.exp(t)) - L, mp.log(mp.mpf(1) / (1 + mp.mpf(z)) ** 2))
    return XM - mp.exp(ld)


def rhat(z, alpha):
    g = mp.asin(x_of_z(z))
    return 2 * mp.atan((g / 2) / (1 - g / (mp.pi / 3)) ** alpha(z))


def H(z, alpha):
    return 1 / mp.diff(lambda zz: rhat(zz, alpha), mp.mpf(z))


def main():
    const = lambda z: mp.mpf("0.5")
    run = lambda z: A_HIGH - (A_HIGH - A_LOW) / mp.sqrt(1 + z)
    H0c, H0r = 1 / mp.diff(lambda zz: rhat(zz, const), 0), 1 / mp.diff(lambda zz: rhat(zz, run), 0)
    out = {"ratio_mp_over_double": []}
    for z in (1e3, 3e8, 4e9):
        rc = float(H(z, const) / H0c) / PH.E_of_z(np.array([z]), PH.alpha_const(0.5))[0]
        rr = float(H(z, run) / H0r) / PH.E_of_z(np.array([z]), lambda zz: PH.alpha_sqrt(zz, A_LOW, A_HIGH))[0]
        out["ratio_mp_over_double"].append({"z": z, "alpha_half": rc, "running": rr})
        print(f"z={z:.0e}: mpmath/double = {rc:.8f} (alpha=1/2), {rr:.8f} (running)")
    zz = np.geomspace(1e3, 4e9, 7)
    n_eff = 1 + PH.q_of_z(zz, PH.alpha_const(0.5))
    out["n_eff_alpha_half"] = dict(zip(map(str, zz), n_eff.tolist()))
    print("n_eff (alpha=1/2) over 1e3-4e9:", np.round(n_eff, 4).tolist())
    (ROOT / "figures" / "verify_high_z.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
