"""Count dipole of an off-axis observer compared with the CatWISE quasar dipole excess.

Data (Secrest et al. 2021, ApJL 908, L51): measured number-count dipole D_obs = 0.01554 towards
(l, b) = (238.2, 28.8) deg; kinematic expectation D_kin ~ 0.007 along the CMB dipole
(l, b) = (264.0, 48.3) deg; mean quasar redshift 1.2; the kinematic interpretation is rejected at
4.9 sigma (joint test of amplitude and direction).  The quantity compared with the model is the
vector excess D_geo = D_obs - D_kin (computed here).  Its uncertainty is taken as
(D_obs - D_kin)/4.9, the amplitude difference divided by the quoted significance; this is a
heuristic, since the 4.9 sigma refers to the joint test.

Model: observer at sector angle theta_obs, axis direction free (the pattern is axisymmetric
about it).  For sources at redshift z in direction n, a flux-limited count with integral slope x
changes by
    delta ln N(z, n) = (2 - 2x) delta ln D_C(z, n) - delta ln E(z, n)
(volume 2 ln D_C - ln E minus the flux dimming 2x ln D_L; delta: departure from the sky average
at fixed z).  The dipole is D(z) = (3/2) int delta ln N cos psi sin psi dpsi, averaged over a
redshift distribution p(z) (z^2 exp(-z/0.4), mean 1.2 before truncation, or flat) on
0.1 <= z <= 2.1.  The slope follows from D_kin = [2 + x(1 + alpha)] beta with alpha = 1.26 and
beta = 369.82/299792.458, which gives x = 1.63; results are also given for x = 1.5 and 1.8.
Sector rates in common units (pantheon_offaxis_test.e_table).  Neglects source evolution,
magnification and the null geodesics of the model.  Writes json/quasar_dipole_fit.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pantheon_offaxis_test as T  # noqa: E402

D_OBS, LB_OBS = 0.01554, (238.2, 28.8)
D_KIN, LB_CMB = 0.007, (264.0, 48.3)
BETA, ALPHA_SPEC = 369.82 / 299792.458, 1.26
X_SLOPE = (D_KIN / BETA - 2.0) / (1.0 + ALPHA_SPEC)
SIG_GEO = (D_OBS - D_KIN) / 4.9
Z_Q = np.linspace(0.1, 2.1, 41)
PSI = np.linspace(0.0, np.pi, 37)
THETA_OBS = [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 35]  # up to sector_geometry.THETA_OBS_MAX_DEG


def unit_icrs(l, b):
    c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
    return np.array([np.cos(c.dec.rad) * np.cos(c.ra.rad), np.cos(c.dec.rad) * np.sin(c.ra.rad), np.sin(c.dec.rad)])


def excess_vector():
    v = D_OBS * unit_icrs(*LB_OBS) - D_KIN * unit_icrs(*LB_CMB)
    g = SkyCoord(x=v[0], y=v[1], z=v[2], representation_type="cartesian", frame="icrs").galactic
    return v, float(np.linalg.norm(v)), (float(g.l.deg), float(g.b.deg))


def pz(kind):
    p = Z_Q**2 * np.exp(-Z_Q / 0.4) if kind == "gamma" else np.ones_like(Z_Q)
    return p / np.trapezoid(p, Z_Q)


def dipole(tab, theta_obs_deg, x, p):
    """Signed dipole (> 0: excess of counts towards the axis direction) and the covered p(z) mass."""
    zz = np.repeat(Z_Q, len(PSI))
    cps = np.tile(np.cos(PSI), len(Z_Q))
    n = np.stack([np.sqrt(1 - cps**2), np.zeros_like(cps), cps], axis=1)
    sky = T.Sky(zz, n)
    e_path = T.bilinear(tab, sky.sector_path(theta_obs_deg, cps), sky.zp)
    dc = np.trapezoid(1.0 / e_path, sky.zp, axis=1).reshape(len(Z_Q), len(PSI))
    lnD, lnE = np.log(dc), np.log(e_path[:, -1].reshape(len(Z_Q), len(PSI)))
    w = np.sin(PSI) / np.trapezoid(np.sin(PSI), PSI)
    dlnD = lnD - np.trapezoid(lnD * w, PSI, axis=1)[:, None]
    dlnE = lnE - np.trapezoid(lnE * w, PSI, axis=1)[:, None]
    d_z = 1.5 * np.trapezoid(((2 - 2 * x) * dlnD - dlnE) * np.cos(PSI) * np.sin(PSI), PSI, axis=1)
    ok = np.isfinite(d_z)
    cover = float(np.trapezoid(p * ok, Z_Q))
    if cover < 0.5:
        return np.nan, cover
    return float(np.trapezoid(np.where(ok, d_z, 0.0) * p, Z_Q) / cover), cover


def solve_theta(d_abs, target):
    for a in range(len(THETA_OBS) - 1):
        if np.isfinite(d_abs[a]) and np.isfinite(d_abs[a + 1]) and (d_abs[a] - target) * (d_abs[a + 1] - target) <= 0 \
                and d_abs[a + 1] != d_abs[a]:
            return float(THETA_OBS[a] + (target - d_abs[a]) * (THETA_OBS[a + 1] - THETA_OBS[a]) / (d_abs[a + 1] - d_abs[a]))
    return None


def options():
    """--from-average-fit: alpha_high of fit_sector_average.py, a dense grid of t0 and outputs with the
    suffix _sector_average_alpha (used by plot_observer_sky.py --landscape)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-average-fit", action="store_true")
    ap.add_argument("--alpha-high", type=float, default=None, help="any alpha_high; outputs get the suffix _ah<value>")
    ap.add_argument("--t0-range", nargs=2, type=float, default=(2.4, 6.0), metavar=("MIN", "MAX"),
                    help="range of the t0 grid (geometric) with --alpha-high or --from-average-fit")
    ap.add_argument("--t0-n", type=int, default=10, help="number of t0 values in that grid")
    ap.add_argument("--theta-obs", nargs="+", type=float, default=None, help="observer sectors [deg] (default THETA_OBS)")
    ap.add_argument("--out-suffix", default="", help="appended to the output suffix, e.g. _wide")
    args = ap.parse_args()
    if args.theta_obs is not None:
        THETA_OBS[:] = args.theta_obs
    if args.alpha_high is not None:
        ah, suffix = args.alpha_high, "_ah" + f"{args.alpha_high:g}".replace(".", "p") + args.out_suffix
    elif args.from_average_fit:
        ah = json.loads((ROOT / "json" / "fit_sector_average.json").read_text())["cases"]["high"]["alpha_high"]
        suffix = "_sector_average_alpha"
    else:
        return None, ""
    T.set_alpha_high(ah)
    return [float(t) for t in np.round(np.geomspace(*args.t0_range, args.t0_n), 3)], suffix


def main():
    t0_list, suffix = options()
    _, d_geo, lb_geo = excess_vector()
    t_ref = float(json.loads((ROOT / "json" / "t0_summary.json").read_text())["figure_t0"])
    print(f"x = {X_SLOPE:.3f}; D_geo = {d_geo:.4f} +- {SIG_GEO:.4f} towards (l,b) = ({lb_geo[0]:.1f}, {lb_geo[1]:.1f})")
    out = {"x": X_SLOPE, "D_geo": d_geo, "sigma": SIG_GEO, "excess_lb": lb_geo, "theta_obs_grid": THETA_OBS,
           "reference_t0": t_ref, "runs": []}
    for t0 in (t0_list or sorted({t_ref, 2.5, 3.0, 4.0, 6.0, 10.0})):
        tab = T.e_table(t0)
        for kind in ("gamma", "flat"):
            for x in (X_SLOPE, 1.5, 1.8):
                res = [dipole(tab, th, x, pz(kind)) for th in THETA_OBS]
                d = [r[0] for r in res]
                da = np.abs(d)
                sol = [solve_theta(da, d_geo + k * SIG_GEO) for k in (-1, 0, 1)]
                out["runs"].append({"t0": t0, "pz": kind, "x": x, "D": d, "coverage": [r[1] for r in res],
                                    "theta_obs_minus1sigma_best_plus1sigma": sol})
                if x == X_SLOPE:
                    print(f"t0={t0:5.2f} p(z)={kind:5s}: theta_obs for D_geo (-1s, best, +1s) = "
                          f"{[None if s is None else round(s, 2) for s in sol]}")
    (ROOT / "json" / f"quasar_dipole_fit{suffix}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
