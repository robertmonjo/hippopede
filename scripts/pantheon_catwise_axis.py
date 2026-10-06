"""Pantheon+ with the axis of the lobe fixed to the CatWISE excess direction.

The observer is placed where the light-cone ball model (fit_observer_ball.py) reproduces the CatWISE
quasar-dipole excess: for a set of t0, theta_obs on the CatWISE curve at the fitted alpha_high
(json/fit_observer_ball_landscape_joint_wide.json, from plot_observer_ball_landscape.py).  At each
point the unbinned Pantheon+ chi2 (pantheon_offaxis_test.py: line-of-sight sector path, full STAT+SYS
covariance, M minimised analytically) is computed for
  - the axis observer (theta_obs = 0, isotropic),
  - the projected axis pointing to the CatWISE excess direction n_C and to -n_C,
  - N_RANDOM random axis directions, which calibrate how special n_C is (fraction of random axes with a
    lower chi2 than n_C).
No parameter is fitted to the supernovae: (alpha_high, t0, theta_obs) come from CC + SN binned +
CatWISE and the axis from the quasar dipole.  Writes json/pantheon_catwise_axis.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pantheon_offaxis_test as T  # noqa: E402
import quasar_dipole_fit as QD  # noqa: E402

JSON = ROOT / "json"   # numerical outputs; figures/ holds the png/pdf
T0_POINTS = (1.6, 1.8, 2.2, 2.7, 3.3, 3.9, 4.5)
N_RANDOM = 2000


def catwise_axis_icrs():
    _, _, (l, b) = QD.excess_vector()
    c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
    ra, dec = c.ra.rad, c.dec.rad
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]), (float(l), float(b))


def points():
    """(t0, theta_obs, alpha_high) on the CatWISE curve at the fitted alpha_high."""
    d = json.loads((JSON / "fit_observer_ball_landscape_joint_wide.json").read_text())
    cw = d["catwise_theta_obs"]
    t, th = np.array(cw["t0"], float), np.array(cw["best"], float)
    ok = np.isfinite(th)
    T_GRID, TH_GRID = np.array(d["t0"]), np.array(d["theta_obs"])
    A = np.array(d["alpha_profiled"], float)
    out = []
    for t0 in T0_POINTS:
        theta = float(np.interp(np.log(t0), np.log(t[ok]), th[ok]))
        j, i = int(np.argmin(np.abs(T_GRID - t0))), int(np.argmin(np.abs(TH_GRID - theta)))
        out.append((t0, theta, float(A[i, j])))
    return out


def run(args):
    t0, theta, ah = args
    T.set_alpha_high(ah)
    z, zh, mb, cov, n = T.load()
    sky = T.Sky(z, n, zh, mb, cov)
    tab = T.e_table(t0)
    n_c, _ = catwise_axis_icrs()
    c_axis = sky.chi2(tab, 0.0, n_c)
    c_plus = sky.chi2(tab, theta, n_c)
    c_minus = sky.chi2(tab, theta, -n_c)
    rng = np.random.default_rng(12345)
    r = rng.normal(size=(N_RANDOM, 3))
    r /= np.linalg.norm(r, axis=1)[:, None]
    c_rand = np.array([sky.chi2(tab, theta, v) for v in r])
    oms = np.linspace(0.2, 0.5, 301)
    c_lcdm = float(min(sky.chi2_model(T.lcdm_model(sky, om)) for om in oms))
    return {"t0": t0, "theta_obs": theta, "alpha_high": ah, "chi2_lcdm": c_lcdm, "chi2_axis_observer": c_axis,
            "chi2_catwise_plus": c_plus, "chi2_catwise_minus": c_minus,
            "frac_random_below_plus": float(np.mean(c_rand < c_plus)),
            "frac_random_below_minus": float(np.mean(c_rand < c_minus)),
            "random_chi2_min_median_max": [float(c_rand.min()), float(np.median(c_rand)), float(c_rand.max())]}


def main():
    n_c, lb = catwise_axis_icrs()
    pts = points()
    with ProcessPoolExecutor(max_workers=len(pts)) as ex:
        rows = list(ex.map(run, pts))
    print(f"CatWISE excess axis (l, b) = ({lb[0]:.1f}, {lb[1]:.1f}); Pantheon+ unbinned, chi2 relative to the axis observer")
    print("  t0    theta  alpha   chi2_axis-LCDM   +n_C     -n_C    random[min, median, max]     frac(random < +n_C, -n_C)")
    for r in rows:
        ca = r["chi2_axis_observer"]
        mn, md, mx = (x - ca for x in r["random_chi2_min_median_max"])
        print(f"{r['t0']:5.2f}  {r['theta_obs']:5.2f}  {r['alpha_high']:.3f}   {ca - r['chi2_lcdm']:+7.2f}        "
              f"{r['chi2_catwise_plus'] - ca:+6.2f}   {r['chi2_catwise_minus'] - ca:+6.2f}   [{mn:+6.2f}, {md:+6.2f}, {mx:+6.2f}]"
              f"     {r['frac_random_below_plus']:.3f}, {r['frac_random_below_minus']:.3f}")
    (JSON / "pantheon_catwise_axis.json").write_text(json.dumps({"catwise_axis_lb": lb, "n_random": N_RANDOM, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
