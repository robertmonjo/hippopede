"""Fit of t0 in which the spread of expansion histories across sectors is a physical scatter.

Hypothesis: supernovae at a given redshift sample sectors of the lobe with weight proportional
to its volume, w(theta) ~ sin^2(2 theta), over 0 < theta < THETA_MAX_DEG = 70 deg (sector_geometry.py), so that the Hubble diagram has a
mean mu_bar(z; t0) and an intrinsic scatter s(z; t0) equal to the weighted mean and standard
deviation of mu_theta(z; t0) across sectors.  Each sector is taken as an independent projected
history, expressed in common units through H_c,theta(t0)/H_c,0(t0); sectors whose history does
not reach a given redshift are left out of the average at that redshift.

Data: unbinned Pantheon+ (Scolnic et al. 2022; Brout et al. 2022), z_HD > 0.01, Cepheid
calibrators excluded, corrected magnitudes m_b_corr with the full STAT+SYS covariance C
(download_pantheon_plus.py).  Likelihood, with the absolute magnitude M minimised analytically:
    -2 ln L(t0) = r^T (C + S)^-1 r + ln det(C + S),  r = m_b_corr - mu_bar - M,  S = diag(s^2).
The limit t0 -> infinity has no scatter and reduces to the axial history.  Output: profile,
best t0 and the Delta(-2 ln L) = 1 and 4 intervals.  Writes json/fit_t0_intrinsic_scatter.json.
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
import sector_geometry as G  # noqa: E402

THETA = np.arange(0.25, G.THETA_MAX_DEG, 0.5)
W = G.lobe_volume_weight(THETA)
Z_MAX = 2.1


def load_sn():
    src = ROOT / "data" / "pantheon_plus"
    d = np.genfromtxt(src / "Pantheon+SH0ES.dat", names=True, dtype=None, encoding="utf-8")
    with open(src / "Pantheon+SH0ES_STAT+SYS.cov") as f:
        n = int(f.readline())
        cov = np.fromstring(f.read(), sep="\n").reshape(n, n)
    sel = np.where((d["zHD"] > 0.01) & (d["zHD"] < Z_MAX) & (d["IS_CALIBRATOR"] == 0))[0]
    return d["zHD"][sel], d["zHEL"][sel], d["m_b_corr"][sel], cov[np.ix_(sel, sel)]


def mu_sectors(t0, z_hd, z_hel):
    """mu_theta(z) - 5 log10(c/H0) for every sector on the SN redshifts (NaN beyond reach)."""
    zz = HM.Z_WORK[HM.Z_WORK >= 0]
    h_ax = HM.h0_sector(0.0, t0)
    out = np.full((len(THETA), len(z_hd)), np.nan)
    for i, a in enumerate(THETA):
        e = HM.projected_sector(a, t0)[0][HM.Z_WORK >= 0] * HM.h0_sector(np.radians(a), t0) / h_ax
        ok = np.isfinite(e) & (e > 0)
        if ok.sum() < 3:
            continue
        zmax = zz[ok].max()
        inv = np.where(ok, 1.0 / np.where(ok, e, 1.0), np.nan)
        dc = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zz))])
        m = np.interp(z_hd, zz, dc)
        m[z_hd > zmax] = np.nan
        out[i] = 5.0 * np.log10((1.0 + z_hel) * m)
    return out


def neg2lnL(t0, data):
    z, zh, mb, C = data
    mu = mu_sectors(t0, z, zh)
    ok = np.isfinite(mu)
    w = np.where(ok, W[:, None], 0.0)
    wsum = w.sum(0)
    mbar = (np.where(ok, mu, 0.0) * w).sum(0) / wsum
    s2 = (np.where(ok, (mu - mbar) ** 2, 0.0) * w).sum(0) / wsum
    Ct = C + np.diag(s2)
    L = np.linalg.cholesky(Ct)
    one = np.ones_like(mb)
    a = np.linalg.solve(L, mb - mbar)
    b = np.linalg.solve(L, one)
    M = (b @ a) / (b @ b)
    r = a - M * b
    return float(r @ r + 2.0 * np.sum(np.log(np.diag(L)))), float(np.sqrt(s2).max()), float(wsum.min() / W.sum())


def main():
    data = load_sn()
    print(f"{len(data[0])} SNe")
    grid = np.geomspace(1.5, 200.0, 36)
    vals = [neg2lnL(t, data) for t in grid]
    c = np.array([v[0] for v in vals])
    i = int(np.argmin(c))
    f0 = lambda t: neg2lnL(t, data)[0]
    if 0 < i < len(grid) - 1:
        r = minimize_scalar(f0, bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-3})
        t_best, c_min = float(r.x), float(r.fun)
    else:
        t_best, c_min = float(grid[i]), float(c[i])
    interval = {}
    for name, lev in (("1sigma", 1.0), ("2sigma", 4.0)):
        f = lambda t: f0(t) - c_min - lev
        lo = brentq(f, grid[0], t_best, xtol=1e-3) if t_best > grid[0] and f(grid[0]) > 0 else None
        hi = brentq(f, t_best, grid[-1], xtol=1e-3) if t_best < grid[-1] and f(grid[-1]) > 0 else None
        interval[name] = [lo, hi]
    out = {"N_SN": int(len(data[0])), "t0_best": t_best, "neg2lnL_min": c_min, "interval": interval,
           "minimum_at_grid_edge": bool(i in (0, len(grid) - 1)),
           "profile": {"t0": grid.tolist(), "neg2lnL": c.tolist(), "max_scatter_mag": [v[1] for v in vals],
                       "min_weight_coverage": [v[2] for v in vals]}}
    (ROOT / "json" / "fit_t0_intrinsic_scatter.json").write_text(json.dumps(out, indent=1))
    print(f"t0 best = {t_best:.3f}  -2lnL = {c_min:.2f}  intervals: {interval}  edge = {out['minimum_at_grid_edge']}")
    for t, v in zip(grid[::5], vals[::5]):
        print(f"  t0={t:7.2f}  -2lnL={v[0]:.2f}  max scatter={v[1]:.4f} mag  coverage={v[2]:.2f}")


if __name__ == "__main__":
    main()
