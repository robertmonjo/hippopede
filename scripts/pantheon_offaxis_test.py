"""Directional test of the hippopede sectors with the unbinned Pantheon+ catalogue.

Observer at sector angle theta_obs; the projected axis points to n_p on the sky.  A source at
redshift z in direction n lies at angular distance rhat(z) from the observer on the lobe, hence
in the sector theta_s = chi_s/2 with
    cos chi_s = cos chi_obs cos rhat + sin chi_obs sin rhat (n . n_p),   chi_obs = 2 theta_obs
(sector_geometry.py); the sector changes along the line of sight.  All sectors share the time t0,
and their rates are expressed in common units through H_c,theta(t0)/H_c,0(t0).  Distance modulus:
    D_C = int_0^{z_HD} dz'/E_{theta_s(z')}(z'),   mu = 5 log10[(1 + z_hel) D_C] + M.
theta_obs = 0 is the axis observer, isotropic on the sky.

Data: Pantheon+ (Scolnic et al. 2022; Brout et al. 2022), z_HD > 0.01, Cepheid calibrators
excluded, m_b_corr with the full STAT+SYS covariance (download_pantheon_plus.py); M minimised
analytically.  For each t0 the axis observer is compared with observers at theta_obs up to
THETA_OBS_MAX_DEG = 35 deg and 192 axis directions; sector tables cover 0 <= theta <= THETA_MAX_DEG = 70 deg
(sector_geometry.py).  Also writes the volume-weighted spread of mu across sectors at the
reference t0 (figures/t0_summary.json).  Writes figures/pantheon_offaxis_test.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import hippopede_model as HM  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import sector_geometry as G  # noqa: E402

RUN = lambda z: PH.alpha_sqrt(z, PH.ALPHA_LOW, PH.load_alpha_high())
Z_MIN, Z_MAX = 0.01, 2.1
Z_TAB = np.linspace(0.0, 2.15, 431)
TH_TAB = np.arange(0.0, G.THETA_MAX_DEG + 0.01, 1.0)
N_PATH = 120
THETA_OBS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, G.THETA_OBS_MAX_DEG)
h0_sector = HM.h0_sector


def load():
    src = ROOT / "data" / "pantheon_plus"
    d = np.genfromtxt(src / "Pantheon+SH0ES.dat", names=True, dtype=None, encoding="utf-8")
    with open(src / "Pantheon+SH0ES_STAT+SYS.cov") as f:
        n = int(f.readline())
        cov = np.fromstring(f.read(), sep="\n").reshape(n, n)
    m = np.where((d["zHD"] > Z_MIN) & (d["zHD"] < Z_MAX) & (d["IS_CALIBRATOR"] == 0))[0]
    ra, dec = np.radians(d["RA"][m]), np.radians(d["DEC"][m])
    nvec = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)
    return d["zHD"][m], d["zHEL"][m], d["m_b_corr"][m], cov[np.ix_(m, m)], nvec


def e_table(t0):
    """Projected E(theta, z) of every sector in the observer's units (NaN beyond reach)."""
    tab = np.empty((len(TH_TAB), len(Z_TAB)))
    h_ax = h0_sector(0.0, t0)
    for i, a in enumerate(TH_TAB):
        e = HM.projected_sector(a, t0)[0]
        tab[i] = np.interp(Z_TAB, HM.Z_WORK, e, left=np.nan, right=np.nan) * h0_sector(np.radians(a), t0) / h_ax
    return tab


def bilinear(tab, th_deg, z):
    th = np.asarray(th_deg, dtype=float)
    fi = np.clip(np.interp(th, TH_TAB, np.arange(len(TH_TAB))), 0, len(TH_TAB) - 1.000001)
    fj = np.clip(z / (Z_TAB[1] - Z_TAB[0]), 0, len(Z_TAB) - 1.000001)
    i, j = fi.astype(int), fj.astype(int)
    a, b = fi - i, fj - j
    out = ((1 - a) * (1 - b) * tab[i, j] + a * (1 - b) * tab[i + 1, j]
           + (1 - a) * b * tab[i, j + 1] + a * b * tab[i + 1, j + 1])
    return np.where(th <= TH_TAB[-1] + 1e-9, out, np.nan)  # sectors beyond THETA_MAX_DEG are not modelled


class Sky:
    """Lines of sight to sources at redshifts z (z_hel for the luminosity factor) in directions n."""

    def __init__(self, z, n, z_hel=None, mu=None, cov=None):
        self.z, self.n = z, n
        self.zh = z if z_hel is None else z_hel
        self.mu = mu
        self.cinv = None if cov is None else np.linalg.inv(cov)
        s = np.linspace(0.0, 1.0, N_PATH)
        self.zp = z[:, None] * s[None, :]
        self.r = PH.rhat_of_z(self.zp.ravel(), RUN).reshape(self.zp.shape)

    def sector_path(self, theta_obs_deg, cpsi):
        """Sector [deg] crossed along each line of sight (sector_geometry.source_sector_deg)."""
        return G.source_sector_deg(theta_obs_deg, self.r, np.asarray(cpsi, dtype=float)[:, None])

    def distance(self, tab, theta_obs_deg, n_pole):
        e = bilinear(tab, self.sector_path(theta_obs_deg, self.n @ n_pole), self.zp)
        if not np.all(np.isfinite(e)) or np.any(e <= 0):
            return None
        return np.trapezoid(1.0 / e, self.zp, axis=1)

    def chi2_model(self, model):
        r = self.mu - model
        one = np.ones_like(r)
        M = (one @ self.cinv @ r) / (one @ self.cinv @ one)
        d = r - M
        return float(d @ self.cinv @ d)

    def chi2(self, tab, theta_obs_deg, n_pole):
        dc = self.distance(tab, theta_obs_deg, n_pole)
        if dc is None:
            return np.inf
        return self.chi2_model(5.0 * np.log10((1.0 + self.zh) * dc))


def fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    lam = np.pi * (1 + 5**0.5) * k
    return np.stack([np.cos(lam) * np.sin(phi), np.sin(lam) * np.sin(phi), np.cos(phi)], axis=1)


def lcdm_model(sky, om):
    zz = np.linspace(0, sky.z.max(), 4000)
    inv = 1 / np.sqrt(om * (1 + zz) ** 3 + 1 - om)
    dc = np.interp(sky.z, zz, np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zz))]))
    return 5 * np.log10((1 + sky.zh) * dc)


def mu_spread(t0, z_nodes):
    """Volume-weighted spread of mu across sectors (independent histories), in magnitudes."""
    zz = HM.Z_WORK[HM.Z_WORK >= 0]
    th = np.arange(0.25, G.THETA_MAX_DEG, 0.5)
    w = G.lobe_volume_weight(th)
    h_ax = h0_sector(0.0, t0)
    mus = []
    for a in th:
        e = HM.projected_sector(a, t0)[0][HM.Z_WORK >= 0] * h0_sector(np.radians(a), t0) / h_ax
        inv = 1.0 / e
        dc = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zz))])
        mus.append(5 * np.log10((1 + z_nodes) * np.interp(z_nodes, zz, dc)))
    mus = np.array(mus)
    ok = np.isfinite(mus)
    ww = np.where(ok, w[:, None], 0.0)
    mbar = (np.where(ok, mus, 0) * ww).sum(0) / ww.sum(0)
    return np.sqrt((np.where(ok, (mus - mbar) ** 2, 0) * ww).sum(0) / ww.sum(0)), ww.sum(0) / w.sum()


def main():
    z, zh, mb, cov, n = load()
    sky = Sky(z, n, zh, mb, cov)
    oms = np.linspace(0.2, 0.5, 301)
    c_l = np.array([sky.chi2_model(lcdm_model(sky, om)) for om in oms])
    om_best, c_lcdm = float(oms[np.argmin(c_l)]), float(c_l.min())
    resid = mb - lcdm_model(sky, om_best)
    resid -= np.average(resid)
    print(f"Pantheon+ SNe used: {len(z)};  flat LCDM: Omega_m = {om_best:.3f}, chi2 = {c_lcdm:.1f}")
    edges = np.array([0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 2.1])
    rms = [{"z": [float(lo), float(hi)], "N": int(((z >= lo) & (z < hi)).sum()),
            "rms_resid": float(np.std(resid[(z >= lo) & (z < hi)]))} for lo, hi in zip(edges[:-1], edges[1:])]
    t_ref = float(json.loads((ROOT / "figures" / "t0_summary.json").read_text())["figure_t0"])
    z_nodes = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.3])
    sp, cover = mu_spread(t_ref, z_nodes)
    print(f"mu spread across sectors at t0 = {t_ref}: {dict(zip(z_nodes.tolist(), np.round(sp, 4).tolist()))} "
          f"(weight coverage {np.round(cover, 2).tolist()})")
    poles = fibonacci_sphere(192)
    rows = []
    for t0 in sorted({t_ref, 2.5, 3.0, 4.0, 6.0, 10.0}):
        tab = e_table(t0)
        c_ax = sky.chi2(tab, 0.0, poles[0])
        best = (c_ax, 0.0, None)
        for th_o in THETA_OBS:
            for k, npole in enumerate(poles):
                c = sky.chi2(tab, th_o, npole)
                if c < best[0]:
                    best = (c, th_o, k)
        radec = None
        if best[2] is not None:
            v = poles[best[2]]
            radec = [float(np.degrees(np.arctan2(v[1], v[0])) % 360), float(np.degrees(np.arcsin(v[2])))]
        rows.append({"t0": t0, "chi2_axis": c_ax, "chi2_best": best[0], "theta_obs_best": best[1], "axis_radec": radec,
                     "dchi2_axis_vs_lcdm": c_ax - c_lcdm, "dchi2_offaxis_vs_axis": best[0] - c_ax})
        print(f"t0={t0}: axis observer chi2 = {c_ax:.2f} (vs LCDM {c_ax - c_lcdm:+.2f}); best off-axis "
              f"{best[0]:.2f} (dchi2 = {best[0] - c_ax:+.2f}) at theta_obs = {best[1]}, axis (RA,Dec) = {radec}")
    out = {"N": int(len(z)), "lcdm": {"Omega_m": om_best, "chi2": c_lcdm}, "rms_residuals": rms,
           "mu_spread": {"t0": t_ref, "z": z_nodes.tolist(), "spread_mag": sp.tolist(), "coverage": cover.tolist()},
           "grid": rows}
    (ROOT / "figures" / "pantheon_offaxis_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
