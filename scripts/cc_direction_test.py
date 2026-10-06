"""Direction dependence of the cosmic chronometers relative to the CatWISE excess axis.

Each chronometer is given the mean unit vector of the fields its galaxies come from (sources in
data/cc_compilation/sources.md).  Narrow fields (COSMOS, GOODS-S/CDFS, UDS, GOODS-N, the four GDDS
fields, the radio galaxies 53W091/53W069, the MUSE cluster fields) enter with their centres; wide
surveys (SDSS, BOSS, DESI, 2SLAQ, ACT) with the mean vector of an approximate footprint, so that their
projection on any axis is diluted; Stern et al. (2010) stack 24 clusters over the sky and get no
direction.  The fractional residuals about flat LCDM fitted to the chronometers alone,
delta_i = (H_i - H_fit)/H_fit, are fitted as delta = A mu_i, mu_i = <v_i> . n_C, with the full
covariance; n_C is the CatWISE excess axis (quasar_dipole_fit.excess_vector).  The significance is
calibrated with random axes, and a dipole of free direction is also fitted.  The prediction of the
model is the relative difference of E(z) between the lines of sight towards and away from the axis.
Writes json/cc_direction_test.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import likelihood as L  # noqa: E402
import quasar_dipole_fit as QD  # noqa: E402

RNG = np.random.default_rng(1)


def gal_vec(ra, dec):
    c = SkyCoord(ra=np.atleast_1d(ra) * u.deg, dec=np.atleast_1d(dec) * u.deg, frame="icrs").galactic
    l, b = c.l.rad, c.b.rad
    return np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], axis=-1)


def box(ra0, ra1, d0, d1, n=4000):
    """Points uniform on the sphere inside an RA/Dec box (RA may wrap through 0)."""
    if ra1 < ra0:
        ra1 += 360
    ra = RNG.uniform(ra0, ra1, n) % 360
    s = RNG.uniform(np.sin(np.radians(d0)), np.sin(np.radians(d1)), n)
    return gal_vec(ra, np.degrees(np.arcsin(s)))


def mean_vec(parts):
    return sum(w * np.atleast_2d(x).mean(axis=0) for w, x in parts) / sum(w for w, _ in parts)


FIELD = {
    "COSMOS": gal_vec(150.12, 2.21), "CDFS": gal_vec(53.12, -27.81), "UDS": gal_vec(34.5, -5.1),
    "GOODSN": gal_vec(189.23, 62.24), "GDDS02": gal_vec(32.42, -4.63), "GDDS12": gal_vec(181.34, -7.37),
    "GDDS15": gal_vec(230.95, -0.09), "GDDS22": gal_vec(334.42, 0.26), "53W": gal_vec(260.64, 50.10),
    "J2222": gal_vec(335.5, 27.8), "M1149": gal_vec(177.4, 22.4), "J1029": gal_vec(157.3, 26.4),
}
SDSS = [(1, box(110, 265, -5, 70))]
BOSS = [(0.75, box(110, 265, -5, 60)), (0.25, box(315, 45, -10, 35))]
DESI = [(0.67, box(100, 280, -10, 80)), (0.33, box(300, 60, -20, 35))]
TWOSLAQ = [(0.5, box(123, 243, -1.25, 1.25)), (0.5, box(309, 59, -1.25, 1.25))]
ACT = [(1, box(0, 360, -60, 20))]
GDDS = [(1, FIELD[k]) for k in ("GDDS02", "GDDS12", "GDDS15", "GDDS22")]
DEEP = [(1, FIELD["COSMOS"]), (1, FIELD["CDFS"]), (1, FIELD["UDS"])] + [(0.25, x) for _, x in GDDS]
ASSIGN = {
    "Zhang2014": SDSS, "Jimenez2003": SDSS, "Moresco2016": BOSS, "Loubser2025b": DESI,
    "Ratsimbazafy2017": TWOSLAQ, "Loubser2025a": ACT,
    "Tomasetti2025": [(1, FIELD["J2222"]), (1, FIELD["M1149"]), (1, FIELD["J1029"])],
    "Borghi2022": [(1, FIELD["COSMOS"])], "Tomasetti2023": [(1, FIELD["CDFS"]), (1, FIELD["UDS"])],
    "Moresco2015": [(1, FIELD["COSMOS"]), (1, FIELD["UDS"]), (1, FIELD["CDFS"]), (1, FIELD["GOODSN"])],
    "Stern2010": None,
}


def direction(ref, z):
    if ref == "Moresco2012":      # SDSS below z = 0.45, deep fields above
        return mean_vec(SDSS if z < 0.45 else DEEP)
    if ref == "Simon2005":        # Treu et al. field ellipticals (GOODS-N) below z = 0.5, GDDS and radio galaxies above
        return mean_vec([(1, FIELD["GOODSN"])] if z < 0.5 else GDDS + [(1, FIELD["53W"])])
    a = ASSIGN[ref]
    return np.zeros(3) if a is None else mean_vec(a)


def main():
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "cc_compilation" / "cc_compilation.csv", encoding="utf-8"))
            if r["use_default"] == "1"]
    z, h, _ = L.CC
    cov = np.linalg.inv(L.CINV_CC)
    V = np.zeros((len(z), 3))
    ref = [""] * len(z)
    for r in rows:
        i = int(np.argmin(np.abs(z - float(r["z"])) + 1e3 * (np.abs(h - float(r["H"])) > 1e-6)))
        V[i], ref[i] = direction(r["reference_key"], float(r["z"])), r["reference_key"]
    model = lambda p: p[0] * np.sqrt(p[1] * (1 + z) ** 3 + 1 - p[1])
    p = minimize(lambda p: (h - model(p)) @ L.CINV_CC @ (h - model(p)), [68, 0.3], method="Nelder-Mead").x
    hf = model(p)
    d = (h - hf) / hf
    cd = cov / np.outer(hf, hf)
    _, _, (l0, b0) = QD.excess_vector()
    l0, b0 = np.radians(l0), np.radians(b0)
    n_c = np.array([np.cos(b0) * np.cos(l0), np.cos(b0) * np.sin(l0), np.sin(b0)])

    def amplitude(mu, sel):
        ci = np.linalg.inv(cd[np.ix_(sel, sel)])
        f = mu[sel] @ ci @ mu[sel]
        a = (mu[sel] @ ci @ d[sel]) / f
        return a, 1 / np.sqrt(f), a**2 * f

    rand = RNG.normal(size=(4000, 3))
    rand /= np.linalg.norm(rand, axis=1)[:, None]
    out = {"lcdm_cc_only": p.tolist(), "points": [], "fits": {}}
    for i in np.argsort(z):
        out["points"].append({"z": float(z[i]), "ref": ref[i], "mean_vector_norm": float(np.linalg.norm(V[i])),
                              "mu": float(V[i] @ n_c), "delta": float(d[i]), "sigma_delta": float(np.sqrt(cd[i, i]))})
    for name, sel in (("all", np.ones(len(z), bool)), ("z>0.5", z > 0.5), ("z<0.5", z < 0.5)):
        a, sa, dchi = amplitude(V @ n_c, sel)
        null = np.array([amplitude(V @ v, sel)[2] for v in rand])
        ci = np.linalg.inv(cd[np.ix_(sel, sel)])
        fm = V[sel].T @ ci @ V[sel]
        dv = np.linalg.solve(fm, V[sel].T @ ci @ d[sel])
        out["fits"][name] = {"A": float(a), "sigma_A": float(sa), "dchi2": float(dchi), "frac_random_larger": float(np.mean(null >= dchi)),
                             "free_dipole_amplitude": float(np.linalg.norm(dv)), "free_dipole_dchi2_3dof": float(dv @ fm @ dv)}
        print(f"{name:6s}: A = {a:+.3f} +- {sa:.3f}; random axes with larger Delta chi2: {np.mean(null >= dchi):.2f}; "
              f"free dipole |D| = {np.linalg.norm(dv):.3f}, Delta chi2 = {dv @ fm @ dv:.2f} (3 dof)")
    out["prediction"] = predicted_amplitude()
    (ROOT / "json" / "cc_direction_test.json").write_text(json.dumps(out, indent=1))


def predicted_amplitude(configs=((0.3347, 1.7412, 1.0), (0.4034, 3.935, 15.0)), zs=(0.5, 1.0, 1.5)):
    """(E_towards - E_away)/(E_towards + E_away) along the projected axis, for (alpha_high, t0, theta_obs)
    on the CatWISE curve (fit_observer_ball.py, plot_observer_ball_landscape.py)."""
    import fit_observer_ball as OB
    import projected_hyperconical as PH
    import sector_geometry as G
    res = []
    for ah, t0, th in configs:
        E, _, _ = OB.sector_table(ah, t0)
        rhat = PH.rhat_of_z(np.array(zs), lambda x: PH.alpha_sqrt(x, PH.ALPHA_LOW, ah))
        amp = []
        for z, r in zip(zs, rhat):
            j = int(np.argmin(np.abs(OB.ZW - z)))
            e = [np.interp(float(G.source_sector_deg(th, np.array([r]), np.array([c]))[0]), OB.THETA, E[:, j]) for c in (1.0, -1.0)]
            amp.append(float((e[0] - e[1]) / (e[0] + e[1])))
        res.append({"alpha_high": ah, "t0": t0, "theta_obs": th, "z": list(zs), "A": amp})
        print(f"prediction alpha_high={ah}, t0={t0}, theta_obs={th}: A(z={zs}) = {np.round(amp, 4).tolist()}")
    return res


if __name__ == "__main__":
    main()
