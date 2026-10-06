"""Joint fit of the light-cone ball model to the chronometers and the unbinned Pantheon+ supernovae.

Model (fit_observer_ball.py, reading (iv) of the paper): an observer in the sector theta_obs receives
the average over the ball rho <= rhat(z) around it.  The chronometers see the ball average <E>(z).  Each
supernova i at redshift z_i in direction n_i has the comoving distance
    D_C,i = <D_C>(z_i) * D_los(z_i, n_i) / <D_los>(z_i),
the ball average modulated by the line-of-sight distance through the sectors crossed in its direction
(pantheon_offaxis_test.Sky, Eq. dc_direction of the paper) relative to its average over the sky at the
same redshift; for theta_obs = 0 the modulation is 1.  mu_i = 5 log10[(1 + z_hel,i) D_C,i] + M.
The projected axis points opposite to the CatWISE excess (the model count dipole is negative).

Likelihood: 38 chronometers with their covariance (H0 analytic) + 1579 unbinned Pantheon+ supernovae
(0.01 < z_HD < 2.1, no Cepheid calibrators, STAT+SYS covariance, M analytic).  The binned supernovae are
not used.  Optionally the CatWISE amplitude, ((|D| - D_geo)/sigma)^2, from the dipole runs of
quasar_dipole_fit.py at alpha_high = 0.30 ... 0.44 (interpolated in alpha_high, ln t0 and theta_obs).
References with the same data: flat LCDM (Omega_m free) and the hyperconical model (the axial sector,
alpha_high free; it does not depend on t0).

Steps: grid in (alpha_high, t0, theta_obs), Nelder-Mead refinement from the best grid point, free-axis
check at the minimum (2000 random axes).  Writes figures/fit_ball_unbinned.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_observer_ball as OB  # noqa: E402
import likelihood as L  # noqa: E402
import pantheon_catwise_axis as PCA  # noqa: E402
import pantheon_offaxis_test as T  # noqa: E402

FIG = ROOT / "figures"
A_GRID = np.round(np.arange(0.30, 0.4301, 0.01), 3)
T_GRID = np.round(np.geomspace(1.5, 8.0, 22), 4)
TH_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0, 20.0, 25.0])
DIPOLE_ALPHAS = (0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44)
DIPOLE_SUFFIX = ""   # suffix of the dipole runs after _ah<value> (set_dipole_runs)
Z_AVG = np.linspace(0.01, 1.92, 60)          # redshifts of the sky average of D_los
U_PSI, W_PSI = np.polynomial.legendre.leggauss(24)   # cos(psi) nodes for that average

_SN = None


def sn():
    """Pantheon+ data and the inverse covariance (loaded once per process)."""
    global _SN
    if _SN is None:
        z, zh, mb, cov, n = T.load()
        _SN = (z, zh, mb, n, np.linalg.inv(cov))
    return _SN


def los_table(ah, t0):
    """Sector table of fit_observer_ball on the grid of pantheon_offaxis_test (TH_TAB x Z_TAB)."""
    E, _, _ = OB.sector_table(ah, t0)
    th = np.clip(T.TH_TAB, OB.THETA[0], OB.THETA[-1])
    rows = np.array([np.interp(th, OB.THETA, E[:, j]) for j in range(E.shape[1])]).T     # TH_TAB x ZW
    return np.array([np.interp(T.Z_TAB, OB.ZW, r, right=np.nan) for r in rows])


def chi2_cc(e_bar):
    zc, hc, _ = L.CC
    e = np.interp(zc, OB.ZW, e_bar)
    h0 = (e @ L.CINV_CC @ hc) / (e @ L.CINV_CC @ e)
    r = hc - h0 * e
    return float(r @ L.CINV_CC @ r)


def chi2_sn_mu(mu_model):
    z, zh, mb, n, ci = sn()
    r = mb - mu_model
    one = np.ones_like(r)
    m = (one @ ci @ r) / (one @ ci @ one)
    d = r - m
    return float(d @ ci @ d)


class Point:
    """Everything that depends on (alpha_high, t0): ball model, line-of-sight tables, skies."""

    def __init__(self, ah, t0):
        self.ah, self.t0 = ah, t0
        T.set_alpha_high(ah)
        z, zh, mb, n, _ = sn()
        self.sky = T.Sky(z, n, zh)
        zz = np.repeat(Z_AVG, len(U_PSI))
        cps = np.tile(U_PSI, len(Z_AVG))
        nn = np.stack([np.sqrt(1 - cps**2), np.zeros_like(cps), cps], axis=1)
        self.avg_sky = T.Sky(zz, nn)
        self.cps = cps
        self.tab = los_table(ah, t0)

    def modulation(self, theta_obs, n_axis):
        if theta_obs == 0:
            return np.ones(len(self.sky.z))
        d = self.sky.distance(self.tab, theta_obs, n_axis)
        e = T.bilinear(self.tab, self.avg_sky.sector_path(theta_obs, self.cps), self.avg_sky.zp)
        if d is None or not np.all(np.isfinite(e)) or np.any(e <= 0):
            return None
        da = np.trapezoid(1.0 / e, self.avg_sky.zp, axis=1).reshape(len(Z_AVG), len(U_PSI))
        mean = (da * W_PSI[None, :]).sum(1) / W_PSI.sum()        # average over cos(psi) in [-1, 1]
        return d / np.interp(self.sky.z, Z_AVG, mean)

    def chi2(self, theta_obs, n_axis):
        m = OB.model(self.ah, self.t0, theta_obs)
        if m is None:
            return np.inf, np.inf
        _, _, _, e_bar, dc_bar = m
        mod = self.modulation(theta_obs, n_axis)
        if mod is None:
            return np.inf, np.inf
        z, zh, *_ = sn()
        mu = 5 * np.log10((1 + zh) * np.interp(z, OB.ZW, dc_bar) * mod)
        return chi2_cc(e_bar), chi2_sn_mu(mu)


def axis_opposite_catwise():
    n_c, _ = PCA.catwise_axis_icrs()
    return -n_c


_DIP = None


def set_dipole_runs(alphas, suffix=""):
    """Use the dipole runs quasar_dipole_fit_ah<alpha><suffix>.json for these alpha_high values."""
    global DIPOLE_ALPHAS, DIPOLE_SUFFIX, _DIP
    DIPOLE_ALPHAS, DIPOLE_SUFFIX, _DIP = tuple(alphas), suffix, None


def catwise_chi2(ah, t0, theta):
    """CatWISE amplitude chi2 interpolated between the dipole runs (alpha_high, ln t0, theta_obs)."""
    global _DIP
    if _DIP is None:
        _DIP = {}
        for a in DIPOLE_ALPHAS:
            qd = json.loads((FIG / ("quasar_dipole_fit_ah" + f"{a:g}".replace(".", "p") + DIPOLE_SUFFIX + ".json")).read_text())
            runs = sorted((r for r in qd["runs"] if r["pz"] == "gamma" and abs(r["x"] - qd["x"]) < 1e-12), key=lambda r: r["t0"])
            _DIP[a] = (np.log([r["t0"] for r in runs]), np.abs(np.array([r["D"] for r in runs], float)),
                       np.array(qd["theta_obs_grid"], float), qd["D_geo"], qd["sigma"])
    al = np.array(DIPOLE_ALPHAS)
    if not al[0] <= ah <= al[-1]:
        return np.nan
    k = min(int(np.searchsorted(al, ah, side="right")) - 1, len(al) - 2)
    w = (ah - al[k]) / (al[k + 1] - al[k])
    vals = []
    for a in (al[k], al[k + 1]):
        lt, D, g, dgeo, sig = _DIP[a]
        x = np.log(t0)
        n = min(int(np.searchsorted(lt, x, side="right")) - 1, len(lt) - 2) if x <= lt[-1] + 1e-12 else len(lt)
        if not 0 <= n < len(lt) - 1:
            return np.nan
        f = (x - lt[n]) / (lt[n + 1] - lt[n])
        row = (1 - f) * D[n] + f * D[n + 1]
        ok = np.isfinite(row)
        if theta > g[ok].max():
            return np.nan
        vals.append(np.interp(theta, g[ok], row[ok]))
    d = (1 - w) * vals[0] + w * vals[1]
    return float(((d - dgeo) / sig) ** 2)


def grid_task(args):
    ah, t0 = args
    p = Point(ah, t0)
    n_ax = axis_opposite_catwise()
    out = [(th, *p.chi2(th, n_ax)) for th in TH_GRID]
    OB._T_CACHE.clear(); OB._W_CACHE.clear(); OB._M_CACHE.clear()
    return ah, t0, out


def total(params, with_catwise=False):
    ah, t0, th = params[0], params[1], abs(params[2])
    p = Point(ah, t0)
    c_cc, c_sn = p.chi2(th, axis_opposite_catwise())
    OB._W_CACHE.clear(); OB._M_CACHE.clear()
    c = c_cc + c_sn
    if with_catwise:
        cw = catwise_chi2(ah, t0, th)
        c = c + (cw if np.isfinite(cw) else 1e6)
    return c


def lcdm_reference():
    z, zh, *_ = sn()
    zc, hc, _ = L.CC

    def f(om):
        zz = np.linspace(0, 2.0, 4001)
        e = np.sqrt(om * (1 + zz) ** 3 + 1 - om)
        inv = 1 / e
        dc = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zz))])
        ec = np.interp(zc, zz, e)
        h0 = (ec @ L.CINV_CC @ hc) / (ec @ L.CINV_CC @ ec)
        r = hc - h0 * ec
        return float(r @ L.CINV_CC @ r) + chi2_sn_mu(5 * np.log10((1 + zh) * np.interp(z, zz, dc)))

    r = minimize_scalar(f, bounds=(0.15, 0.6), method="bounded", options={"xatol": 1e-5})
    return float(r.x), float(r.fun)


def hyperconical_reference():
    z, zh, *_ = sn()

    def f(ah):
        E, _, _ = OB.sector_table(ah, 30.0)
        e = E[0]
        inv = 1 / e
        dc = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(OB.ZW))])
        return chi2_cc(e) + chi2_sn_mu(5 * np.log10((1 + zh) * np.interp(z, OB.ZW, dc)))

    r = minimize_scalar(f, bounds=(0.3, 0.5), method="bounded", options={"xatol": 1e-4})
    return float(r.x), float(r.fun)


def main():
    out = {}
    om, c_l = lcdm_reference()
    ah_h, c_h = hyperconical_reference()
    out["lcdm"] = {"Omega_m": om, "chi2": c_l}
    out["hyperconical"] = {"alpha_high": ah_h, "chi2": c_h, "dchi2_vs_lcdm": c_h - c_l}
    print(f"flat LCDM: Omega_m = {om:.4f}, chi2 = {c_l:.3f};  hyperconical: alpha_high = {ah_h:.4f}, dchi2 = {c_h - c_l:+.3f}", flush=True)

    tasks = [(float(a), float(t)) for a in A_GRID for t in T_GRID]
    grid = []
    with ProcessPoolExecutor(max_workers=26) as ex:
        for ah, t0, rows in ex.map(grid_task, tasks):
            for th, c_cc, c_sn in rows:
                grid.append([ah, t0, th, c_cc, c_sn])
    g = np.array(grid)
    tot = g[:, 3] + g[:, 4]
    out["grid"] = g.tolist()
    res = {}
    for name, wc in (("cc_sn", False), ("cc_sn_catwise", True)):
        tt = tot + (np.array([catwise_chi2(a, t, th) for a, t, th in g[:, :3]]) if wc else 0)
        tt = np.where(np.isfinite(tt), tt, np.inf)
        i = int(np.argmin(tt))
        start = g[i, :3]
        print(f"{name}: grid minimum alpha={start[0]:.3f} t0={start[1]:.3f} theta={start[2]:.2f} chi2={tt[i]:.3f}", flush=True)
        simplex = np.array([start, start + [0.005, 0, 0], start + [0, 0.08 * start[1], 0], start + [0, 0, max(0.5, 0.2 * start[2])]])
        r = minimize(lambda p: total(p, wc), start, method="Nelder-Mead",
                     options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-3, "maxiter": 150})
        ah, t0, th = float(r.x[0]), float(r.x[1]), abs(float(r.x[2]))
        c_cc, c_sn = Point(ah, t0).chi2(th, axis_opposite_catwise())
        cw = catwise_chi2(ah, t0, th)
        res[name] = {"alpha_high": ah, "t0": t0, "theta_obs": th, "chi2_cc": c_cc, "chi2_sn": c_sn,
                     "chi2_cc_sn": c_cc + c_sn, "dchi2_cc_sn_vs_lcdm": c_cc + c_sn - c_l,
                     "dchi2_vs_hyperconical": c_cc + c_sn - c_h, "chi2_catwise": cw}
        print(f"{name}: alpha_high={ah:.4f} t0={t0:.3f} theta_obs={th:.2f}  CC+SN chi2={c_cc + c_sn:.3f} "
              f"(vs LCDM {c_cc + c_sn - c_l:+.3f}, vs hyperconical {c_cc + c_sn - c_h:+.3f}); CatWISE chi2 = {cw:.3f}", flush=True)
    out["fits"] = res
    # isotropic part alone: theta_obs = 0 (no direction dependence), alpha_high and t0 free
    i0 = np.where(g[:, 2] == 0)[0]
    j = i0[int(np.argmin(tot[i0]))]
    r0 = minimize(lambda p: total([p[0], p[1], 0.0]), g[j, :2], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 120})
    out["isotropic"] = {"alpha_high": float(r0.x[0]), "t0": float(r0.x[1]), "chi2": float(r0.fun), "dchi2_vs_lcdm": float(r0.fun - c_l),
                        "dchi2_vs_hyperconical": float(r0.fun - c_h)}
    print(f"theta_obs = 0: alpha_high={r0.x[0]:.4f} t0={r0.x[1]:.3f} dchi2 vs LCDM {r0.fun - c_l:+.3f}, vs hyperconical {r0.fun - c_h:+.3f}", flush=True)
    # free-axis check at the CC+SN+CatWISE minimum
    b = res["cc_sn_catwise"]
    if b["theta_obs"] > 0.05:
        p = Point(b["alpha_high"], b["t0"])
        rng = np.random.default_rng(7)
        axes = rng.normal(size=(2000, 3))
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        cs = np.array([sum(p.chi2(b["theta_obs"], v)) for v in axes])
        c_fix = sum(p.chi2(b["theta_obs"], axis_opposite_catwise()))
        k = int(np.argmin(cs))
        ang = float(np.degrees(np.arccos(np.clip(axes[k] @ axis_opposite_catwise(), -1, 1))))
        out["free_axis"] = {"chi2_fixed_axis": c_fix, "chi2_best_random": float(cs[k]), "frac_random_below_fixed": float(np.mean(cs < c_fix)),
                            "angle_best_to_fixed_deg": ang}
        print(f"free axis at that point: fixed {c_fix:.3f}, best of 2000 random {cs[k]:.3f} (angle {ang:.0f} deg), "
              f"fraction of random axes below the fixed one {np.mean(cs < c_fix):.3f}", flush=True)
    (FIG / "fit_ball_unbinned.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
