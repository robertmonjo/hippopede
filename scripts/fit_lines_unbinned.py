"""Single model lines fitted to the chronometers and the unbinned Pantheon+ supernovae, with the plain
chi2 and with a region-weighted statistic, and what the quasar dipoles say at each fit.

Lines (alpha_high, t0 fitted; the ball curves for an observer at theta_obs = THETA_OBS):
  axis    the sector theta = 0 alone (hyperconical model; independent of t0);
  mean    the average over the light-cone ball: <E> for the chronometers, <D_C> for the supernovae;
  median  the weighted median of E over the ball, D_C = int dz / E_median.
Statistics:
  chi2    CC + unbinned SN, full covariances, H0 and M analytic;
  Q       CC + sum_r w_r chi2_r over SN redshift regions REGIONS, each region with its own covariance
          block (cross-region covariance dropped), w_r = mean(N) / N_r so that every region counts the
          same, one magnitude offset M minimising Q.  Q is not a likelihood: Delta Q has no chi2
          distribution and would need calibration with simulations.
Dipoles at each fit (alpha_high, t0), from the runs of quasar_dipole_fit.py / quaia_zslice_model.py:
  theta_obs required by the CatWISE excess and the chi2 (amplitude) of the three Quaia slices at their
  best theta_obs (scalar amplitudes along the axis); for the axis line the observer is on the axis and
  there is no dipole.  Writes figures/fit_lines_unbinned.json.
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

import compare_ball_curves as C  # noqa: E402
import fit_ball_unbinned as F  # noqa: E402
import fit_observer_ball as OB  # noqa: E402

FIG = ROOT / "figures"
THETA_OBS = 1.0
REGIONS = ((0.0, 0.1), (0.1, 0.4), (0.4, 0.8), (0.8, 3.0))
_REG = None


def regions():
    global _REG
    if _REG is None:
        z, zh, mb, n, ci = F.sn()
        cov = np.linalg.inv(ci)
        sel = [(z >= a) & (z < b) for a, b in REGIONS]
        nr = np.array([s.sum() for s in sel])
        w = nr.mean() / nr
        _REG = [(s, np.linalg.inv(cov[np.ix_(s, s)]), wi) for s, wi in zip(sel, w)]
    return _REG


def q_sn(mu):
    z, zh, mb, n, ci = F.sn()
    r = mb - mu
    reg = regions()
    num = sum(w * np.ones(s.sum()) @ c @ r[s] for s, c, w in reg)
    den = sum(w * np.ones(s.sum()) @ c @ np.ones(s.sum()) for s, c, w in reg)
    m = num / den
    return float(sum(w * (r[s] - m) @ c @ (r[s] - m) for s, c, w in reg))


def line(kind, ah, t0):
    """E on ZW (for CC) and D_C on ZW (for SN)."""
    if kind == "axis":
        E, _, _ = OB.sector_table(ah, 30.0)
        e = E[0]
        return e, C.comoving(e)
    m = OB.model(ah, t0, THETA_OBS)
    if m is None:
        return None, None
    E, _, w, e_bar, dc_bar = m
    if kind == "mean":
        return e_bar, dc_bar
    e_med = np.array([OB._quantiles(E[:, j], w[:, j], [0.5])[0] for j in range(len(OB.ZW))])
    return e_med, C.comoving(e_med)


def stats(kind, ah, t0):
    e, dc = line(kind, ah, t0)
    if e is None or not np.all(np.isfinite(e)) or np.any(e <= 0):
        return np.inf, np.inf
    z, zh, *_ = F.sn()
    mu = 5 * np.log10((1 + zh) * np.interp(z, OB.ZW, dc))
    c_cc = F.chi2_cc(e)
    out = c_cc + F.chi2_sn_mu(mu), c_cc + q_sn(mu)
    OB._W_CACHE.clear(); OB._M_CACHE.clear()
    return out


def fit(args):
    kind, which = args
    k = 0 if which == "chi2" else 1
    if kind == "axis":
        r = minimize_scalar(lambda a: stats("axis", a, 30.0)[k], bounds=(0.3, 0.5), method="bounded", options={"xatol": 1e-4})
        return kind, which, float(r.x), None, float(r.fun)
    best = None
    for s in ([0.34, 1.8], [0.36, 2.4], [0.40, 3.5]):
        r = minimize(lambda p: stats(kind, p[0], p[1])[k], s, method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 120})
        if best is None or r.fun < best.fun:
            best = r
    return kind, which, float(best.x[0]), float(best.x[1]), float(best.fun)


def lcdm(k):
    z, zh, *_ = F.sn()

    def f(om):
        e = np.sqrt(om * (1 + OB.ZW) ** 3 + 1 - om)
        mu = 5 * np.log10((1 + zh) * np.interp(z, OB.ZW, C.comoving(e)))
        return F.chi2_cc(e) + (F.chi2_sn_mu(mu) if k == 0 else q_sn(mu))

    r = minimize_scalar(f, bounds=(0.15, 0.6), method="bounded", options={"xatol": 1e-5})
    return float(r.x), float(r.fun)


def dipoles(ah, t0):
    """theta_obs and chi2 of CatWISE (best over theta) and Quaia (scalar amplitudes, best over theta)."""
    th = np.linspace(0.0, 35.0, 701)
    cw = np.array([F.catwise_chi2(ah, t0, x) for x in th])
    i = int(np.nanargmin(cw)) if np.isfinite(cw).any() else None
    out = {"catwise_theta_obs": None if i is None else float(th[i]), "catwise_chi2": None if i is None else float(cw[i])}
    qa = []
    for a in F.DIPOLE_ALPHAS:
        qz = json.loads((FIG / ("quaia_zslice_model_ah" + f"{a:g}".replace(".", "p") + ".json")).read_text())
        amp = np.array([s["excess_amp"] for s in qz["slices"]]); sig = np.array([s["sigma"] for s in qz["slices"]])
        tq = sorted((float(k), v["f_1deg"]) for k, v in qz["model"].items())
        f = np.array([np.interp(np.log(t0), np.log([x[0] for x in tq]), [x[1][j] for x in tq]) for j in range(3)])
        chi = np.array([np.sum(((f * x - amp) / sig) ** 2) for x in th])
        qa.append(chi)
    al = np.array(F.DIPOLE_ALPHAS)
    if al[0] <= ah <= al[-1]:
        k = min(int(np.searchsorted(al, ah, side="right")) - 1, len(al) - 2)
        w = (ah - al[k]) / (al[k + 1] - al[k])
        chi = (1 - w) * qa[k] + w * qa[k + 1]
        j = int(np.argmin(chi))
        w_ = 1 / sig**2
        const = float(np.sum(((amp - (w_ * amp).sum() / w_.sum()) / sig) ** 2))   # one amplitude for the three slices
        out.update({"quaia_theta_obs": float(th[j]), "quaia_chi2": float(chi[j]), "quaia_chi2_const": const,
                    "quaia_chi2_none": float(np.sum((amp / sig) ** 2))})
    return out


def main():
    jobs = [(k, w) for k in ("axis", "mean", "median") for w in ("chi2", "Q")]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(fit, jobs))
    out = {"theta_obs": THETA_OBS, "regions": REGIONS, "lcdm": {}, "lines": []}
    for k, name in ((0, "chi2"), (1, "Q")):
        om, c = lcdm(k)
        out["lcdm"][name] = {"Omega_m": om, "value": c}
    for kind, which, ah, t0, val in res:
        row = {"line": kind, "statistic": which, "alpha_high": ah, "t0": t0, "value": val,
               "delta_vs_lcdm": val - out["lcdm"][which]["value"]}
        if kind != "axis":
            row.update(dipoles(ah, t0))
        out["lines"].append(row)
        print(f"{which:4s} {kind:6s}: alpha_high={ah:.4f} t0={t0 if t0 is None else round(t0, 3)}  "
              f"delta vs LCDM = {row['delta_vs_lcdm']:+.3f}  " + ("" if kind == "axis" else
              f"CatWISE theta={row['catwise_theta_obs']} chi2={row['catwise_chi2']};  Quaia theta={row.get('quaia_theta_obs')} chi2={row.get('quaia_chi2')}"), flush=True)
    (FIG / "fit_lines_unbinned.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
