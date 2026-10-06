"""Light-cone ball around an observer away from the axis: both readings of the sector weights at once.

The observer sits in the sector theta_obs.  At redshift z the contributing points of the lobe are
those within angular distance rho <= rhat(z) of the observer, in every direction psi, weighted by
their volume on the unit 3-sphere, sin^2(rho) d rho d(cos psi).  A point at (rho, psi) lies in the
sector theta given by
    cos 2 theta = cos 2 theta_obs cos rho + sin 2 theta_obs sin rho cos psi
(sector_geometry.py).  theta_obs = 0 gives the ball of fit_lightcone_ball.py; keeping only the shell
rho = rhat(z) gives the sky of one observer (fit_observer_sky.py).  Each point contributes the
independent projected history of its sector, in the observer's units (H_c,theta(t0)/H_c,theta_obs(t0)):
<E>(z) for the chronometers, <D_C,theta(z)> for the supernovae, and weighted quantiles of q_theta(z)
for the band, compared with GaPP through M (t0_dispersion_match.py).

Usage:
    python scripts/fit_observer_ball.py --scan                (t0, theta_obs) grid at alpha_high = 0.36
    python scripts/fit_observer_ball.py --best T0 THETA_OBS   refit alpha_high there and plot
    python scripts/fit_observer_ball.py --profile             chi2 profile in theta_obs
    python scripts/fit_observer_ball.py --plot-joint          bands at the joint minimum of
                                                              plot_observer_ball_landscape.py and at theta_obs = 1 deg
Writes figures/fit_observer_ball_scan.json, figures/fit_observer_ball_best.json and
figures/hippopede_observer_ball_bands.(png|pdf).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_lightcone_ball as B  # noqa: E402
import fit_sector_average as F  # noqa: E402
import hippopede_model as HM  # noqa: E402
import likelihood as L  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import sector_geometry as G  # noqa: E402
import t0_dispersion_match as DM  # noqa: E402

FIG = ROOT / "figures"
# sectors computed directly every 0.05 deg up to the cut of the analysis, G.THETA_MAX_DEG = 70 deg.  The
# ball reaches theta_obs + rhat(z)/2; a ball that needs sectors beyond the cut is not evaluated
# (chi2 = inf), as the sectors beyond it are excluded from every analysis.
THETA_STEP_DEG, THETA_TOP_DEG = 0.05, G.THETA_MAX_DEG
THETA = np.round(np.arange(THETA_STEP_DEG / 2, THETA_TOP_DEG, THETA_STEP_DEG), 4)
ZW = F.ZW
STEP = THETA[1] - THETA[0]
U, WU = np.polynomial.legendre.leggauss(24)      # radial nodes on [0, 1] after rescaling
C, WC = np.polynomial.legendre.leggauss(32)      # cos psi nodes


# nodes of the band match M for the ball: z = 0.5, ..., 1.9.  Below z = 0.5 the width of the GaPP
# band is taken as instrumental; above it, by hypothesis, the spread of the chronometers can be
# dominated by the anisotropy, which is what the sector spread of the model describes.  1.9 is the last
# node inside the data (ZW ends at 1.97).  The sector-average analysis of the paper keeps
# DM.Z_NODES = 0.1, ..., 1.2.
M_NODES = np.round(np.arange(0.5, 1.91, 0.1), 2)
M_SIG = np.interp(M_NODES, F.GAPP["z"], F.GAPP["q_sigma"])

_W_CACHE, _T_CACHE, _M_CACHE = {}, {}, {}


def weights(ah, theta_obs):
    """w[theta, z] (cached by alpha_high and theta_obs)."""
    key = (round(float(ah), 10), round(float(theta_obs), 10))
    if key not in _W_CACHE:
        _W_CACHE[key] = _weights(ah, theta_obs)
    return _W_CACHE[key]


def sector_table(ah, t0):
    """E_theta (units of the axial present rate) and q_theta on ZW for every sector of THETA,
    cached by alpha_high and t0."""
    key = (round(float(ah), 10), round(float(t0), 10))
    if key not in _T_CACHE:
        h_ax = HM.h0_sector(0.0, t0)
        E = np.empty((len(THETA), len(ZW)))
        Q = np.empty_like(E)
        for i, a in enumerate(THETA):
            e, q = HM.projected_sector(a, t0, ah, PH.ALPHA_LOW)
            E[i] = np.interp(ZW, HM.Z_WORK, e, left=np.nan, right=np.nan) * HM.h0_sector(np.radians(a), t0) / h_ax
            Q[i] = np.interp(ZW, HM.Z_WORK, q, left=np.nan, right=np.nan)
        HM._HIST_CACHE.clear()  # one history per sector and t0: do not keep them all
        inv = 1.0 / E
        DC = np.concatenate([np.zeros((len(THETA), 1)), np.cumsum(0.5 * (inv[:, 1:] + inv[:, :-1]) * np.diff(ZW), axis=1)], axis=1)
        _T_CACHE.clear()
        _T_CACHE[key] = (E, Q, DC)
    return _T_CACHE[key]


def _weights(ah, theta_obs):
    """w[theta, z]: volume of the ball rho <= rhat(z) around the observer falling in each sector."""
    run = lambda z: PH.alpha_sqrt(z, PH.ALPHA_LOW, ah)
    rhat = PH.rhat_of_z(ZW, run)
    w = np.zeros((len(THETA), len(ZW)))
    for j, R in enumerate(rhat):
        rho = 0.5 * R * (U + 1.0)
        wr = 0.5 * R * WU * np.sin(rho) ** 2
        th = G.source_sector_deg(theta_obs, rho[:, None], C[None, :])
        ww = (wr[:, None] * WC[None, :]).ravel()
        # linear (cloud-in-cell) assignment to the two neighbouring tabulated sectors, so that the
        # weights change smoothly with theta_obs
        if th.max() > THETA_TOP_DEG:  # the ball leaves the tabulated sectors
            w[:, j] = np.nan
            continue
        f = np.clip((th.ravel() - THETA[0]) / STEP, 0.0, len(THETA) - 1.000001)
        k = f.astype(int)
        frac = f - k
        np.add.at(w[:, j], k, ww * (1.0 - frac))
        np.add.at(w[:, j], k + 1, ww * frac)
        if w[:, j].sum() <= 0:  # z = 0: the observer's own sector
            w[int(np.clip(round((theta_obs - THETA[0]) / STEP), 0, len(THETA) - 1)), j] = 1.0
    return w / w.sum(axis=0)


def model(ah, t0, theta_obs):
    key = (round(float(ah), 10), round(float(t0), 10), round(float(theta_obs), 10))
    if key not in _M_CACHE:
        if len(_M_CACHE) > 4:
            _M_CACHE.clear()
        _M_CACHE[key] = _model(ah, t0, theta_obs)
    return _M_CACHE[key]


def _model(ah, t0, theta_obs):
    """E, q and weights of the sectors that contribute (rows), and the weighted <E>(z), <D_C>(z)."""
    E, Q, DC = sector_table(ah, t0)                       # E in units of the axial present rate
    w = weights(ah, theta_obs)
    if not np.all(np.isfinite(w)):
        return None
    used = w.sum(axis=1) > 0
    E, Q, DC, w = E[used], Q[used], DC[used], w[used]
    if not np.all(np.isfinite(E)) or np.any(E <= 0):
        return None
    fac = HM.h0_sector(0.0, t0) / HM.h0_sector(np.radians(theta_obs), t0)  # to the observer's units
    return E * fac, Q, w, fac * (w * E).sum(0), (w * DC).sum(0) / fac


def chi2(ah, t0, theta_obs):
    m = model(ah, t0, theta_obs)
    if m is None:
        return np.inf
    _, _, _, e_bar, dc_bar = m
    zc, hc, _ = L.CC
    e = np.interp(zc, ZW, e_bar)
    h0 = (e @ L.CINV_CC @ hc) / (e @ L.CINV_CC @ e)
    r = hc - h0 * e
    mm = np.interp(L.Z_SN, ZW, dc_bar)
    amp = (mm @ L.CINV_SN @ L.D_SN) / (mm @ L.CINV_SN @ mm)
    s = L.D_SN - amp * mm
    return float(r @ L.CINV_CC @ r + s @ L.CINV_SN @ s)


def _quantiles(x, w, probs):
    """Weighted quantiles (same definition as t0_dispersion_match.weighted_quantile), one sort."""
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 1:
        return np.full(len(probs), x[0])
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    cdf = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(probs, cdf, xs)


def mismatch(ah, t0, theta_obs):
    m = model(ah, t0, theta_obs)
    if m is None:
        return np.inf, None
    _, Q, w, _, _ = m
    idx = [int(np.argmin(np.abs(ZW - z))) for z in M_NODES]
    probs = [lo for lo, _ in DM.PROBS.values()] + [hi for _, hi in DM.PROBS.values()]
    qs = np.array([_quantiles(Q[:, j], w[:, j], probs) for j in idx])   # nodes x probs
    n = len(DM.PROBS)
    h = {k: 0.5 * (qs[:, n + i] - qs[:, i]) for i, k in enumerate(DM.PROBS)}
    with np.errstate(divide="ignore"):
        M = float(sum(np.sum((np.log(h[k]) - np.log(k * M_SIG)) ** 2) for k in (1, 2)))
    return M, (h[1] / M_SIG).tolist()


# Evaluation at the redshifts the likelihood and the band match actually use (chronometers, supernova
# bins, band nodes and the top of ZW, which fixes the sectors the ball reaches).  The weights are
# computed only there, so that a grid of many theta_obs fits in memory; chi2_at and mismatch_at agree
# with chi2 and mismatch to the interpolation error of <E> and <D_C> on ZW.
_NODE_IDX = [int(np.argmin(np.abs(ZW - z))) for z in M_NODES]
Z_EVAL, _INV = np.unique(np.concatenate([L.CC[0], L.Z_SN, ZW[_NODE_IDX], [ZW[-1]]]), return_inverse=True)
_I_CC = _INV[:len(L.CC[0])]
_I_SN = _INV[len(L.CC[0]):len(L.CC[0]) + len(L.Z_SN)]
_I_NODE = _INV[len(L.CC[0]) + len(L.Z_SN):len(L.CC[0]) + len(L.Z_SN) + len(_NODE_IDX)]
_WE_CACHE, _TE_CACHE = {}, {}


def _weights_at(ah, theta_obs, z):
    """w[theta, z] at the redshifts z (same construction as _weights)."""
    run = lambda x: PH.alpha_sqrt(x, PH.ALPHA_LOW, ah)
    rhat = PH.rhat_of_z(np.asarray(z), run)
    w = np.zeros((len(THETA), len(z)))
    for j, R in enumerate(rhat):
        rho = 0.5 * R * (U + 1.0)
        wr = 0.5 * R * WU * np.sin(rho) ** 2
        th = G.source_sector_deg(theta_obs, rho[:, None], C[None, :])
        ww = (wr[:, None] * WC[None, :]).ravel()
        if th.max() > THETA_TOP_DEG:  # the ball leaves the tabulated sectors
            w[:, j] = np.nan
            continue
        f = np.clip((th.ravel() - THETA[0]) / STEP, 0.0, len(THETA) - 1.000001)
        k = f.astype(int)
        frac = f - k
        np.add.at(w[:, j], k, ww * (1.0 - frac))
        np.add.at(w[:, j], k + 1, ww * frac)
        if w[:, j].sum() <= 0:
            w[int(np.clip(round((theta_obs - THETA[0]) / STEP), 0, len(THETA) - 1)), j] = 1.0
    return w / w.sum(axis=0)


def _eval_weights(ah, theta_obs):
    key = (round(float(ah), 10), round(float(theta_obs), 10))
    if key not in _WE_CACHE:
        w = _weights_at(ah, theta_obs, Z_EVAL)
        if not np.all(np.isfinite(w)):
            _WE_CACHE[key] = None
        else:
            used = np.where(w.sum(axis=1) > 0)[0]
            _WE_CACHE[key] = (used, w[used])
    return _WE_CACHE[key]


def _eval_table(ah, t0):
    """E, q, D_C of every sector at Z_EVAL, plus whether each sector is defined on all of ZW."""
    key = (round(float(ah), 10), round(float(t0), 10))
    if key not in _TE_CACHE:
        E, Q, DC = sector_table(ah, t0)
        good = np.all(np.isfinite(E), axis=1) & np.all(E > 0, axis=1)
        tab = [np.array([np.interp(Z_EVAL, ZW, row) for row in X]) for X in (E, Q, DC)]
        _TE_CACHE.clear()
        _TE_CACHE[key] = (*tab, good)
    return _TE_CACHE[key]


def _model_at(ah, t0, theta_obs):
    E, Q, DC, good = _eval_table(ah, t0)
    ew = _eval_weights(ah, theta_obs)
    if ew is None:
        return None
    used, w = ew
    if not np.all(good[used]):
        return None
    fac = HM.h0_sector(0.0, t0) / HM.h0_sector(np.radians(theta_obs), t0)
    return Q[used], w, fac * (w * E[used]).sum(0), (w * DC[used]).sum(0) / fac


def chi2_at(ah, t0, theta_obs):
    m = _model_at(ah, t0, theta_obs)
    if m is None:
        return np.inf
    _, _, e_bar, dc_bar = m
    _, hc, _ = L.CC
    e = e_bar[_I_CC]
    h0 = (e @ L.CINV_CC @ hc) / (e @ L.CINV_CC @ e)
    r = hc - h0 * e
    mm = dc_bar[_I_SN]
    amp = (mm @ L.CINV_SN @ L.D_SN) / (mm @ L.CINV_SN @ mm)
    s = L.D_SN - amp * mm
    return float(r @ L.CINV_CC @ r + s @ L.CINV_SN @ s)


def mismatch_at(ah, t0, theta_obs):
    m = _model_at(ah, t0, theta_obs)
    if m is None:
        return np.inf
    Q, w, _, _ = m
    probs = [lo for lo, _ in DM.PROBS.values()] + [hi for _, hi in DM.PROBS.values()]
    qs = np.array([_quantiles(Q[:, j], w[:, j], probs) for j in _I_NODE])
    n = len(DM.PROBS)
    h = {k: 0.5 * (qs[:, n + i] - qs[:, i]) for i, k in enumerate(DM.PROBS)}
    with np.errstate(divide="ignore"):
        return float(sum(np.sum((np.log(h[k]) - np.log(k * M_SIG)) ** 2) for k in (1, 2)))


def scan():
    lc = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]
    out = {"alpha_high": 0.36, "grid": []}
    for th in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        for t0 in (1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.2):
            c = chi2(0.36, t0, th)
            M, r = mismatch(0.36, t0, th)
            out["grid"].append({"theta_obs": th, "t0": t0, "dchi2": c - lc, "M": M, "h1_over_sigma": r})
            rr = np.round(np.array(r)[[2, 5, 8, 11]], 2).tolist() if r else None
            print(f"theta_obs={th:4.1f} t0={t0:3.1f}  dchi2={c - lc:+7.2f}  M={M:8.2f}  h1/sigma(z=0.3,0.6,0.9,1.2)={rr}", flush=True)
    (FIG / "fit_observer_ball_scan.json").write_text(json.dumps(out, indent=1))


def profile(thetas, name):
    """chi2 minimised over alpha_high and t0 for each theta_obs (profile likelihood of theta_obs)."""
    lc = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]
    t_grid = np.round(np.geomspace(1.4, 3.6, 17), 3)
    out = {"lcdm_chi2": lc, "t0_grid": t_grid.tolist(), "rows": []}
    for th in thetas:
        best_row = None
        for t0 in t_grid:
            r = minimize_scalar(lambda a: chi2(a, t0, th), bounds=(0.25, 0.55), method="bounded", options={"xatol": 1e-4})
            if np.isfinite(r.fun) and (best_row is None or r.fun < best_row[2]):
                best_row = (float(t0), float(r.x), float(r.fun))
        if best_row is None:
            continue
        t0, ah, c = best_row
        M, ratio = mismatch(ah, t0, th)
        out["rows"].append({"theta_obs": th, "t0": t0, "alpha_high": ah, "chi2": c, "dchi2": c - lc, "M": M, "h1_over_sigma": ratio,
                            "t0_at_lower_edge_of_feasible_grid": bool(t0 == min(t for t in t_grid if np.isfinite(chi2(ah, t, th))))})
        print(f"theta_obs={th:4.1f}: t0={t0:.3f} alpha_high={ah:.4f} dchi2={c - lc:+.3f} M={M:.1f} "
              f"h1/sigma(z=0.3,0.6,0.9,1.2)={np.round(np.array(ratio)[[2, 5, 8, 11]], 2).tolist()}", flush=True)
    (FIG / f"{name}.json").write_text(json.dumps(out, indent=1))


def best(t0, theta_obs):
    lc = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]
    ah = float(minimize_scalar(lambda a: chi2(a, t0, theta_obs), bounds=(0.25, 0.55), method="bounded", options={"xatol": 1e-4}).x)
    c = chi2(ah, t0, theta_obs)
    M, r = mismatch(ah, t0, theta_obs)
    out = {"t0": t0, "theta_obs": theta_obs, "alpha_high": ah, "dchi2": c - lc, "M": M, "h1_over_sigma": r}
    (FIG / "fit_observer_ball_best.json").write_text(json.dumps(out, indent=1))
    print(f"theta_obs={theta_obs} t0={t0}: alpha_high={ah:.4f} dchi2={c - lc:+.2f} M={M:.2f}")
    print("h1/sigma at z_nodes:", np.round(r, 2).tolist())
    return ah


PLOT_NODES = (96, 128)   # denser quadrature for the drawn bands: their quantiles resolve every 0.05 deg sector


def plot(ah, t0, theta_obs, out_name="hippopede_observer_ball_bands", title=None):
    """Same layout as fit_lightcone_ball.py, with the weights of the ball around the observer.
    The weights are recomputed with PLOT_NODES quadrature nodes, so that the drawn quantiles are not
    sampled more coarsely than the sectors."""
    global U, WU, C, WC
    saved = B.model, B.chi2, B.mismatch, B.THETA, B.M_RANGE
    nodes = U, WU, C, WC
    try:
        U, WU = np.polynomial.legendre.leggauss(PLOT_NODES[0])
        C, WC = np.polynomial.legendre.leggauss(PLOT_NODES[1])
        _W_CACHE.clear()
        _M_CACHE.clear()
        B.THETA = THETA
        B.M_RANGE = (M_NODES[0], M_NODES[-1])
        B.model = lambda a, t: model(a, t, theta_obs)
        B.chi2 = lambda a, t: chi2(a, t, theta_obs)
        B.mismatch = lambda a, t: mismatch(a, t, theta_obs)
        lc = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]
        B.plot(ah, t0, lc, out_name=out_name,
               title=title or rf"Ball $\rho\leq\hat r(z)$ around an observer at $\vartheta_{{\rm obs}}={round(theta_obs, 2):g}^\circ$")
    finally:
        B.model, B.chi2, B.mismatch, B.THETA, B.M_RANGE = saved
        U, WU, C, WC = nodes
        _W_CACHE.clear()
        _M_CACHE.clear()


def joint_fit(theta_obs, t0_start, ah_start):
    """alpha_high and t0 minimising chi2 at fixed theta_obs (Nelder-Mead from the landscape minimum)."""
    from scipy.optimize import minimize
    r = minimize(lambda p: chi2_at(p[0], p[1], theta_obs), [ah_start, t0_start], method="Nelder-Mead",
                 options={"xatol": 2e-4, "fatol": 1e-5})
    return float(r.x[0]), float(r.x[1])


def plot_joint(theta_other):
    """Bands at the joint minimum of the landscape and at theta_obs = theta_other."""
    js = json.loads((FIG / "fit_observer_ball_landscape_joint.json").read_text())["best"]
    lc = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]
    th0 = js["theta_obs"]
    out = {}
    for th, name, title in ((th0, "hippopede_lightcone_ball_bands",
                             rf"Joint minimum: ball around an observer at $\vartheta_{{\rm obs}}={round(th0, 2):g}^\circ$"),
                            (theta_other, "hippopede_observer_ball_bands", None)):
        ah, t0 = joint_fit(th, js["t0"], js["alpha_high"])
        c = chi2(ah, t0, th)
        M, r = mismatch(ah, t0, th)
        out[name] = {"theta_obs": th, "t0": t0, "alpha_high": ah, "dchi2": c - lc, "M": M, "h1_over_sigma": r}
        print(f"{name}: theta_obs={th} t0={t0:.4f} alpha_high={ah:.4f} dchi2={c - lc:+.3f} M={M:.1f}", flush=True)
        plot(ah, t0, th, out_name=name, title=title)
    (FIG / "fit_observer_ball_joint_bands.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--best", nargs=2, type=float, metavar=("T0", "THETA_OBS"))
    ap.add_argument("--profile", nargs="*", type=float, metavar="THETA_OBS",
                    help="chi2 profile at these theta_obs (alpha_high and t0 minimised)")
    ap.add_argument("--plot-joint", action="store_true",
                    help="bands at the joint minimum of figures/fit_observer_ball_landscape_joint.json "
                         "(hippopede_lightcone_ball_bands) and at --plot-joint-theta (hippopede_observer_ball_bands), "
                         "with alpha_high and t0 refitted at each theta_obs")
    ap.add_argument("--plot-joint-theta", type=float, default=1.0)
    a = ap.parse_args()
    if a.plot_joint:
        plot_joint(a.plot_joint_theta)
    if a.profile is not None:
        th = a.profile or [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
        profile(th, "fit_observer_ball_profile" + ("_small" if max(th) <= 2.0 else ""))
    if a.scan:
        scan()
    if a.best:
        t0, th = a.best
        plot(best(t0, th), t0, th)
