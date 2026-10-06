"""(t0, theta_obs) landscape of the light-cone ball around the observer (fit_observer_ball.py).

All three parameters are fitted jointly.  At every (t0, theta_obs) the chi2 (CC + binned Pantheon+)
is profiled over alpha_high: it is computed on the grid A_GRID and minimised by a parabola through
the lowest grid value and its two neighbours; the band mismatch M of q(z) against GaPP is
interpolated to the same alpha_high.  The joint minimum is then located by a recursive zoom around the
lowest grid cell (refine_minimum).  The star marks it and its bars the intervals where the profile of
t0 (minimised over theta_obs and alpha_high) and the profile of theta_obs (minimised over t0 and
alpha_high) stay within chi2_min + 1; an arrow means the interval continues beyond the plotted range.

Colour: M; white contours: profiled chi2 above its minimum; red band: theta_obs(t0) required by the
CatWISE dipole excess (quasar_dipole_fit.py, 1 sigma); orange: theta_obs preferred by the Quaia
redshift slices (quaia_zslice_model.py).  The count dipole is a property of the observer's sky and is
computed along each line of sight, as in those scripts.  With --dipole-alphas A1 A2 ... the dipole
curves are taken at the fitted alpha_high: for every t0, the theta_obs at which the alpha_high of the
dipole run equals the profiled alpha_high of the landscape at that point, interpolated between runs;
with a single value they are drawn at that fixed alpha_high.  The runs must be computed first with
    python scripts/quasar_dipole_fit.py --alpha-high A --t0-range T_MIN T_MAX --t0-n N
    python scripts/quaia_zslice_model.py --alpha-high A --t0-range T_MIN T_MAX --t0-n N
With several dipole runs the script also adds their chi2 to that of CC + SN: CatWISE,
((|D| - D_geo)/sigma)^2, and the three Quaia slices, sum_k ((f_k(t0) theta_obs - A_k)/sigma_k)^2, both
interpolated in alpha_high between runs, and profiles the sum over alpha_high (cyan: minimum and the
regions within 2.30 and 6.18 of it, 68 and 95 per cent for two parameters).
The chi2 and M use fit_observer_ball.chi2_at and mismatch_at (weights only at the redshifts used).
--plot-t0-min, --plot-t0-max and --plot-theta-max crop the plot without changing the computed grid;
--out-suffix is appended to the name of the figure (not of the JSON).
Balls that need sectors beyond the cut of the analysis (70 deg) are not evaluated (white).

Default grids: t0 in [1.5, 3.6], theta_obs in [0, 10] deg.  The wide version is
    python scripts/plot_observer_ball_landscape.py --t0-range 1 10 --nt 80 --theta-max 45 --theta-step 0.25 \
        --alpha-range 0.27 0.47 --tag joint_wide --dipole-alphas 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44
Writes figures/fit_observer_ball_landscape_<tag>.json and
figures/hippopede_observer_ball_landscape_<tag>.(png|pdf) (tag "joint" by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_observer_ball as OB  # noqa: E402

FIG = ROOT / "figures"
RED = "#d62728"
T_BLOCKS = 4
T_GRID = TH_GRID = A_GRID = None
THETA_MAX_PLOT = None


def set_grids(t_range, nt, theta_max, theta_step, a_range, a_step):
    """Grids of t0 (geometric), theta_obs and alpha_high; also run in every worker process."""
    global T_GRID, TH_GRID, A_GRID, THETA_MAX_PLOT
    T_GRID = np.round(np.geomspace(t_range[0], t_range[1], nt), 4)
    THETA_MAX_PLOT = float(theta_max)
    TH_GRID = np.round(np.arange(0.0, theta_max + 1e-9, theta_step), 3)
    A_GRID = np.round(np.arange(a_range[0], a_range[1] + 1e-9, a_step), 3)


def _lcdm():
    return json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]


def _task(args):
    """chi2 and M on TH_GRID for one alpha_high and a block of t0 values."""
    ah, t_idx = args
    lc = _lcdm()
    C = np.full((len(TH_GRID), len(t_idx)), np.nan)
    M = np.full_like(C, np.nan)
    for jj, j in enumerate(t_idx):
        for i, th in enumerate(TH_GRID):
            c = OB.chi2_at(ah, T_GRID[j], th)
            if np.isfinite(c):
                C[i, jj] = c - lc
                M[i, jj] = OB.mismatch_at(ah, T_GRID[j], th)
        OB._T_CACHE.clear()
        OB._TE_CACHE.clear()
    return ah, list(t_idx), C, M


def raw_grid(path, reuse, pool):
    """chi2 - chi2_LCDM and M on A_GRID x TH_GRID x T_GRID."""
    if reuse and path.exists():
        old = json.loads(path.read_text())
        if old["alpha_grid"] == A_GRID.tolist() and old["t0"] == T_GRID.tolist() and old["theta_obs"] == TH_GRID.tolist():
            return np.array(old["dchi2_raw"], float), np.array(old["M_raw"], float)
    C = np.full((len(A_GRID), len(TH_GRID), len(T_GRID)), np.nan)
    M = np.full_like(C, np.nan)
    tasks = [(float(a), b.tolist()) for a in A_GRID for b in np.array_split(np.arange(len(T_GRID)), T_BLOCKS)]
    for ah, t_idx, c, m in pool.map(_task, tasks):
        k = int(np.argmin(np.abs(A_GRID - ah)))
        C[k][:, t_idx], M[k][:, t_idx] = c, m
        print(f"alpha_high = {ah:g}, t0 = {T_GRID[t_idx[0]]:g}-{T_GRID[t_idx[-1]]:g} done", flush=True)
    return C, M


def profile_alpha(C, M):
    """Minimum over alpha_high of each (theta_obs, t0) column: parabola through the lowest grid value
    and its neighbours; M interpolated linearly to that alpha_high."""
    na, nth, nt = C.shape
    Cp, Mp, Ap = (np.full((nth, nt), np.nan) for _ in range(3))
    edge = np.zeros((nth, nt), bool)
    for i in range(nth):
        for j in range(nt):
            c = C[:, i, j]
            if not np.isfinite(c).any():
                continue
            k = int(np.nanargmin(c))
            a = A_GRID[k]
            cmin = c[k]
            if 0 < k < na - 1 and np.isfinite(c[k - 1]) and np.isfinite(c[k + 1]):
                d2 = c[k - 1] - 2 * c[k] + c[k + 1]
                if d2 > 0:
                    x = 0.5 * (c[k - 1] - c[k + 1]) / d2
                    a = A_GRID[k] + x * (A_GRID[1] - A_GRID[0])
                    cmin = c[k] - 0.25 * (c[k - 1] - c[k + 1]) * x
            else:
                edge[i, j] = True
            Ap[i, j], Cp[i, j] = a, cmin
            ok = np.isfinite(M[:, i, j])
            Mp[i, j] = np.interp(a, A_GRID[ok], M[ok, i, j]) if ok.sum() > 1 else M[k, i, j]
    return Cp, Mp, Ap, edge


def _refine_task(args):
    """chi2 - chi2_LCDM at one (alpha_high, t0) for several theta_obs."""
    ah, t0, thetas = args
    lc = _lcdm()
    out = [OB.chi2_at(ah, t0, th) - lc for th in thetas]
    OB._T_CACHE.clear()
    OB._TE_CACHE.clear()
    return ah, t0, out


def refine_minimum(start, span, pool, n=5, steps=8):
    """Recursive zoom on the joint minimum: an n x n x n grid in (alpha_high, t0, theta_obs) around the
    current best point, whose spans are halved at every step (theta_obs >= 0)."""
    best = np.array(start, float)
    span = np.array(span, float)
    c_best = np.inf
    for s in range(steps):
        axes = [best[k] + span[k] * np.linspace(-1, 1, n) for k in range(3)]
        axes[2] = np.unique(np.clip(axes[2], 0.0, None))
        tasks = [(float(a), float(t), axes[2].tolist()) for a in axes[0] for t in axes[1]]
        for ah, t0, vals in pool.map(_refine_task, tasks):
            for th, c in zip(axes[2], vals):
                if c < c_best:
                    c_best, best = c, np.array([ah, t0, th])
        print(f"refine step {s}: alpha_high={best[0]:.5f} t0={best[1]:.5f} theta_obs={best[2]:.4f} dchi2={c_best:+.5f} "
              f"(half-spans {span.round(5).tolist()})", flush=True)
        span = span / 2
    return best, float(c_best)


def _t0_task(args):
    """chi2 - chi2_LCDM minimised over alpha_high at one (t0, theta_obs)."""
    t0, th = args
    r = minimize_scalar(lambda a: OB.chi2_at(a, t0, th), bounds=(0.25, 0.6), method="bounded", options={"xatol": 1e-4})
    OB._T_CACHE.clear()
    OB._TE_CACHE.clear()
    return t0, float(r.x), float(r.fun - _lcdm())


def t0_beyond_grid(theta_obs, level, pool):
    """Profile of t0 (alpha_high minimised, theta_obs at the minimum) just below and well above T_GRID,
    where the 1-sigma interval of t0 may close: its ends (None if not reached) and the profile values."""
    low = np.round(np.linspace(0.9 * T_GRID[0], T_GRID[0], 11), 4)
    high = np.round(T_GRID[-1] * np.array([1.25, 1.5, 2.0, 3.0, 4.0, 6.0]), 3)
    res = {t0: (a, c) for t0, a, c in pool.map(_t0_task, [(float(t), theta_obs) for t in np.concatenate([low, high])])}
    p = np.array([res[float(t)][1] for t in low])
    lo = None
    for i in range(len(low) - 1, 0, -1):
        if p[i - 1] > level >= p[i]:
            lo = float(low[i - 1] + (level - p[i - 1]) * (low[i] - low[i - 1]) / (p[i] - p[i - 1]))
            break
    hi = None
    ph = [res[float(t)][1] for t in high]
    for i, c in enumerate(ph):
        if c > level:
            prev_t, prev_c = (high[i - 1], ph[i - 1]) if i else (T_GRID[-1], level)
            hi = float(prev_t + (level - prev_c) * (high[i] - prev_t) / (c - prev_c))
            break
    rows = [{"t0": float(t), "alpha_high": res[float(t)][0], "dchi2": res[float(t)][1]} for t in np.concatenate([low, high])]
    return lo, hi, rows


def interval(x, prof, level):
    """Range of x where the 1-d profile stays below level (linear interpolation at the crossings);
    None at an end means the interval reaches the end of the grid."""
    ok = np.isfinite(prof)
    x, p = x[ok], prof[ok]
    i0 = int(np.argmin(p))
    lo = hi = None
    for i in range(i0, 0, -1):
        if p[i - 1] > level:
            lo = x[i - 1] + (level - p[i - 1]) * (x[i] - x[i - 1]) / (p[i] - p[i - 1])
            break
    for i in range(i0, len(p) - 1):
        if p[i + 1] > level:
            hi = x[i] + (level - p[i]) * (x[i + 1] - x[i]) / (p[i + 1] - p[i])
            break
    return float(x[i0]), (None if lo is None else float(lo)), (None if hi is None else float(hi))


def _suffix(a):
    return "_ah" + f"{a:g}".replace(".", "p")


def dipole_runs(alphas):
    """{alpha_high: (t0, CatWISE [lo, best, hi], Quaia theta_obs)} from the saved dipole runs."""
    out = {}
    for a in alphas:
        qd = json.loads((FIG / f"quasar_dipole_fit{_suffix(a)}.json").read_text())
        runs = sorted((r for r in qd["runs"] if r["pz"] == "gamma" and abs(r["x"] - qd["x"]) < 1e-12), key=lambda r: r["t0"])
        cat = np.array([[np.nan if v is None else v for v in r["theta_obs_minus1sigma_best_plus1sigma"]] for r in runs], float)
        t = np.array([r["t0"] for r in runs], float)
        qz = json.loads((FIG / f"quaia_zslice_model{_suffix(a)}.json").read_text())["model"]
        # theta_obs beyond the observer range of the dipole model (35 deg) is an extrapolation: dropped
        qmap = {round(float(k), 3): (v["theta_obs"] if v.get("within_observer_range", True) else np.nan) for k, v in qz.items()}
        quaia = np.array([qmap.get(round(x, 3), np.nan) for x in t], float)
        out[a] = (t, cat, quaia)
    return out


def dipole_chi2(alphas):
    """chi2 of the CatWISE excess and of the three Quaia slices on TH_GRID x T_GRID for every dipole run,
    {alpha_high: (chi2_catwise, chi2_quaia)}.  CatWISE: ((|D(theta_obs)| - D_geo) / sigma)^2 with D
    interpolated in theta_obs and ln t0 (NaN where the sky coverage of the run ends); Quaia: the model is
    linear in theta_obs, A_k = f_k(t0) theta_obs (quaia_zslice_model.py), valid up to 35 deg."""
    out = {}
    lt = np.log(T_GRID)
    for a in alphas:
        qd = json.loads((FIG / f"quasar_dipole_fit{_suffix(a)}.json").read_text())
        runs = sorted((r for r in qd["runs"] if r["pz"] == "gamma" and abs(r["x"] - qd["x"]) < 1e-12), key=lambda r: r["t0"])
        t = np.log([r["t0"] for r in runs])
        D = np.abs(np.array([r["D"] for r in runs], float))             # t0 x theta grid of the run
        g = np.array(qd["theta_obs_grid"], float)
        Dt = np.full((len(lt), len(g)), np.nan)                       # T_GRID x theta grid
        for j, x in enumerate(lt):
            n = int(np.searchsorted(t, x, side="right")) - 1
            if 0 <= n < len(t) - 1:                                     # linear between the two runs that bracket t0
                w = (x - t[n]) / (t[n + 1] - t[n])
                Dt[j] = (1 - w) * D[n] + w * D[n + 1]
            elif np.isclose(x, t[-1]):
                Dt[j] = D[-1]
        cat = np.full((len(TH_GRID), len(T_GRID)), np.nan)
        for j in range(len(T_GRID)):
            ok = np.isfinite(Dt[j])
            if ok.sum() > 1:
                d = np.interp(TH_GRID, g[ok], Dt[j, ok], right=np.nan)
                cat[:, j] = ((d - qd["D_geo"]) / qd["sigma"]) ** 2
        qz = json.loads((FIG / f"quaia_zslice_model{_suffix(a)}.json").read_text())
        amp = np.array([s["excess_amp"] for s in qz["slices"]])
        sig = np.array([s["sigma"] for s in qz["slices"]])
        tq = sorted((float(k), v["f_1deg"]) for k, v in qz["model"].items())
        f = np.array([np.interp(lt, np.log([x[0] for x in tq]), [x[1][k] for x in tq], left=np.nan, right=np.nan)
                      for k in range(len(amp))])                         # slices x T_GRID
        th = np.where(TH_GRID <= 35.0, TH_GRID, np.nan)
        qua = sum(((f[k][None, :] * th[:, None] - amp[k]) / sig[k]) ** 2 for k in range(len(amp)))
        out[a] = (cat, qua)
    return out


def combined_profile(C, dip, with_quaia):
    """CC + SN + CatWISE (+ Quaia) chi2 on A_GRID x TH_GRID x T_GRID, the dipole chi2 interpolated
    linearly in alpha_high between runs (NaN outside their range), then profiled over alpha_high."""
    al = np.array(sorted(dip))
    T = np.full_like(C, np.nan)
    for k, a in enumerate(A_GRID):
        if not al[0] - 1e-9 <= a <= al[-1] + 1e-9:
            continue
        n = min(int(np.searchsorted(al, a, side="right")) - 1, len(al) - 2)
        w = (a - al[n]) / (al[n + 1] - al[n])
        extra = [(1 - w) * dip[al[n]][0] + w * dip[al[n + 1]][0]]
        if with_quaia:
            extra.append((1 - w) * dip[al[n]][1] + w * dip[al[n + 1]][1])
        T[k] = C[k] + sum(extra)
    Tp, _, Atp, _ = profile_alpha(T, np.zeros_like(T))
    return Tp, Atp


def alpha_star(Ap, t0, th):
    """Profiled alpha_high of the landscape at (t0, theta_obs), bilinear in (ln t0, theta_obs)."""
    if not (T_GRID[0] <= t0 <= T_GRID[-1] and 0 <= th <= TH_GRID[-1]):
        return np.nan
    x = np.interp(np.log(t0), np.log(T_GRID), np.arange(len(T_GRID)))
    y = np.interp(th, TH_GRID, np.arange(len(TH_GRID)))
    j, i = min(int(x), len(T_GRID) - 2), min(int(y), len(TH_GRID) - 2)
    fx, fy = x - j, y - i
    v = Ap[i:i + 2, j:j + 2]
    return float((1 - fy) * ((1 - fx) * v[0, 0] + fx * v[0, 1]) + fy * ((1 - fx) * v[1, 0] + fx * v[1, 1]))


def self_consistent(curve_by_alpha, Ap):
    """theta_obs(t0) of a dipole curve at the fitted alpha_high: for every t0, the crossing in alpha of
    alpha - alpha_star(t0, theta(alpha)), with theta(alpha) interpolated linearly between the runs."""
    alphas = sorted(curve_by_alpha)
    t = curve_by_alpha[alphas[0]][0]
    out = np.full(len(t), np.nan)
    for n, t0 in enumerate(t):
        th = np.array([curve_by_alpha[a][1][n] for a in alphas])
        g = np.array([a - alpha_star(Ap, t0, x) if np.isfinite(x) else np.nan for a, x in zip(alphas, th)])
        for k in range(len(alphas) - 1):
            if np.isfinite(g[k]) and np.isfinite(g[k + 1]) and g[k] * g[k + 1] <= 0 and g[k] != g[k + 1]:
                f = g[k] / (g[k] - g[k + 1])
                out[n] = th[k] + f * (th[k + 1] - th[k])
                break
    return t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-range", nargs=2, type=float, default=(1.5, 3.6))
    ap.add_argument("--nt", type=int, default=60)
    ap.add_argument("--theta-max", type=float, default=10.0)
    ap.add_argument("--theta-step", type=float, default=0.05)
    ap.add_argument("--alpha-range", nargs=2, type=float, default=(0.29, 0.41))
    ap.add_argument("--alpha-step", type=float, default=0.01)
    ap.add_argument("--tag", default="joint")
    ap.add_argument("--dipole-alphas", nargs="+", type=float, default=[0.338])
    ap.add_argument("--reuse-grid", action="store_true")
    ap.add_argument("--plot-t0-min", type=float, default=None, help="left edge of the plot (the grid is not changed)")
    ap.add_argument("--plot-theta-max", type=float, default=None, help="top edge of the plot (the grid is not changed)")
    ap.add_argument("--plot-t0-max", type=float, default=None, help="right edge of the plot (the grid is not changed)")
    ap.add_argument("--out-suffix", default="", help="suffix of the figure name")
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    grids = (a.t0_range, a.nt, a.theta_max, a.theta_step, a.alpha_range, a.alpha_step)
    set_grids(*grids)
    path = FIG / f"fit_observer_ball_landscape_{a.tag}.json"
    with ProcessPoolExecutor(max_workers=a.workers, initializer=set_grids, initargs=grids) as pool:
        C, M = raw_grid(path, a.reuse_grid, pool)
        Cp, Mp, Ap, edge = profile_alpha(C, M)
        i, j = np.unravel_index(np.nanargmin(Cp), Cp.shape)
        dt = np.log(T_GRID[1] / T_GRID[0])
        (ah_b, t_b, th_b), cmin = refine_minimum((Ap[i, j], T_GRID[j], TH_GRID[i]),
                                                 (A_GRID[1] - A_GRID[0], T_GRID[j] * dt, TH_GRID[1] - TH_GRID[0]), pool)
        cmin = min(cmin, np.nanmin(Cp))
        _, t_lo, t_hi = interval(T_GRID, np.nanmin(Cp, axis=0), cmin + 1)
        _, th_lo, th_hi = interval(TH_GRID, np.nanmin(Cp, axis=1), cmin + 1)
        beyond = None
        if t_lo is None or t_hi is None:
            lo2, hi2, beyond = t0_beyond_grid(th_b, cmin + 1, pool)
            t_lo = t_lo if t_lo is not None else lo2
            t_hi = t_hi if t_hi is not None else hi2
    t_best, th_best = t_b, th_b
    x_left = min(T_GRID[0], 0.985 * t_lo) if t_lo is not None else T_GRID[0]
    if a.plot_t0_min is not None:
        x_left = a.plot_t0_min
    y_top = a.plot_theta_max if a.plot_theta_max is not None else THETA_MAX_PLOT
    x_right = a.plot_t0_max if a.plot_t0_max is not None else T_GRID[-1]

    runs = dipole_runs(a.dipole_alphas)
    if len(a.dipole_alphas) == 1:
        t_c, cat, quaia = runs[a.dipole_alphas[0]]
        cat_lo, cat_mid, cat_hi = cat.T
        t_q = t_c
        dip_label = rf" at $\alpha_{{\rm high}}={a.dipole_alphas[0]:g}$"
    else:
        cat_lo, cat_mid, cat_hi = (self_consistent({al: (r[0], r[1][:, k]) for al, r in runs.items()}, Ap)[1] for k in range(3))
        t_c = runs[a.dipole_alphas[0]][0]
        t_q, quaia = self_consistent({al: (r[0], r[2]) for al, r in runs.items()}, Ap)
        dip_label = r" at the fitted $\alpha_{\rm high}$"
    joint = {}
    if len(a.dipole_alphas) > 1:
        dip = dipole_chi2(a.dipole_alphas)
        for name, wq in (("cc_sn_catwise", False), ("cc_sn_catwise_quaia", True)):
            Tp, Atp = combined_profile(C, dip, wq)
            ii, jj = np.unravel_index(np.nanargmin(Tp), Tp.shape)
            tmin = float(Tp[ii, jj])
            inside = np.isfinite(Tp) & (Tp - tmin <= 2.30)
            joint[name] = {"t0": float(T_GRID[jj]), "theta_obs": float(TH_GRID[ii]), "alpha_high": float(Atp[ii, jj]),
                           "chi2_total_min": tmin,
                           "region68_t0": [float(T_GRID[inside.any(axis=0)].min()), float(T_GRID[inside.any(axis=0)].max())],
                           "region68_theta_obs": [float(TH_GRID[inside.any(axis=1)].min()), float(TH_GRID[inside.any(axis=1)].max())],
                           "profile": np.round(Tp, 4).tolist(), "_Tp": Tp}
            print(f"{name}: minimum t0={T_GRID[jj]:.3f} theta_obs={TH_GRID[ii]:.2f} alpha_high={Atp[ii, jj]:.4f} "
                  f"chi2 (CC+SN - LCDM, plus dipoles)={tmin:.3f}; 68% region (dchi2 <= 2.30): t0 {joint[name]['region68_t0']}, "
                  f"theta_obs {joint[name]['region68_theta_obs']}", flush=True)

    res = {"alpha_grid": A_GRID.tolist(), "t0": T_GRID.tolist(), "theta_obs": TH_GRID.tolist(),
           "best": {"t0": float(t_b), "theta_obs": float(th_b), "alpha_high": float(ah_b), "dchi2": float(cmin),
                    "M": float(OB.mismatch_at(ah_b, t_b, th_b)), "t0_1sigma": [t_lo, t_hi], "theta_obs_1sigma": [th_lo, th_hi]},
           "alpha_at_grid_edge": int(edge.sum()), "t0_profile_beyond_grid": beyond,
           "dipole_alphas": a.dipole_alphas,
           "catwise_theta_obs": {"t0": np.asarray(t_c).tolist(), "lo": np.asarray(cat_lo).tolist(),
                                 "best": np.asarray(cat_mid).tolist(), "hi": np.asarray(cat_hi).tolist()},
           "quaia_theta_obs": {"t0": np.asarray(t_q).tolist(), "best": np.asarray(quaia).tolist()},
           "joint_with_dipoles": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in joint.items()},
           "dchi2_profiled": np.round(Cp, 5).tolist(), "M_profiled": np.round(Mp, 4).tolist(), "alpha_profiled": np.round(Ap, 5).tolist(),
           "dchi2_raw": np.round(C, 5).tolist(), "M_raw": np.round(M, 4).tolist()}
    path.write_text(json.dumps(res, allow_nan=True))

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=180)
    # colour scale = range of log10 M inside the plotted window
    lm = np.log10(Mp)
    win = np.isfinite(lm) & ((T_GRID >= x_left) & (T_GRID <= x_right))[None, :] & (TH_GRID <= y_top)[:, None]
    vmin, vmax = np.floor(10 * lm[win].min()) / 10, np.ceil(10 * lm[win].max()) / 10
    pc = ax.pcolormesh(T_GRID, TH_GRID, lm, shading="nearest", cmap="viridis_r", vmin=vmin, vmax=vmax)
    fig.colorbar(pc, ax=ax, label=r"$\log_{10}\mathcal{M}$ (band mismatch; lower is better)")
    cs = ax.contour(T_GRID, TH_GRID, Cp - cmin, levels=[1, 2, 4], colors="white", linewidths=1.0, linestyles=["-", "--", ":"])
    ax.clabel(cs, fmt={1: r"$\chi^2_{\min}+1$", 2: "+2", 4: "+4"}, fontsize=8)
    ok = np.isfinite(cat_mid)
    ax.fill_between(np.asarray(t_c)[ok], np.where(np.isfinite(cat_lo[ok]), cat_lo[ok], 0.0),
                    np.where(np.isfinite(cat_hi[ok]), cat_hi[ok], y_top), color=RED, alpha=0.25, lw=0)
    ax.plot(np.asarray(t_c)[ok], cat_mid[ok], color=RED, lw=2, label=r"CatWISE dipole excess ($\pm1\sigma$)" + dip_label)
    okq = np.isfinite(quaia)
    ax.plot(np.asarray(t_q)[okq], np.asarray(quaia)[okq], color="orange", lw=2, ls="--", label="Quaia slices, best fit" + dip_label)
    if "cc_sn_catwise_quaia" in joint:
        tz = joint["cc_sn_catwise_quaia"]["_Tp"]
        tz = np.where(np.isfinite(tz), tz - np.nanmin(tz), 1e9)
        ax.contourf(T_GRID, TH_GRID, tz, levels=[-1, 2.30], colors=["cyan"], alpha=0.30)
        ax.contour(T_GRID, TH_GRID, tz, levels=[2.30, 6.18], colors="cyan", linewidths=[1.6, 1.0], linestyles=["-", "--"])
        jb = joint["cc_sn_catwise_quaia"]
        ax.plot(jb["t0"], jb["theta_obs"], marker="D", color="cyan", mec="k", ms=7, zorder=9, ls="none", clip_on=True,
                label="CC+SN+CatWISE+Quaia: minimum, 68% and 95% regions")
    # star with the 1-sigma intervals of the profiles; an arrow means the interval leaves the plot.
    # Drawn without clipping, because the minimum may lie on the edge theta_obs = 0.
    cap = 0.015 * y_top
    x_lo = t_lo if t_lo is not None else x_left
    x_hi = t_hi if t_hi is not None and t_hi <= x_right else x_right
    y_lo = th_lo if th_lo is not None else 0.0
    y_hi = th_hi if th_hi is not None and th_hi <= y_top else y_top
    for x0, x1, y0, y1 in ((x_lo, x_hi, th_best, th_best), (t_best, t_best, y_lo, y_hi)):
        ax.plot([x0, x1], [y0, y1], color="k", lw=3.4, zorder=6, solid_capstyle="butt", clip_on=False)
        ax.plot([x0, x1], [y0, y1], color="white", lw=1.6, zorder=7, clip_on=False)
    arrow = dict(arrowstyle="-|>", color="white", lw=1.6, mutation_scale=14)
    if t_lo is not None:
        ax.plot([x_lo, x_lo], [th_best - cap, th_best + cap], color="white", lw=1.6, zorder=7, clip_on=False)
    if t_hi is None or t_hi > x_right:
        ax.annotate("", xy=(x_right, th_best), xytext=(x_right / 1.06, th_best), arrowprops=arrow, zorder=8,
                    annotation_clip=False)
    else:
        ax.plot([x_hi, x_hi], [th_best - cap, th_best + cap], color="white", lw=1.6, zorder=7, clip_on=False)
    if th_hi is None or th_hi > y_top:
        ax.annotate("", xy=(t_best, y_top), xytext=(t_best, 0.94 * y_top), arrowprops=arrow, zorder=8)
    else:
        ax.plot([t_best / 1.012, t_best * 1.012], [y_hi, y_hi], color="white", lw=1.6, zorder=7)
    ax.plot(t_best, th_best, marker="*", color="white", ms=14, mec="k", zorder=9, ls="none", clip_on=False,
            label=rf"joint $\chi^2$ minimum ($\Delta\chi^2={cmin:+.2f}$ vs $\Lambda$CDM, $\pm1\sigma$)")
    ax.set_xscale("log")
    ticks = [t for t in (1, 1.2, 1.4, 1.5, 1.6, 1.8, 2, 2.2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10) if x_left <= t <= x_right]
    if len(ticks) > 9:
        ticks = [t for t in ticks if t in (1, 1.5, 2, 3, 4, 5, 7, 10)]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xlim(x_left, x_right); ax.set_ylim(0, y_top)
    ax.set_xlabel(r"Present time $t_0$"); ax.set_ylabel(r"Observer sector $\vartheta_{\rm obs}$ [deg]")
    ax.set_title(r"Light-cone ball around the observer, $\alpha_{\rm high}$ fitted at every point;"
                 r" white: $\chi^2$ above its joint minimum", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hippopede_observer_ball_landscape_{a.tag}{a.out_suffix}.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    b = res["best"]
    print(f"joint minimum: t0={b['t0']} [{t_lo}, {t_hi}], theta_obs={b['theta_obs']} [{th_lo}, {th_hi}], "
          f"alpha_high={b['alpha_high']:.4f}, dchi2={cmin:+.3f}, M={b['M']:.1f}; alpha at grid edge in {res['alpha_at_grid_edge']} cells")


if __name__ == "__main__":
    main()
