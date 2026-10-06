"""(t0, theta_obs) map of the light-cone ball model with the unbinned Pantheon+ supernovae.

At every (t0, theta_obs) of a fine grid the chi2 of the chronometers and the 1579 unbinned supernovae
(fit_ball_unbinned.Point: ball average modulated along each line of sight, axis opposite to the CatWISE
excess) is profiled over alpha_high (parabola through the lowest of A_GRID and its neighbours).
Colour: that chi2 relative to flat LCDM with the same data.  Green: CatWISE amplitude chi2 <= 1 at the
same alpha_high (band) and its best theta_obs (line).  Orange: theta_obs that minimises the chi2 of the
Quaia slice amplitudes.  Light green: CC + SN + CatWISE, minimum and 68/95% regions (Delta = 2.30, 6.18).
Default window t0 1.4-2.2, theta_obs 0-3 deg; --t0-range, --nt, --theta-max, --theta-step and --tag
change it.  Writes figures/ball_zoom_unbinned<tag>.json and figures/hippopede_ball_zoom_unbinned<tag>.(png|pdf).
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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_ball_unbinned as F  # noqa: E402
import fit_observer_ball as OB  # noqa: E402

FIG = ROOT / "figures"
GREEN = "#1a9641"   # CatWISE: neither of the colours of the chi2 scale
LIGHT = "#7CFC7C"   # CC + SN + CatWISE region
QUAIA = "#F2A900"   # Quaia line (yellow-orange) and band (yellow)
QUAIA_BAND = "#FFE14D"
A_GRID = np.round(np.arange(0.29, 0.4101, 0.01), 3)
T_GRID = np.round(np.geomspace(1.4, 2.2, 33), 4)
TH_GRID = np.round(np.arange(0.0, 3.0001, 0.1), 2)


def set_grids(t_range, nt, theta_max, theta_step):
    global T_GRID, TH_GRID
    T_GRID = np.round(np.geomspace(t_range[0], t_range[1], nt), 4)
    TH_GRID = np.round(np.arange(0.0, theta_max + 1e-9, theta_step), 3)


def task(args):
    ah, t0 = args
    p = F.Point(ah, t0)
    n_ax = F.axis_opposite_catwise()
    c = []
    for th in TH_GRID:
        c.append(sum(p.chi2(th, n_ax)))
        OB._W_CACHE.clear(); OB._M_CACHE.clear()   # weights are not reused across theta_obs (45 MB each)
    OB._T_CACHE.clear(); OB._W_CACHE.clear(); OB._M_CACHE.clear()
    return ah, t0, c


def profile(C):
    """min over alpha_high of C[alpha, theta, t0] (parabola) and the alpha_high there."""
    na, nth, nt = C.shape
    P, A = np.full((nth, nt), np.nan), np.full((nth, nt), np.nan)
    for i in range(nth):
        for j in range(nt):
            c = C[:, i, j]
            if not np.isfinite(c).any():
                continue
            k = int(np.nanargmin(c))
            a, v = A_GRID[k], c[k]
            if 0 < k < na - 1 and np.isfinite(c[k - 1]) and np.isfinite(c[k + 1]):
                d2 = c[k - 1] - 2 * c[k] + c[k + 1]
                if d2 > 0:
                    x = 0.5 * (c[k - 1] - c[k + 1]) / d2
                    a, v = A_GRID[k] + x * (A_GRID[1] - A_GRID[0]), c[k] - 0.25 * (c[k - 1] - c[k + 1]) * x
            P[i, j], A[i, j] = v, a
    return P, A


_QZ = None


def _quaia_runs():
    """f_k(t0) of the three Quaia slices for every dipole run, and the measured amplitudes."""
    global _QZ
    if _QZ is None:
        _QZ = {}
        for a in F.DIPOLE_ALPHAS:
            qz = json.loads((FIG / ("quaia_zslice_model_ah" + f"{a:g}".replace(".", "p") + ".json")).read_text())
            amp = np.array([x["excess_amp"] for x in qz["slices"]])
            sig = np.array([x["sigma"] for x in qz["slices"]])
            tq = sorted((float(k), v["f_1deg"]) for k, v in qz["model"].items())
            _QZ[a] = (np.log([x[0] for x in tq]), np.array([x[1] for x in tq], float), amp, sig)
    return _QZ


def quaia_chi2(ah, t0, theta):
    """chi2 of the three Quaia slice amplitudes for the model amplitude f_k(t0) theta_obs (linear in
    theta_obs, valid up to 35 deg), interpolated in alpha_high and ln t0 between the dipole runs."""
    runs = _quaia_runs()
    al = np.array(F.DIPOLE_ALPHAS)
    if not al[0] <= ah <= al[-1] or theta > 35.0:
        return np.nan
    k = min(int(np.searchsorted(al, ah, side="right")) - 1, len(al) - 2)
    w = (ah - al[k]) / (al[k + 1] - al[k])
    fs = []
    for a in (al[k], al[k + 1]):
        lt, f, amp, sig = runs[a]
        fs.append(np.array([np.interp(np.log(t0), lt, f[:, j], left=np.nan, right=np.nan) for j in range(f.shape[1])]))
    f = (1 - w) * fs[0] + w * fs[1]
    return float(np.sum(((f * theta - amp) / sig) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--t0-range", nargs=2, type=float, default=(1.4, 2.2))
    ap.add_argument("--nt", type=int, default=33)
    ap.add_argument("--theta-max", type=float, default=3.0)
    ap.add_argument("--theta-step", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    grids = (a.t0_range, a.nt, a.theta_max, a.theta_step)
    set_grids(*grids)
    path = FIG / f"ball_zoom_unbinned{a.tag}.json"
    if a.reuse and path.exists():
        d = json.loads(path.read_text())
        C, c_l = np.array(d["chi2_raw"], float), d["lcdm_chi2"]
    else:
        _, c_l = F.lcdm_reference()
        C = np.full((len(A_GRID), len(TH_GRID), len(T_GRID)), np.nan)
        with ProcessPoolExecutor(max_workers=a.workers, initializer=set_grids, initargs=grids) as ex:
            for ah, t0, c in ex.map(task, [(float(x), float(y)) for x in A_GRID for y in T_GRID]):
                C[int(np.argmin(np.abs(A_GRID - ah))), :, int(np.argmin(np.abs(T_GRID - t0)))] = c
                print(f"alpha_high {ah:g} t0 {t0:g} done", flush=True)
    C = np.where(np.isfinite(C), C, np.nan)
    P, A = profile(C)
    # dipoles at every alpha of the grid, then on the profile
    CW = np.full_like(C, np.nan)
    for k, ah in enumerate(A_GRID):
        for i, th in enumerate(TH_GRID):
            for j, t0 in enumerate(T_GRID):
                CW[k, i, j] = F.catwise_chi2(ah, t0, th)
    PJ, AJ = profile(C + CW)
    # all four data sets: CC + SN + CatWISE + Quaia (amplitudes of the three slices), profiled over alpha_high
    QW = np.full_like(C, np.nan)
    for k, ah in enumerate(A_GRID):
        for i, th in enumerate(TH_GRID):
            for j, t0 in enumerate(T_GRID):
                QW[k, i, j] = quaia_chi2(ah, t0, th)
    PQ, AQ = profile(C + CW + QW)
    iq, jq = np.unravel_index(np.nanargmin(PQ), PQ.shape)
    best_all = {"t0": float(T_GRID[jq]), "theta_obs": float(TH_GRID[iq]), "alpha_high": float(AQ[iq, jq]),
                "dchi2_cc_sn": float(np.interp(AQ[iq, jq], A_GRID, C[:, iq, jq]) - c_l),
                "chi2_catwise": F.catwise_chi2(AQ[iq, jq], T_GRID[jq], TH_GRID[iq]),
                "chi2_quaia": quaia_chi2(AQ[iq, jq], T_GRID[jq], TH_GRID[iq])}
    print("CC + SN + CatWISE + Quaia minimum:", best_all, flush=True)
    cw_on_profile = np.array([[F.catwise_chi2(A[i, j], T_GRID[j], TH_GRID[i]) if np.isfinite(A[i, j]) else np.nan
                               for j in range(len(T_GRID))] for i in range(len(TH_GRID))])
    # Quaia on the profile: best theta_obs for each t0 and the 1-sigma band (chi2 within 1 of that minimum)
    QG = np.array([[quaia_chi2(A[i, j], T_GRID[j], TH_GRID[i]) if np.isfinite(A[i, j]) else np.nan
                    for j in range(len(T_GRID))] for i in range(len(TH_GRID))])
    has_q = np.isfinite(QG).any(axis=0)
    qmin = np.where(has_q, np.nanmin(np.where(np.isfinite(QG), QG, np.inf), axis=0), np.nan)
    quaia = np.array([TH_GRID[int(np.nanargmin(QG[:, j]))] if has_q[j] else np.nan for j in range(len(T_GRID))])
    dq = QG - qmin[None, :]
    # summary: every data set within its own 1 sigma (CC + SN: 68% region of its minimum for two parameters)
    in_ccsn = (P - np.nanmin(P)) <= 2.30
    in_cat = cw_on_profile <= 1.0
    in_quaia = dq <= 1.0
    summary = np.where(np.isfinite(P), in_ccsn & in_cat & in_quaia, False)
    summ = None
    if summary.any():
        ii, jjj = np.where(summary)
        summ = {"t0": [float(T_GRID[jjj].min()), float(T_GRID[jjj].max())], "theta_obs": [float(TH_GRID[ii].min()), float(TH_GRID[ii].max())],
                "dchi2_cc_sn_range": [float((P - c_l)[summary].min()), float((P - c_l)[summary].max())],
                "quaia_chi2_range": [float(QG[summary].min()), float(QG[summary].max())]}
    print("summary region (all within 1 sigma):", summ, flush=True)
    i, j = np.unravel_index(np.nanargmin(P), P.shape)
    ij, jj = np.unravel_index(np.nanargmin(PJ), PJ.shape)
    res = {"lcdm_chi2": c_l, "alpha_grid": A_GRID.tolist(), "t0": T_GRID.tolist(), "theta_obs": TH_GRID.tolist(),
           "chi2_raw": np.round(C, 4).tolist(), "alpha_profiled": np.round(A, 5).tolist(), "best_cc_sn": {"t0": float(T_GRID[j]), "theta_obs": float(TH_GRID[i]), "alpha_high": float(A[i, j]), "dchi2": float(P[i, j] - c_l)},
           "best_cc_sn_catwise": {"t0": float(T_GRID[jj]), "theta_obs": float(TH_GRID[ij]), "alpha_high": float(AJ[ij, jj]),
                                  "dchi2_cc_sn": float(P[ij, jj] - c_l), "total_minus_lcdm_cc_sn": float(PJ[ij, jj] - c_l)},
           "summary_all_within_1sigma": summ, "best_cc_sn_catwise_quaia": best_all}
    path.write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("best")}, indent=1))

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=180)
    D = P - c_l
    # Display only: the fields are interpolated in ln t0 onto TF, and the edge of the allowed region,
    # the largest theta_obs with a finite chi2 at each computed t0, is interpolated between the columns,
    # so that the white region has a continuous edge instead of one step per column.
    TF = np.geomspace(T_GRID[0], T_GRID[-1], 600)
    fin = np.isfinite(P)
    thmax = np.array([TH_GRID[fin[:, c]].max() if fin[:, c].any() else np.nan for c in range(len(T_GRID))])
    okc = np.isfinite(thmax)
    thmax_f = np.interp(np.log(TF), np.log(T_GRID[okc]), thmax[okc], left=np.nan)

    U = np.linspace(0.0, 1.0, 801)

    def fine(Z, fill=None):
        """Interpolation aligned with the edge: at each computed t0 the field is written as a function of
        u = theta_obs / theta_max(t0), interpolated in ln t0 at fixed u, and mapped back to theta_obs."""
        Zu = np.full((len(U), len(T_GRID)), np.nan)
        for c in range(len(T_GRID)):
            m = np.isfinite(Z[:, c])
            if okc[c] and m.sum() >= 2:
                Zu[:, c] = np.interp(U * thmax[c], TH_GRID[m], Z[m, c])
        Zuf = np.full((len(U), len(TF)), np.nan)
        for r in range(len(U)):
            m = np.isfinite(Zu[r])
            if m.sum() >= 2:
                Zuf[r] = np.interp(np.log(TF), np.log(T_GRID[m]), Zu[r, m], left=np.nan, right=np.nan)
        out = np.full((len(TH_GRID), len(TF)), np.nan)
        for c in range(len(TF)):
            if np.isfinite(thmax_f[c]) and np.isfinite(Zuf[:, c]).sum() >= 2:
                m = np.isfinite(Zuf[:, c])
                out[:, c] = np.interp(TH_GRID, U[m] * thmax_f[c], Zuf[m, c], right=np.nan)
        out[~(TH_GRID[:, None] <= thmax_f[None, :])] = np.nan
        return out if fill is None else np.where(np.isfinite(out), out, fill)

    Df, CWf = fine(D), fine(cw_on_profile, 1e9)
    PPf, TZf = fine(P - np.nanmin(P), 1e9), fine(PJ - np.nanmin(PJ), 1e9)
    # Quaia and the summary region on the fine grid: the minimum over theta_obs of the Quaia chi2 is
    # taken at every fine t0, and the extent of the summary region is measured there
    QGf = fine(QG)
    hasf = np.isfinite(QGf).any(axis=0)
    qminf = np.where(hasf, np.nanmin(np.where(np.isfinite(QGf), QGf, np.inf), axis=0), np.nan)
    DQf = QGf - qminf[None, :]
    quaia_f = np.array([TH_GRID[int(np.nanargmin(QGf[:, c]))] if hasf[c] else np.nan for c in range(len(TF))])
    DQf = np.where(np.isfinite(DQf), DQf, 1e9)
    crit = np.maximum(np.maximum(PPf / 2.30, CWf), DQf)
    inside = crit <= 1.0
    if inside.any():
        ii, cc = np.where(inside)
        res["summary_all_within_1sigma"] = {
            "t0": [float(TF[cc].min()), float(TF[cc].max())], "theta_obs": [float(TH_GRID[ii].min()), float(TH_GRID[ii].max())],
            "dchi2_cc_sn_range": [float(np.nanmin(np.where(inside, Df, np.nan))), float(np.nanmax(np.where(inside, Df, np.nan)))],
            "quaia_chi2_range": [float(np.nanmin(np.where(inside, QGf, np.nan))), float(np.nanmax(np.where(inside, QGf, np.nan)))],
            "grid": "interpolated (display grid)"}
        path.write_text(json.dumps(res))
        print("summary region on the fine grid:", res["summary_all_within_1sigma"], flush=True)
    # alpha_high for the red contours, interpolated like the other fields; not drawn in the outer 10% of the
    # allowed range, where the profile reaches the upper edge of A_GRID
    Af = fine(A)
    Af[TH_GRID[:, None] > 0.9 * thmax_f[None, :]] = np.nan
    from matplotlib.colors import TwoSlopeNorm
    # colour scale saturated at -1 and +3 (arrows on the bar)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=3.0)
    pc = ax.pcolormesh(TF, TH_GRID, Df, shading="gouraud", cmap="RdBu_r", norm=norm)
    cb = fig.colorbar(pc, ax=ax, extend="both", shrink=0.85, ticks=[-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], label=r"$\chi^2-\chi^2_{\Lambda\mathrm{CDM}}$ (38 CC + 1579 SN, unbinned; $\alpha_{\rm high}$ fitted)")
    cb.ax.set_yscale("linear")   # proportional bar: -1..0 takes a quarter of it
    cs = ax.contour(TF, TH_GRID, Df, levels=[-1.0, -0.5, 0.0, 0.5, 1.0], colors="k", linewidths=0.6, alpha=0.6)
    ax.clabel(cs, fmt="%+.1f", fontsize=7)
    ax.contourf(TF, TH_GRID, DQf, levels=[-1, 1.0], colors=[QUAIA_BAND], alpha=0.35)
    ax.contourf(TF, TH_GRID, CWf, levels=[-1, 1.0], colors=[GREEN], alpha=0.30)
    ax.contour(TF, TH_GRID, CWf, levels=[1.0], colors=GREEN, linewidths=1.4)
    ax.fill_between([], [], color=GREEN, alpha=0.30, label=r"CatWISE excess within $1\sigma$")
    ok = np.isfinite(quaia_f) & (quaia_f <= TH_GRID[-1])
    ax.plot(TF[ok], quaia_f[ok], color=QUAIA, lw=2, ls="--")
    ax.plot([], [], color=QUAIA, lw=2, ls="--", label="Quaia slices: best fit")
    ax.fill_between([], [], color=QUAIA_BAND, alpha=0.5, label=r"Quaia slices within $1\sigma$")
    ax.contour(TF, TH_GRID, PPf, levels=[2.30], colors="#08306b", linewidths=1.2, linestyles=":")
    ax.plot([], [], color="#08306b", lw=1.2, ls=":", label="CC + SN: 68% region")
    if inside.any():
        # outline: the largest of the three normalised criteria equals 1 on the boundary
        ax.contour(TF, TH_GRID, crit, levels=[1.0], colors="white", linewidths=4.0)
        ax.contour(TF, TH_GRID, crit, levels=[1.0], colors="k", linewidths=2.0)
        ax.plot([], [], color="k", lw=2.0, label=r"all compatible within $1\sigma$")
    # alpha_high fitted to CC + SN at each point (red), inside the range of A_GRID
    ca = ax.contour(TF, TH_GRID, Af, levels=[0.32, 0.34, 0.36, 0.38, 0.40], colors="#c00000", linewidths=0.9, alpha=0.9, linestyles="--")
    ax.clabel(ca, fmt=lambda v: rf"$\alpha_{{\rm high}}={v:.2f}$", fontsize=7, inline_spacing=2)
    ax.plot([], [], color="#c00000", lw=0.9, ls="--", label=r"fitted $\alpha_{\rm high}$")
    ax.contour(TF, TH_GRID, TZf, levels=[2.30, 6.18], colors=LIGHT, linewidths=[1.6, 1.0], linestyles=["-", "--"])
    # a minimum on the edge of the window is not a minimum of the model: it is then named, not drawn
    edge = lambda a, b: a in (0, len(TH_GRID) - 1) or b in (0, len(T_GRID) - 1)
    ax.plot([], [], color=LIGHT, lw=1.6, label="CC + SN + CatWISE: 68% and 95%")
    if not edge(ij, jj):
        ax.plot(T_GRID[jj], TH_GRID[ij], marker="D", color=LIGHT, mec="k", ms=7, ls="none", zorder=8,
                label="CC + SN + CatWISE minimum")
    if not edge(iq, jq):
        ax.plot(T_GRID[jq], TH_GRID[iq], marker="p", color=QUAIA, mec="k", ms=10, ls="none", zorder=9,
                label="CC + SN + CatWISE + Quaia minimum")
    if not edge(i, j):
        ax.plot(T_GRID[j], TH_GRID[i], marker="*", color="white", mec="k", ms=13, ls="none", zorder=9,
                label=rf"CC + SN minimum ($\Delta\chi^2={P[i, j] - c_l:+.2f}$)")
    ax.set_xscale("log")
    ticks = [t for t in (1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0) if T_GRID[0] - 1e-9 <= t <= T_GRID[-1] + 1e-9]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xlim(T_GRID[0], T_GRID[-1]); ax.set_ylim(0, TH_GRID[-1])
    ax.set_xlabel(r"Present time $t_0$"); ax.set_ylabel(r"Observer sector $\vartheta_{\rm obs}$ [deg]")
    ax.set_title("Light-cone ball, unbinned supernovae, axis opposite to the CatWISE excess", fontsize=10)
    # legend in three blocks, lines, areas and points, each ordered by the data combined
    order = ["CC + SN: 68%", "fitted", "CC + SN + CatWISE: 68%", "Quaia slices: best fit", "all compatible",
             "CatWISE excess", "Quaia slices within",
             "CC + SN minimum", "CC + SN + CatWISE minimum", "CC + SN + CatWISE + Quaia minimum"]
    handles, labels = ax.get_legend_handles_labels()
    rank = lambda lab: next((n for n, key in enumerate(order) if lab.startswith(key)), len(order))
    blocks = [[], [], []]   # one column per block, below the axes
    for h, lab in sorted(zip(handles, labels), key=lambda hl: rank(hl[1])):
        r = rank(lab)
        blocks[0 if r < 5 else 1 if r < 7 else 2].append((h, lab))
    rows = max(len(b) for b in blocks)
    blank = (plt.Line2D([], [], alpha=0), "")
    cols = [b + [blank] * (rows - len(b)) for b in blocks]
    ax.legend([h for c in cols for h, _ in c], [lab for c in cols for _, lab in c], loc="upper center",
              bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8, framealpha=0.9)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hippopede_ball_zoom_unbinned{a.tag}.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
