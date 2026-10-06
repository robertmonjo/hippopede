"""Sectors weighted by the part of the lobe that light from redshift z has crossed.

Observer on the axis.  Light from redshift z comes from angular distance rhat(z), i.e. from the
sector rhat(z)/2, and on its way it crosses the sectors 0 <= theta <= rhat(z)/2.  In this reading
the sectors that contribute at redshift z are those inside that ball, weighted by their volume:
    w(theta; z) ~ sin^2(2 theta)   for theta <= rhat(z)/2,   0 otherwise
(normalised at each z; for rhat/2 below the first tabulated sector only that sector is used).
Each sector is an independent projected history in common units (fit_sector_average.py).
Prediction: <E>(z) for the chronometers and <D_C>(z) = sum_theta w(theta; z) D_C,theta(z) for the
supernovae; spread: weighted quantiles of q_theta(z).  Fit, alternating until t0 converges:
alpha_high minimises chi2 at fixed t0, and t0 minimises the band mismatch M at fixed alpha_high
(t0_dispersion_match.py) on a grid 1 <= t0 <= 6.  Only the sectors theta <= 40 deg are used: the ball
reaches rhat(1.97)/2 = 33 deg at most.  Compared with the option of one observer's sky (fit_observer_sky.py).

Usage: python scripts/fit_lightcone_ball.py [--plot]
Writes figures/fit_lightcone_ball.json and, with --plot, figures/hippopede_lightcone_ball_bands.(png|pdf).
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

import fit_sector_average as F  # noqa: E402
import likelihood as L  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import t0_dispersion_match as DM  # noqa: E402

FIG = ROOT / "figures"
THETA_MAX_BALL = 40.0  # the ball never exceeds rhat(1.97)/2 = 33 deg, so larger sectors are not needed
SEL = DM.THETA <= THETA_MAX_BALL
THETA = DM.THETA[SEL]
ZW = F.ZW
M_RANGE = (DM.Z_NODES[0], DM.Z_NODES[-1])  # redshift range of the band match, shaded in the figure


def weights(ah):
    """w[theta, z]: volume weight inside the ball theta <= rhat(z)/2, normalised at each z."""
    run = lambda z: PH.alpha_sqrt(z, PH.ALPHA_LOW, ah)
    th_max = np.degrees(PH.rhat_of_z(ZW, run)) / 2.0
    w = np.sin(2.0 * np.radians(THETA))[:, None] ** 2 * (THETA[:, None] <= np.maximum(th_max, THETA[0])[None, :])
    return w / w.sum(axis=0)


def model(ah, t0):
    E, Q = F.sector_table(PH.ALPHA_LOW, ah, t0)
    E, Q = E[SEL], Q[SEL]
    if not np.all(np.isfinite(E)) or np.any(E <= 0):
        return None
    w = weights(ah)
    inv = 1.0 / E
    dc = np.concatenate([np.zeros((len(THETA), 1)), np.cumsum(0.5 * (inv[:, 1:] + inv[:, :-1]) * np.diff(ZW), axis=1)], axis=1)
    return E, Q, w, (w * E).sum(0), (w * dc).sum(0)


def chi2(ah, t0):
    m = model(ah, t0)
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


def quantile_bands(X, w):
    """Weighted median and 1-, 2-sigma quantiles of X[theta, z] with weights w[theta, z]."""
    out = {"median": np.array([DM.weighted_quantile(X[:, j], w[:, j], 0.5) for j in range(X.shape[1])])}
    for k, (lo, hi) in DM.PROBS.items():
        out[k] = (np.array([DM.weighted_quantile(X[:, j], w[:, j], lo) for j in range(X.shape[1])]),
                  np.array([DM.weighted_quantile(X[:, j], w[:, j], hi) for j in range(X.shape[1])]))
    return out


def mismatch(ah, t0):
    m = model(ah, t0)
    if m is None:
        return np.inf, None
    _, Q, w, _, _ = m
    idx = [int(np.argmin(np.abs(ZW - z))) for z in DM.Z_NODES]
    b = quantile_bands(Q[:, idx], w[:, idx])
    h = {k: 0.5 * (b[k][1] - b[k][0]) for k in (1, 2)}
    with np.errstate(divide="ignore"):
        M = float(sum(np.sum((np.log(h[k]) - np.log(k * F.G_SIG["q"])) ** 2) for k in (1, 2)))
    return M, (h[1] / F.G_SIG["q"]).tolist()


T_GRID = np.round(np.geomspace(1.0, 6.0, 41), 4)


def best_t0(ah):
    """t0 minimising M on T_GRID, refined by a parabola in ln t0 around the grid minimum."""
    v = np.array([mismatch(ah, t)[0] for t in T_GRID])
    i = int(np.nanargmin(np.where(np.isfinite(v), v, np.nan)))
    if 0 < i < len(T_GRID) - 1 and np.all(np.isfinite(v[i - 1:i + 2])):
        x = np.log(T_GRID[i - 1:i + 2])
        c = np.polyfit(x, v[i - 1:i + 2], 2)
        if c[0] > 0:
            return float(np.exp(-c[1] / (2 * c[0])))
    return float(T_GRID[i])


def fit():
    ah, t0 = 0.36, 2.6
    for _ in range(10):
        ah = float(minimize_scalar(lambda a: chi2(a, t0), bounds=(0.25, 0.55), method="bounded", options={"xatol": 1e-4}).x)
        t_new = best_t0(ah)
        print(f"  alpha_high={ah:.4f}  t0={t_new:.3f}  chi2={chi2(ah, t_new):.2f}  M={mismatch(ah, t_new)[0]:.2f}", flush=True)
        if abs(t_new - t0) < 1e-3:
            t0 = t_new
            break
        t0 = t_new
    return ah, t0


def plot(ah, t0, chi_lcdm, out_name="hippopede_lightcone_ball_bands", title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import plot_observer_sky as P
    E, Q, w, _, _ = model(ah, t0)
    e0 = (w[:, 0] * E[:, 0]).sum()
    bands = {"q": quantile_bands(Q, w), "E": quantile_bands(E / e0, w)}
    lc = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]
    g = json.loads((FIG / DM.DEFAULT_GAPP).read_text())
    gm = {"q": (np.interp(ZW, g["z"], g["q"]), np.interp(ZW, g["z"], g["q_sigma"])),
          "E": (np.interp(ZW, g["z"], g["e"]), np.interp(ZW, g["z"], g["e_sigma"]))}
    zz = 1.0 + ZW
    om = round(lc["Omega_m"], 3)
    e2 = 9e-5 * zz**4 + om * zz**3 + 1 - om - 9e-5
    std = {"E": np.sqrt(e2), "q": (9e-5 * zz**4 + 0.5 * om * zz**3 - (1 - om - 9e-5)) / e2}
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
                         "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.2), dpi=180, sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1.2], "hspace": 0.06, "wspace": 0.22})
    for col, name in enumerate(("q", "E")):
        ax, axr = axs[0, col], axs[1, col]
        med_g, sig_g = gm[name]
        for k, c in ((2, P.GREY_2S), (1, P.GREY_1S)):
            ax.fill_between(ZW, med_g - k * sig_g, med_g + k * sig_g, color=c, alpha=0.55, lw=0, zorder=1)
            for sgn in (-1, 1):
                ax.plot(ZW, med_g + sgn * k * sig_g, color="#636363", lw=0.8, zorder=1.5)
        ax.plot(ZW, med_g, color="#737373", lw=2.4, zorder=2)
        m = bands[name]
        ax.fill_between(ZW, *m[2], color=P.BLUE_2S, alpha=0.45, lw=0, zorder=3)
        ax.fill_between(ZW, *m[1], color=P.BLUE_1S, alpha=0.45, lw=0, zorder=4)
        for k in (1, 2):
            for edge in m[k]:
                ax.plot(ZW, edge, color=P.BLUE, lw=0.9, ls="-" if k == 1 else ":", zorder=4.5)
        ax.plot(ZW, m["median"], color=P.BLUE, lw=2.0, zorder=5)
        ax.plot(ZW, std[name], color=P.RED, lw=2.4, ls=(0, (6, 3)), zorder=6)
        ax.set_ylabel(r"Deceleration parameter $q(z)$" if name == "q" else r"$E(z)=H(z)/H_0$")
        ax.set_xlim(0, 2); ax.set_ylim((-2.0, 1.2) if name == "q" else (0.8, 3.4)); ax.grid(alpha=0.25, lw=0.5)
        for k, c in ((1, P.BLUE), (2, P.BLUE_1S)):
            axr.plot(ZW, 0.5 * (m[k][1] - m[k][0]) / (k * sig_g), color=c, lw=2.0)
        axr.axhline(1.0, color="k", lw=0.8); axr.axvspan(*M_RANGE, color="#fff3cd", zorder=0)
        axr.set_ylim(0, 2.5); axr.set_xlabel(r"Redshift $z$"); axr.set_ylabel(r"$h_k^{\rm model}/(k\,\sigma_{\rm GaPP})$")
        axr.grid(alpha=0.25, lw=0.5)
    c, M = chi2(ah, t0), mismatch(ah, t0)[0]
    head = title or r"Observer on the axis, sectors $\vartheta\leq\hat r(z)/2$ weighted by volume"
    fig.suptitle(head + rf": $t_0={t0:.2f}$, $\alpha_{{\rm high}}={ah:.3f}$, $\Delta\chi^2={c - chi_lcdm:+.2f}$ vs $\Lambda$CDM, "
                 rf"$\mathcal{{M}}={M:.2f}$", y=0.95, fontsize=12)
    handles = [Patch(facecolor=P.BLUE_1S, alpha=0.45, edgecolor=P.BLUE), Patch(facecolor=P.BLUE_2S, alpha=0.45, edgecolor=P.BLUE),
               Line2D([0], [0], color=P.BLUE, lw=2), Line2D([0], [0], color=P.RED, lw=2.4, ls=(0, (6, 3))),
               Patch(facecolor=P.GREY_1S, alpha=0.55, edgecolor="#636363"), Patch(facecolor=P.GREY_2S, alpha=0.55, edgecolor="#636363"),
               Line2D([0], [0], color="#737373", lw=2.4), Line2D([0], [0], color=P.BLUE, lw=2),
               Line2D([0], [0], color=P.BLUE_1S, lw=2), Patch(color="#fff3cd")]
    names = [r"Sectors in the ball, $1\sigma$", r"Sectors in the ball, $2\sigma$", "Weighted median",
             rf"Flat $\Lambda$CDM + radiation ($\Omega_m={om:.3f}$)", r"GaPP $1\sigma$ (38 CC)", r"GaPP $2\sigma$ (38 CC)",
             "GaPP median", r"Width ratio, $1\sigma$", r"Width ratio, $2\sigma$", r"Range of $\mathcal{M}$"]
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{out_name}.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print(f"saved figures/{out_name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    chi_lcdm = json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["chi2"]
    ah, t0 = fit()
    c = chi2(ah, t0)
    M, ratio = mismatch(ah, t0)
    out = {"alpha_high": ah, "t0": t0, "chi2": c, "dchi2": c - chi_lcdm, "M": M, "h1_over_sigma_q": ratio,
           "z_nodes": DM.Z_NODES.tolist()}
    print(f"best: alpha_high={ah:.4f} t0={t0:.3f} dchi2={c - chi_lcdm:+.2f} M={M:.2f}")
    print("h1/sigma_G at z_nodes:", np.round(ratio, 2).tolist())
    (FIG / "fit_lightcone_ball.json").write_text(json.dumps(out, indent=1))
    if args.plot:
        plot(ah, t0, chi_lcdm)


if __name__ == "__main__":
    main()
