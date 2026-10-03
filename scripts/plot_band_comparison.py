"""Spread of the projected sector histories compared with the GaPP band, for q(z) and E(z).

At the reference t0 (figures/t0_summary.json) the sectors of the lobe form, at each redshift, a
distribution weighted by the volume of the lobe, w(theta) ~ sin^2(2 theta) (sector_geometry.py);
E is expressed in common units through H_c,theta(t0)/H_c,0(t0) and normalised to the
sector-averaged present rate, the value of H0 an observer measures.

Top row: weighted median and the 1- and 2-sigma quantile bands of that distribution, flat
LCDM + radiation with Omega_m from fit_cc_pantheon.py (figures/fit_cc_pantheon.json) and
Omega_r = 9e-5,
(16-84 and 2.3-97.7 per cent) over the GaPP reconstruction (median, 1 and 2 sigma) used for the
estimate of t0; the outline of a second reconstruction is drawn for comparison.
Bottom row: ratio of the model half-width to the GaPP half-width, h_k / (k sigma_G), k = 1, 2,
for both reconstructions; the shaded interval is the redshift range used by
t0_dispersion_match.py.

Options: --gapp (reconstruction used for t0) and --gapp-alt (outline, default none), files written by
gapp_reconstruction.py; --t0 and --alpha-high (defaults: figures/t0_summary.json and
fit_alpha_high.py), or --from-average-fit to take both from fit_sector_average.py; --suffix for
the output name.  All bands are semi-transparent so that the
overlap of model and reconstruction is visible.  Writes figures/hippopede_band_comparison<suffix>.(png|pdf).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import hippopede_model as HM  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import t0_dispersion_match as DM  # noqa: E402

FIG = ROOT / "figures"
Z = np.round(np.arange(0.02, 2.001, 0.02), 3)
Z_FIT = (float(DM.Z_NODES[0]), float(DM.Z_NODES[-1]))
BLUE, BLUE_1S, BLUE_2S = "#08519c", "#6baed6", "#c6dbef"
LCDM_COLOUR = "#d62728"
OR_REF = 9.0e-5


def lcdm(om):
    """E(z) and q(z) of flat LCDM + radiation on Z."""
    zz = 1.0 + Z
    e2 = OR_REF * zz**4 + om * zz**3 + 1.0 - om - OR_REF
    q = (OR_REF * zz**4 + 0.5 * om * zz**3 - (1.0 - om - OR_REF)) / e2
    return {"E": np.sqrt(e2), "q": q}
GREY_1S, GREY_2S = "#bdbdbd", "#e0e0e0"


def sector_distribution(t0, alpha_high=None):
    """Weighted median and quantile bands of q and E across sectors on Z."""
    h_ax = HM.h0_sector(0.0, t0)
    Q, E = [], []
    for a in DM.THETA:
        e, q = HM.projected_sector(a, t0, alpha_high)
        r = HM.h0_sector(np.radians(a), t0) / h_ax
        Q.append(np.interp(Z, HM.Z_WORK, q, left=np.nan, right=np.nan))
        E.append(r * np.interp(Z, HM.Z_WORK, e, left=np.nan, right=np.nan))
    E = np.array(E)
    e0 = np.array([np.interp(0.0, HM.Z_WORK, HM.projected_sector(a, t0, alpha_high)[0]) * HM.h0_sector(np.radians(a), t0) / h_ax
                   for a in DM.THETA])
    E = E / (DM.W @ e0 / DM.W.sum())
    out = {}
    for name, X in (("q", np.array(Q)), ("E", E)):
        qt = lambda p: np.array([DM.weighted_quantile(X[:, j], DM.W, p) for j in range(len(Z))])
        out[name] = {"median": qt(0.5)}
        for k, (lo, hi) in DM.PROBS.items():
            out[name][k] = (qt(lo), qt(hi))
    return out


def load_gapp(fname):
    g = json.loads((FIG / fname).read_text())
    return {"q": (np.interp(Z, g["z"], g["q"]), np.interp(Z, g["z"], g["q_sigma"])),
            "E": (np.interp(Z, g["z"], g["e"]), np.interp(Z, g["z"], g["e_sigma"])),
            "N_CC": g["N_CC"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapp", default=DM.DEFAULT_GAPP)
    ap.add_argument("--gapp-alt", default="none", help="second reconstruction drawn as outline, or 'none'")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--alpha-high", type=float, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--from-average-fit", action="store_true",
                    help="take t0 and alpha_high from the fit to the sector-averaged history (fit_sector_average.py)")
    args = ap.parse_args()
    if args.from_average_fit:
        fit = json.loads((FIG / "fit_sector_average.json").read_text())["cases"]["high"]
        args.t0, args.alpha_high = fit["t0"], fit["alpha_high"]
    t0 = float(json.loads((FIG / "t0_summary.json").read_text())["figure_t0"]) if args.t0 is None else args.t0
    model = sector_distribution(t0, args.alpha_high)
    g = load_gapp(args.gapp)
    ga = None if args.gapp_alt == "none" else load_gapp(args.gapp_alt)
    om = round(json.loads((FIG / "fit_cc_pantheon.json").read_text())["rows"][0]["Omega_m"], 3)
    std = lcdm(om)

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
                         "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.2), dpi=180, sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1.2], "hspace": 0.06, "wspace": 0.22})
    labels = {"q": r"Deceleration parameter $q(z)$", "E": r"$E(z)=H(z)/H_0$"}
    for col, name in enumerate(("q", "E")):
        ax, axr = axs[0, col], axs[1, col]
        med_g, sig_g = g[name]
        for k, c in ((2, GREY_2S), (1, GREY_1S)):
            ax.fill_between(Z, med_g - k * sig_g, med_g + k * sig_g, color=c, alpha=0.55, lw=0, zorder=1)
            for sgn in (-1, 1):
                ax.plot(Z, med_g + sgn * k * sig_g, color="#636363", lw=0.8, zorder=1.5)
        ax.plot(Z, med_g, color="#737373", lw=2.4, zorder=2)
        if ga is not None:
            med_a, sig_a = ga[name]
            for k in (1, 2):
                for sgn in (-1, 1):
                    ax.plot(Z, med_a + sgn * k * sig_a, color="#525252", lw=1.0, ls=(0, (4, 3)), zorder=2)
        m = model[name]
        ax.fill_between(Z, *m[2], color=BLUE_2S, alpha=0.45, lw=0, zorder=3)
        ax.fill_between(Z, *m[1], color=BLUE_1S, alpha=0.45, lw=0, zorder=4)
        for k in (1, 2):
            for edge in m[k]:
                ax.plot(Z, edge, color=BLUE, lw=0.9, ls="-" if k == 1 else ":", zorder=4.5)
        ax.plot(Z, m["median"], color=BLUE, lw=2.0, zorder=5)
        ax.plot(Z, std[name], color=LCDM_COLOUR, lw=2.4, ls=(0, (6, 3)), alpha=0.95, zorder=6)
        ax.set_ylabel(labels[name])
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim((-2.0, 1.2) if name == "q" else (0.8, 3.4))
        ax.grid(alpha=0.25, lw=0.5)
        for gg, ls in ((g, "-"),) + (((ga, (0, (4, 3))),) if ga is not None else ()):
            _, sig = gg[name]
            for k, c in ((1, BLUE), (2, BLUE_1S)):
                w = 0.5 * (m[k][1] - m[k][0])
                axr.plot(Z, w / (k * sig), color=c, lw=2.0 if ls == "-" else 1.3, ls=ls)
        axr.axhline(1.0, color="k", lw=0.8)
        axr.axvspan(*Z_FIT, color="#fff3cd", zorder=0)
        axr.set_ylim(0.0, 2.5)
        axr.set_xlabel(r"Redshift $z$")
        axr.set_ylabel(r"$h_k^{\rm model}/(k\,\sigma_{\rm GaPP})$")
        axr.grid(alpha=0.25, lw=0.5)
    fig.text(0.075, 0.905, "(a)", fontsize=13); fig.text(0.515, 0.905, "(b)", fontsize=13)
    handles = [Patch(facecolor=BLUE_1S, alpha=0.45, edgecolor=BLUE), Patch(facecolor=BLUE_2S, alpha=0.45, edgecolor=BLUE, ls=":"),
               Line2D([0], [0], color=BLUE, lw=2), Line2D([0], [0], color=LCDM_COLOUR, lw=2.4, ls=(0, (6, 3))),
               Patch(facecolor=GREY_1S, alpha=0.55, edgecolor="#636363"), Patch(facecolor=GREY_2S, alpha=0.55, edgecolor="#636363"), Line2D([0], [0], color="#737373", lw=2.4),
               Line2D([0], [0], color=BLUE, lw=2), Line2D([0], [0], color=BLUE_1S, lw=2), Patch(color="#fff3cd")]
    ah = PH.load_alpha_high() if args.alpha_high is None else args.alpha_high
    names = [rf"Sectors, $1\sigma$ (16–84%), $t_0={t0:.2f}$, $\alpha_{{\rm high}}={ah:.3f}$", r"Sectors, $2\sigma$ (2.3–97.7%)", "Sectors, weighted median",
             rf"Flat $\Lambda$CDM + radiation ($\Omega_m={om:.3f}$)",
             rf"GaPP $1\sigma$ ({g['N_CC']} CC)", rf"GaPP $2\sigma$ ({g['N_CC']} CC)", "GaPP median",
             r"Width ratio, $1\sigma$", r"Width ratio, $2\sigma$", r"Range used to fit $t_0$"]
    if ga is not None:
        handles[7:7] = [Line2D([0], [0], color="#525252", lw=1.0, ls=(0, (4, 3)))]
        names[7:7] = [rf"GaPP $1\sigma$, $2\sigma$ ({ga['N_CC']} CC)"]
        handles.insert(-1, Line2D([0], [0], color=BLUE, lw=1.3, ls=(0, (4, 3))))
        names.insert(-1, rf"Width ratios against the {ga['N_CC']}-CC GaPP")
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    for ext in ("png", "pdf"):
        path = FIG / f"hippopede_band_comparison{args.suffix}.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
