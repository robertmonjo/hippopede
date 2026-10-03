"""Fig. 2: q(z) and E(z) of the hippopede sectors, centred (left) and projected (right).

Sectors theta = 0, 15, 30, 45, 60 deg at the present time t0 given on the command line
(default: the value recorded in figures/t0_summary.json by t0_summary.py, key 'figure_t0').
Each sector is drawn as an independent history (hippopede_model.py).  The projection uses
the running index with alpha_high from fit_alpha_high.py, or the value given with --alpha-high
(for instance the fit to the sector-averaged history, fit_sector_average.py).

Overlays: GaPP reconstruction from CC + binned Pantheon+ (option --gapp, a file written by
gapp_reconstruction.py; 1, 2, 3 sigma bands, extrapolation to z < 0 shaded differently); the CC
points used by that reconstruction, normalised by the same H0; present deceleration and transition redshift estimates
(Myrzakulov et al. 2025, Nucl. Phys. B 1016, 116916: q0 = -0.364 +- 0.032, z_t = 0.597 +- 0.214;
Gao et al. 2024, MNRAS 527, 7861: q0 = -0.50 +- 0.20; Hu et al. 2025, MNRAS 542, 1063:
z_t = 0.64 +- 0.16) and the tangents E = 1 + (1 + q0) z; flat LCDM + radiation with
Omega_m from the flat-LCDM fit of fit_cc_pantheon.py (figures/fit_cc_pantheon.json, run it first)
and Omega_r = 9e-5, as a visual reference.
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

import gapp_reconstruction as GR  # noqa: E402
import hippopede_model as HM  # noqa: E402

OUTDIR = ROOT / "figures"
THETAS = [0, 15, 30, 45, 60]
COLOURS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
Q0_POINTS = [
    {"q": -0.364, "err": 0.032, "marker": "D", "color": "#6a3d9a", "label": r"Observed $q_0$ (CC+Pantheon+SH0ES+BAO; Myrzakulov et al. 2025)"},
    {"q": -0.50, "err": 0.20, "marker": "s", "color": "#1b9e77", "label": r"Observed $q_0$ (FRB+SNe cosmography; Gao et al. 2024)"},
]
ZT_POINTS = [
    {"z": 0.597, "err": 0.214, "marker": "o", "color": "#f16913", "label": r"Observed $z_t$ (CC+Pantheon+SH0ES+BAO; Myrzakulov et al. 2025)"},
    {"z": 0.64, "err": 0.16, "marker": "^", "color": "#fd8d3c", "label": r"Observed $z_t$ ($H(z)$ compilation; Hu et al. 2025)"},
]
OM_REF = round(json.loads((ROOT / "figures" / "fit_cc_pantheon.json").read_text())["rows"][0]["Omega_m"], 3)
OR_REF = 9.0e-5


def e_lcdm_rad(z):
    return np.sqrt(OR_REF * (1 + z) ** 4 + OM_REF * (1 + z) ** 3 + 1 - OM_REF - OR_REF)


def q_lcdm_rad(z):
    e2 = e_lcdm_rad(z) ** 2
    return (OR_REF * (1 + z) ** 4 + 0.5 * OM_REF * (1 + z) ** 3 - (1 - OM_REF - OR_REF)) / e2


def style_panel(ax):
    ax.set_facecolor("#fefefe")
    ax.grid(True, which="major", color="#f6f6f6", lw=0.58, alpha=0.50)
    ax.grid(True, which="minor", color="#fbfbfb", lw=0.38, alpha=0.48)
    ax.minorticks_on()


def gapp_overlay(ax, g, key, zmax):
    z, y, s = g["z"], g[key], g[key + "_sigma"]
    for part, col, alphas in (((z >= 0) & (z <= zmax), "#8e8e8e", (0.15, 0.16, 0.17)),
                              ((z < 0), "#db7272", (0.08, 0.09, 0.10))):
        for k, al in zip((3, 2, 1), alphas[::-1]):
            ax.fill_between(z[part], (y - k * s)[part], (y + k * s)[part], color=col, alpha=al, linewidth=0.0, zorder=1.2)
        ax.plot(z[part], y[part], color="#8f8f8f" if col == "#8e8e8e" else "#c96a6a",
                lw=4.0 if col == "#8e8e8e" else 1.0, alpha=0.72 if col == "#8e8e8e" else 0.32, zorder=1.8)


def top_panel(ax, g, right):
    gapp_overlay(ax, g, "q", 2.0)
    z = np.linspace(-0.5, 2.0, 600)
    ax.plot(z, q_lcdm_rad(z), color="#d62728", lw=2.6, ls=(0, (6, 3)), alpha=0.9, zorder=2.0)
    ax.axhline(-1.0, color="#4d4d4d", lw=1.2, ls=":", alpha=0.95)
    ax.axhline(0.0, color="#777777", lw=1.0, alpha=0.65)
    ax.axvline(0.0, color="#999999", lw=1.0, ls="--", alpha=0.9)
    for it in Q0_POINTS:
        ax.errorbar(0.0, it["q"], yerr=it["err"], fmt=it["marker"], ms=7.2, mfc=it["color"], mec="white", mew=0.9,
                    ecolor=it["color"], elinewidth=1.5, capsize=4, zorder=6)
    for it in ZT_POINTS:
        ax.errorbar(it["z"], 0.0, xerr=it["err"], fmt=it["marker"], ms=7.0, mfc=it["color"], mec="white", mew=0.9,
                    ecolor=it["color"], elinewidth=1.5, capsize=4, zorder=6)
    style_panel(ax)
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(-4.0, 1.0)
    ax.set_ylabel(r"Deceleration parameter $q(z)$")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top(); ax.set_xlabel(r"Redshift $z$")
    if right:
        ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()


def bottom_panel(ax, g, cc, right):
    gapp_overlay(ax, g, "e", 2.0)
    z = np.linspace(-0.5, 2.0, 600)
    ax.plot(z, e_lcdm_rad(z), color="#d62728", lw=2.4, ls=(0, (6, 3)), alpha=0.9, zorder=2.0)
    cz, ch, sp, sm, h0 = cc
    ax.errorbar(cz, ch / h0, yerr=np.vstack([sm, sp]) / h0, fmt="o", ms=5.0, mfc="#2b8cbe", mec="white", mew=0.8,
                ecolor="#2b8cbe", elinewidth=1.1, capsize=2.5, alpha=0.96, zorder=6)
    zt = np.linspace(0.0, 0.8, 200)
    for it in Q0_POINTS:
        s = 1.0 + it["q"]
        ax.plot(zt, 1 + s * zt, color=it["color"], lw=1.8, ls=(0, (5, 3)))
        ax.fill_between(zt, 1 + (s - it["err"]) * zt, 1 + (s + it["err"]) * zt, color=it["color"], alpha=0.14, linewidth=0.0)
    style_panel(ax)
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(0.2, 3.2)
    ax.set_ylabel(r"$E(z)=H(z)/H_0$"); ax.set_xlabel(r"Redshift $z$")
    if right:
        ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()


def legends(fig):
    mh = [Line2D([0], [0], color=c, lw=2.4) for c in COLOURS] + [Line2D([0], [0], color="#d62728", lw=2.6, ls=(0, (6, 3))),
                                                                 Line2D([0], [0], color="#4d4d4d", lw=1.2, ls=":")]
    ml = [rf"Hippopede: $\vartheta={t}^\circ$" + (" (hyperconical)" if t == 0 else "") for t in THETAS] + [
        rf"Flat $\Lambda$CDM + radiation ($\Omega_m={OM_REF:.3f}$)", r"de Sitter: $q=-1$"]
    dh = [Line2D([0], [0], marker="o", color="#2b8cbe", lw=0, markersize=6)] + [
        Line2D([0], [0], marker=it["marker"], color=it["color"], lw=0, markersize=7) for it in Q0_POINTS] + [
        Line2D([0], [0], marker=it["marker"], color=it["color"], lw=1.2, markersize=7) for it in ZT_POINTS] + [
        Line2D([0], [0], color=it["color"], lw=1.8, ls=(0, (5, 3))) for it in Q0_POINTS]
    dl = [r"Cosmic chronometers, $H/H_0$"] + [it["label"] for it in Q0_POINTS] + [
        it["label"] for it in ZT_POINTS] + [r"Tangent $E=1+(1+q_0)z$, Myrzakulov et al. 2025",
                                            r"Tangent $E=1+(1+q_0)z$, Gao et al. 2024"]
    gh = [Line2D([0], [0], color="#8f8f8f", lw=4.0, alpha=0.72)] + [
        Patch(facecolor="#8e8e8e", alpha=a, edgecolor="none") for a in (0.45, 0.30, 0.15)] + [
        Patch(facecolor="#db7272", alpha=0.2, edgecolor="none")]
    gl = [r"GaPP median", r"GaPP $1\sigma$", r"GaPP $2\sigma$", r"GaPP $3\sigma$", r"GaPP extrapolation ($z<0$)"]
    fig.add_artist(fig.legend(mh, ml, loc="lower left", bbox_to_anchor=(0.10, 0.01), title=r"$\bf{Models}$", frameon=False))
    fig.add_artist(fig.legend(dh, dl, loc="lower center", bbox_to_anchor=(0.55, 0.01), title=r"$\bf{Data}$", frameon=False))
    fig.legend(gh, gl, loc="lower right", bbox_to_anchor=(0.93, 0.01), title=r"$\bf{GaPP\ CC{+}Pantheon{+}}$", frameon=False)


def default_t0():
    p = OUTDIR / "t0_summary.json"
    if p.exists():
        return float(json.loads(p.read_text())["figure_t0"])
    raise FileNotFoundError("figures/t0_summary.json not found: run scripts/t0_summary.py or pass --t0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--gapp", default="gapp_reconstruction_compilation_gls_cc.json",
                    help="GaPP reconstruction in figures/ (gapp_reconstruction.py)")
    ap.add_argument("--alpha-high", type=float, default=None,
                    help="asymptotic projection index (default: fit_alpha_high.py)")
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    ap.add_argument("--from-average-fit", action="store_true",
                    help="take t0 and alpha_high from the fit to the sector-averaged history (fit_sector_average.py)")
    args = ap.parse_args()
    if args.from_average_fit:
        fit = json.loads((OUTDIR / "fit_sector_average.json").read_text())["cases"]["high"]
        args.t0, args.alpha_high = fit["t0"], fit["alpha_high"]
    t0 = default_t0() if args.t0 is None else args.t0
    g = {k: (np.array(v) if isinstance(v, list) else v)
         for k, v in json.loads((OUTDIR / args.gapp).read_text()).items()}
    cz, ch, sp, sm = GR.load_cc_points(g["options"]["cc"])
    cc = (cz, ch, sp, sm, g["H0"])
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10.4,
                         "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, axs = plt.subplots(2, 2, figsize=(15.2, 9.6), dpi=180,
                            gridspec_kw={"height_ratios": [3.1, 1.35], "hspace": 0.09, "wspace": 0.07})
    (ax_ul, ax_ur), (ax_ll, ax_lr) = axs
    z = np.linspace(-0.5, 2.0, 1800)
    for th, col in zip(THETAS, COLOURS):
        zd, qd, ed = HM.build_centered_history(np.radians(th), t0)
        ax_ul.plot(z, HM.interp_curve(z, zd, qd), color=col, lw=2.4, zorder=3)
        ax_ll.plot(z, HM.interp_curve(z, zd, ed), color=col, lw=1.9, zorder=3)
        e_p, q_p = HM.projected_sector(th, t0, args.alpha_high)
        ax_ur.plot(z, np.interp(z, HM.Z_WORK, q_p, left=np.nan, right=np.nan), color=col, lw=2.4, zorder=3)
        ax_lr.plot(z, np.interp(z, HM.Z_WORK, e_p, left=np.nan, right=np.nan), color=col, lw=1.9, zorder=3)
    top_panel(ax_ul, g, False); top_panel(ax_ur, g, True)
    bottom_panel(ax_ll, g, cc, False); bottom_panel(ax_lr, g, cc, True)
    fig.text(0.078, 0.965, "(a)", fontsize=14); fig.text(0.520, 0.965, "(b)", fontsize=14)
    legends(fig)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.30)
    for ext in ("png", "pdf"):
        path = OUTDIR / f"hippopede_qz_double_panel_with_gapp{args.suffix}.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print(f"saved {path}")



if __name__ == "__main__":
    main()
