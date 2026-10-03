"""t0 at which the spread of the projected sector histories reproduces the GaPP band.

Hypothesis tested: the width of the Gaussian-process reconstruction reflects a physical spread of
expansion histories across the sectors of the lobe.  At each redshift the sectors form a
distribution weighted by the volume of the lobe, w(theta) ~ sin^2(2 theta), over the sectors
0 < theta < THETA_MAX_DEG = 70 deg (sector_geometry.py).
E is expressed in common units through H_c,theta(t0)/H_c,0(t0); q is unaffected by that factor.

For X = q or E and each node z_j the model bands are the weighted quantiles of X_theta(z_j; t0):
    1-sigma half-width  w_1 = [Q(0.8413) - Q(0.1587)] / 2,
    2-sigma half-width  w_2 = [Q(0.9772) - Q(0.0228)] / 2,
and the GaPP bands are sigma_G(z_j) and 2 sigma_G(z_j) (gapp_reconstruction.py; option --gapp
selects one or more reconstructions, default figures/gapp_reconstruction_compilation_gls_cc.json).  t0 minimises
    S_k(t0) = sum_j [ln w_k(z_j; t0) - ln(k sigma_G(z_j))]^2,   k = 1, 2,
and S_1 + S_2 for the joint match, over the nodes z_j = 0.1, ..., 1.2 (the range of the binned
supernovae).  The bands are computed on a grid of t0 and interpolated linearly in ln t0.  The GaPP band is strongly correlated between nodes (the trained correlation length
is larger than the node range), so S is not a chi-square.  The uncertainty is quoted in two ways:
the 16-84 per cent range of the single-node solutions w_k(z_j; t0) = k sigma_G(z_j), and the
jackknife standard error over nodes.  The offset of the weighted median of the model from the
GaPP median is reported in units of sigma_G.
Writes figures/t0_dispersion_match.json (default reconstruction) or
figures/t0_dispersion_match_<reconstruction>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import hippopede_model as HM  # noqa: E402
import sector_geometry as G  # noqa: E402

Z_NODES = np.round(np.arange(0.1, 1.21, 0.1), 2)
THETA = np.arange(0.25, G.THETA_MAX_DEG, 0.5)
W = G.lobe_volume_weight(THETA)
W /= W.sum()
PROBS = {1: (0.15866, 0.84134), 2: (0.02275, 0.97725)}
T_GRID = np.geomspace(1.2, 40.0, 90)
T_FINE = np.geomspace(1.2, 40.0, 20001)


def weighted_quantile(x, w, p):
    """Quantile p of the discrete distribution x with weights w (linear interpolation of the CDF)."""
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.nan
    o = np.argsort(x[ok])
    xs, ws = x[ok][o], w[ok][o]
    cdf = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return float(np.interp(p, cdf, xs))


def sector_values(t0):
    """q_theta(z_j) and E_theta(z_j) (common units) for every sector; rows = sectors."""
    h_ax = HM.h0_sector(0.0, t0)
    Q, E = [], []
    for a in THETA:
        e, q = HM.projected_sector(a, t0)
        r = HM.h0_sector(np.radians(a), t0) / h_ax
        Q.append(np.interp(Z_NODES, HM.Z_WORK, q, left=np.nan, right=np.nan))
        E.append(r * np.interp(Z_NODES, HM.Z_WORK, e, left=np.nan, right=np.nan))
    return {"q": np.array(Q), "E": np.array(E)}


def band_table():
    """ln half-widths (k = 1, 2) and medians of the sector distribution on T_GRID x Z_NODES."""
    tab = {name: {"ln1": [], "ln2": [], "median": []} for name in ("q", "E")}
    for t0 in T_GRID:
        for name, X in sector_values(t0).items():
            cols = range(len(Z_NODES))
            tab[name]["median"].append([weighted_quantile(X[:, j], W, 0.5) for j in cols])
            for k, (lo, hi) in PROBS.items():
                tab[name][f"ln{k}"].append([np.log(0.5 * (weighted_quantile(X[:, j], W, hi) - weighted_quantile(X[:, j], W, lo)))
                                            for j in cols])
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in tab.items()}


def fine(tab_nk):
    """Interpolate a (T_GRID x nodes) table linearly in ln t0 onto T_FINE."""
    x, xf = np.log(T_GRID), np.log(T_FINE)
    return np.array([np.interp(xf, x, tab_nk[:, j]) for j in range(tab_nk.shape[1])]).T


def fit(lnw, ks, sig, nodes):
    """t0 minimising sum_k sum_{j in nodes} [ln w_k - ln(k sigma_G)]^2 on T_FINE."""
    S = sum(np.nansum((lnw[k][:, nodes] - np.log(k * sig[nodes])) ** 2, axis=1) for k in ks)
    i = int(np.nanargmin(S))
    return float(T_FINE[i]), float(S[i]), bool(i in (0, len(T_FINE) - 1))


def node_solution(lnw, k, sig, j):
    f = lnw[k][:, j] - np.log(k * sig[j])
    idx = np.where(np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (np.sign(f[:-1]) != np.sign(f[1:])))[0]
    if not len(idx):
        return None
    i = idx[0]
    x0, x1 = np.log(T_FINE[i]), np.log(T_FINE[i + 1])
    return float(np.exp(x0 - f[i] * (x1 - x0) / (f[i + 1] - f[i])))


DEFAULT_GAPP = "gapp_reconstruction_compilation_gls_cc.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapp", nargs="+", default=[DEFAULT_GAPP], help="reconstruction files in figures/")
    args = ap.parse_args()
    tab = band_table()
    all_nodes = np.arange(len(Z_NODES))
    for gfile in args.gapp:
        g = json.loads((ROOT / "figures" / gfile).read_text())
        gapp = {"q": (np.interp(Z_NODES, g["z"], g["q"]), np.interp(Z_NODES, g["z"], g["q_sigma"])),
                "E": (np.interp(Z_NODES, g["z"], g["e"]), np.interp(Z_NODES, g["z"], g["e_sigma"]))}
        res = {"gapp": gfile, "z_nodes": Z_NODES.tolist(), "t0_grid": [float(T_GRID[0]), float(T_GRID[-1])]}
        print(f"=== {gfile}")
        for name in ("q", "E"):
            med_g, sig = gapp[name]
            lnw = {1: fine(tab[name]["ln1"]), 2: fine(tab[name]["ln2"])}
            med = fine(tab[name]["median"])
            out = {}
            for label, ks in (("1sigma", (1,)), ("2sigma", (2,)), ("joint", (1, 2))):
                t_best, s_min, edge = fit(lnw, ks, sig, all_nodes)
                jack = np.array([fit(lnw, ks, sig, np.delete(all_nodes, j))[0] for j in all_nodes])
                n = len(jack)
                se_jack = float(np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2)))
                nodes = [node_solution(lnw, k, sig, j) for k in ks for j in all_nodes]
                found = np.array([x for x in nodes if x is not None])
                i = int(np.argmin(np.abs(T_FINE - t_best)))
                w1, w2 = np.exp(lnw[1][i]), np.exp(lnw[2][i])
                out[label] = {
                    "t0_best": t_best, "S_min": s_min, "at_grid_edge": edge, "jackknife_se": se_jack,
                    "node_solutions": nodes,
                    "node_16_84": [float(np.percentile(found, 16)), float(np.percentile(found, 84))] if len(found) else None,
                    "nodes_without_solution": int(sum(x is None for x in nodes)),
                    "model_halfwidth_1sigma": w1.tolist(), "model_halfwidth_2sigma": w2.tolist(),
                    "gapp_sigma": sig.tolist(), "median_offset_in_sigma_G": ((med[i] - med_g) / sig).tolist()}
                r = out[label]
                print(f"{name} {label:6s}: t0 = {t_best:.3f} +- {se_jack:.3f} (jackknife); node 16-84% "
                      f"{[round(x, 2) for x in r['node_16_84']] if r['node_16_84'] else None}; "
                      f"no solution at {r['nodes_without_solution']} nodes; S_min = {s_min:.3f}; edge = {edge}")
                if label == "joint":
                    print("   model/GaPP 1s:", np.round(w1 / sig, 2).tolist())
                    print("   model/GaPP 2s:", np.round(w2 / (2 * sig), 2).tolist())
            res[name] = out
        fname = "t0_dispersion_match.json" if gfile == DEFAULT_GAPP else             "t0_dispersion_match_" + gfile.replace("gapp_reconstruction_", "")
        (ROOT / "figures" / fname).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
