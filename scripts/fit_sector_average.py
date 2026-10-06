"""Running projection index fitted to the sector-averaged history, with t0 fixed by the band match.

Hypothesis (natural variability across sectors): the observed expansion history is the average
over the sectors of the lobe, weighted by their volume w(theta) ~ sin^2(2 theta) over
0 < theta < THETA_MAX_DEG (sector_geometry.py), and the spread across sectors is the width of the
GaPP band (t0_dispersion_match.py).  For the running index
    alpha(z) = alpha_high - (alpha_high - alpha_low)/sqrt(1+z)
and a present time t0 the model predicts
    H(z)    = H0 <E_theta(z)>,     D_C(z) = <int_0^z dz'/E_theta(z')>,
with <.> the weighted average and E_theta in common units, H_c,theta(t0)/H_c,0(t0).
Likelihood: CC compilation and binned Pantheon+ with their full covariances (likelihood.py),
H0 and the SN amplitude minimised analytically.

Fit, alternating until t0 changes by less than 1e-3:
  1. at fixed t0, minimise chi2(alpha_high) with alpha_low = 0.283 (case 'high'; the function
     fit() also accepts case 'both', alpha_low free, which is not used: lower alpha_high shortens
     the reach of the sectors near THETA_MAX_DEG below the highest data redshift);
  2. at fixed alpha, t0 minimises S(t0) = sum_k sum_j [ln w_k(z_j) - ln(k sigma_G(z_j))]^2, k = 1, 2,
     the joint 1- and 2-sigma band match of q(z) at z_j = 0.1, ..., 1.2 (option --z-nodes ZMIN ZMAX
     changes the range and adds the suffix _z<ZMIN>-<ZMAX> to the output).
Reported: parameters, chi2, Delta chi2 and Delta AIC relative to flat LCDM
(figures/fit_cc_pantheon.json), and the offset of the weighted median of q and E from the GaPP
median in units of sigma_G, with E normalised to the sector-averaged present rate.  Writes figures/fit_sector_average.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import hippopede_model as HM  # noqa: E402
import likelihood as L  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import t0_dispersion_match as DM  # noqa: E402

Z_TOP = float(np.ceil(max(L.CC[0].max(), L.Z_SN.max()) * 100) / 100)  # highest data redshift, 1.97
ZW = np.linspace(0.0, Z_TOP, 4001)  # starts exactly at z = 0, as required by D_C
W = DM.W / DM.W.sum()
GAPP = json.loads((ROOT / "figures" / DM.DEFAULT_GAPP).read_text())
NODES = DM.Z_NODES
G_MED = {"q": np.interp(NODES, GAPP["z"], GAPP["q"]), "E": np.interp(NODES, GAPP["z"], GAPP["e"])}
G_SIG = {"q": np.interp(NODES, GAPP["z"], GAPP["q_sigma"]), "E": np.interp(NODES, GAPP["z"], GAPP["e_sigma"])}


def set_nodes(zmin, zmax):
    """Band-match nodes z = zmin, ..., zmax (steps of 0.1) and the GaPP median and width there."""
    global NODES, G_MED, G_SIG
    DM.set_nodes(zmin, zmax)
    NODES = DM.Z_NODES
    G_MED = {"q": np.interp(NODES, GAPP["z"], GAPP["q"]), "E": np.interp(NODES, GAPP["z"], GAPP["e"])}
    G_SIG = {"q": np.interp(NODES, GAPP["z"], GAPP["q_sigma"]), "E": np.interp(NODES, GAPP["z"], GAPP["e_sigma"])}


def sector_table(al, ah, t0):
    """E_theta (common units) and q_theta on ZW for every sector; rows = sectors."""
    h_ax = HM.h0_sector(0.0, t0)
    E, Q = [], []
    for a in DM.THETA:
        e, q = HM.projected_sector(a, t0, ah, al)
        E.append(np.interp(ZW, HM.Z_WORK, e, left=np.nan, right=np.nan) * HM.h0_sector(np.radians(a), t0) / h_ax)
        Q.append(np.interp(ZW, HM.Z_WORK, q, left=np.nan, right=np.nan))
    return np.array(E), np.array(Q)


def chi2_average(al, ah, t0):
    E, _ = sector_table(al, ah, t0)
    ok = np.ones_like(ZW, dtype=bool)
    if not np.all(np.isfinite(E[:, ok])) or np.any(E[:, ok] <= 0):
        return np.inf
    e_bar = W @ E[:, ok]
    inv = 1.0 / E[:, ok]
    dc = np.concatenate([np.zeros((len(W), 1)), np.cumsum(0.5 * (inv[:, 1:] + inv[:, :-1]) * np.diff(ZW[ok]), axis=1)], axis=1)
    dc_bar = W @ dc
    zc, hc, _ = L.CC
    e = np.interp(zc, ZW[ok], e_bar)
    h0 = (e @ L.CINV_CC @ hc) / (e @ L.CINV_CC @ e)
    r = hc - h0 * e
    m = np.interp(L.Z_SN, ZW[ok], dc_bar)
    amp = (m @ L.CINV_SN @ L.D_SN) / (m @ L.CINV_SN @ m)
    s = L.D_SN - amp * m
    return float(r @ L.CINV_CC @ r + s @ L.CINV_SN @ s)


def node_stats(al, ah, t0):
    """Half-widths (k = 1, 2) and weighted medians of q and E at the nodes."""
    E, Q = sector_table(al, ah, t0)
    E = E / (W @ E[:, 0])  # normalised to the sector-averaged present rate, the H0 an observer measures
    out = {}
    for name, X in (("q", Q), ("E", E)):
        Xn = np.array([np.interp(NODES, ZW, row, left=np.nan, right=np.nan) for row in X])
        d = {"median": np.array([DM.weighted_quantile(Xn[:, j], DM.W, 0.5) for j in range(len(NODES))])}
        for k, (lo, hi) in DM.PROBS.items():
            d[k] = np.array([0.5 * (DM.weighted_quantile(Xn[:, j], DM.W, hi) - DM.weighted_quantile(Xn[:, j], DM.W, lo))
                             for j in range(len(NODES))])
        out[name] = d
    return out


def band_S(al, ah, t0):
    b = node_stats(al, ah, t0)["q"]
    return float(sum(np.sum((np.log(b[k]) - np.log(k * G_SIG["q"])) ** 2) for k in (1, 2)))


def fit(case, t0_start=2.6):
    al, ah, t0 = PH.ALPHA_LOW, PH.load_alpha_high(), t0_start
    for _ in range(12):
        if case == "both":
            r = minimize(lambda p: chi2_average(p[0], p[1], t0), [al, ah], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-4})
            al, ah = map(float, r.x)
        else:
            r = minimize_scalar(lambda a: chi2_average(al, a, t0), bounds=(0.2, 1.2), method="bounded",
                                options={"xatol": 1e-5})
            ah = float(r.x)
        r = minimize_scalar(lambda t: band_S(al, ah, t), bounds=(1.8, 6.0), method="bounded", options={"xatol": 1e-4})
        t_new = float(r.x)
        print(f"  {case}: alpha_low={al:.4f} alpha_high={ah:.4f} t0={t_new:.4f} chi2={chi2_average(al, ah, t_new):.3f} "
              f"S={r.fun:.3f}", flush=True)
        if abs(t_new - t0) < 1e-3:
            t0 = t_new
            break
        t0 = t_new
    return al, ah, t0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-nodes", nargs=2, type=float, default=(0.1, 1.2), metavar=("ZMIN", "ZMAX"))
    args = ap.parse_args()
    set_nodes(*args.z_nodes)
    sfx = DM.node_suffix(*args.z_nodes)
    lcdm = json.loads((ROOT / "figures" / "fit_cc_pantheon.json").read_text())["rows"][0]
    out = {"lcdm_chi2": lcdm["chi2"], "lcdm_k": lcdm["k"], "cases": {}}
    om = lcdm["Omega_m"]
    q_lcdm = 1.5 * om * (1 + NODES) ** 3 / (om * (1 + NODES) ** 3 + 1 - om) - 1.0
    out["lcdm_median_offset_q_sigma"] = ((q_lcdm - G_MED["q"]) / G_SIG["q"]).tolist()
    t_ref = float(json.loads((ROOT / "figures" / "t0_summary.json").read_text())["figure_t0"])
    base = (PH.ALPHA_LOW, PH.load_alpha_high(), t_ref)
    for case, params in (("axial_alpha", base), ("high", None)):
        al, ah, t0 = params if params else fit(case)
        c = chi2_average(al, ah, t0)
        k = {"axial_alpha": 3, "high": 4}[case]  # H0, SN amplitude, alpha(s) fitted here, t0
        st = node_stats(al, ah, t0)
        out["cases"][case] = {
            "alpha_low": al, "alpha_high": ah, "t0": t0, "chi2": c, "k": k,
            "dchi2": c - lcdm["chi2"], "dAIC": c - lcdm["chi2"] + 2 * (k - lcdm["k"]),
            "band_S": band_S(al, ah, t0),
            "median_offset_q_sigma": ((st["q"]["median"] - G_MED["q"]) / G_SIG["q"]).tolist(),
            "median_offset_E_sigma": ((st["E"]["median"] - G_MED["E"]) / G_SIG["E"]).tolist(),
            "width_ratio_q_1sigma": (st["q"][1] / G_SIG["q"]).tolist(),
            "width_ratio_q_2sigma": (st["q"][2] / (2 * G_SIG["q"])).tolist()}
        r = out["cases"][case]
        print(f"{case:12s} alpha_low={al:.4f} alpha_high={ah:.4f} t0={t0:.3f} chi2={c:.2f} "
              f"dchi2={r['dchi2']:+.2f} dAIC={r['dAIC']:+.2f} S={r['band_S']:.3f}")
        print("   q median offset/sigma_G:", np.round(r["median_offset_q_sigma"], 2).tolist())
        print("   E median offset/sigma_G:", np.round(r["median_offset_E_sigma"], 2).tolist())
    out["z_nodes"] = NODES.tolist()
    (ROOT / "figures" / f"fit_sector_average{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
