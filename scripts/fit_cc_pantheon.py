"""Table 2: chi-square comparison of expansion histories with CC + binned Pantheon+.

Each hippopede sector is treated as an independent history (hippopede_model.py), unprojected
and projected with the running index (alpha_high from fit_alpha_high.py).  Likelihood:
likelihood.py (H0 and the SN amplitude minimised analytically; full SN covariance).  The GaPP
band of Fig. 2 is a reconstruction from these same data and is not fitted.

Models: flat LCDM (Omega_m free); projected hyperconical history with constant alpha = alpha_low
and with alpha(z); sectors at the reference t0 (json/t0_summary.json, 'figure_t0') and with
t0 free.  For each projected oblique sector the lower bounds on t0 at Delta chi2 = 1 and 4
relative to t0 -> infinity (where every sector reduces to the axial history) are reported.
Delta AIC = Delta chi2 + 2 Delta k, with k counting H0, the SN amplitude, alpha_high when the
projection uses the fitted running index (it is fitted to these same data), Omega_m for LCDM and t0
when free.
Writes json/fit_cc_pantheon.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import likelihood as L  # noqa: E402
import projected_hyperconical as PH  # noqa: E402

SECTORS = [0, 15, 30, 45, 60]


def chi2_tot(E_fn):
    return sum(L.total(E_fn)[:2])


def row(name, E_fn, k, extra=None):
    c_cc, c_sn, h0 = L.total(E_fn)
    out = {"model": name, "chi2_cc": c_cc, "chi2_sn": c_sn, "chi2": c_cc + c_sn, "k": k, "H0": h0}
    zmax_data = float(max(L.CC[0].max(), L.Z_SN.max()))
    if not np.isfinite(c_cc + c_sn):
        # the history ends below the highest data redshift; report its reach instead of a chi2
        zz = np.linspace(0.0, zmax_data, 4001)
        e = np.asarray(E_fn(zz), float)
        bad = ~np.isfinite(e) | (e <= 0)
        out.update({"chi2_cc": None, "chi2_sn": None, "chi2": None, "H0": None,
                    "history_ends_at_z": float(zz[np.argmax(bad)]), "highest_data_z": zmax_data})
    out.update(extra or {})
    return out


def best_t0(th, projected):
    grid = np.geomspace(1.0, 60.0, 40)
    c = [chi2_tot(L.e_sector(th, t, projected)) for t in grid]
    i = int(np.nanargmin(c))
    r = minimize_scalar(lambda t: chi2_tot(L.e_sector(th, t, projected)),
                        bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method="bounded")
    return float(r.x)


def t0_bounds(level):
    c_inf = chi2_tot(L.e_hyperconical())
    out = {}
    for th in SECTORS[1:]:
        f = lambda t: chi2_tot(L.e_sector(th, t, True)) - c_inf - level
        grid = np.geomspace(0.8, 60.0, 30)
        v = [f(t) for t in grid]
        idx = [k for k in range(len(v) - 1) if v[k] > 0 >= v[k + 1]]
        out[str(th)] = float(brentq(f, grid[idx[-1]], grid[idx[-1] + 1], xtol=1e-3)) if idx else None
    return out


def main():
    t0_fig = float(json.loads((ROOT / "json" / "t0_summary.json").read_text())["figure_t0"])
    rows = []
    res = minimize_scalar(lambda om: chi2_tot(L.e_lcdm(om)), bounds=(0.05, 0.8), method="bounded")
    rows.append(row("LCDM", L.e_lcdm(res.x), 3, {"Omega_m": float(res.x)}))
    const = lambda z: PH.E_of_z(np.asarray(z, float), PH.alpha_const(PH.ALPHA_LOW))
    rows.append(row("hyperconical, constant alpha_low", const, 2))
    rows.append(row("hyperconical, alpha(z)", L.e_hyperconical(), 3))
    for projected in (False, True):
        tag = "projected" if projected else "unprojected"
        for th in SECTORS:
            k = 3 if projected else 2
            rows.append(row(f"{tag} {th}", L.e_sector(th, t0_fig, projected), k, {"theta": th, "t0": t0_fig, "projected": projected}))
            if th:
                t = best_t0(th, projected)
                rows.append(row(f"{tag} {th} t0 free", L.e_sector(th, t, projected), k + 1,
                                {"theta": th, "t0": t, "projected": projected}))
    ref = rows[0]["chi2"] + 2 * rows[0]["k"]
    for r in rows:
        r["dchi2"] = None if r["chi2"] is None else r["chi2"] - rows[0]["chi2"]
        r["dAIC"] = None if r["chi2"] is None else r["chi2"] + 2 * r["k"] - ref
    out = {"N_CC": len(L.CC[0]), "N_SN_bins": len(L.Z_SN), "figure_t0": t0_fig, "rows": rows,
           "t0_lower_bound_1sigma": t0_bounds(1.0), "t0_lower_bound_2sigma": t0_bounds(4.0)}
    (ROOT / "json" / "fit_cc_pantheon.json").write_text(json.dumps(out, indent=1))
    print(f"N_CC={out['N_CC']}  N_SN={out['N_SN_bins']}  reference t0={t0_fig}")
    for r in rows:
        ex = {k: round(v, 3) for k, v in r.items() if k in ("Omega_m", "t0")}
        if r["chi2"] is None:
            print(f"{r['model']:34s} history ends at z = {r['history_ends_at_z']:.3f} < {r['highest_data_z']:.3f} {ex}")
            continue
        print(f"{r['model']:34s} {r['chi2_cc']:8.2f} {r['chi2_sn']:8.2f} {r['chi2']:8.2f} k={r['k']} "
              f"dchi2={r['dchi2']:+8.2f} dAIC={r['dAIC']:+8.2f} H0={r['H0']:6.2f} {ex}")
    print("t0 lower bounds 1 sigma:", out["t0_lower_bound_1sigma"])
    print("t0 lower bounds 2 sigma:", out["t0_lower_bound_2sigma"])


if __name__ == "__main__":
    main()
