"""Redshift dependence of the quasar dipole excess: Quaia slices vs the hippopede shape.

Data: dipole_zslices/results.json (Quaia G<20.0, |b|>30 deg, selection function applied;
excess = measured dipole minus kinematic expectation along the CMB dipole, per slice).
Model: for an observer at small theta_obs the count dipole is D_i = theta_obs f_i(t0) along a
common sky direction u.  f_i is the dipole of quasar_dipole_fit.dipole for a flat p(z) inside
slice i, with the slice's own count slope x_i, at theta_obs = 1 deg (the dipole is linear in
theta_obs for small offsets, so the fitted amplitude a is theta_obs in degrees).  Reference t0
from figures/t0_summary.json plus a grid of values.
Fit: chi2 = sum_i |e_i - a f_i u|^2 / sigma_i^2 (per-component mock errors), minimised
analytically over a and u.  The same is done for a redshift-independent shape f_i = 1.
Only slices inside the model tables (z <= 2.1) are used.  Writes figures/quaia_zslice_model.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pantheon_offaxis_test as T  # noqa: E402
import sector_geometry as G  # noqa: E402
import quasar_dipole_fit as Q  # noqa: E402


def load_slices():
    r = json.loads((ROOT / "dipole_zslices" / "results.json").read_text())
    rows = []
    for s in r["slices"]:
        if not isinstance(s["z_range"], list):
            continue  # full sample
        lo, hi = s["z_range"]
        if hi > 1.85:
            continue  # full sample and slices beyond the tables
        d = s["dip"]
        rows.append({"z": (lo, hi), "x": s["x"], "e": np.array(d["excess_vec"]),
                     "sig": d["sigma_mocks_per_component"], "amp": d["excess_amp"]})
    return rows


def shape(tab, zlo, zhi, x):
    zq = Q.Z_Q
    p = ((zq >= zlo) & (zq <= zhi)).astype(float)
    p /= np.trapezoid(p, zq)
    return Q.dipole(tab, 1.0, x, p)[0]


def fit(e, sig, f):
    w = 1.0 / sig**2
    S = np.sum((w * f)[:, None] * e, axis=0)
    F = np.sum(w * f * f)
    chi2 = float(np.sum(w * np.sum(e * e, axis=1)) - S @ S / F)
    a = float(np.linalg.norm(S) / F)
    return chi2, a, S / np.linalg.norm(S)


def main():
    rows = load_slices()
    e = np.array([r["e"] for r in rows]); sig = np.array([r["sig"] for r in rows])
    print("slices:", [r["z"] for r in rows], " excess amplitudes:", [round(r["amp"], 4) for r in rows])
    chi2_0 = float(np.sum(np.sum(e * e, axis=1) / sig**2))
    c_const, a_const, _ = fit(e, sig, np.ones(len(rows)))
    print(f"no dipole: chi2={chi2_0:.2f} (dof {3 * len(rows)});  constant shape: chi2={c_const:.2f}, amplitude={a_const:.4f}")
    out = {"slices": [{"z": r["z"], "x": r["x"], "excess_amp": r["amp"], "sigma": r["sig"]} for r in rows],
           "chi2_null": chi2_0, "chi2_const": c_const, "amp_const": a_const, "model": {}}
    t_ref = float(json.loads((ROOT / "figures" / "t0_summary.json").read_text())["figure_t0"])
    out["reference_t0"] = t_ref
    for t0 in sorted({t_ref, 2.5, 3.0, 4.0, 6.0, 10.0}):
        tab = T.e_table(t0)
        f = np.array([shape(tab, *r["z"], r["x"]) for r in rows])
        c, a, u = fit(e, sig, np.abs(f))
        pred = a * np.abs(f)
        print(f"t0={t0}: f(1 deg)={np.round(np.abs(f), 5)}  best theta_obs={a:.2f} deg  pred={np.round(pred, 4)}  "
              f"chi2={c:.2f}  (constant {c_const:.2f}; delta={c - c_const:+.2f})"
              + ("" if a <= G.THETA_OBS_MAX_DEG else f"  [theta_obs beyond {G.THETA_OBS_MAX_DEG:g} deg]"))
        out["model"][str(t0)] = {"f_1deg": np.abs(f).tolist(), "theta_obs": a, "pred": pred.tolist(), "chi2": c,
                                 "within_observer_range": bool(a <= G.THETA_OBS_MAX_DEG)}
    (ROOT / "figures" / "quaia_zslice_model.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
