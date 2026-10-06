"""Joint chi2 of CC + SN + CatWISE + Quaia for isotropic models with and without a phenomenological dipole,
compared with the light-cone average of the hippopede.

The dipole data enter as in the joint fit of Sect. 3.6 (plot_ball_zoom_unbinned.py, fit_ball_unbinned.py): the
CatWISE excess amplitude, chi2 = ((A - D_geo)/sigma)^2, and the excess amplitudes of the three Quaia slices,
chi2 = sum_i ((A_i - a_i)/s_i)^2; directions are not compared, so every model has a free dipole direction.
Models (the chronometers and supernovae, 38 CC + 1579 unbinned SN, from fit_ball_unbinned.json):
  LCDM, hyperconical      no dipole beyond the kinematic one (A = 0);
  LCDM + common dipole    one amplitude for CatWISE and the three Quaia slices;
  LCDM + two dipoles      one amplitude for CatWISE and one, redshift-independent, for Quaia;
  light-cone ball         amplitudes predicted by the geometry (refine_overview_minimum.json).
Parameters counted beyond the nuisance ones (H0, M): Omega_m or alpha_high, the dipole amplitudes, and t0 and
theta_obs for the ball; the dipole direction is not counted for any model.  Writes json/compare_dipole_models.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "json"   # numerical outputs; figures/ holds the png/pdf


def main():
    fbu = json.loads((JSON / "fit_ball_unbinned.json").read_text())
    qd = json.loads((JSON / "quasar_dipole_fit_ah0p34.json").read_text())
    qz = json.loads((JSON / "quaia_zslice_model_ah0p34.json").read_text())
    ball = json.loads((JSON / "refine_overview_minimum.json").read_text())["continuous_best"]
    d_geo, s_geo = qd["D_geo"], qd["sigma"]
    a = np.array([x["excess_amp"] for x in qz["slices"]])
    s = np.array([x["sigma"] for x in qz["slices"]])
    w = 1.0 / s**2

    cat0, q0 = (d_geo / s_geo) ** 2, float(np.sum(a**2 * w))
    # common amplitude: weighted mean of the four amplitudes
    W = np.r_[1.0 / s_geo**2, w]
    X = np.r_[d_geo, a]
    A_c = float(np.sum(W * X) / np.sum(W))
    cat_c, q_c = ((A_c - d_geo) / s_geo) ** 2, float(np.sum((A_c - a) ** 2 * w))
    # two amplitudes: CatWISE fitted exactly, Quaia by its weighted mean
    A_q = float(np.sum(w * a) / np.sum(w))
    q_2 = float(np.sum((A_q - a) ** 2 * w))

    dh = fbu["hyperconical"]["dchi2_vs_lcdm"]
    rows = [
        ("LCDM", 0.0, cat0, q0, 1, {}),
        ("hyperconical", dh, cat0, q0, 1, {"alpha_high": fbu["hyperconical"]["alpha_high"]}),
        ("LCDM + common dipole", 0.0, cat_c, q_c, 2, {"A": A_c}),
        ("LCDM + two dipoles", 0.0, 0.0, q_2, 3, {"A_CatWISE": d_geo, "A_Quaia": A_q}),
        ("light-cone ball", ball["dchi2_cc_sn"], ball["chi2_catwise"], ball["chi2_quaia"], 3,
         {"alpha_high": ball["alpha_high"], "t0": ball["t0"], "theta_obs": ball["theta_obs"]}),
    ]
    ref = rows[0][1] + rows[0][2] + rows[0][3]
    out = []
    print(f"{'model':22s} {'CC+SN':>7s} {'CatWISE':>8s} {'Quaia':>7s} {'total':>7s} {'dchi2':>7s} {'k':>2s} {'dAIC':>7s}")
    for name, cs, cw, q, k, par in rows:
        tot = cs + cw + q
        r = {"model": name, "dchi2_cc_sn": cs, "chi2_catwise": cw, "chi2_quaia": q, "total": tot,
             "dchi2_vs_lcdm": tot - ref, "k": k, "daic_vs_lcdm": tot - ref + 2 * (k - 1), "parameters": par}
        out.append(r)
        print(f"{name:22s} {cs:+7.2f} {cw:8.2f} {q:7.2f} {tot:7.2f} {tot - ref:+7.2f} {k:2d} {r['daic_vs_lcdm']:+7.2f}")
    (JSON / "compare_dipole_models.json").write_text(json.dumps(
        {"catwise": {"D_geo": d_geo, "sigma": s_geo}, "quaia": {"amp": a.tolist(), "sigma": s.tolist()}, "rows": out}, indent=1))


if __name__ == "__main__":
    main()
