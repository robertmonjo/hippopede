"""Comparison of an off-axis observer with reported directional anomalies.

At the reference t0 (figures/t0_summary.json) and for observers at sector angle theta_obs:
  1. distance asymmetry along the projected axis,
     delta(z) = [D_C(z, +n_ax) - D_C(z, -n_ax)] / [D_C(z, +n_ax) + D_C(z, -n_ax)];
  2. apparent H0 anisotropy between opposite directions, Delta H0/H0 = 2|delta| (from
     D_L ~ c z / H0), as a fraction of the 9% variation of Migkas et al. (2021);
  3. q0 difference between opposite directions, Delta q0 = 4|delta|/z (from
     D_L ~ (c z/H0)[1 + (1 - q0) z/2]), as a fraction of |q_d| = 8.03 of Colin et al. (2019);
  4. the configuration that reproduces the quasar dipole excess (quasar_dipole_fit.py) at the
     reference t0: its unbinned Pantheon+ chi2 with the axis along the excess direction, relative
     to the axis observer, and the distance-modulus difference between opposite directions at z = 1.
Writes figures/directional_anomalies.json.
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
import quasar_dipole_fit as Q  # noqa: E402

THETA_OBS = [1, 2, 5, 10, 20, 30, 35]  # up to sector_geometry.THETA_OBS_MAX_DEG
Z_EVAL = np.array([0.05, 0.1, 0.3, 1.0, 1.2])
MIGKAS, COLIN_QD = 0.09, 8.03


def asymmetry(tab, theta_obs, z):
    pole = np.array([0.0, 0.0, 1.0])
    n = np.array([pole, -pole] * len(z))
    sky = T.Sky(np.repeat(z, 2), n)
    e = T.bilinear(tab, sky.sector_path(theta_obs, sky.n @ pole), sky.zp)
    dc = np.trapezoid(1.0 / e, sky.zp, axis=1)
    return (dc[0::2] - dc[1::2]) / (dc[0::2] + dc[1::2])


def main():
    t0 = float(json.loads((ROOT / "figures" / "t0_summary.json").read_text())["figure_t0"])
    qf = json.loads((ROOT / "figures" / "quasar_dipole_fit.json").read_text())
    tab = T.e_table(t0)
    delta = {th: asymmetry(tab, th, Z_EVAL) for th in THETA_OBS}
    print(f"t0 = {t0}; |delta| [%] at z = {Z_EVAL.tolist()}")
    for th in THETA_OBS:
        print(f"  theta_obs = {th:2d}: " + "  ".join(f"{100 * abs(x):.4f}" for x in delta[th]))
    mig = {th: [2 * abs(delta[th][1]) / MIGKAS, 2 * abs(delta[th][2]) / MIGKAS] for th in THETA_OBS}
    col = {th: 4 * abs(delta[th][0]) / Z_EVAL[0] / COLIN_QD for th in THETA_OBS}
    run = next(r for r in qf["runs"] if abs(r["t0"] - t0) < 1e-9 and r["pz"] == "gamma" and abs(r["x"] - qf["x"]) < 1e-12)
    th_q = run["theta_obs_minus1sigma_best_plus1sigma"][1]
    out = {"t0": t0, "z": Z_EVAL.tolist(), "delta": {str(k): v.tolist() for k, v in delta.items()},
           "migkas_fraction_max": [max(m[i] for m in mig.values()) for i in (0, 1)],
           "colin_fraction_max": max(col.values()), "migkas_fraction": {str(k): v for k, v in mig.items()},
           "colin_fraction": {str(k): v for k, v in col.items()}, "secrest": None}
    print(f"Migkas fraction max (z = 0.1, 0.3): {[round(100 * v, 2) for v in out['migkas_fraction_max']]} %;  "
          f"Colin fraction max: {100 * out['colin_fraction_max']:.3f} %")
    if th_q is not None:
        sign = np.sign(run["D"][-1]) or 1.0
        v, _, lb = Q.excess_vector()
        n_axis = sign * v / np.linalg.norm(v)
        d_q = asymmetry(tab, th_q, Z_EVAL)
        z, zh, mb, cov, n = T.load()
        sky = T.Sky(z, n, zh, mb, cov)
        c_ax = sky.chi2(tab, 0.0, n_axis)
        c_q = sky.chi2(tab, th_q, n_axis)
        out["secrest"] = {"theta_obs": th_q, "excess_lb": lb, "delta": d_q.tolist(),
                          "dmu_z1": float(5 * np.log10((1 + abs(d_q[3])) / (1 - abs(d_q[3])))),
                          "migkas_fraction": [2 * abs(d_q[1]) / MIGKAS, 2 * abs(d_q[2]) / MIGKAS],
                          "colin_fraction": 4 * abs(d_q[0]) / Z_EVAL[0] / COLIN_QD,
                          "pantheon_dchi2_vs_axis": c_q - c_ax}
        print(f"Secrest configuration: theta_obs = {th_q:.2f} deg, excess towards (l,b) = ({lb[0]:.1f}, {lb[1]:.1f}); "
              f"Pantheon+ dchi2 = {c_q - c_ax:+.2f}; Delta mu(z=1) = {out['secrest']['dmu_z1']:.4f} mag; "
              f"Migkas fractions {[round(100 * f, 3) for f in out['secrest']['migkas_fraction']]} %")
    (ROOT / "figures" / "directional_anomalies.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
