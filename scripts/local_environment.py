"""Local environment of an observer near the axis: effective q0 dipole versus redshift.

All sectors share the time t0, so a source sector expands at H_c,theta_s(t0) relative to the
observer's H_c,theta_obs(t0) (pantheon_offaxis_test.e_table carries that factor).  Along the
direction towards the projected axis the source sector is theta_s = |chi_obs - rhat(z)|/2
(sector_geometry.py), which crosses the axis where rhat(z) = chi_obs = 2 theta_obs.

For each direction the effective deceleration parameter is defined from the luminosity distance
in units of the observer's Hubble length,
    D_L = z [1 + (1 - q_eff) z / 2]   =>   q_eff(z, n) = 1 - 2 (D_L/z - 1)/z,
and the dipole is q_d,eff(z) = [q_eff(z, +n_ax) - q_eff(z, -n_ax)]/2, to be compared with the
supernova dipole q_d exp(-z/z_S), q_d = -8.03, z_S = 0.026 (Colin et al. 2019).
Observers: theta_obs = 1, 2, 5 deg and the value reproducing the quasar excess at the reference
t0 (quasar_dipole_fit.json).  Writes figures/local_environment.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pantheon_offaxis_test as T  # noqa: E402
import projected_hyperconical as PH  # noqa: E402

Z = np.array([0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])
Z_S_COLIN = 0.026  # redshift scale of the supernova q0 dipole (Colin et al. 2019)


def z_crossing(theta_obs_deg):
    return brentq(lambda z: np.degrees(PH.rhat_of_z(np.array([z]), T.RUN)[0]) - 2.0 * theta_obs_deg, 1e-7, 2.0)


def q_eff(tab, theta_obs, z, sign):
    pole = np.array([0.0, 0.0, 1.0])
    sky = T.Sky(z, np.repeat((sign * pole)[None, :], len(z), axis=0))
    e = T.bilinear(tab, sky.sector_path(theta_obs, sky.n @ pole), sky.zp)
    e_obs = T.bilinear(tab, np.full(1, float(theta_obs)), np.zeros(1))[0]
    dl = (1 + z) * np.trapezoid(e_obs / e, sky.zp, axis=1)
    return 1 - 2 * (dl / z - 1) / z


def main():
    t_ref = float(json.loads((ROOT / "figures" / "t0_summary.json").read_text())["figure_t0"])
    qf = json.loads((ROOT / "figures" / "quasar_dipole_fit.json").read_text())
    run = next(r for r in qf["runs"] if abs(r["t0"] - t_ref) < 1e-9 and r["pz"] == "gamma" and abs(r["x"] - qf["x"]) < 1e-12)
    th_q = run["theta_obs_minus1sigma_best_plus1sigma"][1]
    thetas = sorted({1.0, 2.0, 5.0} | ({round(th_q, 3)} if th_q else set()))
    tab = T.e_table(t_ref)
    out = {"t0": t_ref, "theta_obs_quasar": th_q, "observers": {}, "z_S_colin": Z_S_COLIN,
           # observer sector whose line of sight crosses the axis at the redshift scale of the q0 dipole
           "theta_obs_crossing_zS": float(np.degrees(PH.rhat_of_z(np.array([Z_S_COLIN]), T.RUN)[0]) / 2.0)}
    for th in thetas:
        qp, qm = q_eff(tab, th, Z, +1), q_eff(tab, th, Z, -1)
        qd = 0.5 * (qp - qm)
        zc = z_crossing(th)
        out["observers"][str(th)] = {"z": Z.tolist(), "q_d_eff": qd.tolist(), "z_crossing": zc}
        print(f"t0={t_ref} theta_obs={th}: axis crossing at z={zc:.4f}; q_d,eff(z) = "
              + " ".join(f"{x:+.2e}" for x in qd[[0, 2, 4, 6, 9, 12, 14]]))
    print("z columns:", Z[[0, 2, 4, 6, 9, 12, 14]].tolist())
    (ROOT / "figures" / "local_environment.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
