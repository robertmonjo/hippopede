"""Kinematic numbers of the hippopede sectors, each taken as an independent history.

At the reference t0 (figures/t0_summary.json) and for theta = 0, 15, 30, 45, 60 deg:
  * q0 of the centred history, unprojected and projected with the running index;
  * transition redshifts of the projected histories (sign change of q for 0 < z < 2);
  * maximum redshift of the unprojected branches;
  * the largest value of q along each unprojected history for 0 <= z <= 2 (numerical check of
    the sign of q, with no asymptotic expansion);
  * spread of the projected E(z) across sectors, (max - min)/mean, at z = 0.1, 0.5, 1;
and the sector of a source at redshift z for an observer on the axis, theta = rhat(z)/2
(sector_geometry.py).  Writes figures/sector_kinematics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import hippopede_model as HM  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import sector_geometry as G  # noqa: E402

SECTORS = [0, 15, 30, 45, 60]


def main():
    t0 = float(json.loads((ROOT / "figures" / "t0_summary.json").read_text())["figure_t0"])
    zw = HM.Z_WORK
    rows, E = [], {}
    for th in SECTORS:
        zd, qd, ed = HM.build_centered_history(np.radians(th), t0)
        e_p, q_p = HM.projected_sector(th, t0)
        ok = np.isfinite(q_p) & (zw > 0.01) & (zw < 2.0)
        zt = [float(zw[ok][i]) for i in np.where(np.diff(np.sign(q_p[ok])) != 0)[0]]
        inside = (zd >= 0) & (zd <= 2.0) & np.isfinite(qd)
        rows.append({"theta": th, "q0_unprojected": float(np.interp(0.0, zd, qd)),
                     "q0_projected": float(np.interp(0.0, zw, q_p)), "z_t_projected": zt,
                     "z_max_unprojected": float(zd.max()), "max_q_unprojected_z_le_2": float(qd[inside].max())})
        E[th] = e_p
        print(f"theta={th:2d}: q0 unproj={rows[-1]['q0_unprojected']:+.4f} proj={rows[-1]['q0_projected']:+.4f} "
              f"z_t={np.round(zt, 3).tolist()} z_max={zd.max():.2f} max q (unproj, z<=2)={rows[-1]['max_q_unprojected_z_le_2']:+.2e}")
    spread = {}
    for z in (0.1, 0.5, 1.0):
        v = np.array([np.interp(z, zw, E[th]) for th in SECTORS])
        spread[str(z)] = float((v.max() - v.min()) / v.mean())
    run = lambda z: PH.alpha_sqrt(z, PH.ALPHA_LOW, PH.load_alpha_high())
    zs = np.array([0.1, 0.5, 1.0, 1.3, 2.0])
    th_z = G.source_sector_deg(0.0, PH.rhat_of_z(zs, run), 1.0)
    print("E spread across sectors (max-min)/mean [%]:", {k: round(100 * v, 2) for k, v in spread.items()})
    print("axis observer: sector theta = rhat(z)/2 [deg] at z =", zs.tolist(), "->", np.round(th_z, 2).tolist())
    out = {"t0": t0, "sectors": rows, "E_spread": spread, "axis_observer_sector_deg": dict(zip(map(str, zs), th_z.tolist()))}
    (ROOT / "figures" / "sector_kinematics.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
