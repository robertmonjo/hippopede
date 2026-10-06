"""Redshift reach of the sectors and the redshift limit of the directional construction.

z_max(theta): largest redshift of the centred history retained after the first minimum of a_c
(unprojected); the observed reach z_reach solves E_hyp^proj(z) - 1 = z_max.  For the observer
on the axis the source at z lies in the sector rhat(z)/2 (sector_geometry.py); the directional
construction holds only up to the redshift z_h at which that sector's reach falls below z.
Reference t0 from json/t0_summary.json.  Writes json/sector_reach.json.
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

import hippopede_model as HM  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import sector_geometry as G  # noqa: E402

RUN = lambda z: PH.alpha_sqrt(z, PH.ALPHA_LOW, PH.load_alpha_high())


def reach(th_deg, t0, n=200000):
    zd, _, _ = HM.build_centered_history(np.radians(th_deg), t0, n=n)
    zm = float(zd.max())
    return zm, float(brentq(lambda z: PH.E_of_z(np.array([z]), RUN)[0] - 1 - zm, 1e-4, 1e7))


def main():
    t0 = float(json.loads((ROOT / "json" / "t0_summary.json").read_text())["figure_t0"])
    rows = []
    for th in (1.0, 5.0, 15.0, 30.0, 45.0, 60.0, 70.0, 75.0, 85.0):
        zm, zr = reach(th, t0)
        rows.append({"theta": th, "z_max_unprojected": zm, "z_reach_observed": zr})
        print(f"theta={th:5.1f} deg: z_max (unprojected) = {zm:8.2f}; observed reach z = {zr:7.3f}")

    def sector_at(z):
        return float(G.source_sector_deg(0.0, PH.rhat_of_z(np.array([z]), RUN)[0], 1.0))

    gap = lambda z: reach(sector_at(z), t0)[1] - z
    zs = np.geomspace(0.5, 1000.0, 40)
    g = np.array([gap(z) for z in zs])
    neg = np.where(g < 0)[0]
    z_h = float(brentq(gap, zs[neg[0] - 1], zs[neg[0]], xtol=1e-3)) if len(neg) and neg[0] > 0 else None
    table = [{"z": float(z), "sector_deg": sector_at(z), "sector_reach": reach(sector_at(z), t0)[1]} for z in (1.0, 2.0, 3.0, 5.0, 1090.0)]
    for r in table:
        print(f"z = {r['z']:7.1f}: source sector {r['sector_deg']:6.2f} deg, observed reach of that sector z = {r['sector_reach']:.3f}")
    print(f"axis observer: directional construction defined up to z_h = {z_h}")
    (ROOT / "json" / "sector_reach.json").write_text(json.dumps(
        {"t0": t0, "sectors": rows, "axis_observer": {"z_h": z_h, "table": table}}, indent=1))


if __name__ == "__main__":
    main()
