"""Collect the estimates of t0 and fix the reference value used in Fig. 2 and Table 2.

Inputs (run the corresponding scripts first):
  t0_dispersion_match.json       sector distribution of q(z) and E(z) matched to the GaPP 1- and
                                 2-sigma bands;
  fit_t0_axis_observer.json      chi2 profile for an observer on the axis;
  fit_t0_intrinsic_scatter.json  unbinned Pantheon+ likelihood with the sector spread added as
                                 intrinsic scatter.
Reference value ('figure_t0'): the joint 1- and 2-sigma band match of q(z).  All estimates are
written to json/t0_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "json"   # numerical outputs; figures/ holds the png/pdf
KEYS = ("t0_best", "jackknife_se", "node_16_84", "at_grid_edge")


def main():
    dm = json.loads((JSON / "t0_dispersion_match.json").read_text())
    ax = json.loads((JSON / "fit_t0_axis_observer.json").read_text())
    sc = json.loads((JSON / "fit_t0_intrinsic_scatter.json").read_text())
    out = {"figure_t0": round(dm["q"]["joint"]["t0_best"], 2),
           "rule": "joint 1- and 2-sigma match of the sector distribution of q(z) to the GaPP band",
           "band_match": {x: {lab: {k: dm[x][lab][k] for k in KEYS} for lab in ("1sigma", "2sigma", "joint")}
                          for x in ("q", "E")},
           "axis_observer": {k: ax[k] for k in ("t0_best", "interval", "chi2_nu", "minimum_at_grid_edge",
                                                "plateau_to_axial_limit")},
           "intrinsic_scatter": {k: sc[k] for k in ("t0_best", "interval", "minimum_at_grid_edge")}}
    (JSON / "t0_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
