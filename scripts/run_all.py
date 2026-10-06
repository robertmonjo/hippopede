"""Run the full analysis of the paper from scratch, in dependency order.

Downloads the public data (Pantheon+, the stellar-population covariance of Moresco et al. 2020,
the Quaia catalogue), builds the derived data sets, runs every fit, writes every figure and JSON
in figures/, and finally compares the numbers quoted in the manuscript with the outputs
(check_paper_numbers.py).  Stops at the first script that fails.  Run from any directory:
    python scripts/run_all.py
The Quaia download is about 270 MB; the full run takes of the order of an hour on a laptop.
With --extended the analyses of the light-cone average, the dipoles and the unbinned supernovae
follow (REPRODUCIBILITY.md); they need a many-core machine and several hours.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    # data
    ["scripts/download_pantheon_plus.py"],
    ["scripts/bin_pantheon_plus.py"],
    ["scripts/build_cc_covariance.py"],
    ["dipole_zslices/download_quaia.py"],
    ["dipole_zslices/measure_dipole.py"],
    # projection index and reconstruction
    ["scripts/fit_alpha_high.py"],
    ["scripts/gapp_reconstruction.py"],
    # present time
    ["scripts/t0_dispersion_match.py"],
    ["scripts/fit_t0_axis_observer.py"],
    ["scripts/fit_t0_intrinsic_scatter.py"],
    ["scripts/t0_summary.py"],
    # fits and kinematics
    ["scripts/fit_cc_pantheon.py"],
    ["scripts/fit_sector_average.py"],
    ["scripts/sector_kinematics.py"],
    ["scripts/sector_kinematics.py", "--from-average-fit"],
    ["scripts/sector_reach.py"],
    ["scripts/verify_high_z.py"],
    # figures
    ["scripts/hippopede_growth_3d.py"],
    ["scripts/plot_hippopede_qz_double_panel_with_gapp.py", "--from-average-fit", "--suffix", "_sector_average_alpha"],
    ["scripts/plot_band_comparison.py", "--from-average-fit", "--suffix", "_sector_average_alpha"],
    # directional analyses
    ["scripts/pantheon_offaxis_test.py"],
    ["scripts/quasar_dipole_fit.py"],
    ["scripts/directional_anomalies.py"],
    ["scripts/quaia_zslice_model.py"],
    ["scripts/local_environment.py"],
    # numbers in the paper
    ["scripts/check_paper_numbers.py"],
]


DIPOLE_ALPHAS = ["0.30", "0.32", "0.34", "0.36", "0.38", "0.40", "0.42", "0.44"]
OVERVIEW_DIPOLE_ALPHAS = [f"{0.30 + 0.02 * k:.2f}" for k in range(21)]   # 0.30 ... 0.70
OVERVIEW_DIPOLE_THETA = ["0", "0.5", "1", "2", "3", "5", "7", "10", "12.5", "15", "17.5", "20", "22.5", "25", "27.5", "30", "32.5", "35", "37.5", "40", "42.5", "45"]
WHOLE_LOBE = {"HIPPOPEDE_THETA_MAX_DEG": "90"}   # environment of a step (sector_geometry.THETA_MAX_DEG)
OVERVIEW_THETA = ["0", "0.25", "0.5", "1", "1.5", "2", "2.5", "3"] + [str(k) for k in range(4, 46)]
OVERVIEW_MAP = ["scripts/plot_ball_zoom_unbinned.py", "--t0-range", "1.4", "20", "--nt", "40", "--theta-nodes", *OVERVIEW_THETA,
                "--alpha-range", "0.29", "0.71", "--display-theta-step", "0.05", "--nt-fine", "1100",
                "--alpha-levels", "0.35", "0.40", "0.45", "--hatch-unresolved",
                "--dipole-alphas", *OVERVIEW_DIPOLE_ALPHAS, "--dipole-suffix", "_wide", "--tag", "_overview"]
EXTENDED = [
    ["scripts/cc_intrinsic_scatter.py"],
    ["scripts/cc_direction_test.py"],
    ["scripts/t0_dispersion_match.py", "--z-nodes", "0.5", "1.9"],
    ["scripts/fit_sector_average.py", "--z-nodes", "0.5", "1.9"],
    *[["scripts/quasar_dipole_fit.py", "--alpha-high", a, "--t0-range", "1", "10", "--t0-n", "30"] for a in DIPOLE_ALPHAS],
    *[["scripts/quaia_zslice_model.py", "--alpha-high", a, "--t0-range", "1", "10", "--t0-n", "30"] for a in DIPOLE_ALPHAS],
    ["scripts/pantheon_catwise_axis.py"],
    ["scripts/plot_observer_ball_landscape.py", "--t0-range", "1", "10", "--nt", "80", "--theta-max", "45", "--theta-step", "0.25",
     "--alpha-range", "0.27", "0.47", "--tag", "joint_wide", "--dipole-alphas", *DIPOLE_ALPHAS,
     "--plot-t0-min", "1.4", "--plot-theta-max", "30"],
    ["scripts/compare_ball_curves.py"],
    ["scripts/fit_ball_unbinned.py"],
    ["scripts/fit_lines_unbinned.py"],
    ["scripts/binning_resolution_test.py"],
    # overview of the (t0, theta_obs) plane with the whole lobe tabulated (sectors up to 90 deg, limited only
    # by their reach): dipole runs up to alpha_high 0.70, t0 20 and theta_obs 45 deg, then the map
    *[(["scripts/quasar_dipole_fit.py", "--alpha-high", a, "--t0-range", "1", "20", "--t0-n", "40",
        "--theta-obs", *OVERVIEW_DIPOLE_THETA, "--out-suffix", "_wide"], WHOLE_LOBE) for a in OVERVIEW_DIPOLE_ALPHAS],
    *[(["scripts/quaia_zslice_model.py", "--alpha-high", a, "--t0-range", "1", "20", "--t0-n", "40",
        "--theta-obs", *OVERVIEW_DIPOLE_THETA, "--out-suffix", "_wide"], WHOLE_LOBE) for a in OVERVIEW_DIPOLE_ALPHAS],
    (OVERVIEW_MAP, WHOLE_LOBE),
    (["scripts/refine_overview_minimum.py"], WHOLE_LOBE),
    # finer map near the minima (sectors up to 70 deg, which these minima do not reach)
    ["scripts/plot_ball_zoom_unbinned.py", "--t0-range", "1.4", "4.0", "--nt", "60", "--theta-max", "20", "--theta-step", "0.05",
     "--tag", "_wide", "--best-all-json", "refine_overview_minimum.json"],
    # figure of the paper (Fig. ball): the overview redrawn up to t0 = 10, minima from the two previous steps
    (OVERVIEW_MAP + ["--reuse", "--best-all-json", "refine_overview_minimum.json", "--markers-json", "ball_zoom_unbinned_wide.json",
                     "--no-title", "--plot-t0-max", "10"], WHOLE_LOBE),
    ["scripts/compare_dipole_models.py"],
    ["scripts/check_paper_numbers.py"],
]


def main():
    steps = STEPS + (EXTENDED if "--extended" in sys.argv[1:] else [])
    for step in steps:
        step, extra = step if isinstance(step, tuple) else (step, {})
        t = time.time()
        print(f"=== {' '.join(f'{k}={v}' for k, v in extra.items())} {' '.join(step)}", flush=True)
        r = subprocess.run([sys.executable, *step], cwd=ROOT, env=dict(os.environ, **extra))
        if r.returncode != 0:
            sys.exit(f"failed: {' '.join(step)} (exit {r.returncode})")
        print(f"    done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
