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
    ["scripts/plot_ball_zoom_unbinned.py", "--t0-range", "1.4", "4.0", "--nt", "60", "--theta-max", "20", "--theta-step", "0.05", "--tag", "_wide"],
    ["scripts/check_paper_numbers.py"],
]


def main():
    steps = STEPS + (EXTENDED if "--extended" in sys.argv[1:] else [])
    for step in steps:
        t = time.time()
        print(f"=== {' '.join(step)}", flush=True)
        r = subprocess.run([sys.executable, *step], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"failed: {' '.join(step)} (exit {r.returncode})")
        print(f"    done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
