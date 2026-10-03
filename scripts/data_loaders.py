"""Observational data used by the analysis scripts (single source for every fit).

* Cosmic chronometers: data/cc_compilation/cc_compilation.csv, rows with use_default = 1
  (independent measurements; see data/cc_compilation/sources.md).  Asymmetric total errors
  are symmetrised as the mean of the upper and lower values.
* Binned Pantheon+: data/gapp/pantheon_plus_binned_50_gls.csv and its 50 x 50 covariance
  data/gapp/pantheon_plus_binned_50_gls_cov_d.csv, produced by bin_pantheon_plus.py from the
  Pantheon+ catalogue (Scolnic et al. 2022; Brout et al. 2022): d = H0 D_C / c with the
  SH0ES distance scale, H0 = 73.04 km/s/Mpc (Riess et al. 2022).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CC_FILE = ROOT / "data" / "cc_compilation" / "cc_compilation.csv"
SN_FILE = ROOT / "data" / "gapp" / "pantheon_plus_binned_50_gls.csv"
SN_COV_FILE = ROOT / "data" / "gapp" / "pantheon_plus_binned_50_gls_cov_d.csv"
H0_SH0ES = 73.04  # km/s/Mpc, distance scale of the Pantheon+ MU_SH0ES moduli


def load_cc(all_rows: bool = False):
    """Return z, H [km/s/Mpc], sigma_H (symmetrised), sigma_plus, sigma_minus, reference keys."""
    rows = list(csv.DictReader(open(CC_FILE, encoding="utf-8")))
    if not all_rows:
        rows = [r for r in rows if r["use_default"].strip() == "1"]
    z = np.array([float(r["z"]) for r in rows])
    h = np.array([float(r["H"]) for r in rows])
    sp = np.array([float(r["sigma_total_plus"]) for r in rows])
    sm = np.array([float(r["sigma_total_minus"]) for r in rows])
    ref = [r["reference_key"] for r in rows]
    order = np.argsort(z)
    return z[order], h[order], (0.5 * (sp + sm))[order], sp[order], sm[order], [ref[i] for i in order]


def load_pantheon_binned():
    """Return z, d_proxy, sigma_d and the full covariance of the binned Pantheon+ distance proxy."""
    d = np.genfromtxt(SN_FILE, names=True, delimiter=",")
    cov = np.loadtxt(SN_COV_FILE, delimiter=",")
    return (np.asarray(d["z"], float), np.asarray(d["d_proxy"], float), np.asarray(d["sigma_d"], float), cov)


def load_cc_covariance():
    """Covariance of the default CC subset, ordered as load_cc() (build_cc_covariance.py)."""
    return np.loadtxt(ROOT / "data" / "cc_compilation" / "cc_covariance.csv", delimiter=",")
