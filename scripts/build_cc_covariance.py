"""Covariance matrix of the cosmic-chronometer compilation (default subset).

Components:
  * diagonal: published total errors, symmetrised (data/cc_compilation/cc_compilation.csv);
  * D4000n points of Moresco et al. 2012, 2015, 2016: the systematic terms of Moresco et al. 2020
    (ApJ 898, 82) that the published totals do not include, namely the initial mass function,
    the stellar library and the stellar-population-synthesis model, each fully correlated between
    redshifts, Cov_X,ij = eta_X(z_i) H_i eta_X(z_j) H_j, with the percentages eta_X(z) of
    data_MM20.dat (columns IMF, stlib, mod) from https://gitlab.com/mmoresco/CCcovariance,
    interpolated linearly in z;
  * Loubser 2025 (MNRAS 544, 3064): the published correlation coefficients between its three
    points, rho(0.46, 0.67) = 0.932, rho(0.46, 0.83) = 0.830, rho(0.67, 0.83) = 0.776, applied to
    the total errors.
The matrix is ordered by redshift as returned by data_loaders.load_cc().  Downloads data_MM20.dat
if absent.  Writes data/cc_compilation/cc_covariance.csv.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import data_loaders as DL  # noqa: E402

MM20 = ROOT / "data" / "cc_compilation" / "data_MM20.dat"
MM20_URL = "https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data/data_MM20.dat"
MORESCO_KEYS = {"Moresco2012", "Moresco2015", "Moresco2016"}
LOUBSER_RHO = {(0.46, 0.67): 0.932, (0.46, 0.83): 0.830, (0.67, 0.83): 0.776}


def main():
    if not MM20.exists():
        urllib.request.urlretrieve(MM20_URL, MM20)
    mm = np.loadtxt(MM20)
    z, h, s, _, _, ref = DL.load_cc()
    rows = [r for r in csv.DictReader(open(DL.CC_FILE, encoding="utf-8")) if r["use_default"].strip() == "1"]
    method = {(round(float(r["z"]), 4), r["reference_key"]): r["method"] for r in rows}
    cov = np.diag(s**2)
    d4 = np.array([ref[i] in MORESCO_KEYS and "D4000" in method[(round(z[i], 4), ref[i])] for i in range(len(z))])
    for col in (1, 2, 3):  # IMF, stellar library, SPS model
        eta = np.interp(z, mm[:, 0], mm[:, col]) / 100.0 * h * d4
        cov += np.outer(eta, eta)
    for (za, zb), rho in LOUBSER_RHO.items():
        ia = [i for i in range(len(z)) if ref[i] == "Loubser2025b" and abs(z[i] - za) < 1e-6]
        ib = [i for i in range(len(z)) if ref[i] == "Loubser2025b" and abs(z[i] - zb) < 1e-6]
        if ia and ib:
            cov[ia[0], ib[0]] = cov[ib[0], ia[0]] = rho * s[ia[0]] * s[ib[0]]
    np.linalg.cholesky(cov)  # positive definite
    np.savetxt(ROOT / "data" / "cc_compilation" / "cc_covariance.csv", cov, delimiter=",", fmt="%.8e")
    print(f"{len(z)} points; {int(d4.sum())} D4000n points with model covariance; Loubser 2025b correlations applied")


if __name__ == "__main__":
    main()
