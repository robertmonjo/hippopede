"""Gaussian-process reconstruction of E(z) and q(z) from cosmic chronometers and binned Pantheon+.

Code: GaPP (Seikel, Clarkson & Smith 2012), bundled in vendor_gapp/, squared-exponential kernel
with trained hyperparameters.  The joint GP of the distance proxy d(z) = H0 D_C / c (supernovae)
and of its derivative d'(z) = H0 / H(z) (chronometers) gives E = 1/d' and
q = -1 - (1+z) d''/d', with first-order error propagation.

Options (the defaults reproduce the reconstruction of the manuscript):
  --cc   compilation  all independent points of data/cc_compilation/cc_compilation.csv
                      (use_default = 1; 38 points, data/cc_compilation/sources.md)
         curated      the 18 cosmic-chronometer rows of data/hz_background/hz_curated_chronometers.csv
  --sn   gls          data/gapp/pantheon_plus_binned_50_gls.csv (bin_pantheon_plus.py)
         curated      data/gapp/pantheon_plus_binned_50.csv
  --h0   cc           H0 at z = 0 of a GP of the chronometer H(z) data alone
         shoes        73.2 km/s/Mpc (Riess et al. 2024)
Chronometer errors are symmetrised as the mean of the upper and lower values; the chronometers
enter as d'(z_i) = H0 / H_i with errors H0 sigma_i / H_i^2.
Writes figures/gapp_reconstruction_<cc>_<sn>_<h0>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for extra in (str(SCRIPT_DIR), str(ROOT / "vendor_gapp"), str(ROOT / "vendor_gapp" / "covfunctions")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import covariance  # noqa: E402
import dgp  # noqa: E402
import gp  # noqa: E402

import data_loaders as DL  # noqa: E402

CURATED_CC = ROOT / "data" / "hz_background" / "hz_curated_chronometers.csv"
SN_FILES = {"gls": ROOT / "data" / "gapp" / "pantheon_plus_binned_50_gls.csv",
            "curated": ROOT / "data" / "gapp" / "pantheon_plus_binned_50.csv"}
H0_SHOES = 73.2  # km/s/Mpc


def load_cc_points(which):
    """z, H, upper and lower errors of the chosen chronometer set, ordered by z."""
    if which == "compilation":
        z, h, _, sp, sm, _ = DL.load_cc()
        return z, h, sp, sm
    rows = [r for r in csv.DictReader(open(CURATED_CC, encoding="utf-8")) if r["source_kind"] == "cosmic_chronometer"]
    z = np.array([float(r["z"]) for r in rows])
    h = np.array([float(r["h_km_s_mpc"]) for r in rows])
    sp = np.array([float(r["sigma_plus"]) for r in rows])
    sm = np.array([float(r["sigma_minus"]) for r in rows])
    o = np.argsort(z)
    return z[o], h[o], sp[o], sm[o]


def load_cc(which):
    """z, H and symmetrised error of the chosen chronometer set."""
    z, h, sp, sm = load_cc_points(which)
    return z, h, 0.5 * (sp + sm)


def load_sn(which):
    d = np.genfromtxt(SN_FILES[which], names=True, dtype=None, encoding="utf-8", delimiter=",")
    return np.asarray(d["z"], float), np.asarray(d["d_proxy"], float), np.asarray(d["sigma_d"], float)


def h0_from_cc(z, h, s):
    """H0 and its GP uncertainty from the chronometer data alone."""
    g = gp.GaussianProcess(z, h, s, covfunction=covariance.SquaredExponential, cXstar=(0.0, 0.1, 11))
    rec, _ = g.gp(thetatrain="True")
    return float(rec[0, 1]), float(rec[0, 2])


def reconstruct(cc="compilation", sn="gls", h0="cc", zmin=-0.5, zmax=2.36, nstar=320):
    cz, ch, cs = load_cc(cc)
    zs, ds, ss = load_sn(sn)
    h0_val, h0_err = h0_from_cc(cz, ch, cs) if h0 == "cc" else (H0_SHOES, None)
    g = dgp.DGaussianProcess(zs, ds, ss, covfunction=covariance.SquaredExponential,
                             dX=cz, dY=h0_val / ch, dSigma=h0_val * cs / ch**2, cXstar=(zmin, zmax, nstar))
    rec, theta = g.gp(thetatrain="True")
    drec, _ = g.dgp(thetatrain="False")
    d2rec, _ = g.d2gp(thetatrain="False")
    z = rec[:, 0]
    dd, dd_s = drec[:, 1], drec[:, 2]
    d2, d2_s = d2rec[:, 1], d2rec[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 1.0 / dd
        e_s = np.abs(dd_s / dd**2)
        q = -1.0 - (1.0 + z) * d2 / dd
        q_s = np.sqrt(((1.0 + z) * d2_s / dd) ** 2 + ((1.0 + z) * d2 * dd_s / dd**2) ** 2)
    return {"z": z, "e": e, "e_sigma": e_s, "q": q, "q_sigma": q_s, "theta": np.asarray(theta),
            "options": {"cc": cc, "sn": sn, "h0": h0}, "H0": h0_val, "H0_err": h0_err,
            "N_CC": int(len(cz)), "N_SN_bins": int(len(zs))}


def output_path(cc, sn, h0):
    return ROOT / "figures" / f"gapp_reconstruction_{cc}_{sn}_{h0}.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", choices=("compilation", "curated"), default="compilation")
    p.add_argument("--sn", choices=("gls", "curated"), default="gls")
    p.add_argument("--h0", choices=("cc", "shoes"), default="cc")
    a = p.parse_args()
    g = reconstruct(a.cc, a.sn, a.h0)
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g.items()}
    path = output_path(a.cc, a.sn, a.h0)
    path.write_text(json.dumps(out, indent=0))
    print(f"cc={a.cc} sn={a.sn} h0={a.h0}: N_CC = {g['N_CC']}, N_SN bins = {g['N_SN_bins']}, "
          f"H0 = {g['H0']:.2f}{'' if g['H0_err'] is None else ' +- %.2f' % g['H0_err']}; "
          f"hyperparameters {g['theta']}; wrote {path.name}")


if __name__ == "__main__":
    main()
