"""Compress the Pantheon+ Hubble diagram into 50 redshift bins with their full covariance.

Input (scripts/download_pantheon_plus.py): Pantheon+SH0ES.dat and Pantheon+SH0ES_STAT+SYS.cov
(Scolnic et al. 2022; Brout et al. 2022).  Selection: z_HD > 0.01 and not a Cepheid calibrator.
Distance moduli: MU_SH0ES (SN Ia absolute magnitude calibrated with SH0ES Cepheids), whose
distance scale corresponds to H0 = 73.04 km/s/Mpc (Riess et al. 2022, ApJL 934, L7).

Compression: the SNe, sorted by z_HD, are split into 50 bins of equal number.  With a flat
LCDM fiducial (Omega_m = 0.334, Brout et al. 2022) the residuals r = mu - mu_fid are
compressed by generalised least squares with the full STAT+SYS covariance C:
    Delta = (A^T C^-1 A)^-1 A^T C^-1 r,     C_bin = (A^T C^-1 A)^-1,
where A_ib = 1 if SN i is in bin b.  The bin distance modulus is mu_b = mu_fid(z_b) + Delta_b,
with z_b the inverse-variance mean redshift of the bin; the fiducial only enters through
the small redshift spread inside each bin.  Each bin also carries the dimensionless comoving
distance d = H0 D_C / c = H0 10^((mu_b - 25)/5) Mpc / [(1 + z_hel,b) c], and its covariance.

Writes data/gapp/pantheon_plus_binned_50_gls.csv (z, z_hel, mu, sigma_mu, d_proxy, sigma_d, n_sn)
and data/gapp/pantheon_plus_binned_50_gls_cov_d.csv (50 x 50 covariance of d_proxy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "pantheon_plus"
OUT = ROOT / "data" / "gapp"
C_KM_S = 299792.458
H0_SH0ES = 73.04
OM_FID = 0.334
N_BINS = 50
Z_MIN = 0.01


def dc_lcdm(z, om=OM_FID):
    zz = np.linspace(0.0, float(np.max(z)) * 1.001, 20001)
    inv = 1.0 / np.sqrt(om * (1 + zz) ** 3 + 1 - om)
    dc = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zz))])
    return np.interp(z, zz, dc) * C_KM_S / H0_SH0ES  # Mpc


def mu_lcdm(z_hd, z_hel):
    return 5.0 * np.log10((1.0 + z_hel) * dc_lcdm(z_hd)) + 25.0


def main():
    d = np.genfromtxt(SRC / "Pantheon+SH0ES.dat", names=True, dtype=None, encoding="utf-8")
    with open(SRC / "Pantheon+SH0ES_STAT+SYS.cov") as f:
        n = int(f.readline())
        cov = np.fromstring(f.read(), sep="\n").reshape(n, n)
    sel = np.where((d["zHD"] > Z_MIN) & (d["IS_CALIBRATOR"] == 0))[0]
    z, zh, mu = d["zHD"][sel], d["zHEL"][sel], d["MU_SH0ES"][sel]
    C = cov[np.ix_(sel, sel)]
    order = np.argsort(z)
    z, zh, mu, C = z[order], zh[order], mu[order], C[np.ix_(order, order)]
    edges = np.linspace(0, len(z), N_BINS + 1).round().astype(int)
    A = np.zeros((len(z), N_BINS))
    for b in range(N_BINS):
        A[edges[b]:edges[b + 1], b] = 1.0
    Ci = np.linalg.inv(C)
    F = A.T @ Ci @ A
    Cb = np.linalg.inv(F)
    r = mu - mu_lcdm(z, zh)
    delta = Cb @ (A.T @ Ci @ r)
    w = 1.0 / np.diag(C)
    zb = np.array([np.average(z[edges[b]:edges[b + 1]], weights=w[edges[b]:edges[b + 1]]) for b in range(N_BINS)])
    zhb = np.array([np.average(zh[edges[b]:edges[b + 1]], weights=w[edges[b]:edges[b + 1]]) for b in range(N_BINS)])
    mub = mu_lcdm(zb, zhb) + delta
    dproxy = H0_SH0ES * 10 ** ((mub - 25.0) / 5.0) / ((1.0 + zhb) * C_KM_S)
    J = dproxy * np.log(10.0) / 5.0
    cov_d = J[:, None] * Cb * J[None, :]
    OUT.mkdir(parents=True, exist_ok=True)
    header = "z,z_hel,mu,sigma_mu,d_proxy,sigma_d,n_sn"
    rows = np.column_stack([zb, zhb, mub, np.sqrt(np.diag(Cb)), dproxy, np.sqrt(np.diag(cov_d)), np.diff(edges)])
    np.savetxt(OUT / "pantheon_plus_binned_50_gls.csv", rows, delimiter=",", header=header, comments="",
               fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%.8e", "%.8e", "%d"])
    np.savetxt(OUT / "pantheon_plus_binned_50_gls_cov_d.csv", cov_d, delimiter=",", fmt="%.10e")
    print(f"{len(z)} SNe in {N_BINS} bins; z from {zb.min():.4f} to {zb.max():.4f}; "
          f"median sigma_mu = {np.median(np.sqrt(np.diag(Cb))):.4f} mag")


if __name__ == "__main__":
    main()
