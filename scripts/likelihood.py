"""Likelihood of an expansion history E(z) given the CC compilation and binned Pantheon+.

  chi2_CC = (H - H0 E)^T C_CC^-1 (H - H0 E),              H0 minimised analytically;
  chi2_SN = (d - A m)^T C^-1 (d - A m),                    A minimised analytically,
with m_b = D_C(z_b) = int_0^{z_b} dz'/E(z'); C_CC is the covariance of the CC compilation
(build_cc_covariance.py) and C the full covariance of the binned distance proxy
(bin_pantheon_plus.py).  The amplitude A absorbs H0 and the SN absolute
magnitude, so the supernovae constrain only the shape of E(z).
"""

from __future__ import annotations

import numpy as np

import data_loaders as DL
import hippopede_model as HM
import projected_hyperconical as PH

CC = DL.load_cc()[:3]
CINV_CC = np.linalg.inv(DL.load_cc_covariance())
Z_SN, D_SN, _S_SN, COV_SN = DL.load_pantheon_binned()
# integration grid for D_C, up to the highest binned supernova redshift
Z_GRID = np.linspace(0.0, np.ceil(Z_SN.max() * 10) / 10, 2601)
CINV_SN = np.linalg.inv(COV_SN)
N_DATA = len(CC[0]) + len(Z_SN)


def chi2_cc(E_at, data=CC, cinv=CINV_CC):
    z, h, _ = data
    e = E_at(z)
    if not np.all(np.isfinite(e)):
        return np.inf, np.nan
    h0 = (e @ cinv @ h) / (e @ cinv @ e)
    r = h - h0 * e
    return float(r @ cinv @ r), float(h0)


def chi2_sn(E_grid):
    if not np.all(np.isfinite(E_grid)) or np.any(E_grid <= 0):
        return np.inf
    inv = 1.0 / E_grid
    dc = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(Z_GRID))])
    m = np.interp(Z_SN, Z_GRID, dc)
    a = (m @ CINV_SN @ D_SN) / (m @ CINV_SN @ m)
    r = D_SN - a * m
    return float(r @ CINV_SN @ r)


def total(E_fn):
    """chi2_CC, chi2_SN and the fitted H0 for an expansion history E_fn(z)."""
    c_cc, h0 = chi2_cc(E_fn)
    return c_cc, chi2_sn(E_fn(Z_GRID)), h0


def e_lcdm(om):
    return lambda z: np.sqrt(om * (1 + np.asarray(z)) ** 3 + 1 - om)


def e_hyperconical(alpha_high=None):
    """Projected hyperconical history (the axial sector) with the running index."""
    ah = PH.load_alpha_high() if alpha_high is None else alpha_high
    return lambda z: PH.E_of_z(np.asarray(z, float), lambda zz: PH.alpha_sqrt(zz, PH.ALPHA_LOW, ah))


def e_sector(theta_deg, t0, projected, alpha_high=None):
    """E(z) of one sector taken as an independent history (unprojected or projected)."""
    if projected:
        e_work = HM.projected_sector(theta_deg, t0, alpha_high)[0]
        return lambda z: np.interp(z, HM.Z_WORK, e_work, left=np.nan, right=np.nan)
    zd, _, ed = HM.build_centered_history(np.radians(theta_deg), t0)
    return lambda z: HM.interp_curve(np.atleast_1d(np.asarray(z, float)), zd, ed)
