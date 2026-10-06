"""Projected hyperconical expansion history with a redshift-dependent index alpha(z).

Projection (Monjo 2018, k=1):
    rhat(z) = 2 arctan[A(z)],   A = (gamma/2) g^{-alpha(z)},   g = 1 - gamma/gamma_U,
    gamma = arcsin x(z),  x(z) = xi_1^{-1}(ln(1+z)),  gamma_U = pi/3,
    H(z) = (d rhat/dz)^{-1}.

Near the boundary g ~ (1+z)^{-2}, so g drops below machine precision at
z ~ 1e8 and x(z) can no longer be resolved.  Above Z_REF the code therefore
evaluates g through its exact asymptotic law, ln g = ln g(Z_REF) - 2 ln[(1+z)/(1+Z_REF)],
whose relative corrections are O(g(Z_REF)) ~ 1e-8.  All large numbers are
handled in logarithms.
"""

from __future__ import annotations

import numpy as np

Z_REF = 1.0e4
GAMMA_U = np.pi / 3.0
# low-redshift projection index of the hyperconical model (Monjo 2018, PRD 98, 043508)
ALPHA_LOW = 0.283


def load_alpha_high():
    """alpha_high fitted to CC + binned Pantheon+ by fit_alpha_high.py (run that script first)."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "json" / "fit_alpha_high.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/fit_alpha_high.py first")
    return float(json.loads(path.read_text())["alpha_high"])


def _lookup(k: float = 1.0):
    """Tabulate ln(1+z) = xi_1(x) on [0, x_max) (same line-of-sight integral as the figure scripts)."""
    T = 1.0

    def ttp(r):
        s = np.sqrt((-k * r**2 + T**2) / T**2)
        return np.sqrt((T**2 * (k - 2) * s - 2 * k * r**2 + 2 * T**2) / (s * T**2 * k))

    def grrp(r):
        s = np.sqrt((-k * r**2 + T**2) / T**2)
        return -(((T**2 * (k - 2) + k * r**2) * s - 2 * k * r**2 + 2 * T**2)
                 / ((T**2 * (k - 2) * s - 2 * k * r**2 + 2 * T**2) * (-k * r**2 + T**2)))

    mx = np.sqrt((1.0 - (1.0 - k / 2.0) ** 2) / k)
    seq = np.linspace(0.0, 12.0, 60000)
    x = np.unique(np.concatenate([np.linspace(0.0, mx, 20000), mx * (1.0 - 10.0 ** (-seq))]))
    x = x[(x >= 0.0) & (x < mx)]
    f = np.sqrt(-grrp(x)) / ttp(x)          # d xi_1 / dx  (> 0)
    lz = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])
    # odd extension to x < 0 (z < 0, future of the observer)
    return (np.concatenate([-x[:0:-1], x]), np.concatenate([-lz[:0:-1], lz]),
            np.concatenate([f[:0:-1], f]), mx)


_X, _LZ, _F, _MX = _lookup()


def alpha_sqrt(z, alpha_low, alpha_high):
    """alpha(z) = alpha_high - (alpha_high - alpha_low)/sqrt(1+z) and its z-derivative."""
    z = np.asarray(z, dtype=float)
    d = alpha_high - alpha_low
    return alpha_high - d / np.sqrt(1.0 + z), 0.5 * d * (1.0 + z) ** -1.5


def alpha_const(alpha):
    return lambda z: (np.full_like(np.asarray(z, dtype=float), alpha), np.zeros_like(np.asarray(z, dtype=float)))


def _geometry(z):
    """gamma, d gamma/dz, ln g, (dg/dz)/g for every z (asymptotic branch above Z_REF)."""
    z = np.asarray(z, dtype=float)
    lz = np.log1p(np.minimum(z, Z_REF))
    x = np.interp(lz, _LZ, _X)
    dxdz = 1.0 / (np.interp(x, _X, _F) * (1.0 + np.minimum(z, Z_REF)))
    gam = np.arcsin(x)
    dgam = dxdz / np.sqrt(1.0 - x**2)
    lng = np.log(1.0 - gam / GAMMA_U)
    dlng = -dgam / (GAMMA_U - gam)

    hi = z > Z_REF
    if np.any(hi):
        x_ref = np.interp(np.log1p(Z_REF), _LZ, _X)
        lng_ref = np.log(1.0 - np.arcsin(x_ref) / GAMMA_U)
        lng[hi] = lng_ref - 2.0 * np.log((1.0 + z[hi]) / (1.0 + Z_REF))
        dlng[hi] = -2.0 / (1.0 + z[hi])
        g = np.exp(lng[hi])
        gam[hi] = GAMMA_U * (1.0 - g)
        dgam[hi] = -GAMMA_U * g * dlng[hi]
    return gam, dgam, lng, dlng


def ln_hubble_unnormalised(z, alpha_fn):
    """ln H(z) up to a constant, H = (d rhat/dz)^{-1}."""
    z = np.asarray(z, dtype=float)
    gam, dgam, lng, dlng = _geometry(z)
    a, da = alpha_fn(z)
    nz = gam != 0.0
    lnA = np.log(np.maximum(np.abs(gam), 1e-300) / 2.0) - a * lng
    # d rhat/dz = 2/(1+A^2) [ (gamma'/2) g^{-a} - A (a' ln g + a g'/g) ]
    #           = 2A/(1+A^2) [ gamma'/gamma - a' ln g - a g'/g ]   (gamma > 0)
    safe = np.where(nz, gam, 1.0)
    bracket = dgam / safe - da * lng - a * dlng
    # |2A/(1+A^2)| = 2/(|A| + 1/|A|); A and gamma share sign, so the product is positive
    ln_drdz = np.log(2.0) - np.logaddexp(lnA, -lnA) + np.log(np.abs(np.where(nz, bracket * np.sign(safe), 1.0)))
    ln_drdz = np.where(nz, ln_drdz, np.log(np.abs(dgam)))  # z = 0: d rhat/dz = gamma'
    return -ln_drdz


def rhat_of_z(z, alpha_fn):
    """Projected angular distance rhat(z) = 2 arctan A on the unit 3-sphere (0 at z=0, -> pi)."""
    z = np.asarray(z, dtype=float)
    gam, _, lng, _ = _geometry(z)
    a, _ = alpha_fn(z)
    lnA = np.log(np.maximum(np.abs(gam), 1e-300) / 2.0) - a * lng
    return np.sign(gam) * 2.0 * np.arctan(np.exp(lnA))


def E_of_z(z, alpha_fn):
    z = np.asarray(z, dtype=float)
    return np.exp(ln_hubble_unnormalised(z, alpha_fn) - ln_hubble_unnormalised(np.array([0.0]), alpha_fn)[0])


def q_of_z(z, alpha_fn):
    """q = -1 + d ln H / d ln(1+z), by centred differences in ln(1+z)."""
    z = np.asarray(z, dtype=float)
    h = 1e-4
    lp = np.log1p(z)
    up = ln_hubble_unnormalised(np.expm1(lp + h), alpha_fn)
    dn = ln_hubble_unnormalised(np.expm1(lp - h), alpha_fn)
    return -1.0 + (up - dn) / (2.0 * h)
