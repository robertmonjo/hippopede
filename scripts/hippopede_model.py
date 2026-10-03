"""Kinematics of the hippopede sectors: centred histories and their projection.

Hypersurface (units in which the constant multiplying x^2 + y^2 + z^2 is one):
    rho(t, theta)^2 = sin^2(theta) + 4 t^2 cos^2(theta),
with theta the polar angle from the distinguished u axis.  Centred scale factor, measured from
the centre (u = t) of the +u lobe:
    a_c(t, theta) = sqrt(rho^2 + t^2 - 2 t rho cos(theta)).
A sector history is retained after the first minimum of a_c before t0 (single-valued
redshift); 1 + z = a_c(t0)/a_c(t), E = H/H(t0), q = -a a''/a'^2.

Projection (Monjo 2018): the projected rate of a sector is E_theta(z*) with
z* = E_hyp^proj(z) - 1, where E_hyp^proj is the projected hyperconical history with the running
index alpha(z) = alpha_high - (alpha_high - alpha_low)/sqrt(1+z) (projected_hyperconical.py).
"""

from __future__ import annotations

import numpy as np

import projected_hyperconical as PH

Z_WORK = np.linspace(-0.85, 2.15, 2500)
_ZMAP_CACHE: dict = {}


def rho_hippopede(t, theta):
    return np.sqrt(np.sin(theta) ** 2 + 4.0 * t**2 * np.cos(theta) ** 2)


def a_centered(t, theta):
    rho = rho_hippopede(t, theta)
    return np.sqrt(rho**2 + t**2 - 2.0 * t * rho * np.cos(theta))


def h0_sector(theta_rad, t0, dt=1e-5):
    """Present expansion rate of the centred history, H = (da_c/dt)/a_c at t0."""
    a = lambda t: a_centered(np.array([t]), theta_rad)[0]
    return (a(t0 + dt) - a(t0 - dt)) / (2.0 * dt) / a(t0)


def build_centered_history(theta, t0, n=60000):
    """Return z, q, E of the centred history of sector theta [rad], sorted by z."""
    t = np.linspace(1.0e-4, max(40.0, 25.0 * t0), n)
    a = a_centered(t, theta)
    a_dot = np.gradient(a, t)
    a_ddot = np.gradient(a_dot, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = a_dot / a
        q = -a * a_ddot / (a_dot**2)
    i0 = min(max(np.searchsorted(t, t0), 1), len(t) - 2)
    first_min = np.argmin(a[: i0 + 1])
    t, a, h, q = t[first_min:], a[first_min:], h[first_min:], q[first_min:]
    z = a_centered(np.array([t0]), theta)[0] / a - 1.0
    e = h / np.interp(t0, t, h)
    order = np.argsort(z)
    uz, idx = np.unique(z[order], return_index=True)
    return uz, q[order][idx], e[order][idx]


def interp_curve(z_grid, z_data, y_data):
    """Linear interpolation, NaN outside the range of the data."""
    ok = np.isfinite(z_data) & np.isfinite(y_data)
    z_data, y_data = z_data[ok], y_data[ok]
    if len(z_data) < 2:
        return np.full_like(np.asarray(z_grid, float), np.nan)
    return np.interp(z_grid, z_data, y_data, left=np.nan, right=np.nan)


def project_curve(z_obs, z_unproj, e_unproj, z_map):
    """Projected E and q on z_obs: E_proj(z) = E(z*(z)), q = -1 + (1+z) E'/E."""
    e_proj = interp_curve(z_map, z_unproj, e_unproj)
    q_proj = np.full_like(z_obs, np.nan, dtype=float)
    ok = np.isfinite(e_proj)
    if ok.sum() > 8:
        de = np.gradient(e_proj[ok], z_obs[ok])
        q_proj[ok] = -1.0 + (1.0 + z_obs[ok]) * de / e_proj[ok]
    return e_proj, q_proj


def z_map(alpha_high=None, alpha_low=None):
    """z*(z) = E_hyp^proj(z) - 1 on Z_WORK for the running index (defaults: fitted alpha_high,
    alpha_low = PH.ALPHA_LOW)."""
    ah = PH.load_alpha_high() if alpha_high is None else float(alpha_high)
    al = PH.ALPHA_LOW if alpha_low is None else float(alpha_low)
    if (al, ah) not in _ZMAP_CACHE:
        _ZMAP_CACHE[(al, ah)] = PH.E_of_z(Z_WORK, lambda z: PH.alpha_sqrt(z, al, ah)) - 1.0
    return _ZMAP_CACHE[(al, ah)]


_HIST_CACHE = {}


def centered_history_cached(theta_deg, t0):
    """build_centered_history for theta [deg], cached by (theta, t0)."""
    key = (float(theta_deg), float(t0))
    if key not in _HIST_CACHE:
        _HIST_CACHE[key] = build_centered_history(np.radians(theta_deg), t0)
    return _HIST_CACHE[key]


def projected_sector(theta_deg, t0, alpha_high=None, alpha_low=None):
    """Projected E and q of sector theta [deg] on Z_WORK (NaN beyond the reach of the sector)."""
    zd, _, ed = centered_history_cached(theta_deg, t0)
    return project_curve(Z_WORK, zd, ed, z_map(alpha_high, alpha_low))
