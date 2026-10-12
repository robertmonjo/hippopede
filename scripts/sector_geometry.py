"""Sector of a source seen by an observer on one lobe of the hippopede.

At large t each lobe is a 3-sphere of radius t centred at u = +t, passing through the origin
(the node).  The sector angle theta is the polar angle measured at the origin from the u axis.
A point of the lobe at sector theta has, in the (r, u) half-plane,
    (r, u) = t (sin 2 theta, 1 + cos 2 theta),
so seen from the centre of the lobe it lies at the angle chi = 2 theta from the pole u = 2t
(inscribed-angle theorem).  Geodesic distances on the lobe are therefore measured in chi.

The projected distance rhat(z) is an angle on the unit 3-sphere.  For an observer at
chi_obs = 2 theta_obs and a source at angular distance rhat in a direction at angle psi from the
projected axis (the direction of the pole), the spherical law of cosines gives
    cos chi_s = cos chi_obs cos rhat + sin chi_obs sin rhat cos psi,   theta_s = chi_s / 2,
with chi in [0, pi] (theta in [0, 90 deg]; theta = 90 deg is the node).
The volume element of the lobe is proportional to sin^2(chi) d chi = 2 sin^2(2 theta) d theta.
"""

from __future__ import annotations

import os

import numpy as np


def source_sector_deg(theta_obs_deg, rhat, cospsi):
    """Sector angle [deg] of a source at angular distance rhat [rad]; broadcasts."""
    chi_o = 2.0 * np.radians(np.asarray(theta_obs_deg, dtype=float))
    c = np.cos(chi_o) * np.cos(rhat) + np.sin(chi_o) * np.sin(rhat) * np.asarray(cospsi, dtype=float)
    return 0.5 * np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# Largest sector angle used in the analysis.  At the reference t0 = 2.6 the observed redshift reach
# of a projected sector (sector_reach.py) is z = 2.35 at theta = 60 deg, 2.07 at 70 deg, 1.97 at
# 75 deg and 1.87 at 85 deg: the sectors up to 70 deg are those whose history covers z <= 2, the
# range of the data and of Fig. 2.  The environment variable HIPPOPEDE_THETA_MAX_DEG replaces it; with
# 90 the whole lobe is tabulated and a model is limited only by the reach of each sector (the tables are
# NaN beyond it), as in the overview map of plot_ball_zoom_unbinned.py.
THETA_MAX_DEG = float(os.environ.get("HIPPOPEDE_THETA_MAX_DEG", "70"))

# Largest observer sector for the directional analyses.  A source at angular distance rhat from
# the observer lies at central angle chi_s <= 2 theta_obs + rhat, so every source up to z = 2.1
# (rhat = 69.5 deg with the running index) stays in a sector theta_s <= THETA_MAX_DEG when
# theta_obs <= (2 THETA_MAX_DEG - 69.5 deg)/2 = 35 deg.
THETA_OBS_MAX_DEG = 35.0


# Light weight of a source sector, switched on with the environment variable HIPPOPEDE_LIGHT_WEIGHT = atan, atan2, cos,
# cos2 or quad (unset or 0: every sector has the same weight).  The light received from sector theta by an observer in sector
# theta_obs is weighted by a function of u = |theta - theta_obs| / (theta_max - theta_obs), with theta_max(z) the
# largest sector whose history reaches the redshift z of the source:
#     atan:  w = (2/pi) atan[(1 - u) / u]        atan2: w = {(2/pi) atan[(1 - u) / u]}^2
#     cos:   w = cos(pi u / 2)                   cos2:  w = cos^2(pi u / 2)
#     quad:  w = (1 - u)^2
# w = 1 in the sector of the observer and w = 0 for u >= 1 and for the sectors that do not reach z.
_LW = os.environ.get("HIPPOPEDE_LIGHT_WEIGHT", "0").strip().lower()
LIGHT_WEIGHT = "" if _LW in ("", "0") else _LW
LIGHT_WEIGHT_FORMS = ("atan", "atan2", "cos", "cos2", "quad")
assert LIGHT_WEIGHT in ("",) + LIGHT_WEIGHT_FORMS, f"unknown HIPPOPEDE_LIGHT_WEIGHT={_LW}"


def light_weight(theta_deg, theta_obs_deg, theta_max_deg, form=None):
    """Light weight of sector theta for an observer in theta_obs (forms above), in [0, 1]; broadcasts."""
    form = form or LIGHT_WEIGHT
    assert form in LIGHT_WEIGHT_FORMS, form
    tp = np.abs(np.asarray(theta_deg, dtype=float) - theta_obs_deg)
    tpm = np.asarray(theta_max_deg, dtype=float) - theta_obs_deg
    with np.errstate(divide="ignore", invalid="ignore"):
        if form in ("atan", "atan2"):
            w = np.clip(np.arctan2(tpm - tp, tp) / (np.pi / 2), 0.0, 1.0)   # arctan2: 1 at u = 0, <= 0 for u >= 1
            w = w**2 if form == "atan2" else w
        else:
            u = np.clip(tp / tpm, 0.0, 1.0)
            w = {"cos": np.cos(0.5 * np.pi * u), "cos2": np.cos(0.5 * np.pi * u) ** 2, "quad": (1.0 - u) ** 2}[form]
    return np.where(np.isfinite(tpm) & (tpm > 0), np.clip(w, 0.0, 1.0), 0.0)


def lobe_volume_weight(theta_deg):
    """Volume weight of sector theta on the lobe, proportional to sin^2(2 theta) (unnormalised)."""
    return np.sin(2.0 * np.radians(np.asarray(theta_deg, dtype=float))) ** 2


def _self_test():
    r = np.radians(np.array([0.0, 10.0, 40.0, 120.0]))
    assert np.allclose(source_sector_deg(0.0, r, 1.0), np.degrees(r) / 2, atol=1e-5)  # observer on the axis
    assert np.allclose(source_sector_deg(20.0, r, 1.0), np.abs(40.0 - np.degrees(r)) / 2, atol=1e-5)  # towards the pole
    assert np.allclose(source_sector_deg(20.0, r, -1.0), np.minimum(40.0 + np.degrees(r), 360.0 - 40.0 - np.degrees(r)) / 2, atol=1e-5)
    # the point (r, u) = t (sin 2 theta, 1 + cos 2 theta) satisfies rho = 2 t cos theta
    th = np.radians(np.linspace(0, 89, 50)); t = 3.0
    rr, uu = t * np.sin(2 * th), t * (1 + np.cos(2 * th))
    assert np.allclose(np.hypot(rr, uu), 2 * t * np.cos(th)) and np.allclose(np.arctan2(rr, uu), th)
    for form in LIGHT_WEIGHT_FORMS:   # 1 in the sector of the observer, 0 at theta_max, decreasing in between
        w = light_weight(np.array([2.0, 11.0, 20.0, 38.0, 50.0]), 2.0, 38.0, form)
        assert np.allclose(w[[0, 3, 4]], [1.0, 0.0, 0.0]) and 1.0 > w[1] > w[2] > 0.0, (form, w)
    print("sector_geometry: self-test passed")


if __name__ == "__main__":
    _self_test()
