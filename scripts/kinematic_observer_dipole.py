"""Kinematic alternative to the geometric dipole: flat LCDM with an observer that moves relative to the sources faster
than relative to the CMB.

To first order in the velocity, a flux-limited sample with integral count slope x and spectral index alpha has the
kinematic dipole [2 + x(1 + alpha)] beta along the velocity (Ellis & Baldwin 1984).  The CMB dipole fixes the velocity
relative to the CMB, beta_CMB = 369.82/c towards (l, b) = (264.0, 48.3) deg.  If the observer moves relative to the
sources with beta_CMB + delta_beta, every sample acquires the excess K delta_beta over its CMB-kinematic dipole, with
K = 2 + x(1 + alpha) its own factor and the same direction for all samples.  delta_beta (amplitude and direction) is
fitted to the CatWISE excess vector D_geo of quasar_dipole_fit.py, which it reproduces exactly; the model then predicts
  - the radio samples: excess relative to that of CatWISE, K_radio / K_CatWISE, for NVSS (x = 0.74, alpha = 0.75) and
    RACS-low (x = 0.72, alpha = 0.88) above 15 mJy (Wagenveld et al. 2025, A&A 697, A112, Sects. 3.1-3.2), compared with
    the measured ratio of their excesses in the same analysis (Table 1 there);
  - the three Quaia slices (dipole_zslices/results.json): excess K_i delta_beta with the count slope and colour
    spectral index of each slice, compared through the amplitude chi2 sum (|D_i| - |e_i|)^2 / sigma_i^2 and the vector
    chi2 sum |e_i - D_i u|^2 / sigma_i^2 (u the direction of the CatWISE excess), both relative to no dipole, as in
    light_weight_comparison.py.
Writes json/kinematic_observer_dipole.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import quasar_dipole_fit as Q  # noqa: E402

C_KMS = 299792.458
RADIO = {"NVSS": {"x": 0.74, "alpha": 0.75}, "RACS-low": {"x": 0.72, "alpha": 0.88}}
# Wagenveld et al. 2025, Table 1: measured dipole, its error and the kinematic expectation for the CMB velocity
RADIO_OBS = {"NVSS": (0.0139, 0.0029, 0.0041), "RACS-low": (0.0143, 0.0024, 0.0041), "CatWISE": (0.0151, 0.0016, 0.0073)}


def unit_gal(l, b):
    lo, la = np.radians(l), np.radians(b)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def main():
    k_cw = 2.0 + Q.X_SLOPE * (1.0 + Q.ALPHA_SPEC)
    _, d_geo, lb_geo = Q.excess_vector()
    u = unit_gal(*lb_geo)
    dbeta = d_geo / k_cw                                   # amplitude of the extra velocity, along the excess
    v_cmb = Q.BETA * C_KMS * unit_gal(*Q.LB_CMB)
    dv = dbeta * C_KMS * u
    v_tot = v_cmb + dv
    out = {"K_catwise": k_cw, "D_geo": d_geo, "excess_lb": lb_geo, "delta_beta": dbeta, "delta_v_kms": dbeta * C_KMS,
           "v_total_kms": float(np.linalg.norm(v_tot)), "v_total_over_v_cmb": float(np.linalg.norm(v_tot) / (Q.BETA * C_KMS)),
           "angle_total_velocity_to_cmb_deg": angle_deg(v_tot, v_cmb), "angle_excess_to_cmb_deg": angle_deg(u, v_cmb)}
    cw, cw_s, cw_k = RADIO_OBS["CatWISE"]
    out["radio"] = {}
    for name, p in RADIO.items():
        k = 2.0 + p["x"] * (1.0 + p["alpha"])
        d, s, kin = RADIO_OBS[name]
        meas = (d - kin) / (cw - cw_k)
        err = float(np.hypot(s / (cw - cw_k), meas * cw_s / (cw - cw_k)))
        out["radio"][name] = {"K": k, "predicted_ratio": k / k_cw, "measured_ratio": meas, "measured_error": err,
                              "pull": (k / k_cw - meas) / err}
    slices = [s for s in json.loads((ROOT / "dipole_zslices" / "results.json").read_text())["slices"]
              if isinstance(s["z_range"], list) and s["z_range"][1] <= 1.8]
    e = np.array([s["dip"]["excess_vec"] for s in slices])
    amp = np.array([s["dip"]["excess_amp"] for s in slices])
    sig = np.array([s["dip"]["sigma_mocks_per_component"] for s in slices])
    k_q = np.array([2.0 + s["x"] * (1.0 + s["alpha_colour_median_edge"]) for s in slices])
    dq = k_q * dbeta
    none = float(np.sum(amp**2 / sig**2))
    chi2_amp = float(np.sum(((dq - amp) / sig) ** 2))
    chi2_vec = float(np.sum(np.sum((e - dq[:, None] * u[None, :]) ** 2, axis=1) / sig**2))
    out["quaia"] = {"z_ranges": [s["z_range"] for s in slices], "K": k_q.tolist(), "predicted_excess": dq.tolist(),
                    "measured_excess": amp.tolist(), "sigma": sig.tolist(), "chi2_none": none,
                    "dchi2_amp": chi2_amp - none, "dchi2_vec": chi2_vec - none}
    (ROOT / "json" / "kinematic_observer_dipole.json").write_text(json.dumps(out, indent=1))
    print(f"delta v = {out['delta_v_kms']:.0f} km/s along the excess; total {out['v_total_kms']:.0f} km/s "
          f"({out['v_total_over_v_cmb']:.2f} x CMB), {out['angle_total_velocity_to_cmb_deg']:.1f} deg from the CMB dipole")
    for name, r in out["radio"].items():
        print(f"{name}: predicted radio/CatWISE {r['predicted_ratio']:.2f}, measured {r['measured_ratio']:.2f} +- "
              f"{r['measured_error']:.2f} (pull {r['pull']:+.1f})")
    q = out["quaia"]
    print(f"Quaia: predicted {np.round(q['predicted_excess'], 4).tolist()} vs {np.round(q['measured_excess'], 4).tolist()}; "
          f"dchi2 amp {q['dchi2_amp']:+.1f}, vec {q['dchi2_vec']:+.1f}")


if __name__ == "__main__":
    main()
