from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

for extra in (str(SCRIPT_ROOT), str(ROOT / "vendor_gapp"), str(ROOT / "vendor_gapp" / "covfunctions")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from analyze_hippopede_dipole_bbn import (  # noqa: E402
    ExtendedProjectedHyperconical,
    H0_SI,
    T0_EV,
    standard_radiation_hubble,
)
from plot_hippopede_qz_double_panel_with_gapp import e_lcdm  # noqa: E402


ALPHA_LOW = 0.283
ALPHA_HIGH = 0.5


def alpha_of_z(z: np.ndarray, zc: float):
    z = np.asarray(z, dtype=float)
    return ALPHA_LOW + (ALPHA_HIGH - ALPHA_LOW) * z / (z + zc)


def projected_distance_variable_alpha(model: ExtendedProjectedHyperconical, z: np.ndarray, zc: float):
    z = np.asarray(z, dtype=float)
    x = model.x_from_lz(np.log1p(z))
    k = model.k
    u = np.sqrt(np.maximum(1.0 / k - x**2, 1e-14))
    y = np.arctan2(x, u)
    alpha = alpha_of_z(z, zc)
    g = np.maximum(1.0 - y / model.y0, 1e-12)
    t = (y / 2.0) / (g**alpha)
    return 2.0 * np.arctan(t)


def e_variable_alpha(model: ExtendedProjectedHyperconical, z: np.ndarray, zc: float):
    z = np.asarray(z, dtype=float)
    rhat = projected_distance_variable_alpha(model, z, zc)
    dr_dz = np.gradient(rhat, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = 1.0 / dr_dz
    valid = np.isfinite(h)
    h0 = np.interp(0.0, z[valid], h[valid])
    return h / h0


def rms(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def analyze_zc(zc: float):
    model = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)

    z_low = np.linspace(0.0, 2.0, 600)
    e_var_low = e_variable_alpha(model, z_low, zc)
    e_ref_low, _ = model.e_and_q(z_low)
    e_std_low = e_lcdm(z_low)

    z_hi = np.logspace(3.0, 6.0, 240)
    e_var_hi = e_variable_alpha(model, z_hi, zc)
    n_eff = np.gradient(np.log(e_var_hi), np.log1p(z_hi))

    alpha_hi = alpha_of_z(z_hi, zc)
    coeff_hi = e_var_hi / ((1.0 + z_hi) ** (1.0 + 2.0 * alpha_hi))
    coeff_med = float(np.nanmedian(coeff_hi))

    temperatures = np.array([0.07, 0.10, 1.0], dtype=float)
    z_t = temperatures * 1.0e6 / T0_EV - 1.0
    alpha_t = alpha_of_z(z_t, zc)
    h_projected = H0_SI * coeff_med * ((temperatures * 1.0e6) / T0_EV) ** (1.0 + 2.0 * alpha_t)
    h_standard = standard_radiation_hubble(temperatures)
    ratio = h_projected / h_standard

    return {
        "z_c": float(zc),
        "alpha_at_z=1": float(alpha_of_z(np.array([1.0]), zc)[0]),
        "alpha_at_z=2": float(alpha_of_z(np.array([2.0]), zc)[0]),
        "rms_vs_alpha_0283_on_0_2": rms(e_var_low, e_ref_low),
        "rms_vs_lcdm_on_0_2": rms(e_var_low, e_std_low),
        "n_eff_median_1e3_1e6": float(np.nanmedian(n_eff)),
        "alpha_median_1e3_1e6": float(np.nanmedian(alpha_hi)),
        "prefactor_median": coeff_med,
        "bbn_temperatures_mev": temperatures.tolist(),
        "hproj_over_hrad": ratio.tolist(),
        "ratio_median": float(np.nanmedian(ratio)),
    }


def main():
    zc_values = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    results = [analyze_zc(zc) for zc in zc_values]

    out_path = FIGURES / "hippopede_variable_alpha_scan.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("z_c scan for alpha(z) = 0.283 + (0.5-0.283) z/(z+z_c)")
    for row in results:
        print(
            f"z_c={row['z_c']:>6.1f}  "
            f"alpha(1)={row['alpha_at_z=1']:.4f}  "
            f"RMS vs 0.283={row['rms_vs_alpha_0283_on_0_2']:.4f}  "
            f"n_med={row['n_eff_median_1e3_1e6']:.4f}  "
            f"H/Hrad@0.1MeV={row['hproj_over_hrad'][1]:.4f}"
        )
    print(f"\nSaved scan: {out_path}")


if __name__ == "__main__":
    main()
