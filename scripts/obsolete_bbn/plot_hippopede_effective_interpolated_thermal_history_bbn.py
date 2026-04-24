from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
    standard_radiation_hubble,
)
from plot_hippopede_projected_thermal_history_bbn import (  # noqa: E402
    ALPHA_HIGH,
    ALPHA_LOW,
    DELTA_NEFF_95,
    DELTA_NEFF_CENTER,
    N_T,
    T_MAX_MEV,
    T_MIN_MEV,
    e_constant_alpha,
    neff_to_scale,
    z_of_temperature_mev,
)

def effective_log_interpolation(h_low: np.ndarray, h_high: np.ndarray, weight: np.ndarray):
    h_low = np.asarray(h_low, dtype=float)
    h_high = np.asarray(h_high, dtype=float)
    weight = np.asarray(weight, dtype=float)
    return np.exp((1.0 - weight) * np.log(h_low) + weight * np.log(h_high))


def effective_alpha_from_h(h_eff: np.ndarray, h_low: np.ndarray, h_high: np.ndarray):
    h_eff = np.asarray(h_eff, dtype=float)
    h_low = np.asarray(h_low, dtype=float)
    h_high = np.asarray(h_high, dtype=float)
    denom = np.log(h_high) - np.log(h_low)
    numer = np.log(h_eff) - np.log(h_low)
    weight = np.where(np.abs(denom) > 1.0e-14, numer / denom, 0.0)
    return ALPHA_LOW + np.clip(weight, 0.0, 1.0) * (ALPHA_HIGH - ALPHA_LOW)


def effective_alpha_from_observations(h_obs: np.ndarray, h_low: np.ndarray, h_high: np.ndarray):
    h_obs = np.asarray(h_obs, dtype=float)
    h_low = np.asarray(h_low, dtype=float)
    h_high = np.asarray(h_high, dtype=float)
    denom = np.log(h_high) - np.log(h_low)
    numer = np.log(h_obs) - np.log(h_low)
    weight = np.where(np.abs(denom) > 1.0e-14, numer / denom, 0.0)
    weight_clipped = np.clip(weight, 0.0, 1.0)
    alpha_eff = ALPHA_LOW + weight_clipped * (ALPHA_HIGH - ALPHA_LOW)
    return alpha_eff, weight, weight_clipped


def y_of_z(model: ExtendedProjectedHyperconical, z: np.ndarray):
    z = np.asarray(z, dtype=float)
    x = model.x_from_lz(np.log1p(z))
    u = np.sqrt(np.maximum(1.0 / model.k - x**2, 1.0e-14))
    return np.arctan2(x, u)


def main():
    temperatures = np.geomspace(T_MIN_MEV, T_MAX_MEV, N_T)
    z = z_of_temperature_mev(temperatures)

    model_low = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)
    model_high = ExtendedProjectedHyperconical(alpha=ALPHA_HIGH)
    z_eval = np.unique(np.concatenate(([0.0], np.geomspace(1.0e3, max(1.0e10, z.max() * 1.05), 5000), z)))
    z_eval.sort()

    h_std = standard_radiation_hubble(temperatures)
    e_low = e_constant_alpha(model_low, z_eval)
    e_high = e_constant_alpha(model_high, z_eval)

    scale_center = neff_to_scale(DELTA_NEFF_CENTER, T_mev=temperatures)
    scale_lo_95 = neff_to_scale(DELTA_NEFF_CENTER - DELTA_NEFF_95, T_mev=temperatures)
    scale_hi_95 = neff_to_scale(DELTA_NEFF_CENTER + DELTA_NEFF_95, T_mev=temperatures)
    h_low = H0_SI * np.interp(z, z_eval, e_low)
    h_high = H0_SI * np.interp(z, z_eval, e_high)
    h_obs = scale_center * h_std
    h_obs_lo = scale_lo_95 * h_std
    h_obs_hi = scale_hi_95 * h_std

    alpha_eff_t, weight_raw, weight_clipped = effective_alpha_from_observations(h_obs, h_low, h_high)
    h_eff = effective_log_interpolation(h_low, h_high, weight_clipped)

    bbn_mask = (temperatures >= 0.07) & (temperatures <= 1.0)
    y_vals = y_of_z(model_low, z[bbn_mask])
    alpha_y_vals = alpha_eff_t[bbn_mask]
    order = np.argsort(y_vals)
    y_sorted = y_vals[order]
    alpha_y_sorted = alpha_y_vals[order]

    resid_std = h_std / h_obs - 1.0
    resid_eff = h_eff / h_obs - 1.0
    resid_lo_95 = h_obs_lo / h_obs - 1.0
    resid_hi_95 = h_obs_hi / h_obs - 1.0

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.6, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 0.95], "hspace": 0.10},
    )

    ax1.fill_between(
        temperatures,
        h_obs_lo,
        h_obs_hi,
        color="#d7c4a3",
        alpha=0.30,
        linewidth=0.0,
        label=(
            r"BBN-compatible expansion band "
            r"($\Delta N_{\mathrm{eff}}=-0.10\pm0.21$, 95\%; parametric, cf.\ Sch\"oneberg 2024)"
        ),
        zorder=1,
    )
    ax1.plot(
        temperatures,
        h_obs,
        color="#b08a53",
        lw=1.8,
        ls="--",
        alpha=0.9,
        label=r"BBN-compatible central expansion ($\Delta N_{\mathrm{eff}}=-0.10$; parametric, cf.\ Sch\"oneberg 2024)",
        zorder=5,
    )
    ax1.plot(
        temperatures,
        h_std,
        color="#2d2d2d",
        lw=1.8,
        ls=(0, (5, 2)),
        label=r"Standard radiation era (variable $g_*(T)$)",
        zorder=6,
    )
    ax1.plot(
        temperatures,
        h_eff,
        color="#1f78b4",
        lw=2.5,
        label=r"Effective branch inferred from BBN central curve",
        zorder=7,
    )
    ax1.plot(
        temperatures,
        h_low,
        color="#6a3d9a",
        lw=1.35,
        ls="-.",
        alpha=0.85,
        label=rf"Projected branch with constant $\alpha={ALPHA_LOW:.3f}$",
        zorder=3,
    )
    ax1.plot(
        temperatures,
        h_high,
        color="#e31a1c",
        lw=1.35,
        ls=":",
        alpha=0.9,
        label=rf"Projected branch with constant $\alpha={ALPHA_HIGH:.1f}$",
        zorder=3,
    )

    ax2.fill_between(
        temperatures,
        resid_lo_95,
        resid_hi_95,
        color="#d7c4a3",
        alpha=0.30,
        linewidth=0.0,
        zorder=1,
    )
    ax2.axhline(0.0, color="#2d2d2d", lw=1.3, ls=(0, (5, 2)), alpha=0.8, zorder=5)
    ax2.plot(
        temperatures,
        resid_std,
        color="#2d2d2d",
        lw=1.8,
        ls=(0, (5, 2)),
        zorder=4,
        label=r"Standard$-$observations",
    )
    ax2.plot(
        temperatures,
        resid_eff,
        color="#1f78b4",
        lw=2.5,
        zorder=6,
        label=r"Interpolated$-$observations",
    )

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.grid(True, which="major", color="#ebebeb", linewidth=0.6)
        ax.grid(True, which="minor", color="#f4f4f4", linewidth=0.45)
        ax.minorticks_on()

    ax1.set_yscale("log")
    ax1.set_ylabel(r"Expansion rate $H(T)\ [{\rm s}^{-1}]$")
    ax1.set_ylim(3.0e-4, 1.0e3)
    ax1.legend(loc="upper left", fontsize=9, frameon=False)

    ax2.set_ylabel(r"Residual $(H-H_{\rm obs})/H_{\rm obs}$")
    ax2.set_xlabel(r"Perceived temperature $T\ [{\rm MeV}]$")
    ax2.set_ylim(-0.08, 0.08)
    ax2.set_xlim(1.0e-2, 1.0e1)

    sample_temperatures = np.array([0.07, 0.10, 0.20, 0.50, 1.00])
    sample_alpha_t = np.interp(sample_temperatures, temperatures, alpha_eff_t)
    sample_h_obs = np.interp(sample_temperatures, temperatures, h_obs)
    sample_h_obs_lo = np.interp(sample_temperatures, temperatures, h_obs_lo)
    sample_h_obs_hi = np.interp(sample_temperatures, temperatures, h_obs_hi)
    sample_resid_eff = np.interp(sample_temperatures, temperatures, resid_eff)
    obs_err = np.vstack((sample_h_obs - sample_h_obs_lo, sample_h_obs_hi - sample_h_obs))
    ax1.errorbar(
        sample_temperatures,
        sample_h_obs,
        yerr=obs_err,
        fmt="o",
        ms=4.2,
        color="#b08a53",
        mfc="white",
        mec="#8b6a3d",
        mew=0.9,
        elinewidth=0.9,
        capsize=2.2,
        zorder=8,
        label=r"BBN-compatible anchor points (parametric; cf.\ Sch\"oneberg 2024)",
    )
    ax2.scatter(
        sample_temperatures,
        np.zeros_like(sample_temperatures),
        s=18,
        facecolor="#b08a53",
        edgecolor="white",
        linewidth=0.6,
        zorder=8,
    )

    png_path = FIGURES / "hippopede_effective_interpolated_thermal_history_bbn.png"
    pdf_path = FIGURES / "hippopede_effective_interpolated_thermal_history_bbn.pdf"
    json_path = FIGURES / "hippopede_effective_interpolated_thermal_history_bbn.json"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "alpha_low": ALPHA_LOW,
        "alpha_high": ALPHA_HIGH,
        "temperature_mev_samples": sample_temperatures.tolist(),
        "alpha_eff_T_samples": sample_alpha_t.tolist(),
        "residual_eff_samples": sample_resid_eff.tolist(),
        "weight_raw_min": float(np.nanmin(weight_raw)),
        "weight_raw_max": float(np.nanmax(weight_raw)),
        "weight_clipped_min": float(np.nanmin(weight_clipped)),
        "weight_clipped_max": float(np.nanmax(weight_clipped)),
        "alpha_y_min": float(np.nanmin(alpha_y_sorted)),
        "alpha_y_max": float(np.nanmax(alpha_y_sorted)),
        "alpha_y_grid": y_sorted[:: max(1, len(y_sorted) // 80)].tolist(),
        "alpha_y_samples": alpha_y_sorted[:: max(1, len(alpha_y_sorted) // 80)].tolist(),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved PNG:  {png_path}")
    print(f"Saved PDF:  {pdf_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
