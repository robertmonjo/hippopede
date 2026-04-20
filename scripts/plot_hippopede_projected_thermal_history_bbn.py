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
    T0_EV,
    standard_radiation_hubble,
)


ALPHA_LOW = 0.283
ALPHA_HIGH = 0.5
ZC = 1.6155e4
DELTA = 2.0

# BBN-inferred observational corridor from primordial abundances, expressed
# as an equivalent extra-radiation contribution (Schoneberg 2024).
NEFF_SM = 3.046
DELTA_NEFF_CENTER = -0.10
DELTA_NEFF_SIGMA = 0.21
DELTA_NEFF_95 = 1.96 * DELTA_NEFF_SIGMA

T_MIN_MEV = 0.01
T_MAX_MEV = 10.0
N_T = 900
RUN_MIN_MEV = 0.03
RUN_MAX_MEV = 1.5
# Mild log-space smoothing used only for the displayed running-alpha thermal
# branch, to suppress derivative artefacts from the dense projected-map lookup.
SMOOTH_WINDOW = 41


def alpha_logistic(z: np.ndarray, zc: float = ZC, delta: float = DELTA):
    z = np.asarray(z, dtype=float)
    return ALPHA_LOW + 0.5 * (ALPHA_HIGH - ALPHA_LOW) * (
        1.0 + np.tanh((np.log1p(z) - np.log1p(zc)) / delta)
    )


def z_of_temperature_mev(T_mev: np.ndarray):
    return np.asarray(T_mev, dtype=float) * 1.0e6 / T0_EV - 1.0


def e_constant_alpha(model: ExtendedProjectedHyperconical, z: np.ndarray):
    h_raw = model.projected_hubble_unnormalized(z)
    h0 = model.projected_hubble_unnormalized(np.array([0.0]))[0]
    return h_raw / h0


def e_variable_alpha(model: ExtendedProjectedHyperconical, z: np.ndarray, zc: float = ZC, delta: float = DELTA):
    z = np.asarray(z, dtype=float)
    x = model.x_from_lz(np.log1p(z))
    u = np.sqrt(np.maximum(1.0 / model.k - x**2, 1.0e-14))
    y = np.arctan2(x, u)
    a = alpha_logistic(z, zc=zc, delta=delta)
    g = np.maximum(1.0 - y / model.y0, 1.0e-12)
    t = (y / 2.0) / (g**a)
    rhat = 2.0 * np.arctan(t)
    dr_dz = np.gradient(rhat, z, edge_order=2)
    dr_dz = np.where(np.abs(dr_dz) < 1.0e-18, np.sign(dr_dz) * 1.0e-18 + (dr_dz == 0.0) * 1.0e-18, dr_dz)
    h_raw = 1.0 / dr_dz
    h0 = np.interp(0.0, z, h_raw)
    return h_raw / h0


def neff_to_scale(delta_neff: np.ndarray | float):
    delta_neff = np.asarray(delta_neff, dtype=float)
    return np.sqrt(1.0 + (7.0 / 43.0) * delta_neff)


def smooth_positive_series(y: np.ndarray, window: int = SMOOTH_WINDOW):
    y = np.maximum(np.asarray(y, dtype=float), 1.0e-30)
    if window <= 1 or len(y) < window:
        return y.copy()
    pad = window // 2
    logy = np.log(y)
    padded = np.pad(logy, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.exp(np.convolve(padded, kernel, mode="valid"))


def main():
    temperatures = np.geomspace(T_MIN_MEV, T_MAX_MEV, N_T)
    z = z_of_temperature_mev(temperatures)
    temperatures_run = np.geomspace(RUN_MIN_MEV, RUN_MAX_MEV, max(400, N_T // 2))
    z_run = z_of_temperature_mev(temperatures_run)

    model_low = ExtendedProjectedHyperconical(alpha=ALPHA_LOW)
    model_half = ExtendedProjectedHyperconical(alpha=ALPHA_HIGH)
    z_eval = np.unique(np.concatenate(([0.0], np.geomspace(1.0e3, max(1.0e10, z.max() * 1.05), 5000), z)))
    z_eval.sort()
    z_eval_run = np.unique(np.concatenate(([0.0], np.geomspace(1.0e3, max(1.0e10, z_run.max() * 1.05), 5000), z_run)))
    z_eval_run.sort()

    h_std = standard_radiation_hubble(temperatures)
    e_low = e_constant_alpha(model_low, z_eval)
    e_half = e_constant_alpha(model_half, z_eval)
    e_run = e_variable_alpha(model_low, z_eval_run, zc=ZC, delta=DELTA)

    h_low = H0_SI * np.interp(z, z_eval, e_low)
    h_half = H0_SI * np.interp(z, z_eval, e_half)
    h_run = H0_SI * np.interp(z_run, z_eval_run, e_run)
    h_run = smooth_positive_series(h_run, window=SMOOTH_WINDOW)

    ratio_low = h_low / h_std
    ratio_half = h_half / h_std
    h_std_run = standard_radiation_hubble(temperatures_run)
    ratio_run = h_run / h_std_run

    scale_center = neff_to_scale(DELTA_NEFF_CENTER)
    scale_lo_95 = neff_to_scale(DELTA_NEFF_CENTER - DELTA_NEFF_95)
    scale_hi_95 = neff_to_scale(DELTA_NEFF_CENTER + DELTA_NEFF_95)
    h_bbn_center = scale_center * h_std
    h_bbn_lo_95 = scale_lo_95 * h_std
    h_bbn_hi_95 = scale_hi_95 * h_std
    h_bbn_center_run = scale_center * h_std_run
    resid_run_plot = h_run / h_bbn_center_run - 1.0
    resid_std = h_std / h_bbn_center - 1.0
    resid_run = h_run / h_bbn_center - 1.0
    resid_lo_95 = h_bbn_lo_95 / h_bbn_center - 1.0
    resid_hi_95 = h_bbn_hi_95 / h_bbn_center - 1.0

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.4, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 0.95], "hspace": 0.10},
    )

    ax1.fill_between(
        temperatures,
        h_bbn_lo_95,
        h_bbn_hi_95,
        color="#d7c4a3",
        alpha=0.30,
        linewidth=0.0,
        label=(
            r"Observed BBN-inferred band "
            r"(Sch\"oneberg 2024; $\Delta N_{\mathrm{eff}}=-0.10\pm0.21$, 95\%)"
        ),
        zorder=1,
    )
    ax1.plot(
        temperatures,
        h_bbn_center,
        color="#b08a53",
        lw=1.8,
        ls="--",
        alpha=0.9,
        label=r"Observed BBN-inferred central expansion (Sch\"oneberg 2024)",
        zorder=5,
    )
    ax1.plot(
        temperatures,
        h_std,
        color="#2d2d2d",
        lw=1.8,
        ls=(0, (5, 2)),
        label=r"Standard radiation era ($g_*=10.75$)",
        zorder=6,
    )
    ax1.plot(
        temperatures_run,
        h_run,
        color="#1f78b4",
        lw=2.4,
        label=rf"Projected hippopede with running $\alpha(z)$ ($z_c={ZC:.3g}$)",
        zorder=4,
    )
    ax1.plot(
        temperatures,
        h_low,
        color="#6a3d9a",
        lw=1.7,
        ls="-.",
        alpha=0.85,
        label=rf"Projected branch with constant $\alpha={ALPHA_LOW:.3f}$",
        zorder=3.5,
    )
    ax1.plot(
        temperatures,
        h_half,
        color="#e31a1c",
        lw=1.7,
        ls=":",
        alpha=0.9,
        label=rf"Projected branch with constant $\alpha={ALPHA_HIGH:.1f}$",
        zorder=3.6,
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
        temperatures_run,
        resid_run_plot,
        color="#1f78b4",
        lw=2.4,
        zorder=6,
        label=r"Hyperconical$_{\rm CMB}-$observations",
    )

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.grid(True, which="major", color="#ebebeb", linewidth=0.6)
        ax.grid(True, which="minor", color="#f4f4f4", linewidth=0.45)
        ax.minorticks_on()

    ax1.set_yscale("log")
    ax1.set_ylabel(r"Expansion rate $H(T)\ [{\rm s}^{-1}]$")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.set_ylim(3.0e-4, 1.0e2)

    ax2.set_ylabel(r"Residual $(H-H_{\rm obs})/H_{\rm obs}$")
    ax2.set_xlabel(r"Perceived temperature $T\ [{\rm MeV}]$")
    ax2.set_ylim(-0.05, 0.05)
    ax2.set_xlim(1.0e-2, 1.0e1)

    sample_temperatures = np.array([0.07, 0.10, 0.20, 0.50, 1.00])
    sample_z = z_of_temperature_mev(sample_temperatures)
    sample_alpha = alpha_logistic(sample_z)
    sample_ratio_run = np.interp(sample_temperatures, temperatures_run, ratio_run)
    sample_h_bbn = np.interp(sample_temperatures, temperatures, h_bbn_center)
    sample_h_bbn_lo = np.interp(sample_temperatures, temperatures, h_bbn_lo_95)
    sample_h_bbn_hi = np.interp(sample_temperatures, temperatures, h_bbn_hi_95)
    sample_resid_run = np.interp(sample_temperatures, temperatures_run, resid_run_plot)
    sample_resid_std = np.interp(sample_temperatures, temperatures, resid_std)
    obs_err = np.vstack((sample_h_bbn - sample_h_bbn_lo, sample_h_bbn_hi - sample_h_bbn))
    ax1.errorbar(
        sample_temperatures,
        sample_h_bbn,
        yerr=obs_err,
        fmt="o",
        ms=4.2,
        color="#b08a53",
        mfc="white",
        mec="#8b6a3d",
        mew=0.9,
        elinewidth=0.9,
        capsize=2.2,
        zorder=7,
        label=r"BBN-inferred anchor points (Sch\"oneberg 2024)",
    )
    ax2.scatter(
        sample_temperatures,
        np.zeros_like(sample_temperatures),
        s=18,
        facecolor="#b08a53",
        edgecolor="white",
        linewidth=0.6,
        zorder=7,
    )

    summary = {
        "alpha_low": ALPHA_LOW,
        "alpha_high": ALPHA_HIGH,
        "z_c": ZC,
        "delta": DELTA,
        "delta_neff_center": DELTA_NEFF_CENTER,
        "delta_neff_sigma": DELTA_NEFF_SIGMA,
        "delta_neff_95": DELTA_NEFF_95,
        "temperature_mev_samples": sample_temperatures.tolist(),
        "alpha_at_temperature_samples": sample_alpha.tolist(),
        "running_ratio_samples": sample_ratio_run.tolist(),
        "running_residual_samples": sample_resid_run.tolist(),
        "standard_residual_samples": sample_resid_std.tolist(),
        "constant_alpha_0.283_ratio_samples": np.interp(sample_temperatures, temperatures, ratio_low).tolist(),
        "constant_alpha_0.5_ratio_samples": np.interp(sample_temperatures, temperatures, ratio_half).tolist(),
        "bbn_band_ratio_95": [float(scale_lo_95), float(scale_hi_95)],
    }

    png_path = FIGURES / "hippopede_projected_thermal_history_bbn.png"
    pdf_path = FIGURES / "hippopede_projected_thermal_history_bbn.pdf"
    json_path = FIGURES / "hippopede_projected_thermal_history_bbn.json"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved PNG:  {png_path}")
    print(f"Saved PDF:  {pdf_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
