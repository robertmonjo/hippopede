"""Generate hippopede_bao_fit_data.png.

Plots projected hyperconical model predictions vs DESI DR1 compressed BAO data
at best-fit alpha_eff = 0.375, beta = 28.85 (analytically marginalized nuisance).
Also shows flat LCDM best-fit (Omega_m = 0.289) for comparison.
Data: data/Ardra/desi_dr1_bao_galqso_lya_mean.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

for extra in (str(SCRIPT_ROOT), str(SCRIPT_ROOT / "obsolete_bbn"), str(ROOT / "vendor_gapp"), str(ROOT / "vendor_gapp" / "covfunctions")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from analyze_hippopede_dipole_bbn import ExtendedProjectedHyperconical  # noqa: E402

ALPHA_EFF = 0.375
BETA = 28.85
OMEGA_M_LCDM = 0.289

DATA_FILE = ROOT / "data" / "Ardra" / "desi_dr1_bao_galqso_lya_mean.csv"
OUT = FIGURES / "hippopede_bao_fit_data.png"


def e_lcdm_flat(z: np.ndarray, omega_m: float = OMEGA_M_LCDM) -> np.ndarray:
    return np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m))


def dm_over_rs(z_vals: np.ndarray, model: ExtendedProjectedHyperconical, beta: float) -> np.ndarray:
    z_fine = np.linspace(0, z_vals.max() * 1.01, 5000)
    E_fine, _ = model.e_and_q(z_fine)
    integrand = 1.0 / E_fine
    comoving = np.array([_trapz(integrand[z_fine <= z], z_fine[z_fine <= z]) for z in z_vals])
    return beta * comoving


def dh_over_rs(z_vals: np.ndarray, model: ExtendedProjectedHyperconical, beta: float) -> np.ndarray:
    E_vals, _ = model.e_and_q(z_vals)
    return beta / E_vals


def dv_over_rs(z_val: float, model: ExtendedProjectedHyperconical, beta: float) -> float:
    dm = dm_over_rs(np.array([z_val]), model, beta)[0]
    dh = dh_over_rs(np.array([z_val]), model, beta)[0]
    return (z_val * dm ** 2 * dh) ** (1.0 / 3.0)


def dm_lcdm(z_vals: np.ndarray, omega_m: float, beta: float) -> np.ndarray:
    z_fine = np.linspace(0, z_vals.max() * 1.01, 5000)
    E_fine = e_lcdm_flat(z_fine, omega_m)
    integrand = 1.0 / E_fine
    comoving = np.array([_trapz(integrand[z_fine <= z], z_fine[z_fine <= z]) for z in z_vals])
    return beta * comoving


def dh_lcdm(z_vals: np.ndarray, omega_m: float, beta: float) -> np.ndarray:
    return beta / e_lcdm_flat(z_vals, omega_m)


def dv_lcdm(z_val: float, omega_m: float, beta: float) -> float:
    z_fine = np.linspace(0, z_val * 1.01, 2000)
    E_fine = e_lcdm_flat(z_fine, omega_m)
    comoving = _trapz(1.0 / E_fine, z_fine)
    dh = beta / e_lcdm_flat(np.array([z_val]), omega_m)[0]
    return (z_val * (beta * comoving) ** 2 * dh) ** (1.0 / 3.0)


# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
dm_data = df[df["observable"] == "DM_over_rs"].copy()
dh_data = df[df["observable"] == "DH_over_rs"].copy()
dv_data = df[df["observable"] == "DV_over_rs"].copy()

# ── model ──────────────────────────────────────────────────────────────────
model_hyp = ExtendedProjectedHyperconical(alpha=ALPHA_EFF)

# Precompute E(z) on dense grid for interpolation (e_and_q needs many points)
z_dense = np.linspace(0.0, 2.7, 2000)
E_dense, _ = model_hyp.e_and_q(z_dense)


def _comoving(z_max: float) -> float:
    mask = z_dense <= z_max
    if mask.sum() < 2:
        return 0.0
    return float(_trapz((1.0 / E_dense)[mask], z_dense[mask]))


def _E_at(z: float) -> float:
    return float(np.interp(z, z_dense, E_dense))


z_curve = np.linspace(0.05, 2.6, 400)
dm_curve = BETA * np.array([_comoving(z) for z in z_curve])
dh_curve = BETA / np.interp(z_curve, z_dense, E_dense)
dv_curve = np.array([(z * (BETA * _comoving(z)) ** 2 * (BETA / _E_at(z))) ** (1.0 / 3.0) for z in z_curve])

dm_lc = dm_lcdm(z_curve, OMEGA_M_LCDM, BETA)
dh_lc = dh_lcdm(z_curve, OMEGA_M_LCDM, BETA)
dv_lc = np.array([dv_lcdm(z, OMEGA_M_LCDM, BETA) for z in z_curve])

# ── plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
plt.rcParams.update({"font.size": 10})

panel_cfg = [
    ("DM_over_rs", dm_curve, dm_lc, dm_data, r"$D_M/r_s$"),
    ("DH_over_rs", dh_curve, dh_lc, dh_data, r"$D_H/r_s$"),
    ("DV_over_rs", dv_curve, dv_lc, dv_data, r"$D_V/r_s$"),
]

for ax, (obs_key, hyp_c, lc_c, data_sub, ylabel) in zip(axes, panel_cfg):
    ax.plot(z_curve, hyp_c, color="#4477AA", lw=1.8,
            label=rf"Hyp. proj. ($\alpha={ALPHA_EFF}$)")
    ax.plot(z_curve, lc_c, color="#EE6677", lw=1.4, ls="--",
            label=rf"$\Lambda$CDM ($\Omega_m={OMEGA_M_LCDM}$)")
    ax.errorbar(data_sub["z"], data_sub["value"],
                fmt="o", color="#222222", ms=5, elinewidth=1.2,
                label="DESI DR1")
    ax.set_xlabel(r"$z_\mathrm{eff}$")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.25, lw=0.5)

fig.suptitle(r"Projected hyperconical model vs DESI DR1 BAO ($\beta=%.2f$)" % BETA, y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
