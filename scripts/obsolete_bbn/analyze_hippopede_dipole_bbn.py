from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

for extra in (str(ROOT), str(ROOT / "vendor_gapp"), str(ROOT / "vendor_gapp" / "covfunctions")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from plot_hippopede_qz_double_panel_with_gapp import (  # noqa: E402
    MonjoProjectedHyperconical,
    build_centered_history,
)


T0_K = 2.7255
KB_EV_PER_K = 8.617333262e-5
T0_EV = T0_K * KB_EV_PER_K
H0_KM_S_MPC = 70.0
MPC_KM = 3.0856775814913673e19
H0_SI = H0_KM_S_MPC / MPC_KM
MPL_GEV = 1.220890e19
HBAR_GEV_S = 6.582119569e-25

# ---------------------------------------------------------------------------
# Fermi-Dirac energy integral for massive species
# h(x) = (120/7π⁴) ∫₀^∞ u² √(u²+x²) / (e^{√(u²+x²)}+1) du, x = m/T
# h(0) = 1 (massless limit), h(∞) = 0.
# Precomputed once at import on a 220-point x-grid; accessed via np.interp.
# ---------------------------------------------------------------------------

def _build_fd_table():
    x_tab = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 60),
        np.linspace(1.0, 5.0, 100),
        np.linspace(5.0, 20.0, 60),
    ]))
    u = np.linspace(1e-5, 60.0, 5000)
    prefac = 120.0 / (7.0 * np.pi ** 4)
    h_tab = np.empty_like(x_tab)
    for i, xi in enumerate(x_tab):
        eps = np.sqrt(u ** 2 + xi ** 2)
        integrand = u ** 2 * eps / (np.exp(np.minimum(eps, 500.0)) + 1.0)
        h_tab[i] = prefac * (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(integrand, u)
    return x_tab, h_tab


_FD_X_TAB, _FD_H_TAB = _build_fd_table()


def _fd_energy_ratio(x):
    """FD energy density ratio for a massive fermion relative to massless limit."""
    return np.interp(np.asarray(x, dtype=float), _FD_X_TAB, _FD_H_TAB,
                     left=1.0, right=0.0)


def effective_g_star(T_mev):
    """
    Effective relativistic DOF g_*(T) for T in [0.01, 100] MeV.

    g_*(T) = 2 [photons]
           + (7/8)*4 * h(m_e/T)            [e+e-  with exact FD integral]
           + (7/8)*6 * (T_nu/T_gamma)^4    [3 nu + 3 nubar, massless]

    Neutrino temperature from entropy conservation during e+e- annihilation:
        (T_nu/T_gamma)^3 = (4 + 7*h(m_e/T)) / 11
    Limits: T_nu/T_gamma -> 1 for T >> m_e;  -> (4/11)^(1/3) for T << m_e.
    g_*(T>>m_e) = 10.75,  g_*(T<<m_e) = 3.36.
    """
    T_mev = np.asarray(T_mev, dtype=float)
    x_e = 0.511 / T_mev
    h_e = _fd_energy_ratio(x_e)
    nu_ratio_4 = ((4.0 + 7.0 * h_e) / 11.0) ** (4.0 / 3.0)
    return 2.0 + (7.0 / 8.0) * 4.0 * h_e + (7.0 / 8.0) * 6.0 * nu_ratio_4


class ExtendedProjectedHyperconical(MonjoProjectedHyperconical):
    """Same Monjo-type map but with a denser boundary lookup for high-z tests."""

    def _prepare_lookup(self):
        k = self.k
        self.mx = np.sqrt((1.0 - (1.0 - k / 2.0) ** 2) / k)
        # The original figure script is tuned for z \lesssim 3e5. For BBN diagnostics
        # we need a wider asymptotic window before the floating-point boundary is hit.
        seq0 = np.linspace(0.0, 35.0, 250000)
        x_linear = np.linspace(0.0, self.mx, 10000)
        x_cluster = self.mx * (1.0 - 10.0 ** (-seq0))
        x_pos = np.unique(np.concatenate([x_linear, x_cluster]))
        x_pos = x_pos[(x_pos >= 0.0) & (x_pos < self.mx)]
        fx_pos = self._f(x_pos)
        dx = np.diff(x_pos)
        integ = np.zeros_like(x_pos)
        integ[1:] = np.cumsum(0.5 * (fx_pos[1:] + fx_pos[:-1]) * dx)
        I_pos = -integ
        self.x_grid = np.concatenate([-x_pos[:0:-1], x_pos])
        self.I_grid = np.concatenate([-I_pos[:0:-1], I_pos])
        self.f_grid = np.concatenate([fx_pos[:0:-1], fx_pos])
        ux = np.sqrt(1.0 / k - self.mx**2)
        self.y0 = np.arctan2(self.mx, ux)

    def projected_hubble_unnormalized(self, z):
        z = np.asarray(z, dtype=float)
        lz = np.log1p(z)
        x = self.x_from_lz(lz)
        dr_dz = self.dinvll_dx(x) * self.dxdLZ(x) / (1.0 + z)
        return 1.0 / dr_dz


def standard_radiation_hubble(T_mev, g_star=None):
    """
    H(T) for the radiation-dominated era in SI units [s^-1].

    Uses effective_g_star(T_mev) by default (variable g_* accounting for
    e+e- annihilation and neutrino decoupling).  Pass g_star=10.75 to
    reproduce the old constant-g_* behaviour.
    """
    T_mev = np.asarray(T_mev, dtype=float)
    T_gev = T_mev / 1000.0
    g = effective_g_star(T_mev) if g_star is None else np.full_like(T_mev, float(g_star))
    h_gev = 1.66 * np.sqrt(g) * (T_gev ** 2) / MPL_GEV
    return h_gev / HBAR_GEV_S


def fit_dipole_like_profile(x, y):
    ones = np.ones_like(x)
    A1 = np.column_stack([ones, x])
    coeff1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    y1 = A1 @ coeff1
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res1 = np.sum((y - y1) ** 2)
    r2_1 = 1.0 - ss_res1 / ss_tot if ss_tot > 0 else np.nan

    A2 = np.column_stack([ones, x, x**2])
    coeff2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    y2 = A2 @ coeff2
    ss_res2 = np.sum((y - y2) ** 2)
    r2_2 = 1.0 - ss_res2 / ss_tot if ss_tot > 0 else np.nan

    return {
        "dipole_constant": float(coeff1[0]),
        "dipole_linear": float(coeff1[1]),
        "dipole_ratio_c1_over_c0": float(coeff1[1] / coeff1[0]),
        "dipole_r2": float(r2_1),
        "dipole_plus_quadrupole_r2": float(r2_2),
    }


def analyze_dipole(alpha=0.283, t0=3.0):
    chis_deg = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], dtype=float)
    chis = np.deg2rad(chis_deg)
    cos_chi = np.cos(chis)
    z_eval = np.array([0.1, 0.3, 0.5, 1.0, 2.0])
    proj = MonjoProjectedHyperconical(alpha=alpha)
    z_obs_grid = np.linspace(-0.49, 2.0, 500)
    e_proj_ref, _ = proj.e_and_q(z_obs_grid)
    z_map_grid = e_proj_ref - 1.0

    histories = {}
    for chi_deg, chi in zip(chis_deg, chis):
        z_hist, q_hist, e_hist = build_centered_history(chi, t0)
        histories[chi_deg] = {"z": z_hist, "q": q_hist, "e": e_hist}

    report = {"angles_deg": chis_deg.tolist(), "z_eval": z_eval.tolist(), "unprojected": [], "projected": []}
    for label in ("unprojected", "projected"):
        for z_val in z_eval:
            values = []
            for chi_deg in chis_deg:
                z_hist = histories[chi_deg]["z"]
                e_hist = histories[chi_deg]["e"]
                if label == "unprojected":
                    values.append(np.interp(z_val, z_hist, e_hist, left=np.nan, right=np.nan))
                else:
                    z_star = np.interp(z_val, z_obs_grid, z_map_grid)
                    values.append(np.interp(z_star, z_hist, e_hist, left=np.nan, right=np.nan))

            values = np.asarray(values, dtype=float)
            valid = np.isfinite(values)
            y = values[valid]
            x = cos_chi[valid]
            fit = fit_dipole_like_profile(x, y)
            fit.update(
                {
                    "z": float(z_val),
                    "fractional_span": float((np.nanmax(y) - np.nanmin(y)) / np.nanmean(y)),
                    "min_value": float(np.nanmin(y)),
                    "max_value": float(np.nanmax(y)),
                }
            )
            report[label].append(fit)

    return report


def analyze_bbn(alpha=0.5, z_fit_min=1.0e3, z_fit_max=1.0e6):
    model = ExtendedProjectedHyperconical(alpha=alpha)
    z_grid = np.logspace(np.log10(z_fit_min), np.log10(z_fit_max), 200)
    h_raw = model.projected_hubble_unnormalized(z_grid)
    exponent = 1.0 + 2.0 * alpha
    coeff_grid = h_raw / ((1.0 + z_grid) ** exponent)
    n_eff = np.gradient(np.log(h_raw), np.log1p(z_grid))
    coeff = float(np.median(coeff_grid))
    n_median = float(np.median(n_eff))

    temperatures = np.array([0.07, 0.10, 1.0], dtype=float)
    h_projected = H0_SI * coeff * ((temperatures * 1.0e6) / T0_EV) ** exponent
    h_standard = standard_radiation_hubble(temperatures)
    ratio = h_projected / h_standard

    return {
        "alpha": alpha,
        "z_fit_window": [float(z_fit_min), float(z_fit_max)],
        "n_eff_median": n_median,
        "prefactor_median": coeff,
        "t0_kelvin": T0_K,
        "h0_km_s_mpc": H0_KM_S_MPC,
        "bbn_temperatures_mev": temperatures.tolist(),
        "projected_hubble_s^-1": h_projected.tolist(),
        "standard_radiation_hubble_s^-1": h_standard.tolist(),
        "projected_to_standard_ratio": ratio.tolist(),
        "ratio_median": float(np.median(ratio)),
    }


def main():
    report = {
        "dipole_like_test": analyze_dipole(alpha=0.283, t0=3.0),
        "bbn_alpha_half_test": analyze_bbn(alpha=0.5),
    }

    out_path = FIGURES / "hippopede_dipole_bbn_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Dipole-like test (projected, alpha=0.283):")
    for row in report["dipole_like_test"]["projected"]:
        print(
            f"  z={row['z']:.1f}: span={row['fractional_span']:.4f}, "
            f"c1/c0={row['dipole_ratio_c1_over_c0']:.4f}, "
            f"R2_dip={row['dipole_r2']:.4f}, R2_dip+quad={row['dipole_plus_quadrupole_r2']:.4f}"
        )

    bbn = report["bbn_alpha_half_test"]
    print("\nBBN-like thermal test (alpha=1/2):")
    print(f"  median n_eff = {bbn['n_eff_median']:.6f}")
    print(f"  median prefactor = {bbn['prefactor_median']:.6f}")
    print(f"  median H_proj/H_rad = {bbn['ratio_median']:.6f}")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
