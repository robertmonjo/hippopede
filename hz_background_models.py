from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

FACTOR_SIZE = 1.25
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "hz_background" / "hz_curated_chronometers.csv"
DEFAULT_BAO_DATA = PROJECT_ROOT / "data" / "hz_background" / "hz_curated_bao.csv"
if not DEFAULT_DATA.exists():
    DEFAULT_DATA = Path(r"C:\Users\rober\OneDrive\Documents\Codex_Portatil\cosmic_acceleration\data\hz_background\hz_curated_chronometers.csv")
if not DEFAULT_BAO_DATA.exists():
    DEFAULT_BAO_DATA = Path(r"C:\Users\rober\OneDrive\Documents\Codex_Portatil\cosmic_acceleration\data\hz_background\hz_curated_bao.csv")
DEFAULT_PNG = PROJECT_ROOT / "figures" / "fig_hz_background_models.png"
DEFAULT_SVG = PROJECT_ROOT / "figures" / "fig_hz_background_models.svg"
DEFAULT_PDF = PROJECT_ROOT / "figures" / "fig_hz_background_models.pdf"
T0_HYPERCONICAL = 3.0
ALPHA_PROJECTED = 0.283
K_PROJECTED = 1.0
H0_FID = 67.4
OMEGA_M_FID = 0.315
GAMMA0_THEORY = 2.0 * np.pi / 3.0
ALPHA_GAMMA_PROJECTION = 0.5
SOURCE_STYLES = {
    "Riess2024": {"color": "#6f42c1", "marker": "o", "label": "Riess+24"},
    "Zhang2014": {"color": "#1f77b4", "marker": "s", "label": "Zhang+14"},
    "Moresco2012": {"color": "#d62728", "marker": "D", "label": "Moresco+12"},
    "Moresco2015": {"color": "#8c564b", "marker": "P", "label": "Moresco+15"},
    "Moresco2016": {"color": "#ff7f0e", "marker": "v", "label": "Moresco+16"},
    "Ratsimbazafy2017": {"color": "#9467bd", "marker": "^", "label": "Ratsimbazafy+17"},
    "Borghi2022": {"color": "#17becf", "marker": "h", "label": "Borghi+22"},
    "Jiao2023": {"color": "#2ca02c", "marker": "X", "label": "Jiao+23"},
    "Tomasetti2023": {"color": "#e377c2", "marker": "*", "label": "Tomasetti+23"},
    "Adame2025a": {"color": "#6d4c41", "marker": "8", "label": "DESI+24 G/Q"},
    "Adame2025b": {"color": "#795548", "marker": "p", "label": "DESI+24 LyA"},
}

BAO_STYLE_GROUPS = {
    "legacy_bao": {
        "refs": {"Gaztanaga2009", "Gaztañaga2009", "GaztaÃ±aga2009", "Xu2013", "Blake2012"},
        "color": "#3a3a3a",
        "marker": "o",
        "label": "Legacy BAO (gal)",
    },
    "desi_gq_bao": {
        "refs": {"Adame2025a"},
        "color": "#6d4c41",
        "marker": "s",
        "label": "DESI DR1 BAO (gal/QSO)",
    },
    "desi_lya_bao": {
        "refs": {"Adame2025b"},
        "color": "#795548",
        "marker": "D",
        "label": "DESI DR1 BAO (Ly$\\alpha$)",
    },
}


@dataclass
class HzPoint:
    z: float
    h_km_s_mpc: float
    sigma_plus: float
    sigma_minus: float
    label: str
    reference_key: str
    source_kind: str
    note: str


def load_points(path: Path) -> list[HzPoint]:
    points: list[HzPoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                HzPoint(
                    z=float(row["z"]),
                    h_km_s_mpc=float(row["h_km_s_mpc"]),
                    sigma_plus=float(row["sigma_plus"]),
                    sigma_minus=float(row["sigma_minus"]),
                    label=row["label"],
                    reference_key=row["reference_key"],
                    source_kind=row["source_kind"],
                    note=row["note"],
                )
            )
    return points


def combined_arrays(points: list[HzPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.array([p.z for p in points])
    h = np.array([p.h_km_s_mpc for p in points])
    sigma = np.array([0.5 * (p.sigma_minus + p.sigma_plus) for p in points])
    return z, h, sigma


def hz_lcdm(z: np.ndarray, h0: float = H0_FID, omega_m: float = OMEGA_M_FID) -> np.ndarray:
    omega_lambda = 1.0 - omega_m
    return h0 * np.sqrt(omega_m * (1.0 + z) ** 3 + omega_lambda)


def hz_linear(z: np.ndarray, h0: float = H0_FID) -> np.ndarray:
    return h0 * (1.0 + z)


def hz_desitter(z: np.ndarray, h0: float = H0_FID) -> np.ndarray:
    return np.full_like(z, h0, dtype=float)


def hz_power_law(z: np.ndarray, n: float, h0: float = H0_FID) -> np.ndarray:
    return h0 * (1.0 + z) ** (1.0 / n)


def hz_wcdm(z: np.ndarray, w: float, h0: float = H0_FID, omega_m: float = OMEGA_M_FID) -> np.ndarray:
    omega_de = 1.0 - omega_m
    return h0 * np.sqrt(omega_m * (1.0 + z) ** 3 + omega_de * (1.0 + z) ** (3.0 * (1.0 + w)))


def hz_hyperconical_projected_gamma(
    z: np.ndarray,
    gamma0: float = GAMMA0_THEORY,
    h0: float = H0_FID,
    alpha: float = ALPHA_GAMMA_PROJECTION,
) -> np.ndarray:
    return h0 * (1.0 + z) / (1.0 + 2.0 * alpha * np.log1p(z) / gamma0)


def rho_hippopede(t: np.ndarray, chi: float) -> np.ndarray:
    s2 = np.sin(chi) ** 2
    c2 = np.cos(chi) ** 2
    return np.sqrt(s2 + 4.0 * t**2 * c2)


def a_centered(t: np.ndarray, chi: float) -> np.ndarray:
    rho = rho_hippopede(t, chi)
    return np.sqrt(rho**2 + t**2 - 2.0 * t * rho * np.cos(chi))


def build_centered_history(chi: float, t0: float, n: int = 8000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_min = 1.0e-4
    t_max = max(40.0, 25.0 * t0)
    t = np.linspace(t_min, t_max, n)

    a = a_centered(t, chi)
    a_dot = np.gradient(a, t)
    a_ddot = np.gradient(a_dot, t)

    with np.errstate(divide="ignore", invalid="ignore"):
        h = a_dot / a
        q = -a * a_ddot / (a_dot**2)

    i0 = np.searchsorted(t, t0)
    i0 = min(max(i0, 1), len(t) - 2)
    bounce_idx = np.argmin(a[: i0 + 1])
    t = t[bounce_idx:]
    a = a[bounce_idx:]
    h = h[bounce_idx:]
    q = q[bounce_idx:]

    a0 = a_centered(np.array([t0]), chi)[0]
    z = a0 / a - 1.0
    h0 = np.interp(t0, t, h)
    e = h / h0

    order = np.argsort(z)
    z_sorted = z[order]
    q_sorted = q[order]
    e_sorted = e[order]
    unique_z, unique_idx = np.unique(z_sorted, return_index=True)
    return unique_z, q_sorted[unique_idx], e_sorted[unique_idx]


def interp_curve(z_grid: np.ndarray, z_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    valid = np.isfinite(z_data) & np.isfinite(y_data)
    z_data = z_data[valid]
    y_data = y_data[valid]
    if len(z_data) < 2:
        return np.full_like(z_grid, np.nan, dtype=float)
    y = np.interp(z_grid, z_data, y_data, left=np.nan, right=np.nan)
    y[z_grid < z_data.min()] = np.nan
    y[z_grid > z_data.max()] = np.nan
    return y


def gamma_sys_from_gamma0(gamma0: float) -> float:
    if gamma0 <= 1.0:
        raise ValueError("gamma0 must be > 1 for the projected hyperconical mapping.")
    lo = 1.0e-9
    hi = 0.5 * np.pi - 1.0e-9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        value = mid / np.cos(mid)
        if value < gamma0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class MonjoProjectedHyperconical:
    def __init__(
        self,
        alpha: float = ALPHA_PROJECTED,
        k: float = K_PROJECTED,
        gamma0: float | None = None,
        y0_override: float | None = None,
    ) -> None:
        self.alpha = alpha
        self.k = k
        self.gamma0 = gamma0
        self.y0_override = y0_override
        self._prepare_lookup()

    def _ttp(self, r: np.ndarray) -> np.ndarray:
        t = 1.0
        T = 1.0
        k = self.k
        return t * np.sqrt(
            (
                T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2)
                - 2 * k * r**2
                + 2 * T**2
            )
            / (np.sqrt((-k * r**2 + T**2) / T**2) * T**2 * k)
        )

    def _grrp(self, r: np.ndarray) -> np.ndarray:
        t = 1.0
        T = 1.0
        k = self.k
        return -t**2 * (
            ((T**2 * (k - 2) + k * r**2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2)
            / ((T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) * (-k * r**2 + T**2))
        )

    def _f(self, x: np.ndarray) -> np.ndarray:
        return -np.sqrt(-self._grrp(x)) / self._ttp(x)

    def _prepare_lookup(self) -> None:
        k = self.k
        self.mx = np.sqrt((1.0 - (1.0 - k / 2.0) ** 2) / k)
        seq0 = np.linspace(0.0, 10.8, 18000)
        x_linear = np.linspace(0.0, self.mx, 4000)
        x_cluster = self.mx * (1.0 - 10.0 ** (-seq0))
        x_pos = np.unique(np.concatenate([x_linear, x_cluster]))
        x_pos = x_pos[(x_pos >= 0.0) & (x_pos < self.mx)]

        fx_pos = self._f(x_pos)
        dx = np.diff(x_pos)
        integ = np.zeros_like(x_pos)
        integ[1:] = np.cumsum(0.5 * (fx_pos[1:] + fx_pos[:-1]) * dx)
        i_pos = -integ

        self.x_grid = np.concatenate([-x_pos[:0:-1], x_pos])
        self.i_grid = np.concatenate([-i_pos[:0:-1], i_pos])
        self.f_grid = np.concatenate([fx_pos[:0:-1], fx_pos])

        ux = np.sqrt(1.0 / k - self.mx**2)
        self.y0 = np.arctan2(self.mx, ux) if self.y0_override is None else self.y0_override

    def x_from_lz(self, lz: np.ndarray) -> np.ndarray:
        return np.interp(lz, self.i_grid, self.x_grid, left=self.x_grid[0], right=self.x_grid[-1])

    def dxdLZ(self, x: np.ndarray) -> np.ndarray:
        f = np.interp(x, self.x_grid, self.f_grid)
        return -1.0 / f

    def inv_ll(self, x: np.ndarray) -> np.ndarray:
        k = self.k
        u = np.sqrt(np.maximum(1.0 / k - x**2, 0.0))
        y = np.arctan2(x, u)
        if self.gamma0 is None:
            g = np.maximum(1.0 - y / self.y0, 1e-12)
        else:
            g = np.maximum(1.0 - y / self.gamma0, 1e-12)
        return 2.0 * np.arctan((y / 2.0) / g**self.alpha)

    def dinvll_dx(self, x: np.ndarray) -> np.ndarray:
        k = self.k
        u = np.sqrt(np.maximum(1.0 / k - x**2, 1e-14))
        y = np.arctan2(x, u)
        dy_dx = 1.0 / u

        s = y / 2.0
        if self.gamma0 is None:
            g = np.maximum(1.0 - y / self.y0, 1e-12)
            inv_scale = 1.0 / self.y0
        else:
            g = np.maximum(1.0 - y / self.gamma0, 1e-12)
            inv_scale = 1.0 / self.gamma0
        t = s / g**self.alpha
        dt_dx = dy_dx * (0.5 / g**self.alpha + s * self.alpha * inv_scale * g ** (-self.alpha - 1.0))
        return 2.0 * dt_dx / (1.0 + t**2)

    def e_and_q(self, z_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_in = np.asarray(z_grid, dtype=float)
        if z_in.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        order = np.argsort(z_in)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)
        z = z_in[order]
        lz = np.log1p(z)
        x = self.x_from_lz(lz)

        with np.errstate(divide="ignore", invalid="ignore"):
            dr_dz = self.dinvll_dx(x) * self.dxdLZ(x) / (1.0 + z)
            h = 1.0 / dr_dz

        kernel = np.ones(9) / 9.0
        valid = np.isfinite(h)
        if valid.sum() > len(kernel):
            hv = h[valid]
            pad = len(kernel) // 2
            hv_pad = np.pad(hv, pad_width=pad, mode="edge")
            hs = np.convolve(hv_pad, kernel, mode="valid")
            h[valid] = hs

        h0 = np.interp(0.0, z[np.isfinite(h)], h[np.isfinite(h)])
        e = h / h0
        dE_dz = np.gradient(e, z)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = -1.0 + (1.0 + z) * dE_dz / e

        e[~np.isfinite(e)] = np.nan
        q[~np.isfinite(q)] = np.nan
        return e[inv_order], q[inv_order]


class MonjoSimpleStereographic:
    """
    Direct stereographic family based on the explicit small-region map

      r_hat = r' / (1 - gamma(r') / gamma0)^alpha

    with gamma(r') = asin(r'/t0).
    This is the projection-map family written explicitly in the manuscript,
    without the outer 2*atan(...) remapping used by MonjoProjectedHyperconical.
    """

    def __init__(self, alpha: float = ALPHA_GAMMA_PROJECTION, gamma0: float = GAMMA0_THEORY, k: float = K_PROJECTED) -> None:
        self.alpha = alpha
        self.gamma0 = gamma0
        self.k = k
        self._prepare_lookup()

    def _ttp(self, r: np.ndarray) -> np.ndarray:
        t = 1.0
        T = 1.0
        k = self.k
        return t * np.sqrt(
            (
                T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2)
                - 2 * k * r**2
                + 2 * T**2
            )
            / (np.sqrt((-k * r**2 + T**2) / T**2) * T**2 * k)
        )

    def _grrp(self, r: np.ndarray) -> np.ndarray:
        t = 1.0
        T = 1.0
        k = self.k
        return -t**2 * (
            ((T**2 * (k - 2) + k * r**2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2)
            / ((T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) * (-k * r**2 + T**2))
        )

    def _f(self, x: np.ndarray) -> np.ndarray:
        return -np.sqrt(-self._grrp(x)) / self._ttp(x)

    def _prepare_lookup(self) -> None:
        k = self.k
        self.mx = np.sqrt((1.0 - (1.0 - k / 2.0) ** 2) / k)
        seq0 = np.linspace(0.0, 10.8, 18000)
        x_linear = np.linspace(0.0, self.mx, 4000)
        x_cluster = self.mx * (1.0 - 10.0 ** (-seq0))
        x_pos = np.unique(np.concatenate([x_linear, x_cluster]))
        x_pos = x_pos[(x_pos >= 0.0) & (x_pos < self.mx)]

        fx_pos = self._f(x_pos)
        dx = np.diff(x_pos)
        integ = np.zeros_like(x_pos)
        integ[1:] = np.cumsum(0.5 * (fx_pos[1:] + fx_pos[:-1]) * dx)
        i_pos = -integ

        self.x_grid = np.concatenate([-x_pos[:0:-1], x_pos])
        self.i_grid = np.concatenate([-i_pos[:0:-1], i_pos])
        self.f_grid = np.concatenate([fx_pos[:0:-1], fx_pos])

    def x_from_lz(self, lz: np.ndarray) -> np.ndarray:
        return np.interp(lz, self.i_grid, self.x_grid, left=self.x_grid[0], right=self.x_grid[-1])

    def dxdLZ(self, x: np.ndarray) -> np.ndarray:
        f = np.interp(x, self.x_grid, self.f_grid)
        return -1.0 / f

    def rhat(self, x: np.ndarray) -> np.ndarray:
        gamma = np.arcsin(np.clip(x, -self.mx, self.mx))
        g = np.maximum(1.0 - gamma / self.gamma0, 1e-12)
        return x / g**self.alpha

    def drhat_dx(self, x: np.ndarray) -> np.ndarray:
        gamma = np.arcsin(np.clip(x, -self.mx, self.mx))
        dgamma_dx = 1.0 / np.sqrt(np.maximum(1.0 - x**2, 1e-12))
        g = np.maximum(1.0 - gamma / self.gamma0, 1e-12)
        return g ** (-self.alpha) + x * self.alpha * dgamma_dx / self.gamma0 * g ** (-self.alpha - 1.0)

    def e_and_q(self, z_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_in = np.asarray(z_grid, dtype=float)
        if z_in.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        order = np.argsort(z_in)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)
        z = z_in[order]
        lz = np.log1p(z)
        x = self.x_from_lz(lz)

        with np.errstate(divide="ignore", invalid="ignore"):
            dr_dz = self.drhat_dx(x) * self.dxdLZ(x) / (1.0 + z)
            h = 1.0 / dr_dz

        kernel = np.ones(9) / 9.0
        valid = np.isfinite(h)
        if valid.sum() > len(kernel):
            hv = h[valid]
            pad = len(kernel) // 2
            hv_pad = np.pad(hv, pad_width=pad, mode="edge")
            hs = np.convolve(hv_pad, kernel, mode="valid")
            h[valid] = hs

        h0 = np.interp(0.0, z[np.isfinite(h)], h[np.isfinite(h)])
        e = h / h0
        dE_dz = np.gradient(e, z)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = -1.0 + (1.0 + z) * dE_dz / e

        e[~np.isfinite(e)] = np.nan
        q[~np.isfinite(q)] = np.nan
        return e[inv_order], q[inv_order]


def hz_hyperconical_unprojected(z: np.ndarray, h0: float = H0_FID, t0: float = T0_HYPERCONICAL) -> np.ndarray:
    z_data, _, e_data = build_centered_history(0.0, t0)
    e_interp = interp_curve(z, z_data, e_data)
    return h0 * e_interp


def hz_hyperconical_projected(z: np.ndarray, h0: float = H0_FID) -> np.ndarray:
    monjo = MonjoProjectedHyperconical(alpha=ALPHA_PROJECTED, k=K_PROJECTED)
    e_data, _ = monjo.e_and_q(z)
    return h0 * e_data


def group_points_by_reference(points: list[HzPoint]) -> dict[str, list[HzPoint]]:
    grouped: dict[str, list[HzPoint]] = {}
    for point in points:
        grouped.setdefault(point.reference_key, []).append(point)
    return grouped


def symmetric_sigma(points: list[HzPoint]) -> np.ndarray:
    return np.array([0.5 * (p.sigma_minus + p.sigma_plus) for p in points])


def chi2_summary(h_obs: np.ndarray, h_model: np.ndarray, sigma: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(h_obs) & np.isfinite(h_model) & np.isfinite(sigma) & (sigma > 0.0)
    nu = int(valid.sum())
    chi2 = float(np.sum(((h_obs[valid] - h_model[valid]) / sigma[valid]) ** 2))
    red = chi2 / nu if nu > 0 else np.nan
    return chi2, red, nu


def best_fit_h0_from_shape(e_model: np.ndarray, h_obs: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(e_model) & np.isfinite(h_obs) & np.isfinite(sigma) & (sigma > 0.0)
    e_valid = e_model[valid]
    h_valid = h_obs[valid]
    sigma_valid = sigma[valid]
    h0_fit = float(np.sum(h_valid * e_valid / sigma_valid**2) / np.sum(e_valid**2 / sigma_valid**2))
    chi2_fit = chi2_summary(h_obs, h0_fit * e_model, sigma)[0]
    return h0_fit, float(chi2_fit)


def fit_lcdm_h0(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    omega_m: float = OMEGA_M_FID,
) -> tuple[float, float]:
    e_model = hz_lcdm(z_obs, h0=1.0, omega_m=omega_m)
    return best_fit_h0_from_shape(e_model, h_obs, sigma)


def fit_desitter_h0(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float]:
    e_model = np.ones_like(z_obs, dtype=float)
    return best_fit_h0_from_shape(e_model, h_obs, sigma)


def fit_power_law_index(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float, float]:
    n_grid = np.linspace(0.85, 3.00, 4301)
    best_n = np.nan
    best_h0 = np.nan
    best_chi2 = np.inf
    for n in n_grid:
        e_model = hz_power_law(z_obs, n, h0=1.0)
        h0_fit, chi2_fit = best_fit_h0_from_shape(e_model, h_obs, sigma)
        if chi2_fit < best_chi2:
            best_n = n
            best_h0 = h0_fit
            best_chi2 = chi2_fit
    return float(best_n), float(best_h0), float(best_chi2)


def fit_wcdm_w(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    omega_m: float = OMEGA_M_FID,
) -> tuple[float, float, float]:
    w_grid = np.linspace(-2.00, -0.35, 3301)
    best_w = np.nan
    best_h0 = np.nan
    best_chi2 = np.inf
    for w in w_grid:
        e_model = hz_wcdm(z_obs, w, h0=1.0, omega_m=omega_m)
        h0_fit, chi2_fit = best_fit_h0_from_shape(e_model, h_obs, sigma)
        if chi2_fit < best_chi2:
            best_w = w
            best_h0 = h0_fit
            best_chi2 = chi2_fit
    return float(best_w), float(best_h0), float(best_chi2)

def fit_hyperconical_gamma0(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    h0: float = H0_FID,
    alpha: float = ALPHA_GAMMA_PROJECTION,
) -> tuple[float, float]:
    gamma_grid = np.linspace(0.8, 8.0, 7201)
    chi2_grid = np.array([chi2_summary(h_obs, hz_hyperconical_projected_gamma(z_obs, g, h0=h0, alpha=alpha), sigma)[0] for g in gamma_grid])
    best = int(np.nanargmin(chi2_grid))
    return float(gamma_grid[best]), float(chi2_grid[best])


def fit_projected_hyperconical_like_standard_h0(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    omega_m: float = 0.30,
) -> tuple[float, float]:
    e_model = hz_lcdm(z_obs, h0=1.0, omega_m=omega_m)
    return best_fit_h0_from_shape(e_model, h_obs, sigma)


def fit_linear_h0(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float]:
    e_model = hz_linear(z_obs, h0=1.0)
    return best_fit_h0_from_shape(e_model, h_obs, sigma)


def fit_monjo_projected_h0_for_alpha(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    k: float = K_PROJECTED,
) -> tuple[float, float]:
    monjo = MonjoProjectedHyperconical(alpha=alpha, k=k)
    e_model, _ = monjo.e_and_q(z_obs)
    return best_fit_h0_from_shape(e_model, h_obs, sigma)


def fit_monjo_projected_alpha_free(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    k: float = K_PROJECTED,
) -> tuple[float, float, float]:
    alpha_grid = np.linspace(0.18, 1.00, 821)
    best_alpha = np.nan
    best_h0 = np.nan
    best_chi2 = np.inf
    for alpha in alpha_grid:
        h0_fit, chi2_fit = fit_monjo_projected_h0_for_alpha(z_obs, h_obs, sigma, alpha=alpha, k=k)
        if chi2_fit < best_chi2:
            best_alpha = alpha
            best_h0 = h0_fit
            best_chi2 = chi2_fit
    return float(best_alpha), float(best_h0), float(best_chi2)


def fit_monjo_projected_alpha_fixed(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    alpha: float = ALPHA_GAMMA_PROJECTION,
    k: float = K_PROJECTED,
) -> tuple[float, float]:
    return fit_monjo_projected_h0_for_alpha(z_obs, h_obs, sigma, alpha=alpha, k=k)


def fit_simple_stereographic_h0_for_params(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    gamma0: float,
    k: float = K_PROJECTED,
) -> tuple[float, float]:
    model = MonjoSimpleStereographic(alpha=alpha, gamma0=gamma0, k=k)
    e_model, _ = model.e_and_q(z_obs)
    return best_fit_h0_from_shape(e_model, h_obs, sigma)


def fit_simple_stereographic_alpha_free(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    gamma0: float = GAMMA0_THEORY,
    k: float = K_PROJECTED,
) -> tuple[float, float, float]:
    alpha_grid = np.linspace(0.10, 1.20, 1101)
    best_alpha = np.nan
    best_h0 = np.nan
    best_chi2 = np.inf
    for alpha in alpha_grid:
        h0_fit, chi2_fit = fit_simple_stereographic_h0_for_params(z_obs, h_obs, sigma, alpha=alpha, gamma0=gamma0, k=k)
        if chi2_fit < best_chi2:
            best_alpha = alpha
            best_h0 = h0_fit
            best_chi2 = chi2_fit
    return float(best_alpha), float(best_h0), float(best_chi2)


def fit_simple_stereographic_gamma0_free(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    alpha: float = ALPHA_GAMMA_PROJECTION,
    k: float = K_PROJECTED,
) -> tuple[float, float, float]:
    gamma0_grid = np.linspace(1.02, 8.0, 699)
    best_gamma0 = np.nan
    best_h0 = np.nan
    best_chi2 = np.inf
    for gamma0 in gamma0_grid:
        h0_fit, chi2_fit = fit_simple_stereographic_h0_for_params(z_obs, h_obs, sigma, alpha=alpha, gamma0=gamma0, k=k)
        if chi2_fit < best_chi2:
            best_gamma0 = gamma0
            best_h0 = h0_fit
            best_chi2 = chi2_fit
    return float(best_gamma0), float(best_h0), float(best_chi2)


def fit_monjo_projected_y0_fixed_alpha(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
    alpha: float = ALPHA_GAMMA_PROJECTION,
    k: float = K_PROJECTED,
) -> tuple[float, float, float]:
    y0_grid = np.linspace(0.55, 1.50, 951)
    best_y0 = np.nan
    best_h0 = np.nan
    best_chi2 = np.inf
    for y0 in y0_grid:
        monjo = MonjoProjectedHyperconical(alpha=alpha, k=k, y0_override=y0)
        e_model, _ = monjo.e_and_q(z_obs)
        h0_fit, chi2_fit = best_fit_h0_from_shape(e_model, h_obs, sigma)
        if chi2_fit < best_chi2:
            best_y0 = y0
            best_h0 = h0_fit
            best_chi2 = chi2_fit
    return float(best_y0), float(best_h0), float(best_chi2)


def compare_projection_models(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
) -> list[dict[str, float | str]]:
    linear_h0, linear_chi2 = fit_linear_h0(z_obs, h_obs, sigma)
    monjo_alpha, monjo_h0, monjo_chi2 = fit_monjo_projected_alpha_free(z_obs, h_obs, sigma)
    monjo_half_y0, monjo_half_h0, monjo_half_chi2 = fit_monjo_projected_y0_fixed_alpha(z_obs, h_obs, sigma, alpha=0.5)
    intrinsic_h0, intrinsic_chi2 = fit_projected_hyperconical_like_standard_h0(z_obs, h_obs, sigma)
    simple_alpha, simple_alpha_h0, simple_alpha_chi2 = fit_simple_stereographic_alpha_free(z_obs, h_obs, sigma)
    simple_gamma0, simple_gamma0_h0, simple_gamma0_chi2 = fit_simple_stereographic_gamma0_free(z_obs, h_obs, sigma)
    return [
        {"model": "Unprojected hyperconical / linear", "param": f"H0={linear_h0:.3f}", "chi2": linear_chi2},
        {"model": "Monjo projected stereographic (alpha free)", "param": f"alpha={monjo_alpha:.3f}, H0={monjo_h0:.3f}", "chi2": monjo_chi2},
        {"model": "Monjo projected stereographic (alpha=0.5, y0 free)", "param": f"y0={monjo_half_y0:.3f}, H0={monjo_half_h0:.3f}", "chi2": monjo_half_chi2},
        {"model": "Simple stereographic without arctan (alpha free)", "param": f"alpha={simple_alpha:.3f}, gamma0={GAMMA0_THEORY:.3f}, H0={simple_alpha_h0:.3f}", "chi2": simple_alpha_chi2},
        {"model": "Simple stereographic without arctan (gamma0 free)", "param": f"alpha={ALPHA_GAMMA_PROJECTION:.3f}, gamma0={simple_gamma0:.3f}, H0={simple_gamma0_h0:.3f}", "chi2": simple_gamma0_chi2},
        {"model": "Projected equalised-to-standard", "param": f"H0={intrinsic_h0:.3f}", "chi2": intrinsic_chi2},
    ]


def compare_background_models(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    sigma: np.ndarray,
) -> list[dict[str, float | str]]:
    lcdm_h0, lcdm_chi2 = fit_lcdm_h0(z_obs, h_obs, sigma)
    w_fit, w_h0, w_chi2 = fit_wcdm_w(z_obs, h_obs, sigma)
    power_n, power_h0, power_chi2 = fit_power_law_index(z_obs, h_obs, sigma)
    ds_h0, ds_chi2 = fit_desitter_h0(z_obs, h_obs, sigma)
    return [
        {"model": "flat LCDM", "param": f"H0={lcdm_h0:.3f}, Om={OMEGA_M_FID:.3f}", "chi2": lcdm_chi2},
        {"model": "wCDM", "param": f"H0={w_h0:.3f}, w={w_fit:.3f}, Om={OMEGA_M_FID:.3f}", "chi2": w_chi2},
        {"model": "Power law", "param": f"H0={power_h0:.3f}, n={power_n:.3f}", "chi2": power_chi2},
        {"model": "de Sitter", "param": f"H0={ds_h0:.3f}", "chi2": ds_chi2},
    ]


def format_report_table(title: str, rows: list[dict[str, float | str]]) -> str:
    lines = [title]
    for row in rows:
        lines.append(f"- {row['model']}: {row['param']}; chi2={row['chi2']:.3f}")
    return "\n".join(lines)


def exclude_source_kind(points: list[HzPoint], source_kind: str) -> list[HzPoint]:
    return [point for point in points if point.source_kind != source_kind]


def scaled_bao_points(points: list[HzPoint], target_z: float, sigma_scale: float) -> list[HzPoint]:
    scaled: list[HzPoint] = []
    for point in points:
        if abs(point.z - target_z) < 1.0e-9:
            scaled.append(
                HzPoint(
                    z=point.z,
                    h_km_s_mpc=point.h_km_s_mpc,
                    sigma_plus=point.sigma_plus * sigma_scale,
                    sigma_minus=point.sigma_minus * sigma_scale,
                    label=point.label,
                    reference_key=point.reference_key,
                    source_kind=point.source_kind,
                    note=point.note + f" [sigma x{sigma_scale:g}]",
                )
            )
        else:
            scaled.append(point)
    return scaled


def plot_grouped_points(
    axis: plt.Axes,
    grouped: dict[str, list[HzPoint]],
    show_labels: bool,
    x_transform=lambda x: x,
    y_transform=lambda x: x,
) -> list[Line2D]:
    handles: list[Line2D] = []
    for reference_key, ref_points in grouped.items():
        style = SOURCE_STYLES.get(
            reference_key,
            {"color": "#222222", "marker": "o", "label": reference_key},
        )
        z = np.array([p.z for p in ref_points])
        h = np.array([p.h_km_s_mpc for p in ref_points])
        h_lo = np.array([max(p.h_km_s_mpc - p.sigma_minus, 1.0e-9) for p in ref_points])
        h_hi = np.array([p.h_km_s_mpc + p.sigma_plus for p in ref_points])
        y = y_transform(h)
        yerr = np.vstack([y - y_transform(h_lo), y_transform(h_hi) - y])
        axis.errorbar(
            x_transform(z),
            y,
            yerr=yerr,
            fmt=style["marker"],
            ms=5.2 if style["marker"] != "*" else 7.2,
            color=style["color"],
            ecolor=style["color"],
            mec="white",
            mew=0.6,
            elinewidth=0.95,
            capsize=0,
            alpha=0.92,
            label=style["label"] if show_labels else "_nolegend_",
            zorder=4,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="None",
                markersize=5.5 if style["marker"] != "*" else 7.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=style["label"],
            )
        )
    return handles


def plot_bao_points(
    axis: plt.Axes,
    points: list[HzPoint],
    model_values: np.ndarray | None = None,
    x_transform=lambda x: x,
    y_transform=lambda x: x,
) -> list[Line2D]:
    handles: list[Line2D] = []
    for group in BAO_STYLE_GROUPS.values():
        group_points = [p for p in points if p.reference_key in group["refs"]]
        if not group_points:
            continue
        z = np.array([p.z for p in group_points])
        h = np.array([p.h_km_s_mpc for p in group_points])
        sigma_sym = np.array([0.5 * (p.sigma_minus + p.sigma_plus) for p in group_points])
        if model_values is None:
            y = y_transform(h)
            y_lo = y_transform(np.maximum(h - sigma_sym, 1.0e-9))
            y_hi = y_transform(h + sigma_sym)
            yerr = np.vstack([y - y_lo, y_hi - y])
        else:
            residual = h - np.array([model_values[points.index(p)] for p in group_points])
            y = residual
            yerr = sigma_sym
        axis.errorbar(
            x_transform(z),
            y,
            yerr=yerr,
            fmt=group["marker"],
            ms=5.8,
            mfc="white",
            mec=group["color"],
            mew=1.0,
            color=group["color"],
            ecolor=group["color"],
            elinewidth=0.9,
            capsize=0,
            alpha=0.95,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=group["color"],
                marker=group["marker"],
                linestyle="None",
                markersize=5.8,
                markerfacecolor="white",
                markeredgecolor=group["color"],
                markeredgewidth=1.0,
                label=group["label"],
            )
        )
    return handles


def fit_panel_models(
    points: list[HzPoint],
    bao_points: list[HzPoint],
    z_grid: np.ndarray,
    include_monjo: bool,
    monjo_override: dict[str, float] | None = None,
) -> dict[str, object]:
    grouped_points = group_points_by_reference(points)
    z_obs, h_obs, sigma_obs = combined_arrays(points)
    z_bao, h_bao, sigma_bao = combined_arrays(bao_points) if bao_points else (np.array([]), np.array([]), np.array([]))
    z_fit = np.concatenate([z_obs, z_bao])
    h_fit = np.concatenate([h_obs, h_bao])
    sigma_fit = np.concatenate([sigma_obs, sigma_bao])
    highz_mask = z_fit > 0.01
    z_fit_highz = z_fit[highz_mask]
    h_fit_highz = h_fit[highz_mask]
    sigma_fit_highz = sigma_fit[highz_mask]

    h0_lcdm_fit, _ = fit_lcdm_h0(z_fit, h_fit, sigma_fit)
    h0_lcdm_highz, _ = fit_lcdm_h0(z_fit_highz, h_fit_highz, sigma_fit_highz)
    h0_lin_fit, _ = fit_linear_h0(z_fit, h_fit, sigma_fit)
    h0_lin_highz, _ = fit_linear_h0(z_fit_highz, h_fit_highz, sigma_fit_highz)
    power_n_fit, h0_power_fit, _ = fit_power_law_index(z_fit, h_fit, sigma_fit)
    power_n_highz, h0_power_highz, _ = fit_power_law_index(z_fit_highz, h_fit_highz, sigma_fit_highz)
    w_fit, h0_w_fit, _ = fit_wcdm_w(z_fit, h_fit, sigma_fit)
    w_highz, h0_w_highz, _ = fit_wcdm_w(z_fit_highz, h_fit_highz, sigma_fit_highz)
    h0_ds_fit, _ = fit_desitter_h0(z_fit, h_fit, sigma_fit)

    model_specs: list[dict[str, object]] = [
        {
            "title": r"flat $\Lambda$CDM & intrinsic hyperconical",
            "curve_color": "#2a5c8a",
            "curve_style": {"linewidth": 2.3, "linestyle": "-"},
            "curve_grid": hz_lcdm(z_grid, h0=h0_lcdm_fit),
            "legend": rf"Flat $\Lambda$CDM & intrinsic hyperconical ($H_0={h0_lcdm_fit:.1f}$)",
            "model_obs": hz_lcdm(z_obs, h0=h0_lcdm_fit),
            "model_bao": hz_lcdm(z_bao, h0=h0_lcdm_fit),
            "model_fit": hz_lcdm(z_fit, h0=h0_lcdm_fit),
            "model_fit_highz": hz_lcdm(z_fit_highz, h0=h0_lcdm_highz),
            "ylabel": r"$\Delta H_{\Lambda{\rm CDM}}$",
        },
        {
            "title": r"$w$CDM",
            "curve_color": "#2a5c8a",
            "curve_style": {"linewidth": 2.0, "linestyle": (0, (4.5, 2.2))},
            "curve_grid": hz_wcdm(z_grid, w_fit, h0=h0_w_fit),
            "legend": rf"$w$CDM ($H_0={h0_w_fit:.1f}$, $w={w_fit:.2f}$)",
            "model_obs": hz_wcdm(z_obs, w_fit, h0=h0_w_fit),
            "model_bao": hz_wcdm(z_bao, w_fit, h0=h0_w_fit),
            "model_fit": hz_wcdm(z_fit, w_fit, h0=h0_w_fit),
            "model_fit_highz": hz_wcdm(z_fit_highz, w_highz, h0=h0_w_highz),
            "ylabel": r"$\Delta H_{w{\rm CDM}}$",
        },
        {
            "title": r"linear coasting",
            "curve_color": "#2f7d32",
            "curve_style": {"linewidth": 2.3, "linestyle": "--"},
            "curve_grid": hz_linear(z_grid, h0=h0_lin_fit),
            "legend": rf"Linear coasting ($R_h=ct$, Milne, extrinsic hyperconical; $H_0={h0_lin_fit:.1f}$)",
            "model_obs": hz_linear(z_obs, h0=h0_lin_fit),
            "model_bao": hz_linear(z_bao, h0=h0_lin_fit),
            "model_fit": hz_linear(z_fit, h0=h0_lin_fit),
            "model_fit_highz": hz_linear(z_fit_highz, h0=h0_lin_highz),
            "ylabel": r"$\Delta H_{\rm lin}$",
        },
        {
            "title": r"power law",
            "curve_color": "#b04a9f",
            "curve_style": {"linewidth": 2.0, "linestyle": (0, (5, 2, 1.2, 2))},
            "curve_grid": hz_power_law(z_grid, power_n_fit, h0=h0_power_fit),
            "legend": rf"Power law ($H_0={h0_power_fit:.1f}$, $n={power_n_fit:.2f}$)",
            "model_obs": hz_power_law(z_obs, power_n_fit, h0=h0_power_fit),
            "model_bao": hz_power_law(z_bao, power_n_fit, h0=h0_power_fit),
            "model_fit": hz_power_law(z_fit, power_n_fit, h0=h0_power_fit),
            "model_fit_highz": hz_power_law(z_fit_highz, power_n_highz, h0=h0_power_highz),
            "ylabel": r"$\Delta H_{\rm pow}$",
        },
    ]

    if include_monjo:
        if monjo_override is None:
            y0_fit, h0_monjo_fit, _ = fit_monjo_projected_y0_fixed_alpha(z_fit, h_fit, sigma_fit, alpha=0.5)
            y0_highz, h0_monjo_highz, _ = fit_monjo_projected_y0_fixed_alpha(z_fit_highz, h_fit_highz, sigma_fit_highz, alpha=0.5)
        else:
            y0_fit = monjo_override["y0_fit"]
            h0_monjo_fit = monjo_override["h0_fit"]
            y0_highz = monjo_override["y0_highz"]
            h0_monjo_highz = monjo_override["h0_highz"]
        monjo_fit = MonjoProjectedHyperconical(alpha=0.5, k=K_PROJECTED, y0_override=y0_fit)
        monjo_highz = MonjoProjectedHyperconical(alpha=0.5, k=K_PROJECTED, y0_override=y0_highz)
        e_grid, _ = monjo_fit.e_and_q(z_grid)
        e_obs, _ = monjo_fit.e_and_q(z_obs)
        e_bao = np.array([])
        if len(z_bao) > 0:
            e_bao, _ = monjo_fit.e_and_q(z_bao)
        e_fit, _ = monjo_fit.e_and_q(z_fit)
        e_fit_highz, _ = monjo_highz.e_and_q(z_fit_highz)
        model_specs.insert(
            3,
            {
                "title": r"projected hyperconical",
                "curve_color": "#d17a00",
                "curve_style": {"linewidth": 2.0, "linestyle": "-."},
                "curve_grid": h0_monjo_fit * e_grid,
                "legend": rf"Projected hyperconical (Monjo 2017 $\alpha \equiv 0.5$; $y_0={y0_fit:.2f}$, $H_0={h0_monjo_fit:.1f}$)",
                "model_obs": h0_monjo_fit * e_obs,
                "model_bao": h0_monjo_fit * e_bao,
                "model_fit": h0_monjo_fit * e_fit,
                "model_fit_highz": h0_monjo_highz * e_fit_highz,
                "ylabel": r"$\Delta H_{\rm Monjo}$",
            },
        )

    return {
        "grouped_points": grouped_points,
        "z_obs": z_obs,
        "h_obs": h_obs,
        "sigma_obs": sigma_obs,
        "z_bao": z_bao,
        "h_bao": h_bao,
        "sigma_bao": sigma_bao,
        "z_fit": z_fit,
        "h_fit": h_fit,
        "sigma_fit": sigma_fit,
        "z_fit_highz": z_fit_highz,
        "h_fit_highz": h_fit_highz,
        "sigma_fit_highz": sigma_fit_highz,
        "model_specs": model_specs,
        "h0_ds_fit": h0_ds_fit,
        "curve_ds": hz_desitter(z_grid, h0=h0_ds_fit),
    }


def make_plot(
    points: list[HzPoint],
    bao_points: list[HzPoint],
    output_png: Path,
    output_svg: Path | None,
    output_pdf: Path | None = None,
    include_monjo: bool = True,
    log1p_axes: bool = True,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "dejavuserif",
        }
    )

    max_z_data = max([p.z for p in points + bao_points], default=1.45)
    z_max = 1.45 if max_z_data <= 1.45 else min(2.55, max_z_data + 0.12)
    y_max = 300.0
    z_grid = np.linspace(0.0, z_max, 800)
    panel_all = fit_panel_models(points, bao_points, z_grid, include_monjo=include_monjo)
    panel_model_ind = fit_panel_models(points, [], z_grid, include_monjo=include_monjo)
    n_residual = len(panel_all["model_specs"])

    fig, axes = plt.subplots(
        1 + n_residual,
        2,
        figsize=(16.4, 11.9 if not include_monjo else 12.9),
        sharex="col",
        gridspec_kw={"height_ratios": [4.25] + [0.58] * n_residual, "wspace": 0.02},
    )
    fig.patch.set_facecolor("#ffffff")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.grid(True, which="both", color="#cfc6b8", linewidth=0.7, alpha=0.28)

    panel_configs = [
        ("(a)", "All sources", panel_all, True),
        ("(b)", "Model-independent sources", panel_model_ind, False),
    ]
    residual_scale = 80.0
    x_transform = np.log1p if log1p_axes else (lambda x: x)
    y_transform = np.log1p if log1p_axes else (lambda x: x)
    x_tick_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_tick_values = [50, 100, 150, 200, 250, 300]

    for col, (tag, source_title, panel_data, show_bao) in enumerate(panel_configs):
        ax = axes[0, col]
        for spec in panel_data["model_specs"]:
            ax.plot(x_transform(z_grid), y_transform(spec["curve_grid"]), color=spec["curve_color"], **spec["curve_style"])
        ax.plot(
            x_transform(z_grid),
            y_transform(panel_data["curve_ds"]),
            color="#6c6c6c",
            linewidth=1.9,
            linestyle=(0, (1.4, 2.2)),
        )

        source_handles = plot_grouped_points(ax, panel_data["grouped_points"], show_labels=False, x_transform=x_transform, y_transform=y_transform)
        source_legend_handles = source_handles.copy()
        if show_bao:
            source_legend_handles.extend(plot_bao_points(ax, bao_points, x_transform=x_transform, y_transform=y_transform))

        if col == 0:
            ax.set_ylabel(r"$H(z)\;[{\rm km\,s^{-1}\,Mpc^{-1}}]$" if log1p_axes else r"$H(z)\;[{\rm km\,s^{-1}\,Mpc^{-1}}]$", fontsize=FACTOR_SIZE*11.5)
        else:
            ax.set_ylabel(r"$H(z)\;[{\rm km\,s^{-1}\,Mpc^{-1}}]$" if log1p_axes else r"$H(z)\;[{\rm km\,s^{-1}\,Mpc^{-1}}]$", fontsize=FACTOR_SIZE*11.5, rotation=270, labelpad=18)
            ax.yaxis.set_label_position("right")
        ax.set_xlim(float(x_transform(np.array([0.0]))[0]), float(x_transform(np.array([z_max]))[0]))
        if log1p_axes:
            ax.set_ylim(float(np.log1p(40.0)), float(np.log1p(y_max)))
        else:
            ax.set_ylim(40.0, y_max)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=True, labelbottom=False)
        ax.tick_params(axis="y", left=(col == 0), labelleft=(col == 0), right=True, labelright=(col == 1))
        secax = ax.secondary_xaxis("top")
        secax.set_xlabel(r"Redshift $z$" if log1p_axes else r"Redshift $z$", fontsize=FACTOR_SIZE*11.0, labelpad=24)
        secax.tick_params(labeltop=False, top=False)

        if log1p_axes:
            ax.xaxis.set_major_locator(FixedLocator(np.log1p(x_tick_values)))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{v:.1f}" for v in x_tick_values]))
            ax.yaxis.set_major_locator(FixedLocator(np.log1p(y_tick_values)))
            ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in y_tick_values]))

        model_handles = []
        for spec in panel_data["model_specs"]:
            model_handles.append(
                Line2D([0], [0], color=spec["curve_color"], label=spec["legend"], **spec["curve_style"])
            )
        model_handles.append(
            Line2D([0], [0], color="#6c6c6c", linewidth=1.9, linestyle=(0, (1.4, 2.2)), label=rf"de Sitter ($H_0={panel_data['h0_ds_fit']:.1f}$)")
        )

        legend_models = ax.legend(
            handles=model_handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.995),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.66,
            fontsize=FACTOR_SIZE*8.8,
            title="Models",
            title_fontsize=FACTOR_SIZE*9.4,
            ncol=1,
            handletextpad=0.55,
        )
        ax.add_artist(legend_models)
        legend_sources = ax.legend(
            handles=source_legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.02),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.66,
            fontsize=FACTOR_SIZE*8.2,
            title=source_title,
            title_fontsize=FACTOR_SIZE*9.0,
            ncol=2,
            columnspacing=0.75,
            handletextpad=0.35,
        )
        ax.add_artist(legend_sources)

        for row, spec in enumerate(panel_data["model_specs"], start=1):
            axis = axes[row, col]
            axis.axhline(0.0, color=spec["curve_color"], linewidth=1.25, linestyle="--", alpha=0.88)
            for reference_key, ref_points in panel_data["grouped_points"].items():
                style = SOURCE_STYLES.get(reference_key, {"color": "#222222", "marker": "o", "label": reference_key})
                ref_z = np.array([p.z for p in ref_points])
                ref_h = np.array([p.h_km_s_mpc for p in ref_points])
                ref_sigma = np.array([0.5 * (p.sigma_minus + p.sigma_plus) for p in ref_points])
                ref_model = np.interp(ref_z, panel_data["z_obs"], spec["model_obs"])
                axis.errorbar(
                    x_transform(ref_z),
                    y_transform(ref_h) - y_transform(ref_model),
                    yerr=ref_sigma,
                    fmt=style["marker"],
                    ms=4.5 if style["marker"] != "*" else 6.6,
                    color=style["color"],
                    ecolor=style["color"],
                    mec="white",
                    mew=0.5,
                    elinewidth=0.85,
                    capsize=0,
                    alpha=0.88,
                    zorder=4,
                )
            if show_bao:
                plot_bao_points(axis, bao_points, model_values=spec["model_bao"])

            chi2_all, _, _ = chi2_summary(panel_data["h_fit"], spec["model_fit"], panel_data["sigma_fit"])
            chi2_highz, _, _ = chi2_summary(panel_data["h_fit_highz"], spec["model_fit_highz"], panel_data["sigma_fit_highz"])
            axis.text(
                0.03,
                75.0,
                rf"$\chi^2={chi2_all:.1f}$ (all), $\chi^2={chi2_highz:.1f}$ ($z>0.01$), " + spec["title"],
                ha="left",
                va="top",
                fontsize=FACTOR_SIZE*7.6,
                color="#4b4b4b",
            )
            axis.set_ylim(-residual_scale, residual_scale)
            axis.tick_params(axis="x", top=True, labeltop=False)
            axis.tick_params(axis="y", left=(col == 0), labelleft=(col == 0), right=True, labelright=(col == 1))
            if col == 0:
                axis.set_ylabel(spec["ylabel"], fontsize=FACTOR_SIZE*9.2)
            else:
                axis.set_ylabel(spec["ylabel"], fontsize=FACTOR_SIZE*9.2, rotation=270, labelpad=14)
                axis.yaxis.set_label_position("right")

    axes[-1, 0].set_xlabel(r"Redshift $z$" if log1p_axes else r"Redshift $z$", fontsize=FACTOR_SIZE*11.5)
    axes[-1, 1].set_xlabel(r"Redshift $z$" if log1p_axes else r"Redshift $z$", fontsize=FACTOR_SIZE*11.5)

    for col, (tag, _, _, _) in enumerate(panel_configs):
        bbox = axes[0, col].get_position()
        x_pos = bbox.x0 - 0.072 if col == 0 else bbox.x0 + 0.002
        fig.text(x_pos, bbox.y1 + 0.048, tag, ha="left", va="bottom", fontsize=FACTOR_SIZE*12.5)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.06, hspace=0.045)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare H(z) background models against curated chronometer data.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Curated CSV input.")
    parser.add_argument("--bao-data", type=Path, default=DEFAULT_BAO_DATA, help="Curated BAO CSV input.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PNG, help="PNG output path.")
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG, help="Optional SVG output path.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Optional PDF output path.")
    parser.add_argument("--report", action="store_true", default=True, help="Print chi^2 summaries with and without the local distance-ladder point.")
    parser.add_argument("--no-bao", action="store_true", default=False, help="Exclude BAO points from fits and plotting.")
    parser.add_argument("--with-monjo", action="store_true", default=True, help="Include the Monjo projected curve and residual panel.")
    parser.add_argument("--z024-sigma-scale", type=float, default=1.0, help="Multiply the BAO uncertainty at the target redshift by this factor for sensitivity tests.")
    parser.add_argument("--sigma-scale-target-z", type=float, default=0.24, help="Target redshift for the BAO uncertainty scaling sensitivity test.")
    parser.add_argument("--log1p-axes", action="store_true", default=True, help="Plot the main panels using log(1+z) and log(1+H) axes.")
    args = parser.parse_args()

    points = load_points(args.data)
    bao_points = [] if args.no_bao else load_points(args.bao_data)
    if bao_points and abs(args.z024_sigma_scale - 1.0) > 1.0e-12:
        bao_points = scaled_bao_points(bao_points, target_z=args.sigma_scale_target_z, sigma_scale=args.z024_sigma_scale)
    if args.report:
        points_no_local = exclude_source_kind(points, "distance_ladder")
        for title, use_points in [
            ("With local Riess point", points),
            ("Without local Riess point", points_no_local),
        ]:
            z_obs, h_obs, sigma = combined_arrays(use_points + bao_points)
            print(format_report_table(f"{title}: background models", compare_background_models(z_obs, h_obs, sigma)))
            print()
            print(format_report_table(f"{title}: hyperconical projection families", compare_projection_models(z_obs, h_obs, sigma)))
            print()
    make_plot(points, bao_points, args.output, args.svg, args.pdf, include_monjo=args.with_monjo, log1p_axes=args.log1p_axes) 

if __name__ == "__main__":
    main()
