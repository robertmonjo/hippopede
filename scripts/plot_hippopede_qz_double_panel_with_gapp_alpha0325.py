from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUTDIR = Path(r"C:\Users\rober\Downloads")
SHOW_TITLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
for extra in (str(SCRIPT_DIR), str(SCRIPT_DIR / "vendor_gapp"), str(SCRIPT_DIR / "vendor_gapp" / "covfunctions")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import reconstruct_qz_mukherjee2021_gapp as qg  # noqa: E402
import dgp  # noqa: E402
import covariance  # noqa: E402

CC_Z = np.array([0.07, 0.12, 0.20, 0.28, 0.40, 0.48, 0.75, 0.80, 1.26])
CC_H = np.array([69.0, 68.6, 72.9, 88.8, 95.0, 97.0, 105.0, 113.1, 135.0])
CC_ERR = np.array([19.6, 26.2, 29.6, 36.6, 17.0, 62.0, 10.8, 15.1, 35.0])
H0_CC = 70.0
CC_E = CC_H / H0_CC
CC_E_ERR = CC_ERR / H0_CC

Q0_POINTS = [
    {"q": -0.364, "err": 0.032, "marker": "D", "color": "#6a3d9a", "label": r"Observed $q_0$ (CC+Pantheon+SH0ES+BAO; Myrzakulov et al. 2025)"},
    {"q": -0.50, "err": 0.20, "marker": "s", "color": "#1b9e77", "label": r"Observed $q_0$ (FRB+SNe cosmography; Gao et al. 2024)"},
]
ZT_POINTS = [
    {"z": 0.597, "err": 0.214, "marker": "o", "color": "#f16913", "label": r"Obs. $z_t$: CC+Pantheon+SH0ES+BAO (2025)"},
    {"z": 0.64, "err": 0.16, "marker": "^", "color": "#fd8d3c", "label": r"Obs. $z_t$: latest $H(z)$ meta-analysis (2025)"},
]


def rho_hippopede(t, chi):
    s2 = np.sin(chi) ** 2
    c2 = np.cos(chi) ** 2
    return np.sqrt(s2 + 4.0 * t**2 * c2)


def a_centered(t, chi):
    rho = rho_hippopede(t, chi)
    return np.sqrt(rho**2 + t**2 - 2.0 * t * rho * np.cos(chi))


def q_lcdm_of_z(z, omega_r=9.0e-5, omega_m=0.3, omega_l=0.69991):
    e2 = omega_r * (1.0 + z) ** 4 + omega_m * (1.0 + z) ** 3 + omega_l
    return (omega_r * (1.0 + z) ** 4 + 0.5 * omega_m * (1.0 + z) ** 3 - omega_l) / e2


def e_lcdm(z, omega_r=9.0e-5, omega_m=0.3, omega_l=0.69991):
    return np.sqrt(omega_r * (1.0 + z) ** 4 + omega_m * (1.0 + z) ** 3 + omega_l)


def build_centered_history(chi, t0, n=8000):
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


def interp_curve(z_grid, z_data, y_data):
    valid = np.isfinite(z_data) & np.isfinite(y_data)
    z_data = z_data[valid]
    y_data = y_data[valid]
    if len(z_data) < 2:
        return np.full_like(z_grid, np.nan, dtype=float)
    y = np.interp(z_grid, z_data, y_data, left=np.nan, right=np.nan)
    y[z_grid < z_data.min()] = np.nan
    y[z_grid > z_data.max()] = np.nan
    return y


def tangent_band(z, q0, dq0):
    slope = 1.0 + q0
    dslope = dq0
    y = 1.0 + slope * z
    y_lo = 1.0 + (slope - dslope) * z
    y_hi = 1.0 + (slope + dslope) * z
    return y, y_lo, y_hi


class MonjoProjectedHyperconical:
    def __init__(self, alpha=0.283, k=1.0):
        self.alpha = alpha
        self.k = k
        self._prepare_lookup()

    def _ttp(self, r):
        T = 1.0
        t = 1.0
        k = self.k
        return t * np.sqrt(((T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) / (np.sqrt((-k * r**2 + T**2) / T**2) * T**2 * k)))

    def _grrp(self, r):
        T = 1.0
        t = 1.0
        k = self.k
        return -t**2 * (((T**2 * (k - 2) + k * r**2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) / ((T**2 * (k - 2) * np.sqrt((-k * r**2 + T**2) / T**2) - 2 * k * r**2 + 2 * T**2) * (-k * r**2 + T**2)))

    def _f(self, x):
        return -np.sqrt(-self._grrp(x)) / self._ttp(x)

    def _prepare_lookup(self):
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
        I_pos = -integ
        self.x_grid = np.concatenate([-x_pos[:0:-1], x_pos])
        self.I_grid = np.concatenate([-I_pos[:0:-1], I_pos])
        self.f_grid = np.concatenate([fx_pos[:0:-1], fx_pos])
        ux = np.sqrt(1.0 / k - self.mx**2)
        self.y0 = np.arctan2(self.mx, ux)

    def x_from_lz(self, lz):
        return np.interp(lz, self.I_grid, self.x_grid, left=self.x_grid[0], right=self.x_grid[-1])

    def dxdLZ(self, x):
        f = np.interp(x, self.x_grid, self.f_grid)
        return -1.0 / f

    def dinvll_dx(self, x):
        k = self.k
        u = np.sqrt(np.maximum(1.0 / k - x**2, 1e-14))
        y = np.arctan2(x, u)
        dy_dx = 1.0 / u
        s = y / 2.0
        g = np.maximum(1.0 - y / self.y0, 1e-12)
        t = s / g**self.alpha
        dt_dx = dy_dx * (0.5 / g**self.alpha + s * self.alpha / self.y0 * g ** (-self.alpha - 1.0))
        return 2.0 * dt_dx / (1.0 + t**2)

    def e_and_q(self, z_grid):
        z = np.asarray(z_grid, dtype=float)
        lz = np.log1p(z)
        x = self.x_from_lz(lz)
        with np.errstate(divide="ignore", invalid="ignore"):
            dr_dz = self.dinvll_dx(x) * self.dxdLZ(x) / (1.0 + z)
            H = 1.0 / dr_dz
        kernel = np.ones(9) / 9.0
        valid = np.isfinite(H)
        if valid.sum() > len(kernel):
            Hv = H[valid]
            pad = len(kernel) // 2
            Hv_pad = np.pad(Hv, pad_width=pad, mode="edge")
            H[valid] = np.convolve(Hv_pad, kernel, mode="valid")
        H0 = np.interp(0.0, z[np.isfinite(H)], H[np.isfinite(H)])
        E = H / H0
        dE_dz = np.gradient(E, z)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = -1.0 + (1.0 + z) * dE_dz / E
        E[~np.isfinite(E)] = np.nan
        q[~np.isfinite(q)] = np.nan
        return E, q


def project_curve(z_obs, z_unproj, e_unproj, z_map):
    e_proj = interp_curve(z_map, z_unproj, e_unproj)
    valid = np.isfinite(e_proj)
    if valid.sum() > 8:
        e_valid = e_proj[valid]
        z_valid = z_obs[valid]
        de_dz = np.gradient(e_valid, z_valid)
        q_valid = -1.0 + (1.0 + z_valid) * de_dz / e_valid
        q_proj = np.full_like(z_obs, np.nan, dtype=float)
        q_proj[valid] = q_valid
    else:
        q_proj = np.full_like(z_obs, np.nan, dtype=float)
    return e_proj, q_proj


def reconstruct_gapp_cc_pantheon(zmin=-0.5, zmax=2.36, nstar=320):
    cc_z, cc_h, cc_sigma = qg.load_dataset("CC")
    dX = cc_z
    dY = qg.H0_LOCAL / cc_h
    dSigma = qg.H0_LOCAL * cc_sigma / (cc_h**2)
    z_data, d_data, sigma_data = qg.load_pantheon_plus_binned()
    gp_obj = dgp.DGaussianProcess(
        z_data,
        d_data,
        sigma_data,
        covfunction=covariance.SquaredExponential,
        dX=dX,
        dY=dY,
        dSigma=dSigma,
        cXstar=(zmin, zmax, nstar),
    )
    rec, theta = gp_obj.gp(thetatrain="True")
    drec, _ = gp_obj.dgp(thetatrain="False")
    d2rec, _ = gp_obj.d2gp(thetatrain="False")
    z = rec[:, 0]
    d = rec[:, 1]
    d_sigma = rec[:, 2]
    dd = drec[:, 1]
    dd_sigma = drec[:, 2]
    d2d = d2rec[:, 1]
    d2d_sigma = d2rec[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 1.0 / dd
        e_sigma = np.abs(dd_sigma / (dd**2))
        q = -1.0 - (1.0 + z) * d2d / dd
        q_sigma = np.sqrt(((1.0 + z) * d2d_sigma / dd) ** 2 + (((1.0 + z) * d2d * dd_sigma) / (dd**2)) ** 2)
    return {"z": z, "e": e, "e_sigma": e_sigma, "q": q, "q_sigma": q_sigma, "theta": np.asarray(theta)}


def style_panel(ax, face="#fefefe"):
    ax.set_facecolor(face)
    ax.grid(True, which="major", color="#f6f6f6", lw=0.58, alpha=0.50)
    ax.grid(True, which="minor", color="#fbfbfb", lw=0.38, alpha=0.48)
    ax.minorticks_on()


def add_gapp_q_overlay(ax, gapp, zmin=-0.5, zmax=2.0):
    z = gapp["z"]
    q = gapp["q"]
    s = gapp["q_sigma"]
    core = (z >= 0.0) & (z <= zmax)
    extra = (z >= zmin) & (z < 0.0)
    ax.fill_between(z[core], (q - 3*s)[core], (q + 3*s)[core], color="#d7d7d7", alpha=0.17, linewidth=0.0, zorder=1.2)
    ax.fill_between(z[core], (q - 2*s)[core], (q + 2*s)[core], color="#b5b5b5", alpha=0.16, linewidth=0.0, zorder=1.3)
    ax.fill_between(z[core], (q - 1*s)[core], (q + 1*s)[core], color="#8e8e8e", alpha=0.15, linewidth=0.0, zorder=1.4)
    ax.plot(z[core], q[core], color="#8f8f8f", lw=4.0, alpha=0.72, zorder=1.8)
    ax.fill_between(z[extra], (q - 3*s)[extra], (q + 3*s)[extra], color="#efb0b0", alpha=0.10, linewidth=0.0, zorder=0.9)
    ax.fill_between(z[extra], (q - 2*s)[extra], (q + 2*s)[extra], color="#e59292", alpha=0.09, linewidth=0.0, zorder=1.0)
    ax.fill_between(z[extra], (q - 1*s)[extra], (q + 1*s)[extra], color="#db7272", alpha=0.08, linewidth=0.0, zorder=1.1)
    ax.plot(z[extra], q[extra], color="#c96a6a", lw=1.0, alpha=0.32, zorder=1.7)


def add_gapp_e_overlay(ax, gapp, zmin=-0.5, zmax=2.0):
    z = gapp["z"]
    e = gapp["e"]
    s = gapp["e_sigma"]
    core = (z >= 0.0) & (z <= zmax)
    extra = (z >= zmin) & (z < 0.0)
    ax.fill_between(z[core], (e - 3*s)[core], (e + 3*s)[core], color="#d7d7d7", alpha=0.17, linewidth=0.0, zorder=1.2)
    ax.fill_between(z[core], (e - 2*s)[core], (e + 2*s)[core], color="#b5b5b5", alpha=0.16, linewidth=0.0, zorder=1.3)
    ax.fill_between(z[core], (e - 1*s)[core], (e + 1*s)[core], color="#8e8e8e", alpha=0.15, linewidth=0.0, zorder=1.4)
    ax.plot(z[core], e[core], color="#8f8f8f", lw=4.0, alpha=0.72, zorder=1.8)
    ax.fill_between(z[extra], (e - 3*s)[extra], (e + 3*s)[extra], color="#efb0b0", alpha=0.10, linewidth=0.0, zorder=0.9)
    ax.fill_between(z[extra], (e - 2*s)[extra], (e + 2*s)[extra], color="#e59292", alpha=0.09, linewidth=0.0, zorder=1.0)
    ax.fill_between(z[extra], (e - 1*s)[extra], (e + 1*s)[extra], color="#db7272", alpha=0.08, linewidth=0.0, zorder=1.1)
    ax.plot(z[extra], e[extra], color="#c96a6a", lw=1.0, alpha=0.32, zorder=1.7)


def add_common_top(ax, z, zmin, zmax, gapp, right_side=False):
    add_gapp_q_overlay(ax, gapp, zmin=zmin, zmax=zmax)
    ax.plot(z, q_lcdm_of_z(z), color="#c8a27a", lw=3.6, alpha=0.95, zorder=2.0, label=r"Standard model: flat $\Lambda$CDM + radiation")
    ax.axhline(-1.0, color="#4d4d4d", lw=1.2, ls=":", alpha=0.95, label=r"de Sitter / inflation: $q=-1$")
    ax.axhline(0.0, color="#777777", lw=1.0, alpha=0.65)
    ax.axvline(0.0, color="#999999", lw=1.0, ls="--", alpha=0.9)
    ax.text(0.0, 0.72, "today", rotation=90, ha="center", va="top", fontsize=10, color="#666666")
    for item in Q0_POINTS:
        ax.errorbar(0.0, item["q"], yerr=item["err"], fmt=item["marker"], ms=7.2, mfc=item["color"], mec="white", mew=0.9, ecolor=item["color"], elinewidth=1.5, capsize=4, label=item["label"], zorder=6)
    for item in ZT_POINTS:
        ax.errorbar(item["z"], 0.0, xerr=item["err"], fmt=item["marker"], ms=7.0, mfc=item["color"], mec="white", mew=0.9, ecolor=item["color"], elinewidth=1.5, capsize=4, label=item["label"], zorder=6)
    style_panel(ax)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-4.0, 1.0)
    ax.set_ylabel(r"Deceleration parameter $q(z)$")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel(r"Redshift $z$")
    ax.tick_params(axis="x", labelbottom=False)
    if right_side:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelleft=False, labelright=True)


def add_common_bottom(ax, z_model, z_tan, zmin, zmax, gapp, right_side=False):
    add_gapp_e_overlay(ax, gapp, zmin=zmin, zmax=zmax)
    ax.plot(z_model, e_lcdm(z_model), color="#c8a27a", lw=3.4, alpha=0.95, label="_nolegend_", zorder=2.0)
    ax.errorbar(CC_Z, CC_E, yerr=CC_E_ERR, fmt="o", ms=5.8, mfc="#2b8cbe", mec="white", mew=0.8, ecolor="#2b8cbe", elinewidth=1.2, capsize=3, alpha=0.96, label=r"Cosmic chronometers $H(z)$ data", zorder=6)
    for item in Q0_POINTS:
        y, y_lo, y_hi = tangent_band(z_tan, item["q"], item["err"])
        ax.plot(z_tan, y, color=item["color"], lw=1.8, ls=(0, (5, 3)), label=rf"Tangent from {item['label'].replace('Observed ', '')}")
        ax.fill_between(z_tan, y_lo, y_hi, color=item["color"], alpha=0.14, linewidth=0.0)
    style_panel(ax)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(0.2, 3.2)
    ax.set_ylabel(r"Normalized Hubble rate $E(z)$")
    ax.set_xlabel(r"Redshift $z$")
    if right_side:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelleft=False, labelright=True)


def make_legends(fig):
    hip_colors = ["#d0e1f2", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    chis_deg = [0, 15, 30, 45, 60, 85]
    model_handles = []
    model_labels = []
    for chi_deg, color in zip(chis_deg, hip_colors):
        line = Line2D([0], [0], color=color, lw=2.4)
        label = rf"Hippopede: $\chi={chi_deg}^\circ$"
        if chi_deg == 0:
            label += r" & hyperconical"
        model_handles.append(line)
        model_labels.append(label)
    model_handles.extend([
        Line2D([0], [0], color="#c8a27a", lw=3.6),
        Line2D([0], [0], color="#4d4d4d", lw=1.2, ls=":"),
    ])
    model_labels.extend([
        r"Standard model: flat $\Lambda$CDM + radiation",
        r"de Sitter / inflation: $q=-1$",
    ])

    data_handles = [
        Line2D([0], [0], marker="o", color="#2b8cbe", lw=0, markersize=6),
        Line2D([0], [0], marker="D", color="#6a3d9a", lw=0, markersize=7),
        Line2D([0], [0], marker="s", color="#1b9e77", lw=0, markersize=7),
        Line2D([0], [0], marker="o", color="#f16913", lw=1.2, markersize=7),
        Line2D([0], [0], marker="^", color="#fd8d3c", lw=1.2, markersize=7),
        Line2D([0], [0], color="#6a3d9a", lw=1.8, ls=(0, (5, 3))),
        Line2D([0], [0], color="#1b9e77", lw=1.8, ls=(0, (5, 3))),
    ]
    data_labels = [
        r"Cosmic chronometers $H(z)$ data",
        r"Observed $q_0$ (CC+Pantheon+SH0ES+BAO; Myrzakulov et al. 2025)",
        r"Observed $q_0$ (FRB+SNe cosmography; Gao et al. 2024)",
        r"Obs. $z_t$: CC+Pantheon+SH0ES+BAO (2025)",
        r"Obs. $z_t$: latest $H(z)$ meta-analysis (2025)",
        r"Tangent from $q_0$ (Myrzakulov et al. 2025)",
        r"Tangent from $q_0$ (Gao et al. 2024)",
    ]

    gapp_handles = [
        Line2D([0], [0], color="#8f8f8f", lw=4.0, alpha=0.72),
        Patch(facecolor="#8e8e8e", alpha=0.15, edgecolor="none"),
        Patch(facecolor="#b5b5b5", alpha=0.16, edgecolor="none"),
        Patch(facecolor="#d7d7d7", alpha=0.17, edgecolor="none"),
        Patch(facecolor="#db7272", alpha=0.08, edgecolor="none"),
    ]
    gapp_labels = [r"GaPP median", r"GaPP $1\sigma$", r"GaPP $2\sigma$", r"GaPP $3\sigma$", r"GaPP extrapolation"]

    leg1 = fig.legend(model_handles, model_labels, loc="lower left", bbox_to_anchor=(0.15, 0.01), title=r"$\bf{Models}$", frameon=False, handlelength=2.8, handletextpad=0.8, ncol=1)
    fig.add_artist(leg1)
    leg2 = fig.legend(data_handles, data_labels, loc="lower center", bbox_to_anchor=(0.555, 0.01), title=r"$\bf{Data}$", frameon=False, handlelength=2.0, handletextpad=0.8, ncol=1)
    fig.add_artist(leg2)
    fig.legend(gapp_handles, gapp_labels, loc="lower right", bbox_to_anchor=(0.84, 0.01), title=r"$\bf{GaPP\ CC{+}Pantheon{+}}$", frameon=False, handlelength=2.0, handletextpad=0.8, ncol=1)


def main():
    t0 = 3.0
    z_unproj = np.linspace(-0.5, 2.0, 1800)
    z_unproj_model = np.linspace(-0.5, 2.0, 1200)
    z_proj = np.linspace(-0.5, 2.0, 1700)
    z_proj_model = np.linspace(-0.5, 2.0, 1100)
    z_tan = np.linspace(0.0, 0.8, 300)
    z_work = np.linspace(-0.85, 2.15, 2500)
    chis_deg = [0, 15, 30, 45, 60, 85]
    chis = [np.deg2rad(c) for c in chis_deg]
    hip_colors = ["#d0e1f2", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    gapp = reconstruct_gapp_cc_pantheon(zmin=-0.5, zmax=2.36, nstar=320)

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10.6, "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, axs = plt.subplots(2, 2, figsize=(15.2, 9.6), dpi=180, gridspec_kw={"height_ratios": [3.1, 1.35], "hspace": 0.09, "wspace": 0.07}, sharex=False)
    (ax_ul, ax_ur), (ax_ll, ax_lr) = axs
    fig.patch.set_facecolor("white")

    for chi_deg, chi, color in zip(chis_deg, chis, hip_colors):
        z_data, q_data, e_data = build_centered_history(chi, t0)
        q_curve = interp_curve(z_unproj, z_data, q_data)
        e_curve = interp_curve(z_unproj_model, z_data, e_data)
        label = rf"Hippopede: $\chi={chi_deg}^\circ$"
        if chi_deg == 0:
            label += r" (hyperconical)"
        ax_ul.plot(z_unproj, q_curve, color=color, lw=2.4, label=label, zorder=3)
        ax_ll.plot(z_unproj_model, e_curve, color=color, lw=1.9, alpha=0.95, label="_nolegend_", zorder=3)
        zmax = np.nanmax(z_data)
        if np.isfinite(zmax) and zmax < z_unproj.max():
            ax_ul.axvline(zmax, color=color, lw=0.85, ls=":", alpha=0.40)

    add_common_top(ax_ul, z_unproj, -0.5, 2.0, gapp, right_side=False)
    add_common_bottom(ax_ll, z_unproj_model, z_tan, -0.5, 2.0, gapp, right_side=False)

    monjo = MonjoProjectedHyperconical(alpha=0.325, k=1.0)
    e_hyp_proj_work, _ = monjo.e_and_q(z_work)
    z_map_work = e_hyp_proj_work - 1.0
    for chi_deg, chi, color in zip(chis_deg, chis, hip_colors):
        z_data, _, e_data = build_centered_history(chi, t0)
        e_proj_work, q_proj_work = project_curve(z_work, z_data, e_data, z_map_work)
        e_proj = np.interp(z_proj, z_work, e_proj_work, left=np.nan, right=np.nan)
        q_proj = np.interp(z_proj, z_work, q_proj_work, left=np.nan, right=np.nan)
        ax_ur.plot(z_proj, q_proj, color=color, lw=2.4, zorder=3)
        ax_lr.plot(z_proj_model, np.interp(z_proj_model, z_proj, e_proj, left=np.nan, right=np.nan), color=color, lw=1.9, alpha=0.95, label="_nolegend_", zorder=3)

    add_common_top(ax_ur, z_proj, -0.5, 2.0, gapp, right_side=True)
    add_common_bottom(ax_lr, z_proj_model, z_tan, -0.5, 2.0, gapp, right_side=True)

    fig.text(0.078, 0.965, r"(a)", ha="left", va="center", fontsize=14)
    fig.text(0.520, 0.965, r"(b)", ha="left", va="center", fontsize=14)

    make_legends(fig)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.30)

    png_path = OUTDIR / "hippopede_qz_centered_t0_3p0_double_panel_with_gapp_alpha0325.png"
    pdf_path = OUTDIR / "hippopede_qz_centered_t0_3p0_double_panel_with_gapp_alpha0325.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
