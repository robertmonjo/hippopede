from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hz_background_models as hz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAPP_ROOT = Path(__file__).resolve().parent / "vendor_gapp"
GAPP_COV_ROOT = GAPP_ROOT / "covfunctions"
for extra in (str(GAPP_ROOT), str(GAPP_COV_ROOT)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import dgp  # type: ignore  # noqa: E402
import covariance  # type: ignore  # noqa: E402

DEFAULT_PNG = PROJECT_ROOT / "figures" / "fig_qz_mukherjee2021_gapp.png"
DEFAULT_PDF = PROJECT_ROOT / "figures" / "fig_qz_mukherjee2021_gapp.pdf"
DEFAULT_CSV = PROJECT_ROOT / "data" / "hz_background" / "qz_mukherjee2021_gapp_summary.csv"
PANTHEON_PLUS_PATH = Path(r"C:\Users\rober\OneDrive\Documents\Codex_Portatil\tmp\PantheonPlus_DataRelease\Pantheon+_Data\4_DISTANCES_AND_COVAR\Pantheon+SH0ES.dat")
C_KM_S = 299792.458
H0_LOCAL = 73.2


@dataclass
class GappResult:
    name: str
    z: np.ndarray
    h: np.ndarray
    h_sigma: np.ndarray
    dh: np.ndarray
    dh_sigma: np.ndarray
    q: np.ndarray
    q_sigma: np.ndarray
    theta: np.ndarray
    q0: float
    q0_sigma: float
    zt: float | None
    n_points: int


def load_dataset(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cc_points = [
        p
        for p in hz.load_points(hz.DEFAULT_DATA)
        if p.source_kind == "cosmic_chronometer"
    ]
    bao_points = hz.load_points(hz.DEFAULT_BAO_DATA)
    if kind == "CC":
        points = cc_points
    elif kind == "CC+BAO":
        points = cc_points + bao_points
    else:
        raise ValueError(kind)
    z, h, sigma = hz.combined_arrays(points)
    order = np.argsort(z)
    return z[order], h[order], sigma[order]


def load_pantheon_plus_binned(nbins: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.genfromtxt(PANTHEON_PLUS_PATH, names=True, dtype=None, encoding="utf-8")
    z = np.asarray(raw["zHD"], dtype=float)
    mu = np.asarray(raw["MU_SH0ES"], dtype=float)
    mu_err = np.asarray(raw["MU_SH0ES_ERR_DIAG"], dtype=float)
    valid = np.isfinite(z) & np.isfinite(mu) & np.isfinite(mu_err) & (z > 0.0) & (mu_err > 0.0)
    z = z[valid]
    mu = mu[valid]
    mu_err = mu_err[valid]

    d_l_mpc = 10.0 ** ((mu - 25.0) / 5.0)
    d_proxy = (H0_LOCAL / C_KM_S) * d_l_mpc / (1.0 + z)
    sigma_d = d_proxy * (np.log(10.0) / 5.0) * mu_err

    order = np.argsort(z)
    z = z[order]
    d_proxy = d_proxy[order]
    sigma_d = sigma_d[order]

    bins = np.array_split(np.arange(len(z)), nbins)
    z_bin = []
    d_bin = []
    s_bin = []
    for idx in bins:
        if len(idx) == 0:
            continue
        w = 1.0 / np.maximum(sigma_d[idx], 1.0e-12) ** 2
        z_mean = float(np.average(z[idx], weights=w))
        d_mean = float(np.average(d_proxy[idx], weights=w))
        stat = float(np.sqrt(1.0 / np.sum(w)))
        scatter = float(np.sqrt(np.average((d_proxy[idx] - d_mean) ** 2, weights=w)))
        sigma = max(stat, scatter / np.sqrt(len(idx)))
        z_bin.append(z_mean)
        d_bin.append(d_mean)
        s_bin.append(sigma)
    return np.asarray(z_bin), np.asarray(d_bin), np.asarray(s_bin)


def transition_redshift(z: np.ndarray, q: np.ndarray) -> float | None:
    changes = np.where(np.signbit(q[:-1]) != np.signbit(q[1:]))[0]
    if len(changes) == 0:
        return None
    i = int(changes[0])
    z0, z1 = z[i], z[i + 1]
    q0, q1 = q[i], q[i + 1]
    if q1 == q0:
        return float(0.5 * (z0 + z1))
    return float(z0 - q0 * (z1 - z0) / (q1 - q0))


def reconstruct(kind: str, zmax: float, nstar: int) -> GappResult:
    z_data, h_data, sigma_data = load_dataset(kind)
    gp_obj = dgp.DGaussianProcess(
        z_data,
        h_data,
        sigma_data,
        covfunction=covariance.SquaredExponential,
        cXstar=(-0.5, zmax, nstar),
    )
    rec, theta = gp_obj.gp(thetatrain="True")
    drec, _ = gp_obj.dgp(thetatrain="False")

    z = rec[:, 0]
    h = rec[:, 1]
    h_sigma = rec[:, 2]
    dh = drec[:, 1]
    dh_sigma = drec[:, 2]

    q = -1.0 + (1.0 + z) * dh / h
    q_sigma = np.sqrt(((1.0 + z) * dh_sigma / h) ** 2 + (((1.0 + z) * dh * h_sigma) / (h**2)) ** 2)

    return GappResult(
        name=kind,
        z=z,
        h=h,
        h_sigma=h_sigma,
        dh=dh,
        dh_sigma=dh_sigma,
        q=q,
        q_sigma=q_sigma,
        theta=np.asarray(theta),
        q0=float(np.interp(0.0, z, q)),
        q0_sigma=float(np.interp(0.0, z, q_sigma)),
        zt=transition_redshift(z[z >= 0.0], q[z >= 0.0]),
        n_points=len(z_data),
    )


def reconstruct_from_distance(kind: str, zmax: float, nstar: int, dX=None, dY=None, dSigma=None) -> GappResult:
    z_data, d_data, sigma_data = load_pantheon_plus_binned()
    gp_obj = dgp.DGaussianProcess(
        z_data,
        d_data,
        sigma_data,
        covfunction=covariance.SquaredExponential,
        dX=dX,
        dY=dY,
        dSigma=dSigma,
        cXstar=(-0.5, zmax, nstar),
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

    q = -1.0 - (1.0 + z) * d2d / dd
    q_sigma = np.sqrt(((1.0 + z) * d2d_sigma / dd) ** 2 + (((1.0 + z) * d2d * dd_sigma) / (dd**2)) ** 2)

    n_points = len(z_data) + (0 if dX is None else len(np.atleast_1d(dX)))
    return GappResult(
        name=kind,
        z=z,
        h=d,
        h_sigma=d_sigma,
        dh=dd,
        dh_sigma=dd_sigma,
        q=q,
        q_sigma=q_sigma,
        theta=np.asarray(theta),
        q0=float(np.interp(0.0, z, q)),
        q0_sigma=float(np.interp(0.0, z, q_sigma)),
        zt=transition_redshift(z[z >= 0.0], q[z >= 0.0]),
        n_points=n_points,
    )


def add_panel(ax: plt.Axes, result: GappResult, lcdm_q: np.ndarray, title: str) -> None:
    core = (result.z >= 0.0) & (result.z <= 2.0)
    extra_left = (result.z >= -0.5) & (result.z < 0.0)
    extra_right = (result.z > 2.0) & (result.z <= 2.5)

    ax.fill_between(result.z[core], (result.q - 3.0 * result.q_sigma)[core], (result.q + 3.0 * result.q_sigma)[core], color="#d9d9d9", alpha=0.45, linewidth=0.0, zorder=1)
    ax.fill_between(result.z[core], (result.q - 2.0 * result.q_sigma)[core], (result.q + 2.0 * result.q_sigma)[core], color="#b0b0b0", alpha=0.55, linewidth=0.0, zorder=2)
    ax.fill_between(result.z[core], (result.q - 1.0 * result.q_sigma)[core], (result.q + 1.0 * result.q_sigma)[core], color="#7f7f7f", alpha=0.65, linewidth=0.0, zorder=3)
    ax.plot(result.z[core], result.q[core], color="black", lw=1.7, zorder=4)

    for extra in (extra_left, extra_right):
        ax.fill_between(result.z[extra], (result.q - 3.0 * result.q_sigma)[extra], (result.q + 3.0 * result.q_sigma)[extra], color="#efb0b0", alpha=0.20, linewidth=0.0, zorder=0.9)
        ax.fill_between(result.z[extra], (result.q - 2.0 * result.q_sigma)[extra], (result.q + 2.0 * result.q_sigma)[extra], color="#e59292", alpha=0.22, linewidth=0.0, zorder=1.0)
        ax.fill_between(result.z[extra], (result.q - 1.0 * result.q_sigma)[extra], (result.q + 1.0 * result.q_sigma)[extra], color="#db7272", alpha=0.24, linewidth=0.0, zorder=1.1)
        ax.plot(result.z[extra], result.q[extra], color="#c96a6a", lw=1.2, alpha=0.65, zorder=3.2)

    ax.plot(result.z, lcdm_q, color="black", ls="--", lw=1.2, alpha=0.8, zorder=4)
    ax.axhline(0.0, color="#555555", ls=":", lw=1.0, alpha=0.9, zorder=0)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.55, 1.35)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.5, alpha=0.45)
    ax.grid(True, which="minor", color="#f1f1f1", linewidth=0.4, alpha=0.35)
    ax.minorticks_on()
    ax.text(
        0.04,
        0.06,
        rf"$q_0={result.q0:.2f}\pm{result.q0_sigma:.2f}$",
        transform=ax.transAxes,
        fontsize=9.5,
        ha="left",
        va="bottom",
        color="#222222",
    )
    if result.zt is not None:
        ax.text(
            0.04,
            0.15,
            rf"$z_t={result.zt:.2f}$",
            transform=ax.transAxes,
            fontsize=9.5,
            ha="left",
            va="bottom",
            color="#222222",
        )


def save_summary(results: list[GappResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "n_points", "theta0", "theta1", "q0", "q0_sigma", "zt"])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.n_points,
                    float(result.theta[0]),
                    float(result.theta[1]),
                    result.q0,
                    result.q0_sigma,
                    "" if result.zt is None else result.zt,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct q(z) with the original GaPP package logic using local H(z) datasets.")
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--zmax", type=float, default=2.36)
    parser.add_argument("--nstar", type=int, default=220)
    args = parser.parse_args()

    z_grid = np.linspace(0.0, args.zmax, args.nstar)
    lcdm_q = 1.5 * hz.OMEGA_M_FID * (1.0 + z_grid) ** 3 / (
        hz.OMEGA_M_FID * (1.0 + z_grid) ** 3 + (1.0 - hz.OMEGA_M_FID)
    ) - 1.0

    cc = reconstruct("CC", args.zmax, args.nstar)
    ccbao = reconstruct("CC+BAO", args.zmax, args.nstar)

    cc_z, cc_h, cc_sigma = load_dataset("CC")
    bao_z, bao_h, bao_sigma = load_dataset("CC+BAO")
    dX_cc = cc_z
    dY_cc = H0_LOCAL / cc_h
    dSigma_cc = H0_LOCAL * cc_sigma / (cc_h**2)
    dX_ccbao = bao_z
    dY_ccbao = H0_LOCAL / bao_h
    dSigma_ccbao = H0_LOCAL * bao_sigma / (bao_h**2)

    pantheon = reconstruct_from_distance("Pantheon+", args.zmax, args.nstar)
    ccp = reconstruct_from_distance("CC+Pantheon+", args.zmax, args.nstar, dX=dX_cc, dY=dY_cc, dSigma=dSigma_cc)
    ccbaop = reconstruct_from_distance("CC+BAO+Pantheon+", args.zmax, args.nstar, dX=dX_ccbao, dY=dY_ccbao, dSigma=dSigma_ccbao)

    results = [cc, ccp, ccbao, ccbaop]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.9), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    flat_axes = axes.ravel()
    add_panel(flat_axes[0], results[0], lcdm_q, "CC")
    add_panel(flat_axes[1], results[1], lcdm_q, "CC + Pantheon+")
    add_panel(flat_axes[2], results[2], lcdm_q, "CC + BAO")
    add_panel(flat_axes[3], results[3], lcdm_q, "CC + BAO + Pantheon+")
    axes[0, 0].set_ylabel(r"Deceleration parameter $q(z)$")
    axes[1, 0].set_ylabel(r"Deceleration parameter $q(z)$")
    axes[1, 0].set_xlabel(r"Redshift $z$")
    axes[1, 1].set_xlabel(r"Redshift $z$")
    fig.suptitle(r"GaPP reconstruction of $q(z)$ from local $H(z)$ datasets", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))

    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(args.pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    save_summary(results, args.csv)

    print("GaPP q(z) reconstruction summary:")
    for result in results:
        print(
            f"  {result.name}: theta={result.theta}, q0={result.q0:.3f} +- {result.q0_sigma:.3f}, "
            + (f"zt={result.zt:.3f}" if result.zt is not None else "zt=undetermined")
        )


if __name__ == "__main__":
    main()
