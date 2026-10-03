"""Quaia (G<20.0) number-count dipole in photometric-redshift slices.

Data: Quaia, Storey-Fisher et al. 2024, ApJ 964, 69; Zenodo 10.5281/zenodo.10403370
(files quaia_G20.0.fits, quaia_G20.5.fits, selection_function_NSIDE64_G20.0.fits in ./data).

Estimator: Poisson maximum likelihood on HEALPix NSIDE=64 pixels (equatorial RING, the
frame of the Quaia selection-function map), model
    lambda_p = S_p * N0 * (1 + d.n_p [+ sum_k q_k Y2_k(n_p)]),
n_p = pixel-centre unit vector in Galactic coordinates. S_p is the Quaia selection function.
Mask: Galactic latitude of pixel centre |b| > 30 deg and S_p > S_MIN.
Errors: inverse Fisher matrix, and 100 isotropic Poisson mocks (lambda_p = S_p * N0_fit).
"""
import json
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy_healpix import HEALPix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NSIDE = 64
BCUT = 30.0
S_MIN = 0.3
NMOCK = 100
BETA = 369.82 / 299792.458
L_CMB, B_CMB = 264.0, 48.3
ALPHA_FIXED = 1.26
SLICES = [(0.1, 0.8), (0.8, 1.3), (1.3, 1.8), (1.8, 2.5), (2.5, 4.0)]
RNG = np.random.default_rng(20260928)

# Gaia DR3 photometric zero points (Riello et al. 2021, A&A 649, A3, Table 3): Vega and AB,
# and approximate pivot wavelengths (nm) used only for the colour-based spectral index.
ZP_VEGA = {"BP": 25.3385, "RP": 24.7479}
ZP_AB = {"BP": 25.3540, "RP": 25.1040}
LAM_BP, LAM_RP = 511.0, 777.0


def unitvec(lon_deg, lat_deg):
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def vec2lb(v):
    v = np.asarray(v) / np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])) % 360.0, np.degrees(np.arcsin(v[2]))


def angle(v1, v2):
    c = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def design(n, quad):
    cols = [np.ones(len(n)), n[:, 0], n[:, 1], n[:, 2]]
    if quad:  # real traceless quadrupole basis
        x, y, z = n.T
        cols += [x * y, x * z, y * z, x**2 - y**2, 3 * z**2 - 1]
    return np.stack(cols, axis=1)


def poisson_fit(N, S, Y):
    """Maximise sum N log(lambda) - lambda, lambda = S * (Y @ c). Newton iterations."""
    c = np.linalg.lstsq(Y * S[:, None], N.astype(float), rcond=None)[0]
    for _ in range(50):
        lam = S * (Y @ c)
        if np.any(lam <= 0):
            raise RuntimeError("non-positive intensity")
        g = (Y * (S * (N / lam - 1.0))[:, None]).sum(0)
        H = -(Y * (N * S**2 / lam**2)[:, None]).T @ Y
        step = np.linalg.solve(H, g)
        c = c - step
        if np.max(np.abs(step)) < 1e-12 * max(1.0, abs(c[0])):
            break
    lam = S * (Y @ c)
    F = (Y * (S**2 / lam)[:, None]).T @ Y  # Fisher matrix
    cov = np.linalg.inv(F)
    d = c[1:4] / c[0]
    J = np.zeros((3, len(c)))  # Jacobian of d wrt c
    J[:, 0] = -c[1:4] / c[0] ** 2
    J[:, 1:4] = np.eye(3) / c[0]
    return c, d, J @ cov @ J.T


def main():
    cat = fits.getdata(os.path.join(DATA, "quaia_G20.0.fits"))
    cat5 = fits.getdata(os.path.join(DATA, "quaia_G20.5.fits"))
    S_all = fits.getdata(os.path.join(DATA, "selection_function_NSIDE64_G20.0.fits"))["T"].ravel().astype(float)
    hp = HEALPix(nside=NSIDE, order="ring")  # equatorial, like the selection map
    assert S_all.size == hp.npix

    ra_c, dec_c = hp.healpix_to_lonlat(np.arange(hp.npix))
    gal = SkyCoord(ra=ra_c.to(u.deg), dec=dec_c.to(u.deg), frame="icrs").galactic
    n_all = unitvec(gal.l.deg, gal.b.deg)
    good = (np.abs(gal.b.deg) > BCUT) & (S_all > S_MIN)
    n_pix, S_pix = n_all[good], S_all[good]
    print(f"unmasked pixels: {good.sum()} / {hp.npix}  (f_sky = {good.mean():.3f})")

    pix_src = hp.lonlat_to_healpix(cat["ra"] * u.deg, cat["dec"] * u.deg)
    b_src = cat["b"]
    z = cat["redshift_quaia"]
    G = cat["phot_g_mean_mag"]
    bp, rp = cat["phot_bp_mean_mag"], cat["phot_rp_mean_mag"]
    in_mask = good[pix_src]
    b5 = cat5["b"]
    ra5, dec5 = cat5["ra"], cat5["dec"]
    pix5 = hp.lonlat_to_healpix(ra5 * u.deg, dec5 * u.deg)
    in5 = good[pix5]
    z5, G5 = cat5["redshift_quaia"], cat5["phot_g_mean_mag"]

    v_cmb = unitvec(L_CMB, B_CMB)
    slices = [("full", 0.0, 10.0)] + [(f"{a}-{b}", a, b) for a, b in SLICES]
    results = []
    for name, zlo, zhi in slices:
        sel = in_mask & (z >= zlo) & (z < zhi)
        N = np.bincount(pix_src[sel], minlength=hp.npix)[good]
        row = {"slice": name, "z_range": [zlo, zhi] if name != "full" else "all",
               "N": int(sel.sum()), "median_z": float(np.median(z[sel])),
               "median_z_err": float(np.median(cat["redshift_quaia_err"][sel]))}

        # --- counts slope x = d log10 N(<G) / d(0.4 G) = (dlnN/dG)/(0.4 ln10), centred at G=20.0
        s5 = in5 & (z5 >= zlo) & (z5 < zhi)
        n_hi, n_lo = np.sum(s5 & (G5 < 20.1)), np.sum(s5 & (G5 < 19.9))
        x = (np.log(n_hi) - np.log(n_lo)) / 0.2 / (0.4 * np.log(10))
        # one-sided check from G<20.0 catalogue alone: 19.8 -> 20.0
        n_a, n_b = np.sum(sel & (G < 20.0)), np.sum(sel & (G < 19.8))
        x_one = (np.log(n_a) - np.log(n_b)) / 0.2 / (0.4 * np.log(10))
        # Poisson error on x (counts in shell vs below)
        x_err = np.sqrt(1.0 / (n_hi - n_lo) + 1.0 / n_lo) / 0.2 / (0.4 * np.log(10))

        # --- colour spectral index, S_nu ~ nu^-alpha, sources with 19.8 < G < 20.0
        edge = sel & (G > 19.8) & np.isfinite(bp) & np.isfinite(rp)
        col_ab = (bp[edge] - rp[edge]) + (ZP_AB["BP"] - ZP_VEGA["BP"]) - (ZP_AB["RP"] - ZP_VEGA["RP"])
        alpha_i = col_ab / (2.5 * np.log10(LAM_RP / LAM_BP))
        alpha_col = float(np.median(alpha_i))

        row.update({"x": float(x), "x_err_poisson": float(x_err), "x_onesided_19.8_20.0": float(x_one),
                    "alpha_fixed": ALPHA_FIXED, "alpha_colour_median_edge": alpha_col,
                    "D_kin": float((2 + x * (1 + ALPHA_FIXED)) * BETA),
                    "D_kin_alpha_colour": float((2 + x * (1 + alpha_col)) * BETA)})

        for quad in (False, True):
            Y = design(n_pix, quad)
            c, d, cov = poisson_fit(N, S_pix, Y)
            D = np.linalg.norm(d)
            l, b = vec2lb(d)
            sig_fit_D = float(np.sqrt(d @ cov @ d) / D)
            sig_fit_comp = float(np.sqrt(np.trace(cov) / 3))
            mocks = np.empty((NMOCK, 3))
            lam0 = S_pix * c[0]
            for i in range(NMOCK):
                mocks[i] = poisson_fit(RNG.poisson(lam0), S_pix, Y)[1]
            sig_comp_m = float(np.sqrt(np.mean(np.sum(mocks**2, 1)) / 3))
            Dm = np.linalg.norm(d[None, :] + mocks, axis=1)
            angm = np.array([angle(d + m, d) for m in mocks])
            dkin_vec = row["D_kin"] * v_cmb
            ex = d - dkin_vec
            exm = np.linalg.norm(ex[None, :] + mocks, axis=1)
            p_iso = float(np.mean(np.linalg.norm(mocks, axis=1) >= D))
            key = "dipquad" if quad else "dip"
            ex_c = d - row["D_kin_alpha_colour"] * v_cmb
            exm_c = np.linalg.norm(ex_c[None, :] + mocks, axis=1)
            radec = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
            row[key] = {"D": float(D), "l": float(l), "b": float(b),
                        "sigma_fit_D": sig_fit_D, "sigma_fit_per_component": sig_fit_comp,
                        "sigma_mocks_D": float(np.std(Dm)), "sigma_mocks_per_component": sig_comp_m,
                        "mock_mean_noise_amplitude": float(np.mean(np.linalg.norm(mocks, axis=1))),
                        "dir_err_mocks_deg_68": float(np.percentile(angm, 68)),
                        "angle_to_CMB_deg": float(angle(d, v_cmb)),
                        "p_iso_mocks": p_iso,
                        "D_over_Dkin": float(D / row["D_kin"]),
                        "excess_vec": ex.tolist(), "excess_amp": float(np.linalg.norm(ex)),
                        "excess_sigma_mocks": float(np.std(exm)),
                        "excess_sigma_fit": float(np.sqrt(ex @ cov @ ex) / np.linalg.norm(ex)),
                        "excess_amp_alpha_colour": float(np.linalg.norm(ex_c)),
                        "excess_sigma_mocks_alpha_colour": float(np.std(exm_c)),
                        "D_debiased": float(np.sqrt(max(D**2 - 3 * sig_comp_m**2, 0.0))),
                        "d_par_CMB": float(d @ v_cmb), "d_par_CMB_sigma_mocks": float(np.std(mocks @ v_cmb)),
                        "ra": float(radec.ra.deg), "dec": float(radec.dec.deg)}
        # no-selection-function variant (S=1), dipole only, fit errors only
        c, d, cov = poisson_fit(N, np.ones_like(S_pix), design(n_pix, False))
        l, b = vec2lb(d)
        row["dip_noSF"] = {"D": float(np.linalg.norm(d)), "l": float(l), "b": float(b),
                           "sigma_fit_D": float(np.sqrt(d @ cov @ d) / np.linalg.norm(d)),
                           "angle_to_CMB_deg": float(angle(d, v_cmb))}
        results.append(row)
        r = row["dip"]
        print(f"{name:8s} N={row['N']:7d} x={x:.3f} (1-sided {x_one:.3f}) a_col={alpha_col:.2f} "
              f"Dkin={row['D_kin']*1e2:.3f}e-2 | D={r['D']*1e2:.3f}+-{r['sigma_fit_D']*1e2:.3f}(fit)"
              f"+-{r['sigma_mocks_D']*1e2:.3f}(mock) (l,b)=({r['l']:.0f},{r['b']:.0f}) "
              f"ang={r['angle_to_CMB_deg']:.0f} RADec=({r['ra']:.0f},{r['dec']:.0f}) dpar={r['d_par_CMB']*1e2:.3f}+-{r['d_par_CMB_sigma_mocks']*1e2:.3f} Ddeb={r['D_debiased']*1e2:.3f} excC={r['excess_amp_alpha_colour']*1e2:.3f} exc={r['excess_amp']*1e2:.3f}+-{r['excess_sigma_mocks']*1e2:.3f} "
              f"| quad: D={row['dipquad']['D']*1e2:.3f}+-{row['dipquad']['sigma_mocks_D']*1e2:.3f} "
              f"ang={row['dipquad']['angle_to_CMB_deg']:.0f} | noSF D={row['dip_noSF']['D']*1e2:.3f}")

    # --- trend with redshift (disjoint slices; shot noise only)
    trend = {}
    zs = np.array([r["median_z"] for r in results[1:]])
    for key in ("dip", "dipquad"):
        for q, sq in (("D", "sigma_mocks_D"), ("excess_amp", "excess_sigma_mocks"),
                      ("d_par_CMB", "d_par_CMB_sigma_mocks")):
            y = np.array([r[key][q] for r in results[1:]])
            s = np.array([r[key][sq] for r in results[1:]])
            w = 1 / s**2
            A = np.stack([np.ones_like(zs), zs], 1)
            C = np.linalg.inv((A * w[:, None]).T @ A)
            p = C @ ((A * w[:, None]).T @ y)
            mean = np.sum(w * y) / np.sum(w)
            chi2_const = float(np.sum(w * (y - mean) ** 2))
            trend[f"{key}_{q}"] = {"slope_per_unit_z": float(p[1]), "slope_sigma": float(np.sqrt(C[1, 1])),
                                   "slope_signif_sigma": float(p[1] / np.sqrt(C[1, 1])),
                                   "chi2_constant": chi2_const, "dof": len(y) - 1}
            print(key, q, trend[f"{key}_{q}"])

    # --- full sample with the stricter |b| > 40 deg mask (comparison with Mittal et al. 2024)
    good40 = (np.abs(gal.b.deg) > 40.0) & (S_all > S_MIN)
    N40 = np.bincount(pix_src[good40[pix_src]], minlength=hp.npix)[good40]
    full40 = {"f_sky": float(good40.mean())}
    for key, S_use in (("dip", S_all[good40]), ("dip_noSF", np.ones(int(good40.sum())))):
        c, d, cov = poisson_fit(N40, S_use, design(n_all[good40], False))
        l, b = vec2lb(d)
        full40[key] = {"D": float(np.linalg.norm(d)), "l": float(l), "b": float(b),
                       "sigma_fit_D": float(np.sqrt(d @ cov @ d) / np.linalg.norm(d)),
                       "angle_to_CMB_deg": float(angle(d, v_cmb))}
    print(f"full sample, |b|>40: f_sky={full40['f_sky']:.3f}  D={full40['dip']['D']*1e2:.3f}+-{full40['dip']['sigma_fit_D']*1e2:.3f}e-2 "
          f"(no SF {full40['dip_noSF']['D']*1e2:.3f}e-2)")

    out = {"data": "Quaia G<20.0 (Zenodo 10403370)", "nside": NSIDE, "mask": f"|b|>{BCUT} (pixel centre) and S>{S_MIN}",
           "f_sky": float(good.mean()), "full_sample_b40": full40,
           "selection_function": "S_p multiplies model intensity (Poisson likelihood)",
           "beta": BETA, "cmb_dir_lb": [L_CMB, B_CMB], "n_mocks": NMOCK,
           "x_definition": "x = [ln N(<20.1) - ln N(<19.9)] / (0.2 * 0.4 ln10), from G<20.5 catalogue, same mask and z slice",
           "alpha_colour_definition": "median over 19.8<G<20.0 of (BP-RP)_AB / (2.5 log10(777/511)), Riello+2021 AB-Vega offsets",
           "slices": results, "trend": trend}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results.json")


if __name__ == "__main__":
    main()
