"""CatWISE AGN number-count dipole in W1 magnitude and W1-W2 colour subsamples.

Data: CatWISE2020 AGN sample of Secrest et al. 2021 (ApJL 908, L51), public release
"A Test of the Cosmological Principle with Quasars", Zenodo 10.5281/zenodo.8303800 (concept DOI
10.5281/zenodo.4431089): catwise_agns_masked_final_w1lt16p5_alpha.fits (1,604,392 sources with
9 < W1 < 16.5, extinction-corrected W1, W2, W1-W2 >= 0.8, source-level mask of 292 regions and |b| > 30 deg
already applied, spectral index alpha_W1 from W1-W2) and MASKS_exclude_master_final.fits.  Both files are
read from ./data/catwise (download them there; they are not part of the repository).

Pixel mask (as in the CatWISE_Dipole_Results notebook of the release): circular and elliptical masks and
|b| > 30 deg on a HEALPix NSIDE=1024 map (pixel centres), degraded to NSIDE=64 keeping only fully unmasked
pixels (Galactic, RING).

Estimators.
(1) Reproduction of Secrest et al.: density map at NSIDE=64; the linear trend of the density with |ecliptic
latitude| is fitted on 2 x 20 ecliptic-latitude bins of an NSIDE=1024 mask built in ecliptic coordinates and
subtracted; healpy.remove_dipole on the unmasked pixels; D = |dipole| / monopole.
(2) Main estimator, used for every subsample: Poisson maximum likelihood on the unmasked NSIDE=64 pixels,
    lambda_p = c0 + c.n_p + c4 (|beta_ecl,p| - <|beta_ecl|>),     d = c / c0,
n_p = Galactic unit vector of the pixel centre; the ecliptic term is fitted for every subsample.  Errors:
covariance of d over N_MOCK Poisson mocks of the fitted dipole-free intensity c0 + c4(...).  The Pearson
dispersion chi2/dof of the fit is stored (excess pixel variance from clustering; the mock errors are pure
shot noise).

Subsamples: the full sample W1 <= 16.4 (N = 1,355,352); cumulative limits W1 < 15.0, 15.5, 16.0, 16.4;
disjoint W1 bins with edges 9, 15.4, 15.9, 16.2, 16.4 (about the quartiles); W1-W2 terciles at W1 <= 16.4
(edges 0.8, 0.95, 1.15); and the 4 x 2 grid of the W1 bins and the two W1-W2 halves (split at 1.04).

Kinematic and geometric weights.  For a sample with counts N(<m) (same colour selection and mask), the integral
slope at magnitude m is x(m) = d log10 N(<m) / d(0.4 m), from N(<m +- 0.05).  The spectral index of the
sources at the limit, alpha(m) = mean of -alpha_W1 over |W1 - m| < 0.05 (S_nu ~ nu^-alpha).  For a bin
m_lo <= W1 < m_hi (N_bin = N_hi - N_lo, N_lo = 0 for a cumulative sample) the counts change by
    kinematic (Ellis & Baldwin):  D_kin = beta { 2 + [x_hi (1 + alpha_hi) N_hi - x_lo (1 + alpha_lo) N_lo] / N_bin },
    geometric:  x enters only through (2 - 2x) and the bin behaves as x_eff = (x_hi N_hi - x_lo N_lo) / N_bin,
so each subsample is summarised by K = D_kin / beta, x_eff and alpha_eff = (K - 2)/x_eff - 1.  Below
m = 9 the counts vanish (N_lo = 0 for the brightest bin).

Writes ../json/catwise_cuts_dipoles.json.  Robustness option: --bcut B also removes the pixels with |b| <= B deg
from the subsample fits (the reproduction keeps |b| > 30 deg); the output then gets the suffix _b<B>.

Further options (all off by default; the default output is unchanged):
  --jackknife NJK   delete-one jackknife over sky regions: the unmasked NSIDE=64 pixels are grouped by their
                    NSIDE=NJK parent pixel (Galactic, RING); regions with fewer than half the median number of
                    unmasked pixels are merged into the nearest large region (angular distance of the centres).
                    Every subsample is refitted with each region removed; the dipole vectors d_(r) are stored
                    ("jackknife" -> "d_jk", same region order for all subsamples, so the cross-covariance
                    between subsamples follows from them) together with
                        cov_jk = (R - 1)/R sum_r (d_(r) - <d>) (d_(r) - <d>)^T.
                    The jackknife includes clustering and any large-scale systematic that varies across the
                    regions; the Poisson mocks include shot noise only.  Suffix _jk for NJK = 4, _jk<NJK> otherwise.
  --exclude-gc R    also removes the pixels within R deg of the Galactic centre (0, 0) and anticentre (180, 0).
                    Suffix _gc<R>.
  --ebv-template    adds the per-pixel mean E(B-V) of the catalogue sources (column ebv, all sources with
                    W1 <= 16.4) minus its mean as a further linear template of every fit.  Suffix _ebv.
  --n-mock N        number of Poisson mocks (default 1000).
Suffix order: _jk, _b<B>, _gc<R>, _ebv (e.g. catwise_cuts_dipoles_jk_b40.json).
"""
import argparse
import json
import os
import time

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "catwise")
OUT = os.path.join(HERE, "..", "json", "catwise_cuts_dipoles.json")
CAT = "catwise_agns_masked_final_w1lt16p5_alpha.fits"
MASKS = "MASKS_exclude_master_final.fits"
NSIDE, NSIDE_HI, GALCUT = 64, 1024, 30.0
W1_MAX = 16.4
N_MOCK = 1000
DM = 0.05  # half width of the magnitude step for x(m) and alpha(m)
BETA = 369.82 / 299792.458
L_CMB, B_CMB = 264.0, 48.3
D_SECREST, LB_SECREST = 0.01554, (238.2, 28.8)  # Secrest et al. 2021
D_KIN_SECREST = 0.007
CUMULATIVE = [15.0, 15.5, 16.0, 16.4]
MAG_EDGES = [9.0, 15.4, 15.9, 16.2, 16.4]
COL_EDGES = [0.8, 0.95, 1.15, 9.0]
COL_SPLIT = 1.04
RNG = np.random.default_rng(20261011)


def unitvec(lon_deg, lat_deg):
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def vec2lb(v):
    v = np.asarray(v) / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2])))


def angle(v1, v2):
    c = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ---------------------------------------------------------------- mask (Secrest et al. release notebook)
def offset_by(lon, lat, posang, distance):
    """Point at angular distance `distance` and position angle `posang` from (lon, lat); all in degrees."""
    cos_a, sin_a = np.cos(np.radians(distance)), np.sin(np.radians(distance))
    cos_c, sin_c = np.sin(np.radians(lat)), np.cos(np.radians(lat))
    cos_B, sin_B = np.cos(np.radians(posang)), np.sin(np.radians(posang))
    cos_b = cos_c * cos_a + sin_c * sin_a * cos_B
    A = np.arctan2(sin_a * sin_B * sin_c, cos_a - cos_b * cos_c)
    return (lon + np.degrees(A)) % 360.0, np.degrees(np.arcsin(cos_b))


def in_ellipse(lon, lat, lon0, lat0, semi_major, e, theta):
    """Pixels inside the ellipse of the release code (foci at semi_major * e along position angle theta)."""
    c = semi_major * e
    f1, f2 = offset_by(lon0, lat0, 90 - theta, c), offset_by(lon0, lat0, 270 - theta, c)
    n = unitvec(lon, lat)
    s1 = np.degrees(np.arccos(np.clip(n @ unitvec(*f1), -1, 1)))
    s2 = np.degrees(np.arccos(np.clip(n @ unitvec(*f2), -1, 1)))
    return s1 + s2 <= 2 * semi_major


def make_mask(masks, nside, frame):
    """Masked regions (one-sided) and |latitude of the Galactic plane| > GALCUT on a map in `frame`."""
    mask = np.ones(hp.nside2npix(nside))
    plon, plat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    to = (lambda c: (c.galactic.l.deg, c.galactic.b.deg)) if frame == "galactic" else \
        (lambda c: (c.barycentricmeanecliptic.lon.deg, c.barycentricmeanecliptic.lat.deg))
    cm = masks[(masks["pa"] <= 2) & (masks["radius"] < 15)]
    em = masks[masks["pa"] > 2]
    lon, lat = to(SkyCoord(ra=cm["ra"] * u.deg, dec=cm["dec"] * u.deg, frame="icrs"))
    for lo, la, r in zip(lon, lat, cm["radius"]):
        mask[hp.query_disc(nside, hp.ang2vec(lo, la, lonlat=True), np.radians(r))] = 0
    lon, lat = to(SkyCoord(ra=em["ra"] * u.deg, dec=em["dec"] * u.deg, frame="icrs"))
    for lo, la, r, ba, pa in zip(lon, lat, em["radius"], em["ba"], em["pa"]):
        mask[in_ellipse(plon, plat, lo, la, r, ba, pa)] = 0
    ngp = SkyCoord(l=0 * u.deg, b=90 * u.deg, frame="galactic")
    plo, pla = (0.0, 90.0) if frame == "galactic" else to(ngp)
    v = hp.ang2vec(plo, pla, lonlat=True)
    plane = np.ones_like(mask)
    plane[hp.query_disc(nside, v, np.radians(90 + GALCUT))] = 0
    plane[hp.query_disc(nside, v, np.radians(90 - GALCUT))] = 1
    return mask * plane


# ---------------------------------------------------------------- estimators
def secrest_dipole(l, b, elat, masks):
    """Reproduction of the estimator of Secrest et al. 2021 (release notebook)."""
    hi_gal = make_mask(masks, NSIDE_HI, "galactic")
    lo_mask = hp.ud_grade(hi_gal, nside_out=NSIDE)
    lo_mask = (lo_mask == 1)
    hi_ecl = make_mask(masks, NSIDE_HI, "ecliptic")
    latmap = hp.pix2ang(NSIDE_HI, np.arange(hp.nside2npix(NSIDE_HI)), lonlat=True)[1]
    e1, e2 = np.linspace(0, 90, 21), np.linspace(-90, 0, 21)
    cen, unm, num = [], [], []
    for edges in (e1, e2):
        for a, c in zip(edges[:-1], edges[1:]):
            unm.append(hi_ecl[(latmap > a) & (latmap < c)].sum())
            num.append(np.sum((elat > a) & (elat < c)))
            cen.append(0.5 * (a + c))
    cen, unm, num = map(np.asarray, (cen, unm, num))
    ok = unm != 0
    pfit = np.polyfit(np.abs(cen[ok]), num[ok] / unm[ok], 1, w=np.sqrt(num[ok]))
    f_hi = (60.0 / hp.nside2resol(NSIDE_HI, arcmin=True)) ** 2
    f_lo = (60.0 / hp.nside2resol(NSIDE, arcmin=True)) ** 2
    dens = np.bincount(hp.ang2pix(NSIDE, l, b, lonlat=True), minlength=hp.nside2npix(NSIDE)) * f_lo
    plon, plat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    pel = SkyCoord(l=plon * u.deg, b=plat * u.deg, frame="galactic").barycentricmeanecliptic.lat.deg
    dens = dens - pfit[0] * f_hi * np.abs(pel)
    dens[~lo_mask] = hp.UNSEEN
    _, mono, dip = hp.remove_dipole(dens, fitval=True, bad=hp.UNSEEN)
    D = float(np.linalg.norm(dip) / mono)
    return {"D": D, "lb": vec2lb(dip), "ecl_slope_per_deg3": float(pfit[0] * f_hi),
            "ecl_offset_per_deg2": float(pfit[1] * f_hi), "monopole_per_deg2": float(mono),
            "n_pix": int(lo_mask.sum())}, lo_mask


def poisson_fit(N, Y):
    """Maximise sum N log(lambda) - lambda with lambda = Y @ c (Newton); returns c and its Fisher covariance."""
    c = np.linalg.lstsq(Y, N.astype(float), rcond=None)[0]
    for _ in range(50):
        lam = Y @ c
        g = Y.T @ (N / lam - 1.0)
        H = -(Y * (N / lam**2)[:, None]).T @ Y
        step = np.linalg.solve(H, g)
        c = c - step
        if np.max(np.abs(step)) < 1e-10 * abs(c[0]):
            break
    lam = Y @ c
    return c, np.linalg.inv((Y / lam[:, None]).T @ Y)


def dvec(c):
    return c[1:4] / c[0]


# ---------------------------------------------------------------- subsamples
def counts_below(w1_sel, m):
    return int(np.sum(w1_sel < m))


def edge_quantities(w1_c, alpha_c, m):
    """N(<m), x(m), alpha(m) for the colour-selected sources (W1 values w1_c, spectral indices alpha_c)."""
    if m <= 9.0:
        return 0, 0.0, 0.0
    n_hi, n_lo = counts_below(w1_c, m + DM), counts_below(w1_c, m - DM)
    x = (np.log10(n_hi) - np.log10(n_lo)) / (0.4 * 2 * DM)
    near = np.abs(w1_c - m) < DM
    return counts_below(w1_c, m), float(x), float(np.mean(alpha_c[near]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bcut", type=float, default=GALCUT, help="|b| cut of the subsample fits [deg]")
    ap.add_argument("--jackknife", type=int, default=0, help="NSIDE of the jackknife regions (0: no jackknife)")
    ap.add_argument("--exclude-gc", type=float, default=0.0, help="radius around the Galactic centre/anticentre [deg]")
    ap.add_argument("--ebv-template", action="store_true", help="add a per-pixel mean E(B-V) template")
    ap.add_argument("--n-mock", type=int, default=N_MOCK, help="number of Poisson mocks")
    args = ap.parse_args()
    sfx = ("" if not args.jackknife else "_jk" if args.jackknife == 4 else f"_jk{args.jackknife}") + ("" if args.bcut == GALCUT else f"_b{args.bcut:g}") + \
        (f"_gc{args.exclude_gc:g}" if args.exclude_gc > 0 else "") + ("_ebv" if args.ebv_template else "")
    out_path = OUT.replace(".json", f"{sfx}.json")
    t_start = time.time()
    cat = fits.getdata(os.path.join(DATA, CAT), 1)
    masks = fits.getdata(os.path.join(DATA, MASKS), 1)
    w1, w12, alpha = cat["w1"], cat["w12"], -cat["alpha_W1"]
    l, b = cat["l"], cat["b"]
    radec = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic")
    elat = radec.barycentricmeanecliptic.lat.deg
    full = w1 <= W1_MAX
    print(f"catalogue: {len(w1)} sources; W1 <= {W1_MAX}: {full.sum()}")

    # ---- reproduction
    sec, good = secrest_dipole(l[full], b[full], elat[full], masks)
    v_cmb = unitvec(L_CMB, B_CMB)
    v_sec = unitvec(*LB_SECREST)
    v_exc = D_SECREST * v_sec - D_KIN_SECREST * v_cmb
    sec["angle_to_published_deg"] = angle(unitvec(*sec["lb"]), v_sec)
    print(f"Secrest estimator: D = {sec['D']:.5f} towards (l,b) = ({sec['lb'][0]:.1f}, {sec['lb'][1]:.1f}); "
          f"published {D_SECREST} towards {LB_SECREST}; {sec['n_pix']} pixels; ecliptic slope {sec['ecl_slope_per_deg3']:.4f}/deg^3")

    plon, plat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    good = good & (np.abs(plat) > args.bcut)
    if args.exclude_gc > 0:
        v_all = unitvec(plon, plat)
        for lb0 in ((0.0, 0.0), (180.0, 0.0)):
            good &= np.degrees(np.arccos(np.clip(v_all @ unitvec(*lb0), -1, 1))) > args.exclude_gc
    n_pix = unitvec(plon, plat)[good]
    pel = np.abs(SkyCoord(l=plon * u.deg, b=plat * u.deg, frame="galactic").barycentricmeanecliptic.lat.deg)[good]
    tmpl = pel - pel.mean()
    Y = np.column_stack([np.ones(len(n_pix)), n_pix, tmpl])
    pix_src = hp.ang2pix(NSIDE, l, b, lonlat=True)
    if args.ebv_template:
        npx = hp.nside2npix(NSIDE)
        ebv_pix = np.bincount(pix_src[full], weights=cat["ebv"][full], minlength=npx) / \
            np.maximum(np.bincount(pix_src[full], minlength=npx), 1)
        ebv_t = ebv_pix[good] - ebv_pix[good].mean()
        Y = np.column_stack([Y, ebv_t])
    in_mask = good[pix_src]
    jk = None
    if args.jackknife:
        reg = hp.ang2pix(args.jackknife, plon[good], plat[good], lonlat=True)
        ids, cnt = np.unique(reg, return_counts=True)
        small = cnt < 0.5 * np.median(cnt)
        cen = np.array(hp.pix2vec(args.jackknife, ids)).T
        big = ids[~small]
        for i_s in ids[small]:
            j = np.argmax(cen[~small] @ cen[ids == i_s][0])
            reg[reg == i_s] = big[j]
        regions = np.unique(reg)
        jk = {"nside_regions": args.jackknife, "n_regions": int(len(regions)),
              "region_npix": [int(np.sum(reg == r)) for r in regions], "region_ids": regions.tolist(),
              "merge_rule": "regions with < 0.5 x median unmasked pixels merged into the nearest large region",
              "cov_definition": "(R-1)/R sum_r (d_(r) - <d>)(d_(r) - <d>)^T, R = n_regions"}
        print(f"jackknife: {len(ids)} parent pixels at NSIDE {args.jackknife}, {small.sum()} merged -> "
              f"{len(regions)} regions of {min(jk['region_npix'])}-{max(jk['region_npix'])} pixels")

    def subsample(name, kind, m_lo, m_hi, c_lo, c_hi):
        col = (w12 >= c_lo) & (w12 < c_hi)
        sel = in_mask & col & (w1 >= m_lo) & (w1 < m_hi if m_hi < W1_MAX else w1 <= m_hi)
        N = np.bincount(pix_src[sel], minlength=len(good))[good]
        cc = col & in_mask  # x and alpha from the same mask and colour selection
        n_hi, x_hi, a_hi = edge_quantities(w1[cc], alpha[cc], m_hi)
        n_lo, x_lo, a_lo = edge_quantities(w1[cc], alpha[cc], m_lo)
        n_bin = n_hi - n_lo
        x_eff = (x_hi * n_hi - x_lo * n_lo) / n_bin
        K = 2 + (x_hi * (1 + a_hi) * n_hi - x_lo * (1 + a_lo) * n_lo) / n_bin
        c, cov_c = poisson_fit(N, Y)
        d = dvec(c)
        J = np.zeros((3, Y.shape[1]))
        J[:, 0] = -c[1:4] / c[0] ** 2
        J[:, 1:4] = np.eye(3) / c[0]
        cov_fit = J @ cov_c @ J.T
        lam = Y @ c
        disp = float(np.sum((N - lam) ** 2 / lam) / (len(N) - Y.shape[1]))
        lam0 = c[0] + c[4] * tmpl
        if Y.shape[1] > 5:
            lam0 = lam0 + Y[:, 5:] @ c[5:]
        mocks = np.array([dvec(poisson_fit(RNG.poisson(lam0), Y)[0]) for _ in range(args.n_mock)])
        cov_m = np.cov(mocks.T)
        D = float(np.linalg.norm(d))
        sig_c = float(np.sqrt(np.trace(cov_m) / 3))
        angm = np.array([angle(d + m, d) for m in mocks[:300]])
        # least-squares (Secrest-like, unweighted) on the same design, for comparison
        cl = np.linalg.lstsq(Y, N.astype(float), rcond=None)[0]
        row = {"name": name, "kind": kind, "W1_range": [m_lo, m_hi], "W1W2_range": [c_lo, c_hi],
               "N": int(sel.sum()), "N_check_bin": int(n_bin),
               "x_hi": x_hi, "x_lo": x_lo, "alpha_hi": a_hi, "alpha_lo": a_lo, "N_hi": n_hi, "N_lo": n_lo,
               "x_eff": float(x_eff), "K": float(K), "alpha_eff": float((K - 2) / x_eff - 1),
               "alpha_mean": float(np.mean(alpha[sel])), "W1W2_median": float(np.median(w12[sel])),
               "D_kin_CMB": float(K * BETA),
               "d": d.tolist(), "cov_mocks": cov_m.tolist(), "cov_fit": cov_fit.tolist(),
               "D": D, "lb": vec2lb(d), "sigma_per_component_mocks": sig_c,
               "sigma_D_mocks": float(np.sqrt(d @ cov_m @ d) / D),
               "dir_err_68_deg": float(np.percentile(angm, 68)),
               "D_debiased": float(np.sqrt(max(D**2 - np.trace(cov_m), 0.0))),
               "angle_to_CMB_deg": angle(d, v_cmb), "angle_to_catwise_excess_deg": angle(d, v_exc),
               "d_par_CMB": float(d @ v_cmb), "d_par_excess": float(d @ v_exc / np.linalg.norm(v_exc)),
               "ecl_coeff_rel_per_deg": float(c[4] / c[0]), "pearson_dispersion": disp,
               "lsq": {"D": float(np.linalg.norm(dvec(cl))), "lb": vec2lb(dvec(cl))}}
        if args.ebv_template:
            row["ebv_coeff_rel"] = float(c[5] / c[0])
        if jk is not None:
            d_jk = np.array([dvec(poisson_fit(N[reg != r], Y[reg != r])[0]) for r in regions])
            dd = d_jk - d_jk.mean(0)
            cov_jk = (len(regions) - 1) / len(regions) * dd.T @ dd
            row.update({"d_jk": d_jk.tolist(), "cov_jk": cov_jk.tolist(),
                        "sigma_per_component_jk": float(np.sqrt(np.trace(cov_jk) / 3)),
                        "jk_to_mock_variance_ratio": float(np.trace(cov_jk) / np.trace(cov_m)),
                        "sigma_D_jk": float(np.sqrt(d @ cov_jk @ d) / D)})
            print(f"{'':22s} jackknife sigma/comp {row['sigma_per_component_jk']*1e2:.3f}e-2, "
                  f"var ratio jk/mock {row['jk_to_mock_variance_ratio']:.2f}", flush=True)
        print(f"{name:22s} N={row['N']:8d} x_eff={x_eff:.3f} a_eff={row['alpha_eff']:.3f} K={K:.3f} "
              f"D={D*1e2:.3f}+-{sig_c*1e2:.3f}e-2 (l,b)=({row['lb'][0]:.0f},{row['lb'][1]:.0f}) "
              f"ang_CMB={row['angle_to_CMB_deg']:.0f} ang_exc={row['angle_to_catwise_excess_deg']:.0f} "
              f"disp={disp:.3f} lsq D={row['lsq']['D']*1e2:.3f}", flush=True)
        return row

    rows = [subsample("full_W1<=16.4", "full", 9.0, W1_MAX, 0.8, 9.0)]
    rows += [subsample(f"cum_W1<{m}", "cumulative", 9.0, m, 0.8, 9.0) for m in CUMULATIVE[:-1]]
    rows += [subsample(f"bin_W1_{a}-{c}", "mag_bin", a, c, 0.8, 9.0) for a, c in zip(MAG_EDGES[:-1], MAG_EDGES[1:])]
    rows += [subsample(f"col_W1W2_{a}-{c}", "colour_bin", 9.0, W1_MAX, a, c) for a, c in zip(COL_EDGES[:-1], COL_EDGES[1:])]
    for a, c in zip(MAG_EDGES[:-1], MAG_EDGES[1:]):
        for cl_, ch_, tag in ((0.8, COL_SPLIT, "blue"), (COL_SPLIT, 9.0, "red")):
            rows.append(subsample(f"grid_W1_{a}-{c}_{tag}", "grid", a, c, cl_, ch_))

    out = {"data": "Secrest et al. 2021, Zenodo 10.5281/zenodo.8303800 (concept 10.5281/zenodo.4431089), " + CAT,
           "mask": "release-notebook pixel mask: masks + |b|>30 at NSIDE 1024, fully unmasked NSIDE 64 pixels (Galactic)",
           "n_pix": int(good.sum()), "f_sky": float(good.mean()),
           "estimator": "Poisson ML: lambda = c0 + c.n + c4(|elat| - mean); d = c/c0; errors from Poisson mocks of the dipole-free fit",
           "n_mocks": args.n_mock, "beta_CMB": BETA, "cmb_lb": [L_CMB, B_CMB],
           "catwise_excess_vector_published": v_exc.tolist(), "catwise_excess_lb": vec2lb(v_exc),
           "x_definition": f"x(m) = [log10 N(<m+{DM}) - log10 N(<m-{DM})] / (0.4*{2*DM}), same mask and colour selection",
           "alpha_definition": f"alpha = -alpha_W1 of the release (S_nu ~ nu^-alpha), mean over |W1 - m| < {DM}",
           "K_definition": "D_kin/beta = 2 + [x_hi(1+a_hi)N_hi - x_lo(1+a_lo)N_lo]/N_bin; x_eff = (x_hi N_hi - x_lo N_lo)/N_bin",
           "secrest_reproduction": sec, "published": {"D": D_SECREST, "lb": list(LB_SECREST)},
           "subsamples": rows, "runtime_s": time.time() - t_start}
    out["bcut_subsamples_deg"] = args.bcut
    if args.exclude_gc > 0 or args.ebv_template or jk is not None or args.n_mock != N_MOCK:
        out["options"] = {"exclude_gc_deg": args.exclude_gc, "ebv_template": args.ebv_template, "n_mock": args.n_mock}
    if jk is not None:
        out["jackknife"] = jk
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path} ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
