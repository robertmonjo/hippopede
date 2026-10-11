"""Redshift distributions of the CatWISE AGN subsamples from cross-matches with Quaia and SDSS DR16Q.

Data.
  CatWISE AGN sample of Secrest et al. 2021 (ApJL 908, L51), Zenodo 10.5281/zenodo.8303800,
  ./data/catwise/catwise_agns_masked_final_w1lt16p5_alpha.fits, with the pixel mask and the subsample
  definitions of measure_catwise_cuts.py (|b| > 30 deg, W1 <= 16.4; full sample, cumulative W1 limits,
  W1 bins, W1-W2 terciles and the 4 x 2 grid).
  Quaia G < 20 (Storey-Fisher et al. 2024, ApJ 964, 69; Zenodo 10.5281/zenodo.10403370),
  ./data/quaia_G20.0.fits: all-sky Gaia quasar candidates with spectrophotometric redshifts redshift_quaia.
  SDSS DR16Q (Lyke et al. 2020, ApJS 250, 8), ./data/sdss/DR16Q_v4.fits, downloaded from
  https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits: spectroscopic quasars, best redshift Z (> 0).

Cross-match.  Nearest neighbour on the sphere (KD-tree on unit vectors) of every CatWISE source within the
match radius R_MATCH (arcsec).  The rate of chance matches is measured by repeating the match with the CatWISE
positions displaced by 1 arcmin in declination.  SDSS footprint: HEALPix NSIDE=64 pixels (Galactic) holding
at least FOOT_MIN DR16Q quasars; matched fractions for SDSS refer to the CatWISE sources inside it.

Redshift distribution of a subsample, from the redshifts of its matched sources:
  mean, median, fractions below 0.1 and above 2.1, a histogram on 0-5 in steps of 0.05, and p(z) on the grid
  Z_Q = 0.1, 0.15, ..., 2.1 of quasar_dipole_fit (the model dipole is computed only there): p_i = n_i /
  (N_match w_i), n_i the counts in [Z_Q_i - 0.025, Z_Q_i + 0.025) clipped to [0.1, 2.1] and w_i the trapezoid
  weights (0.05 inside, 0.025 at the ends), so that trapezoid(p, Z_Q) is the matched fraction in 0.1-2.1.
Selection bias.  The matched sources are optically brighter than the unmatched ones (Quaia: G < 20; DR16Q:
  target selection to i ~ 20-22).  Stored per subsample: matched fraction; mean z, median z and fraction above
  2.1 in bins of optical magnitude (Gaia G for Quaia, SDSS PSF i for DR16Q); in the SDSS footprint, the SDSS
  redshifts of the DR16Q-matched sources split into those also matched to Quaia and those that are not
  (direct measure of how the Quaia p(z) is biased by G < 20); and the agreement of the two redshifts for the
  sources in both catalogues.  A third p(z), "sdss_faint", uses only the DR16Q matches with i > 20 as a proxy for
  the optically fainter, unmatched sources (the trend of p(z) with optical magnitude extrapolated one step).
Writes ../json/catwise_bins_pz.json.  With --plot, draws ../figures/hippopede_catwise_cuts_pz.(png|pdf) from
json/catwise_bins_pz.json, json/catwise_cuts_dipoles_jk.json, json/catwise_kinematic_vs_geometric_jk.json and
json/catwise_kinematic_vs_geometric_pz_{quaia,sdss,sdss_faint}_jk.json (scripts/catwise_kinematic_vs_geometric.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "json", "catwise_bins_pz.json")
QUAIA = os.path.join(HERE, "data", "quaia_G20.0.fits")
DR16Q = os.path.join(HERE, "data", "sdss", "DR16Q_v4.fits")
DR16Q_URL = "https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits"
R_MATCH = 1.5  # arcsec
R_SEARCH = 10.0  # arcsec, for the separation histogram
SHIFT_ARCMIN = 1.0
FOOT_MIN = 10
Z_Q = np.linspace(0.1, 2.1, 41)
Z_HIST = np.arange(0.0, 5.0001, 0.05)
G_EDGES = [0.0, 18.5, 19.25, 20.0]
I_EDGES = [0.0, 18.5, 19.5, 20.5, 30.0]
I_FAINT = 20.0  # optically faint DR16Q matches, proxy for the unmatched sources


def unit_radec(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def match(tree, ra, dec):
    """Separation (arcsec, inf if none within R_SEARCH) and index of the nearest catalogue source."""
    chord = 2 * np.sin(np.radians(R_SEARCH / 3600) / 2)
    dist, idx = tree.query(unit_radec(ra, dec), distance_upper_bound=chord, workers=8)
    sep = np.where(np.isfinite(dist), np.degrees(2 * np.arcsin(np.minimum(dist, 2) / 2)) * 3600, np.inf)
    return sep, idx


def pz_grid(z):
    """p on Z_Q with trapezoid(p, Z_Q) = fraction of z in [0.1, 2.1] (see the module docstring)."""
    edges = np.concatenate([[Z_Q[0]], 0.5 * (Z_Q[1:] + Z_Q[:-1]), [Z_Q[-1]]])
    n = np.histogram(np.clip(z, None, Z_Q[-1] - 1e-12)[(z >= Z_Q[0]) & (z <= Z_Q[-1])], edges)[0]
    w = np.diff(edges)
    return n / (len(z) * w)


def zstats(z):
    if len(z) == 0:
        return {"n": 0}
    return {"n": int(len(z)), "mean": float(np.mean(z)), "median": float(np.median(z)),
            "f_below_0.1": float(np.mean(z < 0.1)), "f_above_2.1": float(np.mean(z > 2.1)),
            "mean_in_0.1_2.1": float(np.mean(z[(z >= 0.1) & (z <= 2.1)])) if np.any((z >= 0.1) & (z <= 2.1)) else None}


def by_mag(z, mag, edges):
    return [{"range": [a, b], **zstats(z[(mag >= a) & (mag < b)])} for a, b in zip(edges[:-1], edges[1:])]


def build():
    import healpy as hp
    from astropy.io import fits
    from scipy.spatial import cKDTree
    sys.path.insert(0, HERE)
    import measure_catwise_cuts as M

    t0 = time.time()
    cat = fits.getdata(os.path.join(M.DATA, M.CAT), 1)
    masks = fits.getdata(os.path.join(M.DATA, M.MASKS), 1)
    w1, w12, l, b = cat["w1"], cat["w12"], cat["l"], cat["b"]
    good = hp.ud_grade(M.make_mask(masks, M.NSIDE_HI, "galactic"), nside_out=M.NSIDE) == 1
    plat = hp.pix2ang(M.NSIDE, np.arange(hp.nside2npix(M.NSIDE)), lonlat=True)[1]
    good &= np.abs(plat) > M.GALCUT
    pix = hp.ang2pix(M.NSIDE, l, b, lonlat=True)
    base = good[pix] & (w1 <= M.W1_MAX)
    ra, dec = cat["ra"], cat["dec"]
    print(f"CatWISE: {base.sum()} sources in the mask with W1 <= {M.W1_MAX}", flush=True)

    q = fits.getdata(QUAIA, 1)
    s = fits.getdata(DR16Q, 1)
    sz = s["Z"]
    ok = np.isfinite(sz) & (sz > 0)
    s_ra, s_dec, s_z, s_i = s["RA"][ok], s["DEC"][ok], sz[ok], s["PSFMAG"][ok][:, 3]
    print(f"Quaia: {len(q)}; DR16Q: {len(s)} ({ok.sum()} with Z > 0)", flush=True)
    cats = {"quaia": {"tree": cKDTree(unit_radec(q["ra"], q["dec"])), "z": q["redshift_quaia"],
                      "z_err": q["redshift_quaia_err"], "mag": q["phot_g_mean_mag"], "mag_name": "Gaia G",
                      "mag_edges": G_EDGES},
            "sdss": {"tree": cKDTree(unit_radec(s_ra, s_dec)), "z": s_z, "mag": s_i, "mag_name": "SDSS PSF i",
                     "mag_edges": I_EDGES}}
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    sg = SkyCoord(ra=s_ra * u.deg, dec=s_dec * u.deg, frame="icrs").galactic
    s_cnt = np.bincount(hp.ang2pix(M.NSIDE, sg.l.deg, sg.b.deg, lonlat=True), minlength=hp.nside2npix(M.NSIDE))
    foot = s_cnt >= FOOT_MIN
    in_foot = foot[pix]

    match_info, mz, mmag, mok = {}, {}, {}, {}
    for name, c in cats.items():
        sep, idx = match(c["tree"], ra[base], dec[base])
        sep_r, _ = match(c["tree"], ra[base], np.clip(dec[base] + SHIFT_ARCMIN / 60, -90, 90))
        m = sep < R_MATCH
        dom = in_foot[base] if name == "sdss" else np.ones(base.sum(), bool)
        h = np.histogram(sep[np.isfinite(sep)], np.arange(0, R_SEARCH + 1e-9, 0.25))[0]
        h_r = np.histogram(sep_r[np.isfinite(sep_r)], np.arange(0, R_SEARCH + 1e-9, 0.25))[0]
        match_info[name] = {
            "radius_arcsec": R_MATCH, "n_catwise_domain": int(dom.sum()), "n_matched": int(m.sum()),
            "matched_fraction_domain": float(m.sum() / dom.sum()),
            "chance_rate_shifted": float(np.mean(sep_r[dom] < R_MATCH)),
            "matched_fraction_by_radius": {f"{r:g}": float(np.mean(sep[dom] < r)) for r in (0.5, 1, 1.5, 2, 3, 5)},
            "chance_fraction_by_radius": {f"{r:g}": float(np.mean(sep_r[dom] < r)) for r in (0.5, 1, 1.5, 2, 3, 5)},
            "sep_hist_edges_arcsec": [0.0, R_SEARCH, 0.25], "sep_hist": h.tolist(), "sep_hist_shifted": h_r.tolist(),
            "median_sep_matched_arcsec": float(np.median(sep[m]))}
        full_idx = np.full(len(w1), -1)
        full_idx[np.where(base)[0][m]] = idx[m]
        mok[name] = full_idx >= 0
        mz[name] = np.where(mok[name], c["z"][np.maximum(full_idx, 0)], np.nan)
        mmag[name] = np.where(mok[name], c["mag"][np.maximum(full_idx, 0)], np.nan)
        if name == "quaia":
            q_zerr = np.where(mok[name], c["z_err"][np.maximum(full_idx, 0)], np.nan)
        print(f"{name}: matched {m.sum()} / {dom.sum()} = {m.sum() / dom.sum():.3f} within {R_MATCH}\"; chance "
              f"{match_info[name]['chance_rate_shifted']:.4f}; by radius {match_info[name]['matched_fraction_by_radius']}", flush=True)
    match_info["sdss"]["footprint"] = {"nside": M.NSIDE, "min_dr16q_per_pixel": FOOT_MIN,
                                       "f_catwise_in_footprint": float(np.mean(in_foot[base])),
                                       "n_pix_footprint_in_mask": int(np.sum(foot & good))}
    both = mok["quaia"] & mok["sdss"]
    dz = (mz["quaia"][both] - mz["sdss"][both]) / (1 + mz["sdss"][both])
    agreement = {"n_both": int(both.sum()), "median_dz_1pz": float(np.median(dz)),
                 "nmad_dz_1pz": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
                 "f_abs_dz_1pz_gt_0.2": float(np.mean(np.abs(dz) > 0.2)),
                 "mean_z_sdss": float(np.mean(mz["sdss"][both])), "mean_z_quaia": float(np.mean(mz["quaia"][both]))}
    print(f"Quaia vs SDSS: {agreement}", flush=True)

    defs = [("full_W1<=16.4", "full", 9.0, M.W1_MAX, 0.8, 9.0)]
    defs += [(f"cum_W1<{m}", "cumulative", 9.0, m, 0.8, 9.0) for m in M.CUMULATIVE[:-1]]
    defs += [(f"bin_W1_{a}-{c}", "mag_bin", a, c, 0.8, 9.0) for a, c in zip(M.MAG_EDGES[:-1], M.MAG_EDGES[1:])]
    defs += [(f"col_W1W2_{a}-{c}", "colour_bin", 9.0, M.W1_MAX, a, c) for a, c in zip(M.COL_EDGES[:-1], M.COL_EDGES[1:])]
    for a, c in zip(M.MAG_EDGES[:-1], M.MAG_EDGES[1:]):
        for cl_, ch_, tag in ((0.8, M.COL_SPLIT, "blue"), (M.COL_SPLIT, 9.0, "red")):
            defs.append((f"grid_W1_{a}-{c}_{tag}", "grid", a, c, cl_, ch_))
    rows = []
    for name, kind, m_lo, m_hi, c_lo, c_hi in defs:
        sel = base & (w12 >= c_lo) & (w12 < c_hi) & (w1 >= m_lo) & (w1 < m_hi if m_hi < M.W1_MAX else w1 <= m_hi)
        row = {"name": name, "kind": kind, "W1_range": [m_lo, m_hi], "W1W2_range": [c_lo, c_hi], "N": int(sel.sum()),
               "f_in_sdss_footprint": float(np.mean(in_foot[sel]))}
        for cn, c in cats.items():
            dom = sel & (in_foot if cn == "sdss" else True)
            mm = dom & mok[cn]
            z, mag = mz[cn][mm], mmag[cn][mm]
            row[cn] = {"n_domain": int(dom.sum()), "n_matched": int(mm.sum()),
                       "matched_fraction": float(mm.sum() / max(dom.sum(), 1)), **zstats(z),
                       "hist": np.histogram(z, Z_HIST)[0].tolist(), "p_ZQ": pz_grid(z).tolist(),
                       "median_optical_mag": float(np.median(mag)), "by_optical_mag": by_mag(z, mag, c["mag_edges"]),
                       "optical_mag": c["mag_name"]}
            if cn == "quaia":
                row[cn]["median_z_err"] = float(np.median(q_zerr[mm]))
        fs = sel & in_foot & mok["sdss"]
        ff = fs & (mmag["sdss"] > I_FAINT)
        zf = mz["sdss"][ff]
        row["sdss_faint"] = {"n_domain": int(np.sum(sel & in_foot)), "n_matched": int(ff.sum()),
                             "matched_fraction": float(ff.sum() / max(np.sum(sel & in_foot), 1)), **zstats(zf),
                             "hist": np.histogram(zf, Z_HIST)[0].tolist(), "p_ZQ": pz_grid(zf).tolist(),
                             "selection": f"DR16Q-matched sources with SDSS PSF i > {I_FAINT}"}
        row["sdss_split_by_quaia"] = {"in_quaia": zstats(mz["sdss"][fs & mok["quaia"]]),
                                      "not_in_quaia": zstats(mz["sdss"][fs & ~mok["quaia"]]),
                                      "quaia_matched_fraction_in_sdss_footprint":
                                          float(np.mean(mok["quaia"][sel & in_foot])) if np.any(sel & in_foot) else None}
        rows.append(row)
        qq, ss = row["quaia"], row["sdss"]
        print(f"{name:24s} N={row['N']:8d} Quaia f={qq['matched_fraction']:.3f} <z>={qq['mean']:.3f} med={qq['median']:.3f} "
              f">2.1:{qq['f_above_2.1']:.3f} | SDSS f={ss['matched_fraction']:.3f} <z>={ss['mean']:.3f} "
              f"med={ss['median']:.3f} >2.1:{ss['f_above_2.1']:.3f} | SDSS z in/not Quaia "
              f"{row['sdss_split_by_quaia']['in_quaia'].get('mean', np.nan):.3f}/"
              f"{row['sdss_split_by_quaia']['not_in_quaia'].get('mean', np.nan):.3f}", flush=True)

    gamma = Z_Q**2 * np.exp(-Z_Q / 0.4)
    gamma = gamma / np.trapezoid(gamma, Z_Q)
    out = {"sources": {"catwise": "Secrest et al. 2021, Zenodo 10.5281/zenodo.8303800",
                       "quaia": "Storey-Fisher et al. 2024, quaia_G20.0.fits (Zenodo 10.5281/zenodo.10403370)",
                       "sdss_dr16q": f"Lyke et al. 2020, {DR16Q_URL}; column Z (best redshift), Z > 0; PSFMAG[3] = i"},
           "catwise_selection": "measure_catwise_cuts.py pixel mask, |b| > 30 deg, W1 <= 16.4",
           "match": match_info, "quaia_vs_sdss_redshifts": agreement,
           "z_grid": Z_Q.tolist(), "z_hist_edges": Z_HIST.tolist(), "gamma_p_ZQ": gamma.tolist(),
           "p_ZQ_definition": "n_i/(N_matched w_i) on Z_Q with trapezoid weights w_i; trapezoid(p) = fraction in 0.1-2.1",
           "subsamples": rows, "runtime_s": time.time() - t0}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")


def plot():
    """Three panels: p(z) of the W1-W2 terciles; geometric amplitude at the reference point with each p(z);
    colour-bin dipoles projected on the full-sample direction with jackknife and Poisson errors and the fits
    with the SDSS p(z) and the jackknife covariance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    js = os.path.join(ROOT, "json")
    pzj = json.load(open(OUT))
    fits = {c: json.load(open(os.path.join(js, f"catwise_kinematic_vs_geometric_pz_{c}_jk.json")))
            for c in ("quaia", "sdss", "sdss_faint")}
    gam = json.load(open(os.path.join(js, "catwise_kinematic_vs_geometric_jk.json")))
    rows = {r["name"]: r for r in pzj["subsamples"]}
    cols = ["col_W1W2_0.8-0.95", "col_W1W2_0.95-1.15", "col_W1W2_1.15-9.0"]
    colour = {cols[0]: "#0072B2", cols[1]: "#009E73", cols[2]: "#D55E00"}
    label = {cols[0]: "W1$-$W2 0.80$-$0.95", cols[1]: "W1$-$W2 0.95$-$1.15", cols[2]: "W1$-$W2 $\\geq$ 1.15"}
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.2), dpi=180)
    zc = 0.5 * (np.array(pzj["z_hist_edges"][1:]) + np.array(pzj["z_hist_edges"][:-1]))
    for c in cols:
        for cat, ls in (("sdss", "-"), ("quaia", "--")):
            h = np.array(rows[c][cat]["hist"], float)
            ax[0].plot(zc, h / h.sum() / 0.05, ls=ls, color=colour[c], lw=1.4,
                       label=f"{label[c]}, {'DR16Q' if cat == 'sdss' else 'Quaia'}")
    ax[0].plot(Z_Q, pzj["gamma_p_ZQ"], ":", color="k", lw=1.4, label="common gamma p(z)")
    ax[0].axvline(2.1, color="0.5", lw=0.8)
    ax[0].set_xlim(0, 4)
    ax[0].set_xlabel("redshift $z$")
    ax[0].set_ylabel("p(z) of the matched sources")
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].set_title("(a) redshift distributions", fontsize=9)

    mk = {"gamma": ("o", "common gamma"), "quaia": ("s", "Quaia"), "sdss": ("D", "DR16Q"),
          "sdss_faint": ("^", "DR16Q, $i > 20$")}
    for k, (m, lab) in mk.items():
        if k == "gamma":
            per = fits["sdss"]["pz"]["per_subsample_reference"]
            g = [-per[c]["g_gamma"] for c in cols]
        else:
            per = fits[k]["pz"]["per_subsample_reference"]
            g = [-per[c]["g"] for c in cols]
        x = [per[c]["x_eff"] for c in cols]
        ax[1].plot(x, g, marker=m, ls="-", lw=1.0, ms=6, color="0.25" if k == "gamma" else
                   {"quaia": "#CC79A7", "sdss": "#E69F00", "sdss_faint": "#56B4E9"}[k], label=lab)
    ax[1].set_xlabel("count slope $x$ of the colour bin")
    ax[1].set_ylabel("geometric dipole $-g$ at the reference point")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("(b) model amplitude with each p(z)", fontsize=9)

    r = fits["sdss"]["fits"]["colour_bins_jk"]
    rg = gam["fits"]["colour_bins_jk"]
    x = np.array(r["x_eff"])
    meas = json.load(open(os.path.join(js, "catwise_cuts_dipoles_jk.json")))
    mrows = {m["name"]: m for m in meas["subsamples"]}
    u_full = np.array(meas["subsamples"][0]["d"]) / np.linalg.norm(meas["subsamples"][0]["d"])
    sig_p = [np.sqrt(u_full @ np.array(mrows[c]["cov_mocks"]) @ u_full) for c in cols]
    for i, c in enumerate(cols):
        ax[2].errorbar(x[i], r["proj_full_dir"][i], yerr=r["proj_sigma"][i], fmt="o", color=colour[c], ms=6,
                       capsize=3, lw=1.4, zorder=5)
        ax[2].errorbar(x[i] + 0.04, r["proj_full_dir"][i], yerr=sig_p[i], fmt="none", ecolor=colour[c], capsize=2,
                       lw=0.8, alpha=0.6)
    order = np.argsort(x)
    for name, res, lab, col, ls in (("M1b", r, "kinematic, $\\vec\\beta$ free (M1b)", "#1f77b4", "-"),
                                    ("M4", r, "$\\beta_{\\rm CMB}$ + constant vector (M4)", "#2ca02c", "-."),
                                    ("M2s", rg, "geometric shape, gamma p(z) (M2s)", "0.4", ":"),
                                    ("M2s", r, "geometric shape, DR16Q p(z) (M2s)", "#d62728", "-")):
        pred = np.array(res["models"][name]["pred"]) @ u_full
        ax[2].plot(x[order], pred[order], ls=ls, color=col, lw=1.4,
                   label=f"{lab}: $\\chi^2$ = {res['models'][name]['chi2']:.1f}")
    ax[2].set_ylim(0, 0.035)
    ax[2].set_xlabel("count slope $x$ of the colour bin")
    ax[2].set_ylabel("dipole projected on the full-sample direction")
    ax[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    ax[2].set_title("(c) colour bins, errors: jackknife (thick), Poisson (thin) ($|b| > 30^\\circ$)", fontsize=9)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"hippopede_catwise_cuts_pz.{ext}"), dpi=220 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="draw the figure from the existing json files")
    if ap.parse_args().plot:
        plot()
    else:
        build()


if __name__ == "__main__":
    main()
