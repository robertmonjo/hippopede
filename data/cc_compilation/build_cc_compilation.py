"""Build cc_compilation.csv: cosmic-chronometer H(z) measurements, each checked against the original paper.

Values are typed from the papers (see sources.md). Totals are recomputed as the quadrature sum of the
statistical and systematic components when both are published; the total printed in the paper is kept
in sigma_total_published for comparison.
"""
import csv
import math
from pathlib import Path

OUT = Path(__file__).with_name("cc_compilation.csv")

REFS = {
    "Jimenez2003": dict(doi="10.1086/376595", arxiv="astro-ph/0302560", journal="ApJ 593, 622 (2003)", peer_reviewed=1),
    "Simon2005": dict(doi="10.1103/PhysRevD.71.123001", arxiv="astro-ph/0412269", journal="PRD 71, 123001 (2005)", peer_reviewed=1),
    "Stern2010": dict(doi="10.1088/1475-7516/2010/02/008", arxiv="0907.3149", journal="JCAP 02 (2010) 008", peer_reviewed=1),
    "Moresco2012": dict(doi="10.1088/1475-7516/2012/08/006", arxiv="1201.3609", journal="JCAP 08 (2012) 006", peer_reviewed=1),
    "Zhang2014": dict(doi="10.1088/1674-4527/14/10/002", arxiv="1207.4541", journal="RAA 14, 1221 (2014)", peer_reviewed=1),
    "Moresco2015": dict(doi="10.1093/mnrasl/slv037", arxiv="1503.01116", journal="MNRAS 450, L16 (2015)", peer_reviewed=1),
    "Moresco2016": dict(doi="10.1088/1475-7516/2016/05/014", arxiv="1601.01701", journal="JCAP 05 (2016) 014", peer_reviewed=1),
    "Ratsimbazafy2017": dict(doi="10.1093/mnras/stx301", arxiv="1702.00418", journal="MNRAS 467, 3239 (2017)", peer_reviewed=1),
    "Borghi2022": dict(doi="10.3847/2041-8213/ac3fb2", arxiv="2110.04304", journal="ApJL 928, L4 (2022)", peer_reviewed=1),
    "Jiao2023": dict(doi="10.3847/1538-4365/acbc77", arxiv="2205.05701", journal="ApJS 265, 48 (2023)", peer_reviewed=1),
    "Jimenez2023": dict(doi="10.1088/1475-7516/2023/11/047", arxiv="2306.11425", journal="JCAP 11 (2023) 047", peer_reviewed=1),
    "Tomasetti2023": dict(doi="10.1051/0004-6361/202346992", arxiv="2305.16387", journal="A&A 679, A96 (2023)", peer_reviewed=1),
    "Loubser2025a": dict(doi="10.1093/mnras/staf915", arxiv="2506.03836", journal="MNRAS 540, 3135 (2025)", peer_reviewed=1),
    "Loubser2025b": dict(doi="10.1093/mnras/staf1939", arxiv="2511.02730", journal="MNRAS 544, 3064 (2025)", peer_reviewed=1),
    "Tomasetti2025": dict(doi="10.1051/0004-6361/202558259", arxiv="2512.02109", journal="A&A 708, A145 (2026)", peer_reviewed=1),
    "Pradhan2026": dict(doi="", arxiv="2606.07298", journal="preprint (submitted to A&A)", peer_reviewed=0),
    "Lisiecki2026": dict(doi="", arxiv="2608.10163", journal="preprint (A&A manuscript)", peer_reviewed=0),
    "Alvarez2026": dict(doi="", arxiv="2608.13178", journal="preprint (A&A manuscript)", peer_reviewed=0),
}

# Columns of each row:
# ref, z, H, stat_plus, stat_minus, sys_plus, sys_minus, total_published_plus, total_published_minus,
# method, sample, dependence_group, verified, use_default, default_note
# stat/sys = None when the paper publishes only a total error.
N = None
ROWS = [
    # --- Jimenez et al. 2003 ---
    ("Jimenez2003", 0.09, 69.0, N, N, N, N, 12.0, 12.0, "full-spectrum (SSP age fit, SPEED models)",
     "SDSS early-type galaxies, z < 0.17", "SDSS-lowz",
     "arXiv v1 p.11 (Sect. 4): zeff = 0.09, 'H0 = 69 +/- 12'; also Stern+10 Table 2 lists z = 0.1, 69 +/- 12",
     1, "Published value is H0 = H(zeff)/E(zeff) with a 4% LCDM correction (Om=0.27); standard compilations list it as H(0.09)"),
    # --- Simon, Verde & Jimenez 2005 (numbers tabulated in Stern+10 Table 2) ---
    *[("Simon2005", z, h, N, N, N, N, e, e, "full-spectrum (SSP age fit, SPEED models)",
       "field ellipticals (Treu+), GDDS, radio galaxies 53W091/53W069 (32 galaxies)", "Simon05-sample",
       "Simon+05 Fig. 1 right panel (no table); values from Stern+10 arXiv v1 Table 2 p.14", 1, "")
      for z, h, e in [(0.17, 83.0, 8.0), (0.27, 77.0, 14.0), (0.40, 95.0, 17.0), (0.90, 117.0, 23.0),
                      (1.30, 168.0, 17.0), (1.43, 177.0, 18.0), (1.53, 140.0, 14.0), (1.75, 202.0, 40.0)]],
    # --- Stern et al. 2010 ---
    ("Stern2010", 0.48, 97.0, N, N, N, N, 62.0, 62.0, "full-spectrum (BC03 SSP age fit)",
     "Keck-LRIS red-envelope galaxies in 24 clusters + SPICES + VVDS, 0.35<z<0.6", "Stern10-sample",
     "arXiv v1 abstract p.1: 97 +/- 62 at z~0.5; Table 2 p.14 prints 97 +/- 60 at z = 0.48", 1,
     "Internal inconsistency in the paper (62 abstract vs 60 table); 62 adopted as in Moresco+22 and Moresco+24"),
    ("Stern2010", 0.88, 90.0, N, N, N, N, 40.0, 40.0, "full-spectrum (BC03 SSP age fit)",
     "Keck-LRIS red-envelope galaxies in 24 clusters + SPICES + VVDS, 0.6<z<1.0", "Stern10-sample",
     "arXiv v1 abstract p.1 (z~0.8) and Table 2 p.14 (z = 0.88): 90 +/- 40", 1, ""),
    # --- Moresco et al. 2012 (BC03 baseline) ---
    *[("Moresco2012", z, h, st, st, sy, sy, tot, tot, "D4000n (BC03 calibration)",
       "SDSS DR6 MGS, SDSS DR7 LRG, Stern+10 cluster stacks, zCOSMOS, K20, GOODS-S, cluster BCGs, GDDS, UDS",
       "Moresco12-sample", "arXiv v4 Table 4 p.22 (BC03 columns: H, sigma_stat, sigma_syst, sigma_tot)", 1, "")
      for z, h, st, sy, tot in [(0.1791, 75, 3.8, 0.5, 4), (0.1993, 75, 4.9, 0.6, 5), (0.3519, 83, 13, 4.8, 14),
                                (0.5929, 104, 11.6, 4.5, 13), (0.6797, 92, 6.4, 4.3, 8), (0.7812, 105, 9.4, 6.1, 12),
                                (0.8754, 125, 15.3, 6.0, 17), (1.037, 154, 13.6, 14.9, 20)]],
    # --- Zhang et al. 2014 ---
    *[("Zhang2014", z, h, N, N, N, N, e, e, "full-spectrum (UlySS SSP fit)",
       f"SDSS DR7 LRGs, subsample {k} ({rng})", "SDSS-DR7-LRG",
       "arXiv v3 abstract p.1 and Sect. 4 p.9-10 (subsample results, Fig. 4 caption for z ranges)", 1,
       note)
      for k, z, h, e, rng, note in [
          (1, 0.07, 69.0, 19.6, "0.033<z<0.109", "Shares galaxies with subsample 2 in 0.090<z<0.109"),
          (2, 0.12, 68.6, 26.2, "0.090<z<0.156", "Shares galaxies with subsample 1 in 0.090<z<0.109"),
          (3, 0.20, 72.9, 29.6, "0.170<z<0.236", ""),
          (4, 0.28, 88.8, 36.6, "0.243<z<0.315", "")]],
    # --- Moresco 2015 ---
    ("Moresco2015", 1.363, 160.0, N, N, N, N, 33.6, 33.6, "D4000n (BC03 and MaStro calibration; model difference in error)",
     "29 massive passive galaxies at 1.4<z<2.2 from literature NIR spectra, joined to the last Moresco+12 D4000n point",
     "Moresco15-sample", "arXiv Table 1 p.3", 1,
     "Slope uses the last D4000n point of Moresco+12, hence correlated with Moresco2012 z=1.037"),
    ("Moresco2015", 1.965, 186.5, N, N, N, N, 50.4, 50.4, "D4000n (BC03 and MaStro calibration; model difference in error)",
     "29 massive passive galaxies at 1.4<z<2.2 from literature NIR spectra", "Moresco15-sample",
     "arXiv Table 1 p.3", 1, ""),
    # --- Moresco et al. 2016 (BC03, individual bins) ---
    *[("Moresco2016", z, h, st, st, sy, sy, tot, tot, "D4000n (BC03 calibration)",
       "BOSS DR9 massive passive galaxies (stacked spectra)", "BOSS-DR9",
       "arXiv v2 Table 3 p.15 (BC03 columns)", 1, "")
      for z, h, st, sy, tot in [(0.3802, 83.0, 4.3, 12.9, 13.5), (0.4004, 77.0, 2.1, 10.0, 10.2),
                                (0.4247, 87.1, 2.4, 11.0, 11.2), (0.4497, 92.8, 4.5, 12.1, 12.9),
                                (0.4783, 80.9, 2.1, 8.8, 9.0)]],
    ("Moresco2016", 0.4293, 85.7, 1.0, 1.0, 5.1, 5.1, 5.2, 5.2, "D4000n (BC03 calibration)",
     "BOSS DR9 massive passive galaxies, all bins averaged", "BOSS-DR9",
     "arXiv v2 Table 3 p.15 (BC03, averaged row) and text p.13", 0,
     "Average of the five BOSS bins; the paper states it must not be combined with them"),
    ("Moresco2016", 0.4293, 91.8, 1.0, 1.0, 5.1, 5.1, 5.3, 5.3, "D4000n (M11 calibration)",
     "BOSS DR9 massive passive galaxies, all bins averaged", "BOSS-DR9",
     "arXiv v2 abstract, Table 3 p.15 (M11, averaged row) and text p.13", 0,
     "Same galaxies as the five BOSS bins, and M11 calibration whereas every other D4000n point uses BC03"),
    # --- Ratsimbazafy et al. 2017 ---
    ("Ratsimbazafy2017", 0.47, 89.0, 23.0, 23.0, 44.0, 44.0, N, N, "full-spectrum (ULySS; model spread as systematic)",
     "SALT spectra of 2SLAQ luminous red galaxies", "SALT-2SLAQ",
     "arXiv v2 abstract p.1 and Sect. 5 p.7: 89 +/- 23(stat) +/- 44(syst)", 1, ""),
    # --- Borghi et al. 2022 ---
    ("Borghi2022", 0.75, 98.8, 24.8, 24.8, 22.7, 22.7, 33.6, 33.6, "Lick indices",
     "LEGA-C DR2, 140 massive passive galaxies", "LEGA-C",
     "arXiv Eq. (2) p.4 (98.8 +/- 33.6), Table 1 p.3 (joint: zeff = 0.753, stat 24.8), text p.4 (syst 22.7)", 1,
     "Chosen for the LEGA-C group: in the Moresco+22 32-point set, symmetric errors, SPS systematics budgeted"),
    # --- Jiao et al. 2023 ---
    ("Jiao2023", 0.80, 113.1, 15.1, 15.1, 29.1, 11.3, N, N, "full-spectrum (BAGPIPES, spectra + photometry)",
     "LEGA-C DR2, 350 massive passive galaxies (parent sample of Borghi+22a)", "LEGA-C",
     "arXiv abstract p.1, Sect. 5 p.14 and Conclusions p.16: 113.1 +/- 15.1(stat) +29.1/-11.3(syst)", 0,
     "Same LEGA-C DR2 galaxies as Borghi2022 (superset)"),
    # --- Jimenez et al. 2023 ---
    ("Jimenez2023", 0.75, 105.0, 7.9, 7.9, 7.3, 7.3, N, N, "photometric ages (neural network trained on LEGA-C Lick-index ages)",
     "COSMOS2015 photometric passive galaxies, 0.65<z<0.85; training set = Borghi+22 LEGA-C sample", "LEGA-C",
     "arXiv v1 abstract p.1 and Eq. (3.2) p.8", 0,
     "Age scale calibrated on the Borghi2022 LEGA-C galaxies, same COSMOS field; Moresco+24 Table 1 flags it"),
    # --- Tomasetti et al. 2023 ---
    ("Tomasetti2023", 1.26, 135.0, 60.0, 60.0, 27.1, 27.1, 65.0, 65.0, "full-spectrum (BAGPIPES)",
     "VANDELS DR4, 39 passive galaxies at 1.07<z<1.5", "VANDELS",
     "arXiv abstract p.1 and Sect. 4.3.1 p.13: 135 +/- 60(stat) +/- 27(sys) +/- 2.4(bin); total 65", 1,
     "sys = 27 (SFH) and 2.4 (binning) in quadrature"),
    # --- Loubser et al. 2025 (BCGs) ---
    ("Loubser2025a", 0.50, 72.1, 33.9, 33.9, 7.3, 7.3, N, N, "D4000n (MILES/FSPS calibration)",
     "SALT spectra of 53 BCGs in ACT SZ-selected clusters, 0.3<z<0.7", "SALT-ACT-BCG",
     "arXiv v1 abstract p.1 and Eq. (6) p.9", 1, ""),
    # --- Loubser 2025 (DESI DR1) ---
    *[("Loubser2025b", z, h, st, st, sy, sy, N, N, "D4000n",
       f"DESI DR1 massive passive galaxies (log M > 10.75, sigma > 280 km/s), {rng}", "DESI-DR1",
       f"arXiv v1 abstract p.1 and Eq. ({eq}) p.7-8; correlations p.8", 1,
       "Systematic errors highly correlated between the three points: rho(0.46,0.67)=0.932, rho(0.46,0.83)=0.830, rho(0.67,0.83)=0.776")
      for z, h, st, sy, rng, eq in [(0.46, 88.48, 0.57, 12.32, "0.30<z<0.55", 4),
                                    (0.67, 119.45, 6.39, 16.64, "0.55<z<0.78", 5),
                                    (0.83, 108.28, 10.07, 15.08, "0.78<z<1.0", 6)]],
    # --- Tomasetti et al. 2025 (clusters) ---
    ("Tomasetti2025", 0.542, 66.0, 81.0, 29.0, 13.0, 13.0, 82.0, 32.0, "full-spectrum (BAGPIPES)",
     "VLT/MUSE members of SDSS J2222+2745, MACS J1149.5+2223, SDSS J1029+2623 (35-38 CCs)", "MUSE-clusters",
     "arXiv v1 abstract p.1 and Sect. 4.1 p.5: 66 +81/-29(stat) +/- 13(syst) = 66 +82/-32", 1, ""),
    # --- Pradhan et al. 2026 (VIPERS) ---
    ("Pradhan2026", 0.65, 93.68, 28.27, 28.27, 10.67, 10.67, N, N, "D4000 (Bayesian age posteriors, photometric selection)",
     "VIPERS PDR2 massive passive galaxies, 0.5<z<0.8", "VIPERS",
     "arXiv v1 abstract p.1 and Sect. 5 p.11", 0, "Preprint, not yet peer-reviewed"),
    # --- Lisiecki et al. 2026 (CORN) ---
    ("Lisiecki2026", 0.15, 85.0, 51.5, 51.5, 10.2, 8.9, 53.0, 53.0, "Dn4000 (MILES, alpha-enhancement propagated)",
     "189 relic UCMGs from E-INSPIRE (SDSS spectra), 0.07<z<0.22", "SDSS-lowz",
     "arXiv v2 abstract p.1 and Eq. (7) p.6", 0,
     "Preprint, not yet peer-reviewed; SDSS galaxies possibly shared with SDSS-based low-z points"),
    # --- Alvarez et al. 2026 (DESI DR1) ---
    ("Alvarez2026", 0.57, 95.1, 10.9, 6.0, 11.3, 11.3, N, N, "Lick indices on stacked spectra; pivotal cosmographic fit",
     "DESI DR1 LRG stacks, 0.3<z<1.2", "DESI-DR1",
     "arXiv v1 Sect. 5.1 p.9 and Conclusions p.12", 0,
     "Cosmographic H(z0) from a fit, not a discrete CC point; same DESI DR1 galaxies as Loubser2025b; preprint"),
    ("Alvarez2026", 0.55, 104.5, 13.2, 7.6, 22.4, 22.4, N, N, "Lick indices on stacked spectra; finite difference",
     "DESI DR1 LRG stacks, massive velocity-dispersion group", "DESI-DR1",
     "arXiv v1 Sect. 5.2 p.11 and Conclusions p.12", 0,
     "Same DESI DR1 galaxies as Loubser2025b; preprint"),
    ("Alvarez2026", 0.61, 88.5, 6.7, 12.6, 8.1, 8.1, N, N, "Lick indices on stacked spectra; finite difference",
     "DESI DR1 LRG stacks, most massive velocity-dispersion group", "DESI-DR1",
     "arXiv v1 abstract p.1, Sect. 5.2 p.11", 0,
     "Same DESI DR1 galaxies as Loubser2025b; preprint"),
]

OVERLAP = {
    "SDSS-lowz": "SDSS low-z passive galaxies; may share objects with Zhang2014 (SDSS-DR7-LRG) and Moresco2012 SDSS MGS bin",
    "SDSS-DR7-LRG": "SDSS DR7 LRGs; Moresco2012 uses SDSS DR6 MGS and DR7 LRGs at 0.15<z<0.4 (possible partial overlap)",
    "Simon05-sample": "GDDS galaxies also enter Moresco2012 (0.91<z<1.13 bin)",
    "Stern10-sample": "Keck cluster spectra also enter Moresco2012 as stacked spectra (0.38<z<0.75)",
    "Moresco12-sample": "Includes Stern10 cluster stacks and GDDS galaxies used by Simon2005",
    "Moresco15-sample": "z=1.363 slope anchored on Moresco2012 z~1 D4000n point",
}


def q(a, b):
    return None if a is None or b is None else round(math.hypot(a, b), 2)


def fmt(x):
    return "" if x is None else (f"{x:g}")


cols = ["id", "z", "H", "sigma_stat_plus", "sigma_stat_minus", "sigma_sys_plus", "sigma_sys_minus",
        "sigma_total_plus", "sigma_total_minus", "sigma_total_published", "error_type", "method", "sample",
        "dependence_group", "overlap_note", "reference_key", "journal", "doi", "arxiv", "peer_reviewed",
        "verified", "use_default", "default_note"]

rows = []
for i, r in enumerate(sorted(ROWS, key=lambda r: (r[1], r[0])), 1):
    ref, z, H, stp, stm, syp, sym, tpp, tpm, method, sample, grp, ver, use, note = r
    if stp is not None and syp is not None:
        tp, tm = q(stp, syp), q(stm, sym)
        etype = "stat+sys (quadrature)"
    else:
        tp, tm = tpp, tpm
        etype = "total only (as published)"
    pub = "" if tpp is None else (fmt(tpp) if tpp == tpm else f"+{fmt(tpp)}/-{fmt(tpm)}")
    ref_meta = REFS[ref]
    rows.append({
        "id": i, "z": fmt(z), "H": fmt(H),
        "sigma_stat_plus": fmt(stp), "sigma_stat_minus": fmt(stm),
        "sigma_sys_plus": fmt(syp), "sigma_sys_minus": fmt(sym),
        "sigma_total_plus": fmt(tp), "sigma_total_minus": fmt(tm), "sigma_total_published": pub,
        "error_type": etype, "method": method, "sample": sample, "dependence_group": grp,
        "overlap_note": OVERLAP.get(grp, ""), "reference_key": ref, "journal": ref_meta["journal"],
        "doi": ref_meta["doi"], "arxiv": ref_meta["arxiv"], "peer_reviewed": ref_meta["peer_reviewed"],
        "verified": ver, "use_default": use, "default_note": note,
    })

with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

n_def = sum(r["use_default"] for r in rows)
print(f"wrote {OUT.name}: N = {len(rows)}, N_default = {n_def}")
for r in rows:
    print(r["id"], r["z"], r["H"], r["sigma_total_plus"], r["sigma_total_minus"], r["sigma_total_published"],
          r["reference_key"], r["use_default"])
