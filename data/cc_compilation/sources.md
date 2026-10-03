# Cosmic-chronometer H(z) compilation: sources and verification

`cc_compilation.csv` lists 47 H(z) measurements obtained with the differential-age (cosmic chronometer, CC) method, from 18 papers published or posted between 2003 and 2026. `build_cc_compilation.py` writes the file; every value in it was read in the original paper (arXiv PDF, version stated below). H(z) and all errors are in km s⁻¹ Mpc⁻¹.

## Columns

| column | content |
|---|---|
| `z`, `H` | redshift and H(z) as printed in the paper |
| `sigma_stat_plus/minus` | statistical error (empty when the paper gives only a total) |
| `sigma_sys_plus/minus` | systematic error (empty when the paper gives only a total) |
| `sigma_total_plus/minus` | stat and sys in quadrature when both are published; otherwise the published total |
| `sigma_total_published` | total error printed in the paper, when printed |
| `error_type` | `stat+sys (quadrature)` or `total only (as published)` |
| `dependence_group` | label of the galaxy sample; rows with the same label share galaxies or calibration |
| `overlap_note` | partial overlaps between different groups |
| `verified` | where the value was read (arXiv version, table, equation, PDF page) |
| `use_default` | 1 = member of the default independent subset |
| `default_note` | reason for exclusion, or a caveat on an included point |

PDF page numbers refer to the arXiv PDF, not the journal pagination. Where a paper prints a total that differs from the quadrature sum of its rounded components (Moresco 2012, Tomasetti 2023), both are kept: `sigma_total_*` holds the quadrature sum and `sigma_total_published` the printed value. The Moresco 2012 printed totals are rounded to integers. The covariance repository lists them unrounded (e.g. 12.50 at z = 0.5929, 12.20 at z = 0.7812); these differ from the quadrature sum of the rounded components by up to 1.0 (z = 0.7812: 11.21 versus 12.20), so `sigma_total_published` or the repository values reproduce the Moresco analyses more closely than `sigma_total_*`.

## Per-paper verification

### Jimenez et al. 2003 (ApJ 593, 622; astro-ph/0302560)
- z = 0.09, 69 ± 12. Read on p. 11 of the arXiv v1 PDF (Sect. 4): "zmax = 0.17, P = 0.32, zeff = 0.09, and the correction from H(zeff) to H0 is a 4% effect. We obtain H0 = 69 ± 12".
- The published number is H0, obtained from H(zeff) divided by E(zeff) for Ωm = 0.27, ΩΛ = 0.7. By the paper's own 4% correction, H(0.09) is about 72. Every compilation checked (Stern et al. 2010 Table 2 at z = 0.1; Moresco et al. 2022 Table 1 and Moresco 2024 Table 1 at z = 0.09, both attributing it to Simon et al.) lists 69 ± 12 as an H(z) point. The row keeps the published value and flags the difference.

### Simon, Verde & Jimenez 2005 (PRD 71, 123001; astro-ph/0412269)
- Eight points: z = 0.17, 0.27, 0.40, 0.90, 1.30, 1.43, 1.53, 1.75 with H = 83 ± 8, 77 ± 14, 95 ± 17, 117 ± 23, 168 ± 17, 177 ± 18, 140 ± 14, 202 ± 40.
- The paper shows these points only in Fig. 1 (right panel, p. 9); it has no table. The numbers were read in Stern et al. 2010, Table 2 (arXiv v1 p. 14), which tabulates the Simon et al. points together with the two new Stern et al. ones (Fig. 9 caption: "The points at z > 1 are taken from Ref. [23]", Ref. [23] = Simon et al. 2005).
- Sample (Simon et al. Sect. IV): 32 galaxies, i.e. field early-type galaxies from Treu et al., the GDDS old sample re-analysed with SPEED models, and the radio galaxies 53W091 and 53W069. Errors are totals; no separate systematic budget.

### Stern et al. 2010 (JCAP 02 (2010) 008; 0907.3149)
- z = 0.48, 97 ± 62 and z = 0.88, 90 ± 40.
- The abstract (p. 1) gives "H(z) = 97 ± 62 at z ≈ 0.5 and H(z) = 90 ± 40 at z ≈ 0.8"; Table 2 (p. 14) prints 97 ± 60 at z = 0.48 and 90 ± 40 at z = 0.88. The value 62 is adopted, as in Moresco et al. 2022 and 2024.
- Sample: Keck-LRIS spectra of red-envelope galaxies in 24 clusters (0.2 < z < 1.0), SPICES and VVDS archival spectra; BC03 SSP fits; the range 0.35 < z < 1.0 is split into 0.35 to 0.6 and 0.6 to 1.0 (p. 13).

### Moresco et al. 2012 (JCAP 08 (2012) 006; 1201.3609v4)
- Eight points, BC03 baseline, Table 4 (p. 22), columns H, σstat, σsyst, σtot:
  0.1791: 75, 3.8, 0.5, 4 · 0.1993: 75, 4.9, 0.6, 5 · 0.3519: 83, 13, 4.8, 14 · 0.5929: 104, 11.6, 4.5, 13 · 0.6797: 92, 6.4, 4.3, 8 · 0.7812: 105, 9.4, 6.1, 12 · 0.8754: 125, 15.3, 6, 17 · 1.037: 154, 13.6, 14.9, 20.
- The same table gives MaStro-calibrated values (81, 81, 88, 110, 98, 88, 124, 113); they are not included. Table 5 (p. 28) repeats the BC03 median values with totals only.
- σsyst here (Sect. 4.1, p. 15 to 16) covers the SFH assumption and the metallicity range only. The dependence on the SPS model, IMF and stellar library is not in σsyst; see "Covariance" below.
- Sample (Table 1, p. 10): SDSS DR6 MGS (0.15 to 0.23), SDSS DR7 LRGs (0.3 to 0.4), stacked spectra of the Stern et al. clusters (0.38 to 0.75), zCOSMOS, K20, GOODS-S, cluster BCGs, GDDS (0.91 to 1.13), UDS, three z > 1.8 galaxies.

### Zhang et al. 2014 (RAA 14, 1221; 1207.4541v3)
- z = 0.07, 0.12, 0.20, 0.28 with 69.0 ± 19.6, 68.6 ± 26.2, 72.9 ± 29.6, 88.8 ± 36.6: abstract (p. 1) and subsample results (p. 9 to 10).
- Subsample redshift ranges from the Fig. 4 caption: 0.033 to 0.109 (zeff 0.07), 0.090 to 0.156 (0.12), 0.170 to 0.236 (0.20), 0.243 to 0.315 (0.28). Subsamples 1 and 2 overlap in 0.090 < z < 0.109.
- Method: UlySS full-spectrum SSP fit of 17 832 SDSS DR7 LRGs. Errors come from the regression on the oldest-age envelope; no separate systematic budget is given.
- Journal DOI checked on Crossref (the arXiv abstract page prints it with a doubled hyphen).

### Moresco 2015 (MNRAS 450, L16; 1503.01116)
- z = 1.363, 160 ± 33.6 and z = 1.965, 186.5 ± 50.4: Table 1 (p. 3).
- The errors are totals that include SFH, metallicity and the BC03 versus MaStro difference (abstract; Sect. 3). No stat/sys split is published. The covariance repository (below) splits each total equally, which is an assumption, not a published number.
- The first point uses the last D4000n point of Moresco et al. 2012 as the low-redshift end of the slope (Sect. 3), so it is correlated with the Moresco 2012 point at z = 1.037.

### Moresco et al. 2016 (JCAP 05 (2016) 014; 1601.01701v2)
- Table 3 (p. 15), BC03 columns: 0.3802: 83.0, 4.3, 12.9, 13.5 · 0.4004: 77.0, 2.1, 10, 10.2 · 0.4247: 87.1, 2.4, 11, 11.2 · 0.4497: 92.8, 4.5, 12.1, 12.9 · 0.4783: 80.9, 2.1, 8.8, 9.0 · averaged 0.4293: 85.7, 1, 5.1, 5.2.
- M11 columns: 89.3, 82.8, 93.7, 99.7, 86.6 and averaged 91.8 ± 5.3 (stat 1, syst 5.1). The abstract quotes the M11 average, H(0.4293) = 91.8 ± 5.3.
- Text p. 13: the averaged point "is clearly not independent of the results reported in Tab. 3, and should not be used in combination with them".
- Sample: BOSS DR9 massive passive galaxies, stacked in velocity-dispersion and redshift bins.

### Ratsimbazafy et al. 2017 (MNRAS 467, 3239; 1702.00418v2)
- z = 0.47, 89 ± 23 (stat) ± 44 (syst): abstract (p. 1); statistical part in Sect. 5 (p. 7). Quadrature total 49.65.
- Sample: SALT spectra of 2SLAQ luminous red galaxies; full-spectrum fitting, systematic from the spread between SPS models.

### Borghi et al. 2022 (ApJL 928, L4; 2110.04304)
- z = 0.75, 98.8 ± 33.6: Eq. (2), p. 4. Table 1 (p. 3): joint zeff = 0.753, H = 98.8, σstat = 24.8. Systematic 22.7 (p. 4), from 15 variations of Lick index set, binning, SPS model and SFH, added in quadrature.
- Sample: 140 massive passive galaxies in LEGA-C DR2.

### Jiao et al. 2023 (ApJS 265, 48; 2205.05701)
- z = 0.80, 113.1 ± 15.1 (stat) +29.1/−11.3 (syst): abstract (p. 1), Sect. 5 (p. 14), Conclusions (p. 16). Quadrature totals +32.78/−18.86.
- Sample: 350 LEGA-C DR2 massive passive galaxies selected in Borghi et al. 2022a (the parent sample of the 140 used by Borghi et al. 2022). BAGPIPES full-spectrum plus photometry.
- Moresco 2024 (Table 1) lists this point with a symmetric 25.22, which is 15.1 and the mean systematic (20.2) in quadrature.

### Jimenez et al. 2023 (JCAP 11 (2023) 047; 2306.11425v1)
- z = 0.75, 105.0 ± 7.9 (stat) ± 7.3 (sys): abstract (p. 1) and Eq. (3.2), p. 8. The systematic is a 6.6% SPS term added following Moresco et al. 2020.
- Method: a neural network trained on the Lick-index ages of the 140 LEGA-C galaxies of Borghi et al. 2022 assigns ages and redshifts to about 19 000 COSMOS2015 photometric galaxies at 0.65 < z < 0.85. The age scale, and part of the field, are shared with the LEGA-C measurements.

### Tomasetti et al. 2023 (A&A 679, A96; 2305.16387)
- z = 1.26, 135 ± 60 (stat) ± 27 (sys) ± 2.4 (bin) = 135 ± 65: Sect. 4.3.1 (p. 13); total also in the abstract.
- The quadrature sum of 60, 27 and 2.4 is 65.8; the paper prints 65. `sigma_sys` = 27.1 combines the SFH and binning terms.
- Sample: VANDELS DR4, 39 passive galaxies at 1.07 < z < 1.5, BAGPIPES without a cosmological age prior.

### Loubser et al. 2025 (MNRAS 540, 3135; 2506.03836v1), key `Loubser2025a`
- z = 0.5, 72.1 ± 33.9 (stat) ± 7.3 (syst): abstract (p. 1) and Eq. (6), p. 9.
- Sample: SALT spectra of 53 BCGs in ACT SZ-selected clusters at 0.3 < z < 0.7; D4000n.

### Loubser 2025 (MNRAS 544, 3064; 2511.02730v1), key `Loubser2025b`
- z = 0.46: 88.48 ± 0.57 (stat) ± 12.32 (syst); z = 0.67: 119.45 ± 6.39 ± 16.64; z = 0.83: 108.28 ± 10.07 ± 15.08. Abstract (p. 1) and Eqs. (4) to (6), p. 7 to 8.
- The systematic is 13.9% (SFH 3%, stellar library 7%, SPS model 6%, metallicity 10%). The paper gives correlation coefficients between the three points: ρ(0.46, 0.67) = 0.932, ρ(0.46, 0.83) = 0.830, ρ(0.67, 0.83) = 0.776 (p. 8).
- Sample: about 360 000 DESI DR1 galaxies with log M > 10.75, σ > 280 km s⁻¹ and no [OII] emission; D4000n.

### Tomasetti et al. 2025 (A&A 708, A145; 2512.02109v1)
- z = 0.542, 66 +81/−29 (stat) ± 13 (syst) = 66 +82/−32 (stat+syst): abstract (p. 1) and Sect. 4.1 (p. 5).
- Sample: VLT/MUSE members of SDSS J2222+2745, MACS J1149.5+2223 and SDSS J1029+2623; BAGPIPES.

### Pradhan et al. 2026 (2606.07298v1, preprint)
- z = 0.65, 93.68 ± 28.27 (stat) ± 10.67 (syst): abstract (p. 1) and Sect. 5 (p. 11), where σsyst = 0.1139 × 93.68.
- Sample: VIPERS PDR2 massive passive galaxies at 0.5 < z < 0.8, selected from rest-frame colours; Bayesian D4000 age posteriors.

### Lisiecki et al. 2026 (2608.10163v2, preprint)
- z = 0.15, 85.0 ± 51.5 (stat) +10.2/−8.9 (sys): Eq. (7), p. 6. The abstract rounds the total to ± 53; the quadrature totals are +52.50/−52.26.
- Sample: 189 relic ultra-compact massive galaxies from E-INSPIRE (SDSS spectra), 0.07 < z < 0.22; Dn4000 with α-enhancement in the systematic budget.

### Álvarez et al. 2026 (2608.13178v1, preprint)
- Cosmographic fit: H(z0 = 0.57) = 95.1 +10.9/−6.0 (stat) ± 11.3 (syst), Sect. 5.1 (p. 9) and p. 12.
- Finite differences on two velocity-dispersion groups: H(0.55) = 104.5 +13.2/−7.6 (stat) ± 22.4 (syst) and H(0.61) = 88.5 +6.7/−12.6 (stat) ± 8.1 (syst), Sect. 5.2 (p. 11).
- The paper provides a full statistical plus systematic covariance for its H(z) array (Sect. 5.1). Sample: stacked spectra of DESI DR1 LRGs, Lick indices.

## Dependence and the default subset

`use_default = 1` selects one measurement per galaxy sample: 38 points. Excluded rows (9):

| row | reason |
|---|---|
| Moresco2016, z = 0.4293, BC03 85.7 ± 5.2 | Average of the five BOSS bins, which are kept; the paper says not to combine them |
| Moresco2016, z = 0.4293, M11 91.8 ± 5.3 | Same galaxies as the five BOSS bins; M11 calibration, while all other D4000n points use BC03 |
| Jiao2023, z = 0.80 | Same LEGA-C DR2 galaxies as Borghi2022 (superset) |
| Jimenez2023, z = 0.75 | Ages calibrated on the Borghi2022 LEGA-C galaxies, same COSMOS field |
| Alvarez2026 (three rows) | Same DESI DR1 galaxies as Loubser2025b; the z0 = 0.57 row is a cosmographic fit; not peer-reviewed |
| Pradhan2026, z = 0.65 | Not peer-reviewed as of 2026-09-30; sample independent of all others |
| Lisiecki2026, z = 0.15 | Not peer-reviewed as of 2026-09-30; SDSS galaxies possibly shared with SDSS low-z points |

Choices within a dependent group:
- **LEGA-C** (Borghi 2022, Jiao 2023, Jimenez 2023): Borghi 2022 is kept. It is the LEGA-C point in the 32-point set of Moresco et al. 2022, has symmetric errors, and its systematic budget follows the Moresco et al. 2020 decomposition. Moresco 2024 (Table 1, note a) states that these three measurements must not be used together. Replacing Borghi 2022 by Jiao 2023 changes the error at z ≈ 0.8 from ±33.6 to +32.8/−18.9.
- **DESI DR1** (Loubser 2025b, Álvarez 2026): Loubser 2025b is kept because it is refereed and publishes the correlation of its three points.
- **BOSS DR9** (Moresco 2016): the five BC03 bins are kept, matching the BC03 calibration of Moresco 2012 and 2015 and the covariance repository.

Partial overlaps kept in the default, as in Moresco et al. 2022 ("all these measurements are independent, since they consider different datasets"), and flagged in `overlap_note`:
- Moresco 2012 uses stacked spectra of the Stern et al. clusters (0.38 < z < 0.75) and GDDS galaxies (0.91 < z < 1.13) that Simon 2005 also used.
- Moresco 2015 at z = 1.363 anchors its slope on the Moresco 2012 D4000n point near z = 1.
- Zhang 2014 subsamples 1 and 2 share galaxies in 0.090 < z < 0.109.
- Jimenez 2003, Zhang 2014 and the Moresco 2012 SDSS bins all draw on SDSS passive galaxies below z = 0.4, with different selections (early-type, LRG, MGS mass cut).

## Covariance and systematics

- **Moresco et al. 2020 (ApJ 898, 82; arXiv 2003.07362) covariance.** Available for the 15 D4000n points of Moresco 2012, 2015 and 2016 (BC03 and M11 versions) at https://gitlab.com/mmoresco/CCcovariance. `data/HzTable_MM_BC03.dat` lists z, H, the total error, and its statistical and metallicity parts (identical to the published σstat and σsyst of Moresco 2012 and 2016; for Moresco 2015 the total is split equally). `data/data_MM20.dat` gives the percentage systematic from IMF, stellar library and SPS model (with and without outliers) in redshift steps of 0.05. The recipe treats metallicity and young-component terms as diagonal and the model terms as fully correlated between redshifts. The published totals of these 15 points do **not** include the model terms; the full covariance adds them.
- **Borghi 2022, Jimenez 2023, Loubser 2025a and 2025b** include SPS-model terms in the published systematic, following the Moresco et al. 2020 components; Loubser 2025b also publishes the inter-point correlations. No cross-covariance with other papers is published.
- **Álvarez 2026** publishes a full covariance for its own H(z) array.
- **Jimenez 2003, Simon 2005, Stern 2010, Zhang 2014, Ratsimbazafy 2017, Jiao 2023, Tomasetti 2023 and 2025, Pradhan 2026, Lisiecki 2026**: no covariance; diagonal errors only.
- Moresco 2024 (arXiv 2412.01994, Sect. 5.1) states that its Table 1 reports "only the statistical uncertainties"; in fact its entries match the published totals (for example Borghi 98.8 ± 33.6, Jimenez 105 ± 10.76). In both readings, the SPS model covariance still has to be added for the D4000n points.
