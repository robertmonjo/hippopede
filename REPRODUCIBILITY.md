# Reproducibility

Every figure and every number of the paper is produced by the scripts in `scripts/` (and two in
`dipole_zslices/`). The whole analysis runs from scratch, data downloads included, with

```
pip install -r requirements.txt
python scripts/run_all.py
```

`run_all.py` executes the steps below in order, stops at the first failure, and ends with
`scripts/check_paper_numbers.py`, which compares the values quoted in the manuscript with the
outputs. All outputs are written to `figures/` (and `dipole_zslices/results.json`).

## Data

| File | Content | Obtained by |
|---|---|---|
| `data/cc_compilation/cc_compilation.csv` | 47 cosmic-chronometer measurements, 38 independent (`use_default = 1`), with references, DOIs and the reason for each exclusion | compiled from the publications listed in `data/cc_compilation/sources.md` |
| `data/cc_compilation/data_MM20.dat` | stellar-population systematics of Moresco et al. (2020) | `scripts/build_cc_covariance.py` (download) |
| `data/cc_compilation/cc_covariance.csv` | covariance of the 38 chronometers | `scripts/build_cc_covariance.py` |
| `data/pantheon_plus/` | Pantheon+ catalogue and STAT+SYS covariance (Scolnic et al. 2022; Brout et al. 2022) | `scripts/download_pantheon_plus.py` |
| `data/gapp/pantheon_plus_binned_50_gls.csv`, `..._gls_cov_d.csv` | 50 bins of Pantheon+ and their covariance | `scripts/bin_pantheon_plus.py` |
| `dipole_zslices/data/` | Quaia catalogue and selection function (Storey-Fisher et al. 2024), about 270 MB | `dipole_zslices/download_quaia.py` |
| `vendor_gapp/` | GaPP code (Seikel, Clarkson & Smith 2012) | bundled |

`data/hz_background/hz_curated_chronometers.csv` and `data/gapp/pantheon_plus_binned_50.csv`
are an earlier selection of 18 chronometers and an earlier binning; they are used only by the
optional reconstruction `gapp_reconstruction.py --cc curated --sn curated --h0 shoes`, for
comparison, and not in the paper.

## Results and the scripts that produce them

| Result in the paper | Script | Output |
|---|---|---|
| Table 1, chronometer covariance (Sect. 2) | `build_cc_covariance.py` | `data/cc_compilation/cc_covariance.csv` |
| GaPP reconstruction (Sect. 2; Figs. 2, 3) | `gapp_reconstruction.py` | `figures/gapp_reconstruction_compilation_gls_cc.json` |
| alpha_high fitted to the axial sector (Sects. 4.1, 4.2) | `fit_alpha_high.py` | `figures/fit_alpha_high.json` |
| t0 from the band match (Sect. 4.3, Eq. of M(t0)) | `t0_dispersion_match.py` | `figures/t0_dispersion_match.json` |
| Observer on the axis, t0 bounds (Sect. 4.3) | `fit_t0_axis_observer.py` | `figures/fit_t0_axis_observer.json` |
| Sector spread as intrinsic scatter of the supernovae (Sect. 4.3) | `fit_t0_intrinsic_scatter.py` | `figures/fit_t0_intrinsic_scatter.json` |
| Reference t0 used by the other scripts | `t0_summary.py` | `figures/t0_summary.json` |
| Table 2, t0 bounds per sector (Sect. 4.2) | `fit_cc_pantheon.py` | `figures/fit_cc_pantheon.json` |
| Fit to the sector-averaged history, offsets from the GaPP median (Sect. 4.3, Table 2) | `fit_sector_average.py` | `figures/fit_sector_average.json` |
| q0, z_t, spread of E, sector of a source for the axis observer (Sects. 4.1, 4.3) | `sector_kinematics.py` | `figures/sector_kinematics.json` |
| Redshift reach of the sectors, z_h (Sects. 3.2, 4.4; Appendix G) | `sector_reach.py` | `figures/sector_reach.json` |
| Double-precision evaluation at high redshift (Appendix G) | `verify_high_z.py` | `figures/verify_high_z.json` |
| Fig. 1 | `hippopede_growth_3d.py` | `figures/hippopede_growth_3d.png` |
| Fig. 2 | `plot_hippopede_qz_double_panel_with_gapp.py` | `figures/hippopede_qz_double_panel_with_gapp.(png, pdf)` |
| Fig. 3 | `plot_band_comparison.py --from-average-fit --suffix _sector_average_alpha` | `figures/hippopede_band_comparison_sector_average_alpha.(png, pdf)` |
| Direction dependence with unbinned Pantheon+, spread of mu (Sect. 4.4) | `pantheon_offaxis_test.py` | `figures/pantheon_offaxis_test.json` |
| Count dipole and the CatWISE excess, theta_obs(t0) (Sect. 4.5) | `quasar_dipole_fit.py` | `figures/quasar_dipole_fit.json` |
| Distance asymmetry, H0 and q0 anisotropy, CatWISE configuration (Sect. 4.5) | `directional_anomalies.py` | `figures/directional_anomalies.json` |
| Quaia dipole, full sample and redshift slices (Sect. 4.5; Appendix I) | `dipole_zslices/measure_dipole.py` | `dipole_zslices/results.json` |
| Model shape against the Quaia slices (Sect. 4.5) | `quaia_zslice_model.py` | `figures/quaia_zslice_model.json` |
| Effective q0 dipole and axis crossing (Sect. 5) | `local_environment.py` | `figures/local_environment.json` |
| Every number quoted in the paper | `check_paper_numbers.py` | printed report |

## Modules

`sector_geometry.py` (sector of a source, volume weight of a sector, the limits
THETA_MAX_DEG = 70 deg and THETA_OBS_MAX_DEG = 35 deg), `hippopede_model.py` (centred sector
histories and their projection), `projected_hyperconical.py` (distorted projection with running
index, stable to z ~ 1e10), `data_loaders.py` and `likelihood.py` (data and chi-square with full
covariances).

`scripts/archive/` holds earlier scripts that the paper does not use.
