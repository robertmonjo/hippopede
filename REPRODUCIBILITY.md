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
| q0, z_t and spread of E for Fig. 2 (Sect. 4.1) | `sector_kinematics.py --from-average-fit` | `figures/sector_kinematics_sector_average_alpha.json` |
| z_t with the index fitted to the axial sector, sector of a source for the axis observer (Sects. 4.1, 4.3) | `sector_kinematics.py` | `figures/sector_kinematics.json` |
| Redshift reach of the sectors, z_h (Sects. 3.2, 4.4; Appendix G) | `sector_reach.py` | `figures/sector_reach.json` |
| Double-precision evaluation at high redshift (Appendix G) | `verify_high_z.py` | `figures/verify_high_z.json` |
| Fig. 1 | `hippopede_growth_3d.py` | `figures/hippopede_growth_3d.png` |
| Fig. 2 | `plot_hippopede_qz_double_panel_with_gapp.py --from-average-fit --suffix _sector_average_alpha` | `figures/hippopede_qz_double_panel_with_gapp_sector_average_alpha.(png, pdf)` |
| Fig. 3 | `plot_band_comparison.py --from-average-fit --suffix _sector_average_alpha` | `figures/hippopede_band_comparison_sector_average_alpha.(png, pdf)` |
| Direction dependence with unbinned Pantheon+, spread of mu (Sect. 4.4) | `pantheon_offaxis_test.py` | `figures/pantheon_offaxis_test.json` |
| Count dipole and the CatWISE excess, theta_obs(t0) (Sect. 4.5) | `quasar_dipole_fit.py` | `figures/quasar_dipole_fit.json` |
| Distance asymmetry, H0 and q0 anisotropy, CatWISE configuration (Sect. 4.5) | `directional_anomalies.py` | `figures/directional_anomalies.json` |
| Quaia dipole, full sample and redshift slices (Sect. 4.5; Appendix I) | `dipole_zslices/measure_dipole.py` | `dipole_zslices/results.json` |
| Model shape against the Quaia slices (Sect. 4.5) | `quaia_zslice_model.py` | `figures/quaia_zslice_model.json` |
| Effective q0 dipole and axis crossing (Sect. 5) | `local_environment.py` | `figures/local_environment.json` |
| Every number quoted in the paper | `check_paper_numbers.py` | printed report |

## Light-cone average, dipoles and unbinned supernovae

These analyses use the average over the light-cone ball of the observer (reading (iv) of the
statistical analysis, Sects. 3.3, 3.5 and 3.6) and are run with `python scripts/run_all.py --extended`
after the main pipeline. They are computationally heavy: the grids of the last two steps take of the
order of an hour on 28 cores and several gigabytes of memory per hundred values of theta_obs if the
caches of `fit_observer_ball.py` are kept, which the scripts avoid.

| Result | Script | Output |
|---|---|---|
| Intrinsic scatter of the chronometers and spread of H across sectors (Sect. 3.3) | `cc_intrinsic_scatter.py` | `figures/cc_intrinsic_scatter.json` |
| Direction dependence of the chronometers about the CatWISE axis, and the model prediction (Sect. 3.5) | `cc_direction_test.py` | `figures/cc_direction_test.json` |
| Band match and sector-averaged fit with the nodes z = 0.5, ..., 1.9 | `t0_dispersion_match.py --z-nodes 0.5 1.9`, `fit_sector_average.py --z-nodes 0.5 1.9` | `figures/t0_dispersion_match_z0p5-1p9.json`, `figures/fit_sector_average_z0p5-1p9.json` |
| Count dipoles at alpha_high = 0.30, ..., 0.44 for 1 <= t0 <= 10 (input of the next steps) | `quasar_dipole_fit.py --alpha-high A --t0-range 1 10 --t0-n 30`, `quaia_zslice_model.py --alpha-high A --t0-range 1 10 --t0-n 30` | `figures/quasar_dipole_fit_ah<A>.json`, `figures/quaia_zslice_model_ah<A>.json` |
| Pantheon+ with the axis fixed by the CatWISE excess, random-axis calibration (Sect. 3.5) | `pantheon_catwise_axis.py` | `figures/pantheon_catwise_axis.json` |
| Landscape of the light-cone average with binned supernovae, joint fit with the dipoles (Sect. 3.6) | `plot_observer_ball_landscape.py --t0-range 1 10 --nt 80 --theta-max 45 --theta-step 0.25 --alpha-range 0.27 0.47 --tag joint_wide --dipole-alphas 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44` | `figures/fit_observer_ball_landscape_joint_wide.json`, `figures/hippopede_observer_ball_landscape_joint_wide.png` |
| Single lines of the ball model (sector 0, mean, median) with binned supernovae | `compare_ball_curves.py` | `figures/compare_ball_curves.json` |
| Joint fit with the unbinned supernovae, isotropic and directional, with CatWISE, free-axis check | `fit_ball_unbinned.py` | `figures/fit_ball_unbinned.json` |
| Single lines with the unbinned supernovae, plain and region-weighted statistic, dipoles at each fit | `fit_lines_unbinned.py` | `figures/fit_lines_unbinned.json` |
| Effect of the binning of the supernovae on model comparisons | `binning_resolution_test.py` | printed report |
| Map of the light-cone average with the unbinned supernovae, CatWISE and Quaia bands, region compatible with all data within 1 sigma | `plot_ball_zoom_unbinned.py --t0-range 1.4 4.0 --nt 60 --theta-max 20 --theta-step 0.05 --tag _wide` | `figures/ball_zoom_unbinned_wide.json`, `figures/hippopede_ball_zoom_unbinned_wide.png` |

The map is computed on the grid given by the options; the figure interpolates the fields bilinearly
(in ln t0 and in theta_obs relative to the largest allowed theta_obs at each t0) onto a finer display
grid, and the bands and the summary region are evaluated there. `check_paper_numbers.py` covers the
numbers of Sects. 3.3 and 3.5.

## Modules

`sector_geometry.py` (sector of a source, volume weight of a sector, the limits
THETA_MAX_DEG = 70 deg and THETA_OBS_MAX_DEG = 35 deg), `hippopede_model.py` (centred sector
histories and their projection), `projected_hyperconical.py` (distorted projection with running
index, stable to z ~ 1e10), `data_loaders.py` and `likelihood.py` (data and chi-square with full
covariances), `fit_observer_ball.py` (average over the light-cone ball of an observer, used by the
analyses of Sects. 3.5 and 3.6) and `fit_lightcone_ball.py` (its band figure).

`scripts/archive/` holds earlier scripts that the paper does not use.
