# Reproducibility

Every figure and every number of the paper is produced by the scripts in `scripts/` (and two in
`dipole_zslices/`). The whole analysis runs from scratch, data downloads included, with

```
pip install -r requirements.txt
python scripts/run_all.py
```

`run_all.py` executes the steps below in order, stops at the first failure, and ends with
`scripts/check_paper_numbers.py`, which compares the values quoted in the manuscript with the
outputs. Figures are written to `figures/`, numerical outputs to `json/` (and `dipole_zslices/results.json`).

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
| GaPP reconstruction (Sect. 2; Figs. 2, 3) | `gapp_reconstruction.py` | `json/gapp_reconstruction_compilation_gls_cc.json` |
| alpha_high fitted to the axial sector (Sects. 4.1, 4.2) | `fit_alpha_high.py` | `json/fit_alpha_high.json` |
| t0 from the band match (Sect. 4.3, Eq. of M(t0)) | `t0_dispersion_match.py` | `json/t0_dispersion_match.json` |
| Observer on the axis, t0 bounds (Sect. 4.3) | `fit_t0_axis_observer.py` | `json/fit_t0_axis_observer.json` |
| Sector spread as intrinsic scatter of the supernovae (Sect. 4.3) | `fit_t0_intrinsic_scatter.py` | `json/fit_t0_intrinsic_scatter.json` |
| Reference t0 used by the other scripts | `t0_summary.py` | `json/t0_summary.json` |
| Table 2, t0 bounds per sector (Sect. 4.2) | `fit_cc_pantheon.py` | `json/fit_cc_pantheon.json` |
| Fit to the sector-averaged history, offsets from the GaPP median (Sect. 4.3, Table 2) | `fit_sector_average.py` | `json/fit_sector_average.json` |
| q0, z_t and spread of E for Fig. 2 (Sect. 4.1) | `sector_kinematics.py --from-average-fit` | `json/sector_kinematics_sector_average_alpha.json` |
| z_t with the index fitted to the axial sector, sector of a source for the axis observer (Sects. 4.1, 4.3) | `sector_kinematics.py` | `json/sector_kinematics.json` |
| Redshift reach of the sectors, z_h (Sects. 3.2, 4.4; Appendix G) | `sector_reach.py` | `json/sector_reach.json` |
| Double-precision evaluation at high redshift (Appendix G) | `verify_high_z.py` | `json/verify_high_z.json` |
| Fig. 1 | `hippopede_growth_3d.py` | `figures/hippopede_growth_3d.png` |
| Fig. 2 | `plot_hippopede_qz_double_panel_with_gapp.py --from-average-fit --suffix _sector_average_alpha` | `figures/hippopede_qz_double_panel_with_gapp_sector_average_alpha.(png, pdf)` |
| Fig. 3 | `plot_band_comparison.py --from-average-fit --suffix _sector_average_alpha` | `figures/hippopede_band_comparison_sector_average_alpha.(png, pdf)` |
| Direction dependence with unbinned Pantheon+, spread of mu (Sect. 4.4) | `pantheon_offaxis_test.py` | `json/pantheon_offaxis_test.json` |
| Count dipole and the CatWISE excess, theta_obs(t0) (Sect. 4.5) | `quasar_dipole_fit.py` | `json/quasar_dipole_fit.json` |
| Distance asymmetry, H0 and q0 anisotropy, CatWISE configuration (Sect. 4.5) | `directional_anomalies.py` | `json/directional_anomalies.json` |
| Quaia dipole, full sample and redshift slices (Sect. 4.5; Appendix I) | `dipole_zslices/measure_dipole.py` | `dipole_zslices/results.json` |
| Model shape against the Quaia slices (Sect. 4.5) | `quaia_zslice_model.py` | `json/quaia_zslice_model.json` |
| Effective q0 dipole and axis crossing (Sect. 5) | `local_environment.py` | `json/local_environment.json` |
| Every number quoted in the paper | `check_paper_numbers.py` | printed report |

## Light-cone average, dipoles and unbinned supernovae

These analyses use the average over the light-cone ball of the observer (reading (iv) of the
statistical analysis, Sects. 3.3, 3.5 and 3.6) and are run with `python scripts/run_all.py --extended`
after the main pipeline. They are computationally heavy: the grids of the last two steps take of the
order of an hour on 28 cores and several gigabytes of memory per hundred values of theta_obs if the
caches of `fit_observer_ball.py` are kept, which the scripts avoid.

| Result | Script | Output |
|---|---|---|
| Intrinsic scatter of the chronometers and spread of H across sectors (Sect. 3.3) | `cc_intrinsic_scatter.py` | `json/cc_intrinsic_scatter.json` |
| Direction dependence of the chronometers about the CatWISE axis, and the model prediction (Sect. 3.5) | `cc_direction_test.py` | `json/cc_direction_test.json` |
| Band match and sector-averaged fit with the nodes z = 0.5, ..., 1.9 | `t0_dispersion_match.py --z-nodes 0.5 1.9`, `fit_sector_average.py --z-nodes 0.5 1.9` | `json/t0_dispersion_match_z0p5-1p9.json`, `json/fit_sector_average_z0p5-1p9.json` |
| Count dipoles at alpha_high = 0.30, ..., 0.44 for 1 <= t0 <= 10 (input of the next steps) | `quasar_dipole_fit.py --alpha-high A --t0-range 1 10 --t0-n 30`, `quaia_zslice_model.py --alpha-high A --t0-range 1 10 --t0-n 30` | `json/quasar_dipole_fit_ah<A>.json`, `json/quaia_zslice_model_ah<A>.json` |
| Pantheon+ with the axis fixed by the CatWISE excess, random-axis calibration (Sect. 3.5) | `pantheon_catwise_axis.py` | `json/pantheon_catwise_axis.json` |
| Landscape of the light-cone average with binned supernovae, joint fit with the dipoles (Sect. 3.6) | `plot_observer_ball_landscape.py --t0-range 1 10 --nt 80 --theta-max 45 --theta-step 0.25 --alpha-range 0.27 0.47 --tag joint_wide --dipole-alphas 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44` | `json/fit_observer_ball_landscape_joint_wide.json`, `figures/hippopede_observer_ball_landscape_joint_wide.png` |
| Single lines of the ball model (sector 0, mean, median) with binned supernovae | `compare_ball_curves.py` | `json/compare_ball_curves.json` |
| Joint fit with the unbinned supernovae, isotropic and directional, with CatWISE, free-axis check | `fit_ball_unbinned.py` | `json/fit_ball_unbinned.json` |
| Single lines with the unbinned supernovae, plain and region-weighted statistic, dipoles at each fit | `fit_lines_unbinned.py` | `json/fit_lines_unbinned.json` |
| Effect of the binning of the supernovae on model comparisons | `binning_resolution_test.py` | printed report |
| Finer map near the minima, 1.4 <= t0 <= 4 and theta_obs <= 20 deg (CC + SN and CC + SN + CatWISE minima of Sect. 3.6) | `plot_ball_zoom_unbinned.py --t0-range 1.4 4.0 --nt 60 --theta-max 20 --theta-step 0.05 --tag _wide --best-all-json refine_overview_minimum.json` (after the overview steps below) | `json/ball_zoom_unbinned_wide.json`, `figures/hippopede_ball_zoom_unbinned_wide.png` |
| Count dipoles at alpha_high = 0.30, ..., 0.70 for 1 <= t0 <= 20 and theta_obs <= 45 deg, whole lobe tabulated (input of the next step; the Quaia slices are computed at each theta_obs) | `HIPPOPEDE_THETA_MAX_DEG=90 quasar_dipole_fit.py --alpha-high A --t0-range 1 20 --t0-n 40 --theta-obs 0 0.5 1 2 3 5 7 10 12.5 15 ... 45 --out-suffix _wide`, the same with `quaia_zslice_model.py` | `json/quasar_dipole_fit_ah<A>_wide.json`, `json/quaia_zslice_model_ah<A>_wide.json` |
| Map of the light-cone average (Sect. 3.6, Fig. ball), 1.4 <= t0 <= 20 and theta_obs <= 45 deg, sectors up to 90 deg limited only by their reach; region where alpha_high is at an end of its allowed range hatched | `HIPPOPEDE_THETA_MAX_DEG=90 plot_ball_zoom_unbinned.py --t0-range 1.4 20 --nt 40 --theta-nodes 0 0.25 0.5 1 1.5 2 2.5 3 4 5 ... 45 --alpha-range 0.29 0.71 --display-theta-step 0.05 --nt-fine 1100 --alpha-levels 0.35 0.40 0.45 --hatch-unresolved --dipole-alphas 0.30 0.32 ... 0.70 --dipole-suffix _wide --tag _overview`; after the two rows below, the figure of the paper is redrawn with `--reuse --best-all-json refine_overview_minimum.json --markers-json ball_zoom_unbinned_wide.json --no-title --plot-t0-max 10` | `json/ball_zoom_unbinned_overview.json`, `figures/hippopede_ball_zoom_unbinned_overview.png` |
| Profile along t0 of the CC + SN + CatWISE + Quaia chi2 and its continuous minimum (Nelder-Mead, CC + SN recomputed at each point) | `HIPPOPEDE_THETA_MAX_DEG=90 refine_overview_minimum.py` | `json/refine_overview_minimum.json` |
| Joint chi2 of CC + SN + CatWISE + Quaia for LCDM, the hyperconical model, LCDM with a phenomenological dipole (one common amplitude, or one for CatWISE and one for Quaia) and the light-cone ball | `compare_dipole_models.py` | `json/compare_dipole_models.json` |
| Blind prediction of the quasar dipoles by the light-cone ball fitted to CC + SN only (predictive distribution of the CatWISE amplitude, predictive likelihood against LCDM and LCDM with an ad hoc amplitude) | `HIPPOPEDE_THETA_MAX_DEG=90 predict_dipole_from_ccsn.py` | `json/predict_dipole_from_ccsn.json` |

The map is computed on the grid given by the options; the figure interpolates the chi2 of the chronometers
and supernovae and the fitted alpha_high with monotone piecewise-cubic (PCHIP) interpolation (in ln t0 and in
theta_obs relative to the largest allowed theta_obs at each t0) onto a finer display grid. The CatWISE and Quaia
chi2 are formed at every point of that grid from the dipole amplitudes of the runs, interpolated in the same way
and in alpha_high, and the bands and the summary region are evaluated there. The minimum with the Quaia slices
comes from the continuous fit of `refine_overview_minimum.py` (`--best-all-json`). With irregular theta_obs nodes the largest
allowed theta_obs at each (alpha_high, t0) is located by bisection, so the edge does not depend on the node
spacing. The sector tables of the paper stop at THETA_MAX_DEG = 70 deg (sector_geometry.py); the environment
variable HIPPOPEDE_THETA_MAX_DEG replaces it, and `run_all.py` sets it to 90 for the overview steps.
`check_paper_numbers.py` covers the
numbers of Sects. 3.3 and 3.5.

## Modules

`sector_geometry.py` (sector of a source, volume weight of a sector, the limits
THETA_MAX_DEG = 70 deg and THETA_OBS_MAX_DEG = 35 deg), `hippopede_model.py` (centred sector
histories and their projection), `projected_hyperconical.py` (distorted projection with running
index, stable to z ~ 1e10), `data_loaders.py` and `likelihood.py` (data and chi-square with full
covariances), `fit_observer_ball.py` (average over the light-cone ball of an observer, used by the
analyses of Sects. 3.5 and 3.6) and `fit_lightcone_ball.py` (its band figure).

`scripts/archive/` holds earlier scripts that the paper does not use.
