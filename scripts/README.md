# Scripts

All scripts are run from the **project root** (the directory containing `scripts/`, `data/`, `vendor_gapp/`, and `figures/`).

## Active scripts for this paper

| Script | Purpose |
|---|---|
| `hippopede_growth_3d.py` | Fig. 1 — 3D growth and projection |
| `plot_hippopede_qz_double_panel_with_gapp.py` | Fig. 2 — q(z)/E(z) with GaPP |
| `plot_hippopede_projected_thermal_history_bbn.py` | Fig. 3 — BBN thermal history |
| `hippopede_bao_fit_compute.py` | DESI DR1 BAO chi2 fit; writes `data/Ardra/hippopede_bao_fit_results.json` |
| `hippopede_bao_fit_data.py` | BAO fit diagnostic plot (fixed best-fit values) |
| `analyze_hippopede_variable_alpha.py` | Running-alpha scan (uses a rational form; for the tanh form used in the appendix, see `plot_hippopede_projected_thermal_history_bbn.py`) |
| `hz_background_models.py` | Background H(z) comparison |
| `reconstruct_qz_mukherjee2021_gapp.py` | GaPP q(z) reconstruction |

## obsolete_bbn/

`obsolete_bbn/analyze_hippopede_dipole_bbn.py` and related files are kept for internal traceability. The active BBN implementation for this paper is `plot_hippopede_projected_thermal_history_bbn.py`, which imports from `obsolete_bbn/` but adds the running-alpha thermal history used in the manuscript.
