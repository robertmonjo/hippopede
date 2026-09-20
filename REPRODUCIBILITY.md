# Hippopede figure reproducibility

Run all scripts from the **project root** (the directory that contains `scripts/`, `data/`, `vendor_gapp/`, and `figures/`).

## GaPP double-panel figure (Fig. 2)

```bash
python scripts/plot_hippopede_qz_double_panel_with_gapp.py
```

Writes `figures/hippopede_qz_centered_t0_3p0_double_panel_with_gapp.(png|pdf)`.

Requires `vendor_gapp/` (bundled). NumPy ≥ 2.0 is supported via the compatibility shim already present in `vendor_gapp/`.

## BBN thermal history figure (Fig. 3)

```bash
python scripts/plot_hippopede_projected_thermal_history_bbn.py
```

Writes `figures/hippopede_projected_thermal_history_bbn.(png|pdf|json)`.

## DESI DR1 BAO fit

```bash
python scripts/hippopede_bao_fit_compute.py
```

Writes `data/Ardra/hippopede_bao_fit_results.json`.
Reproduces: 10-pt (no Lyα) alpha=0.375 (+0.017/-0.016), chi2=12.33, DeltaAIC=0.68.

## 3D growth figure (Fig. 1)

```bash
python scripts/hippopede_growth_3d.py
```

## Bundled inputs

```
data/hz_background/hz_curated_chronometers.csv   — cosmic-chronometer H(z) compilation
data/hz_background/hz_curated_bao.csv            — BAO H(z) measurements
data/gapp/pantheon_plus_binned_50.csv            — Pantheon+ binned distance moduli
data/Ardra/desi_dr1_bao_galqso_lya_mean.csv      — DESI DR1 compressed BAO means
data/Ardra/desi_dr1_bao_galqso_lya_cov.csv       — DESI DR1 covariance matrix
vendor_gapp/                                      — GaPP code (Seikel et al. 2012)
```

## Python requirements

```
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.10
```
