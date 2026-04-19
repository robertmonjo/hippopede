# Hippopede figure reproducibility

This folder is self-contained for the final four-panel figure:
`figures/hippopede_qz_centered_t0_3p0_double_panel_with_gapp.(png|pdf)`.

## Main script

- `plot_hippopede_qz_double_panel_with_gapp.py`

## Bundled local inputs

- `data/hz_background/hz_curated_chronometers.csv`
- `data/hz_background/hz_curated_bao.csv`
- `data/gapp/pantheon_plus_binned_50.csv`
- `vendor_gapp/`

## Python requirements

Install the minimal dependencies with:

```bash
pip install -r requirements_hippopede_figure.txt
```

## Rebuild

Run:

```bash
python plot_hippopede_qz_double_panel_with_gapp.py
```

The script writes the final outputs to `figures/`.

## Notes

- The project uses a bundled local copy of the GaPP code under `vendor_gapp/`.
- The Pantheon+ contribution is provided here as the pre-binned file `pantheon_plus_binned_50.csv`, exactly in the form consumed by the plotting and reconstruction scripts.
