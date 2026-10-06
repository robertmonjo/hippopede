# Hippopede universe: reproduction scripts

Scripts and data to reproduce the figures and numerical results in:

> R. Monjo, M. Romero, A. E. Sasi,
> *Hippopede universe to model sector-dependent acceleration*,
> submitted to The European Physical Journal C (2026).

## Running

```
pip install -r requirements.txt
python scripts/run_all.py
```

downloads the public data, runs every fit, writes the figures to `figures/` and the JSON outputs to `json/`,
and checks the numbers quoted in the paper. The analyses of the average over the light cone of the
observer, the quasar dipoles and the unbinned supernovae follow with

```
python scripts/run_all.py --extended
```

which needs a many-core machine and several hours; their outputs are included in `figures/` and `json/`. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the data, the script behind each result and the order
of the steps.

## Authors

Robert Monjo, Manel Romero, Ardra E. Sasi

## Licence

Code: MIT (see `LICENSE`). Data keep the licences of their sources.
