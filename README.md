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

downloads the public data, runs every fit, writes the figures and the JSON outputs to `figures/`,
and checks the numbers quoted in the paper. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the
data, the script behind each result and the order of the steps.

## Authors

Robert Monjo, Manel Romero, Ardra E. Sasi

## Licence

Code: MIT (see `LICENSE`). Data keep the licences of their sources.
