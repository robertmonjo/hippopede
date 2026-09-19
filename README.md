# Hippopede Universe — reproduction scripts

<!-- MEMORIA-ESTRUCTURADA:INICI -->
## 🧠 memory/ — memoria estructurada (LLEGIR PRIMER)

Aquest projecte manté el seu estat de treball a **[`memory/`](memory/)**, no a la
conversa. Qualsevol agent (Claude Code, Codex, ...) ha de:

1. Obrir **[`memory/INDEX.md`](memory/INDEX.md)** abans de respondre qualsevol prompt i
   anar al `.md` que corresponga (l'entrada 0 es el protocol complet:
   [`memory/README.md`](memory/README.md)).
2. Actualitzar `memory/ESTAT.md`, `memory/PENDENTS.md` i el `.md` tematic **despres de
   cada pas rellevant**. Si no consta a `memory/`, no s'ha fet.
3. Abans de qualsevol compactacio de context, bolcar-hi tot el que encara nomes viu a la
   conversa.

La conversa es volatil i es compacta; `memory/` no. No preguntes a l'usuari res que ja
estiga escrit ací.
<!-- MEMORIA-ESTRUCTURADA:FI -->


Scripts and data to reproduce the figures and numerical diagnostics in:

> R. Monjo, M. Romero, A. E. Sasi,  
> *Hippopede Universe as a toy model of late-linear expansion*,  
> Physics of the Dark Universe (2026), submitted.

---

## Scripts

| Script | Output |
|---|---|
| `scripts/plot_hippopede_qz_double_panel_with_gapp.py` | Fig. 2 — $q(z)$ and $E(z)$, centered and projected hippopede histories with GaPP reconstruction |
| `scripts/plot_hippopede_projected_thermal_history_bbn.py` | Fig. 3 — projected thermal history versus standard BBN corridor |
| `scripts/analyze_hippopede_variable_alpha.py` | Running-$\alpha$ scan; BBN thermal window diagnostics (Appendix) |
| `scripts/hz_background_models.py` | Background $H(z)$ comparison curves |
| `scripts/reconstruct_qz_mukherjee2021_gapp.py` | GaPP reconstruction of $q(z)$ from CC+Pantheon+ |
| `scripts/obsolete_bbn/analyze_hippopede_dipole_bbn.py` | Dipole fit and constant-$\alpha$ BBN check (superseded by `analyze_hippopede_variable_alpha.py`) |

Fig. 1 (3D hippopede growth) and the BAO fit figure are produced by scripts not yet deposited here; the output figures are in `figures/`.

## Data

```
data/hz_background/hz_curated_chronometers.csv   — cosmic-chronometer H(z) compilation
data/hz_background/hz_curated_bao.csv            — BAO H(z) measurements
data/gapp/pantheon_plus_binned_50.csv            — Pantheon+ binned distance moduli (input for GaPP)
data/Ardra/desi_dr1_bao_galqso_lya_mean.csv      — DESI DR1 compressed BAO means
data/Ardra/desi_dr1_bao_galqso_lya_cov.csv       — DESI DR1 covariance matrix
data/Ardra/hippopede_bao_fit_summary.json         — projected-model BAO fit results
```

The GaPP Gaussian-process code used by `reconstruct_qz_mukherjee2021_gapp.py` is included under `vendor_gapp/` (original: Seikel et al. 2012, [arXiv:1204.2832](https://arxiv.org/abs/1204.2832)).

## Requirements

```
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.10
```

Install with:

```bash
pip install numpy matplotlib scipy
```

## Usage

```bash
python scripts/plot_hippopede_qz_double_panel_with_gapp.py
python scripts/analyze_hippopede_variable_alpha.py
```

Figures are written to `figures/`.

## Authors

Robert Monjo, Manel Romero, Ardra E. Sasi

## Licence

MIT
