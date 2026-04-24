# Scripts Status

The `hippopede/scripts` folder no longer contains the active BBN workflow.

## BBN status

The older BBN scripts that were developed inside `hippopede` have been moved to:

- `obsolete_bbn/`

They are kept only for internal traceability and should be treated as superseded.

## Use these scripts instead

The active and maintained BBN / thermal-history workflow now lives in the sibling project:

- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\cmb_hyperconical\scripts\cmb_bbn_effective_branch.py`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\cmb_hyperconical\scripts\cmb_bbn_asymptotic_diagnostics.py`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\cmb_hyperconical\scripts\cmb_hyperconical_bbn_thermal_history.py`

## Why

The BBN treatment has been consolidated in `cmb_hyperconical` so there is a single maintained implementation for:

- the radiation-calibrated temperature mapping,
- the asymptotic `H(T)` diagnostics,
- and the effective BBN interpolation branch.

This avoids duplicate logic across projects and reduces the risk of mixing obsolete and current results.
