# BBN Transfer Note for `cmb_hyperconical`

This note summarizes the BBN-related work done in the `hippopede` thread up to April 24, 2026, so it can be continued from the `cmb_hyperconical` thread without losing context.

## Scope

This document is only about the BBN/thermal-history side of the discussion.
The `hippopede` thread should remain focused on the anisotropic/hippopede model itself.

## Exact location of this note

`C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\docs\BBN_TRANSFER_NOTE_FOR_CMB_HYPERCONICAL.md`

## Repositories involved

- Main repo used for the present note:
  `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede`
- Sibling repo where the BBN discussion should continue:
  `C:\Users\rober\OneDrive\Documents\Codex_Portatil\cmb_hyperconical`

## Executive summary

There are now **two distinct BBN constructions** in play:

1. **Literal projected-map BBN test**
   This uses the Monjo-type projected map directly and computes the thermal history from the literal derivative of the projected radius.

2. **Effective interpolated BBN completion**
   This constructs a phenomenological effective thermal history by interpolating between the two constant-`\alpha` projected branches, `\alpha=0.283` and `\alpha=0.5`.

The important conclusion is:

- The **literal running-`\alpha(z)` map is not robustly extendable to the full `0.07–1 MeV` BBN window** with the present implementation.
- The **effective interpolated branch does cover the full `0.07–1 MeV` window** and can be made to reproduce the BBN-inferred central expansion history essentially exactly.

So, if the next thread wants to continue the BBN programme in a productive way, the recommended path is to work from the **effective interpolation** and only later try to derive a geometrically more fundamental map that reproduces it.

## What was tested

### A. Constant-`\alpha` projected branches

Two baseline projected branches were used throughout:

- `\alpha = 0.283`
- `\alpha = 0.5`

These are implemented via the standard projected hyperconical map.

Main script for the literal BBN figure:

- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\scripts\plot_hippopede_projected_thermal_history_bbn.py`

Outputs:

- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_projected_thermal_history_bbn.png`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_projected_thermal_history_bbn.pdf`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_projected_thermal_history_bbn.json`

### B. Literal running-`\alpha(z)` projected map

Several versions of a running-`\alpha` branch were tried, with

`\alpha(z)` moving between `0.283` and `0.5`.

Main findings:

- The branch can be kept near the BBN-inferred band only in a **limited low-temperature window**.
- In the present literal implementation, the branch starts to leave the desired `R(T) = H(T)/H_obs(T) ~ 1` regime around
  `T ~ 0.1–0.15 MeV`.
- Beyond that, the deviation is **not just plotting noise**. It reflects the current limitations of the literal map.

### C. Rigorous derivative checks

This was an important debugging step.

We explicitly checked that:

- for **constant** `\alpha`, the analytic derivative of the projected map reproduces the existing implementation to machine precision;
- for **running** `\alpha`, the fully rigorous analytic derivative gives a result very different from the earlier optimistic numerical extension.

This means:

- the earlier apparent agreement of the literal running branch up to `1 MeV` was **too optimistic**;
- it relied on a numerically forgiving differentiation procedure;
- the mathematically stricter treatment shows that the present running-`\alpha(z)` literal map does **not** yet sustain a clean BBN extension to `1 MeV`.

## Why the literal map is not enough yet

The present literal running map behaves acceptably only in a limited window because:

- it is normalized at `H(0)=H_0`,
- it inherits the near-boundary sensitivity of the projected map,
- and once pushed to very high redshift/temperature it becomes dominated by the geometry of the map itself rather than by the desired BBN-like target behaviour.

This is why the blue branch in the literal figure was finally shown only on its **stable domain**, rather than being extended dishonestly.

## Effective interpolation: the productive way forward

The physically useful observation was:

we already have two projected branches,

- `H_{0.283}(T)`
- `H_{0.5}(T)`

so we can construct an **effective branch** between them.

### Formula used

The interpolation is done in logarithmic space:

`\ln H_eff(T) = [1 - w(T)] \ln H_{0.283}(T) + w(T)\ln H_{0.5}(T)`

equivalently,

`H_eff(T) = H_{0.283}(T)^{1-w(T)} H_{0.5}(T)^{w(T)}`

The key point is that the weight was **not** imposed with an arbitrary logistic ansatz in the final version.

Instead, it was **deduced from the BBN-inferred central curve**:

`w(T) = [\ln H_obs(T) - \ln H_{0.283}(T)] / [\ln H_{0.5}(T) - \ln H_{0.283}(T)]`

and then clipped to the physical interval:

`0 <= w(T) <= 1`.

This directly yields

`alpha_eff(T) = 0.283 + (0.5 - 0.283) w(T)`.

### Effective-interpolation script

- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\scripts\plot_hippopede_effective_interpolated_thermal_history_bbn.py`

Outputs:

- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_effective_interpolated_thermal_history_bbn.png`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_effective_interpolated_thermal_history_bbn.pdf`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_effective_interpolated_thermal_history_bbn.json`

## Current effective results

The effective branch reproduces the BBN-inferred central expansion history essentially exactly in the window:

- `0.07 MeV`
- `0.10 MeV`
- `0.20 MeV`
- `0.50 MeV`
- `1.00 MeV`

The current JSON reports:

- `alpha_eff_T(0.07 MeV) ≈ 0.3121`
- `alpha_eff_T(0.10 MeV) ≈ 0.3264`
- `alpha_eff_T(0.20 MeV) ≈ 0.3541`
- `alpha_eff_T(0.50 MeV) ≈ 0.3907`
- `alpha_eff_T(1.00 MeV) ≈ 0.4184`

and the residuals of the effective branch relative to the BBN-inferred central curve are numerically zero up to floating-point precision.

## `alpha(T)` versus `alpha(y)`

Two effective quantities are now available:

### 1. `alpha_eff(T)`

This is the most useful one physically.
It tells us which effective mixing of the two projected branches reproduces the BBN-inferred central expansion history at each temperature.

### 2. `alpha_eff(y)`

This was also computed by mapping the same effective solution onto the projected angular variable `y`.

However, one should be careful:

- `alpha_eff(y)` is less intuitive than `alpha_eff(T)`,
- because the `y`-coordinate is strongly compressed in the BBN window,
- so it should be treated as a secondary diagnostic, not as the primary object.

In the BBN-relevant window, the effective range is roughly:

- `alpha_eff(y) ~ 0.312 – 0.418`

## Interpretation

This does **not** prove that full primordial abundances have been derived.

What it does prove is:

- there exists a **well-defined effective thermal history** built from the two projected branches,
- and this effective history can reproduce the **BBN-inferred central expansion curve** over the full `0.07–1 MeV` window.

So the present status is:

- **literal geometric completion**: not yet solved;
- **effective thermal completion**: yes, available now.

This is already strong enough to motivate continuing the BBN discussion in the `cmb_hyperconical` thread.

## What should be done next in the `cmb_hyperconical` thread

Recommended order:

1. Treat the effective interpolated branch as the current working BBN completion.
2. Decide whether the next target is:
   - matching the BBN-inferred **central curve** only, or
   - fitting the full **BBN-inferred band**.
3. Use the effective branch to build:
   - a clean `alpha_eff(T)` plot,
   - a clean `alpha_eff(y)` plot,
   - and, if desired, an effective `w(T)` plot.
4. Only after that, attempt a **geometric reconstruction**:
   - either a better running `\alpha(y)`,
   - or a more general effective projection function `f(y)`.
5. If the aim becomes full BBN rather than just `H(T)` matching, then the next missing step is:
   - weak freeze-out,
   - `g_*(T)` / `g_{*S}(T)`,
   - and the nuclear network.

## Important caution

The literal projected-map figure and the effective interpolated figure should not be conflated:

- the literal figure tests what the current map already gives;
- the effective figure shows what can be obtained as a physically motivated effective completion between the two constant-`\alpha` branches.

That distinction should be kept explicit in future discussion and in any manuscript wording.

## Current git state relevant to this note

As of this transfer note, the following new files exist locally in `hippopede` and are not yet committed:

- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\scripts\plot_hippopede_effective_interpolated_thermal_history_bbn.py`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_effective_interpolated_thermal_history_bbn.png`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_effective_interpolated_thermal_history_bbn.pdf`
- `C:\Users\rober\OneDrive\Documents\Codex_Portatil\hippopede\figures\hippopede_effective_interpolated_thermal_history_bbn.json`

These are the files to pick up from the `cmb_hyperconical` thread if continuing this line of work.

