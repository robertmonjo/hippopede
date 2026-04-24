# BBN Scripts Review — 2026-04-24

Revisió de tots els scripts BBN/CMB del model hipercònic.
Fitxers examinats:

- `scripts/analyze_hippopede_dipole_bbn.py`
- `scripts/plot_hippopede_projected_thermal_history_bbn.py`
- `scripts/plot_hippopede_effective_interpolated_thermal_history_bbn.py`
- `scripts/analyze_hippopede_variable_alpha.py`
- `scripts/plot_hippopede_qz_double_panel_with_gapp.py` (classe base)

---

## Problemes corregits en aquesta revisió

### Bug 1 — Directori de sortida incorrecte (`analyze_hippopede_dipole_bbn.py`, línia 9)

**Problema:** `ROOT = Path(__file__).resolve().parent` apuntava a `hippopede/scripts/`,
de manera que el JSON de sortida es desava a `hippopede/scripts/figures/` en lloc de
`hippopede/figures/`, inconsistent amb tots els altres scripts.

**Correcció:** `ROOT = Path(__file__).resolve().parents[1]` (= `hippopede/`).

---

### Bug 2 — `g_*(T) = 10.75` constant a tot el rang 0.01–10 MeV

**Problema:** La funció `standard_radiation_hubble` usava `g_* = 10.75` per a totes les
temperatures. Però en el rang BBN clau 0.07–0.5 MeV el parell e⁺e⁻ s'anihila i `g_*`
decreix contínuament des de 10.75 fins a ~3.36, resultant en errors de fins a ~15% en
H(T) en aquest interval.

**Correcció:** S'implementa `effective_g_star(T_mev)` basat en:
- Integral de Fermi-Dirac exacta per al parell e⁺e⁻ (tabulada numèricament en 220 punts,
  interpolada amb `np.interp`).
- Temperatura dels neutrins T_ν/T_γ derivada de la conservació de l'entropia durant
  l'anihilació e⁺e⁻:
  ```
  (T_ν/T_γ)^3 = (4 + 7·h(m_e/T)) / 11
  ```
  on `h(x)` és el rati d'energia de Fermi-Dirac massiva respecte al límit sense massa.
  Dóna T_ν/T_γ = 1 per T ≫ m_e i (4/11)^(1/3) per T ≪ m_e. ✓

La fórmula completa per a `g_*`:
```
g_*(T) = 2  [fotons]
       + (7/8)·4·h(m_e/T)          [e⁺e⁻]
       + (7/8)·6·(T_ν/T_γ)^4       [3ν + 3ν̄]
```
Comprova:
- T ≫ m_e: g_* = 2 + 3.5 + 5.25 = 10.75 ✓
- T ≪ m_e: g_* = 2 + 0 + 5.25·(4/11)^(4/3) ≈ 3.36 ✓

`standard_radiation_hubble` accepta ara un paràmetre opcional `g_star=None`; quan és
`None` usa `effective_g_star(T_mev)` (comportament per defecte). Passar `g_star=10.75`
recupera el comportament antic (retrocompatible).

---

### Bug 3 — Factor ΔN_eff per a la banda BBN independent de T

**Problema:** `neff_to_scale` usava `sqrt(1 + 7·ΔN_eff/43)`, que és la fórmula exacta
per a g_* = 10.75 constant i T_ν = T_γ, però no per a g_*(T) variable.

**Correcció:** `neff_to_scale(delta_neff, T_mev=None)` usa ara:
```
scale(T) = sqrt(1 + (7/4)·(T_ν/T_γ)^4·ΔN_eff / g_*(T))
```
Quan `T_mev=None` es recupera la fórmula antiga (retrocompatible).

---

### Bug 4 — Fórmula de llei de potència mixta a `analyze_hippopede_variable_alpha.py`

**Problema:** `analyze_zc` calculava un "coeficient" com:
```python
coeff_hi = e_var_hi / (1 + z_hi)^(1 + 2·alpha(z_hi))
```
i llavors:
```python
h_projected = H0_SI * coeff_med * (1+z_t)^(1 + 2·alpha(z_t))
```
L'exponent `1 + 2α` per a H ∝ (1+z)^(1+2α) és vàlid únicament per a α **constant**.
Per a α(z) variable, l'índex efectiu local és diferent. El "coeficient" no és una constant,
sinó que varia amb z, i la seva mediana no té un significat físic precís.

**Correcció:** La H als redshifts BBN s'obté ara per interpolació directa de
`e_variable_alpha` avaluada en una graella estesa fins a z ≫ 10^9:
```python
z_all = [0, geomspace(0.1, 1, 10), geomspace(1, z_max, 6000)]
e_all = e_variable_alpha(model, z_all, zc)
h_projected = H0_SI * interp(z_t, z_all, e_all)
```
S'ha eliminat `prefactor_median` del JSON de sortida; s'ha mantingut `n_eff_median`
(que sí és una derivada logarítmica genuïna) i `alpha_median_1e3_1e6`.

---

### Anotació — Etiquetes de la banda BBN a les figures

**Problema:** Les etiquetes de les figures citaven "Observed BBN-inferred band
(Schöneberg 2024)" però la corba es genera *des de zero* via la parametrització
ΔN_eff, no llegint dades del paper. La forma de la banda és una aproximació.

**Correcció:** Les etiquetes indiquen ara "BBN-compatible expansion band
(ΔN_eff parametrization; cf. Schöneberg 2024)" per ser honestos sobre el que
es mostra.

---

## Aspectes verificats com a correctes (no modificats)

| Element | Estat |
|---|---|
| Constants físiques (T₀, kB, H₀, Mpl, ħ) | ✓ Correctes |
| Fórmula BBN: H = 1.66√g_* T²/Mpl, conversió SI | ✓ Correcta |
| Derivades analítiques `dinvll_dx`, `dxdLZ` | ✓ Correctes (regla de la cadena verificada) |
| dy/dx = 1/u on u = √(1/k − x²) | ✓ Correcte |
| x(lz) independent de α (ve de la mètrica) | ✓ Correcte |
| Interpolació log-espai i inversió exacta | ✓ Tautològicament exacte |
| Conversió z(T): T_mev × 10⁶ eV / T₀_ev − 1 | ✓ Correcte |
| Gradient numèric en `e_variable_alpha` (script 1) | Aproximació declarada al document |
| Suavitzat 41 punts en la branca literal | Documentat i limitat al domini estable |

---

## Limitació persistent (no corregida, però documentada)

La "banda BBN observacional" es construeix com `h_obs = scale(T) * h_std(T)` on
`scale(T) = neff_to_scale(ΔN_eff, T_mev=T)`. Això és una **parametrització efectiva**
del constraint de Schöneberg 2024, no les dades reals de xarxa nuclear BBN. La forma
espectral correcta requeriria integrar la xarxa nuclear (freeze-out del deuteri, He-4,
Li-7, etc.) amb H(T) del model. Fins que no es faci aquest càlcul, la banda es pot usar
com a guia orientativa però no com a test definitiu de les abundàncies primordials.

---

## Problema estructural — Pendent logarítmica incorrecta a l'alta temperatura

Aquesta secció documenta un problema de naturalesa **teòrica fonamental**, no un error
d'implementació. Ja estava identificat i documentat al repositori `cmb_hyperconical` en
els fitxers indicats més avall; aquí s'inclou per completesa del registre.

### El problema

El límit d'alta temperatura del model projectat dóna:

```
d ln H_proj / d ln T  →  1   (per a qualsevol α constant)
```

mentre que la cosmologia estàndard de radiació exigeix:

```
d ln H_std / d ln T  →  2   (H ∝ T²)
```

Aquest factor 2 no és una qüestió de normalització: és la pendent de la llei de potència.
Una renormalització multiplicativa pot corregir una amplitud, però **no pot canviar
una pendent 1 en una pendent 2**.

### Confirmació numèrica

La diagnosi asimptòtica del repositori `cmb_hyperconical` confirma el resultat per a tot
el rang 0.07–100 MeV. Valors representatius del fitxer CSV de diagnosi:

| T (MeV) | pendent H_std | pendent α=0.283 | pendent α=0.5 |
|---------|--------------|-----------------|----------------|
| 1       | 2.0153       | 1.0000          | 1.0000         |
| 10      | 2.0002       | 1.0000          | 1.0000         |
| 100     | 2.0000       | 1.0000          | 1.0000         |

Totes les branques projectives constants donen pendent exactament 1, independentment de α.

### Origen físic

La pendent prové del comportament asimptòtic del map proyectiu hiperchònic: la funció
`E(z) = H_proj(z)/H0` creix com `(1+z)^1` a z ≫ 1 per a qualsevol α constant,
en lloc de `(1+z)^2` com exigeix la radiació. Amb la identificació mínima
`1 + z = T/T₀`, la pendent en T queda fixada a 1.

### Conseqüència per al treball actual

1. **El test BBN de finestra finita roman vàlid** en el rang 0.07–1 MeV: la branca
   efectiva interpolada reprodueix H_obs(T) per construcció en aquest interval.
2. **El límit asimptòtic de radiació no està resolt**: no es pot presentar α = 0.5 com
   a branca "de radiació" que reprodueixi H ∝ T² a T ≫ 1 MeV.
3. **No és un bug de codi**: els scripts implementen correctament el model actual.
   La discrepància és un resultat físic del mapa projectiu, no un error de programació.

### Pas teòric necessari

Per recuperar la pendent 2 calen, probablement, una de les dues coses:
- Una relació T(z) revisada (no simplement `T/T₀ = 1 + z`), o
- Una derivació de primers principis de l'observable tèrmic projectat que modifiqui
  l'índex efectiu de `E(z)` a l'alta z.

### Documentació preexistent a `cmb_hyperconical`

- Document principal:
  `cmb_hyperconical/docs/BBN_HIGH_T_DISCUSSION_AND_RESOLUTION.md`
- Nota complementària:
  `cmb_hyperconical/docs/BBN_ASYMPTOTIC_SLOPE_NOTE.md`
- Script de diagnosi:
  `cmb_hyperconical/scripts/cmb_bbn_asymptotic_diagnostics.py`
- Dades numèriques:
  `cmb_hyperconical/data/cmb_hyperconical_bbn_asymptotic_diagnostics.csv`

---

## Nota — Implementació paral·lela a `cmb_hyperconical`

El fitxer `cmb_hyperconical/scripts/cmb_bbn_effective_branch.py` conté una implementació
independent —i prèvia a aquesta revisió— de les mateixes correccions aplicades ara als
scripts de `hippopede`:

- `effective_g_star(T_mev)` amb integral de Fermi-Dirac tabulada (Bug 2 d'aquí)
- `neff_to_scale(delta_neff, T_mev=None)` dependent de T (Bug 3 d'aquí)
- `standard_radiation_hubble` amb `g_*` variable

Això confirma la correcció de la lògica física aplicada als scripts `hippopede`.
Les dues implementacions son numèricament equivalents.

---

## Fitxers modificats en aquesta sessió

```
hippopede/scripts/analyze_hippopede_dipole_bbn.py          (bugs 1, 2, 3)
hippopede/scripts/plot_hippopede_projected_thermal_history_bbn.py   (bug 3, anotació)
hippopede/scripts/plot_hippopede_effective_interpolated_thermal_history_bbn.py  (bug 3, anotació)
hippopede/scripts/analyze_hippopede_variable_alpha.py       (bug 4)
```
