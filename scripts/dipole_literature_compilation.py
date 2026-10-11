"""Published number-count dipoles of high-redshift tracers and a single-velocity (Ellis & Baldwin) fit.

Every entry below is transcribed from the quoted table or section of the primary source; nothing is
measured here.  Amplitudes D, 1-sigma errors, directions, count slopes x, spectral indices alpha
(convention S ~ nu^-alpha), the kinematic expectation D_kin quoted by the authors and a typical
redshift of the sample.  Fields marked "derived" are computed in this script from quoted numbers
(conversion of a 95% interval to 1 sigma, of a velocity to an amplitude, of RA/Dec to Galactic).

Kinematic prediction (Ellis & Baldwin 1984, MNRAS 206, 377):  D_i = K_i beta,  K_i = 2 + x_i (1 + alpha_i).
K_i is taken as D_kin,i / beta_CMB when the authors quote D_kin (this carries their own x, alpha and
any refinement), otherwise from x_i and alpha_i.  beta_CMB = 369.82 / c (Planck 2018 I, 1807.06205).

Fits (chi2 = sum_i (D_i - model_i)^2 / sigma_i^2, symmetric sigma = mean of the two quoted errors):
  K1   single beta, D_i = K_i beta                                   (LCDM with free observer speed)  k = 1
  K0   beta = beta_CMB fixed                                                                        k = 0
  C    constant amplitude, D_i = c (excess independent of x, alpha)                                 k = 1
  KC   D_i = K_i beta_CMB + c                                                                       k = 1
  G    D_i = K_i beta_CMB + (2 - 2 x_i) a + b  (sign structure of the geometric volume term)        k = 2
       (needs x_i; entries without x are skipped in G)
  R    weighted linear fit of the ratio D_i / D_kin,i against x_i                                   k = 2
Two versions of D_i: the full amplitude ("amp") and its projection on the CMB dipole direction
("proj", D cos theta, with theta the angle to (l, b) = (264.021, 48.253) deg; entries measured with the
direction fixed to the CMB are already projections).  The direction error is not propagated into the
projection.  Samples overlap on the sky (NVSS, RACS, VLASS, TGSS, WENSS, SUMSS share sources; Quaia and
CatWISE share unWISE photometry), so the chi2 values are indicative.

Also stores the CMB aberration / Doppler-modulation measurements that bound an intrinsic CMB dipole.
Writes json/dipole_literature_compilation.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "json" / "dipole_literature_compilation.json"

C_KM_S = 299792.458
V_CMB = 369.82  # km/s, Planck 2018 I
BETA_CMB = V_CMB / C_KM_S
L_CMB, B_CMB = 264.021, 48.253  # deg, Planck 2018 I
T_DIPOLE_MK = 3.36208  # mK, Planck 2018 I, as quoted in Ferreira & Quartin (2021), Introduction

# J2000 equatorial -> Galactic rotation matrix (Hipparcos convention)
_R_EQ2GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
])


def unit(lon_deg, lat_deg):
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def eq_to_gal(ra, dec):
    v = _R_EQ2GAL @ unit(ra, dec)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2])))


def angle_to_cmb(l, b):
    return float(np.degrees(np.arccos(np.clip(unit(l, b) @ unit(L_CMB, B_CMB), -1.0, 1.0))))


# ----------------------------------------------------------------------------------------------
# Compilation.  D and errors in absolute units; dir = ("gal", l, b) | ("eq", RA, Dec) | ("cmb",) when
# the amplitude was measured with the direction fixed to the CMB dipole | None when not quoted.
# ----------------------------------------------------------------------------------------------
S = []


def add(**kw):
    kw.setdefault("x", None)
    kw.setdefault("alpha", None)
    kw.setdefault("D_kin", None)
    kw.setdefault("z", None)
    kw.setdefault("notes", "")
    kw.setdefault("fits", [])
    S.append(kw)


# --- infrared quasars ---------------------------------------------------------------------------
add(id="CatWISE_Secrest21", tracer="CatWISE2020 quasars (W1<16.4, |b|>30), N=1,355,352", band="IR",
    D=0.01554, err=(None, None), dir=("gal", 238.2, 28.8), x=1.7, alpha=1.26, D_kin=0.007,
    z="mean 1.2 (Sect. II, from SDSS Stripe 82 matches)",
    ref="Secrest et al. 2021, ApJL 908, L51; arXiv:2009.14826", where="Sect. IV (D, direction, 4.9 sigma), Sect. II (alpha, z); x=1.7 as quoted for this sample by Dam et al. 2023 (Fig. 2 caption)",
    notes="no amplitude error quoted; significance 4.9 sigma one-sided (p=5e-7)")
add(id="CatWISE_Secrest22", tracer="CatWISE2020 quasars (W1<16.5), N~1.6 million", band="IR",
    D=0.0148, err=(0.0016, 0.0016), dir=("gal", 238.0, 31.0), x=1.89, alpha=1.06, D_kin=0.0073,
    z="mean 1.2 (Sect. I, for the 1.36M Secrest 2021 catalogue)",
    ref="Secrest et al. 2022, ApJL 937, L31; arXiv:2206.05624", where="Sect. III (D, direction), Sect. II.3 (x, alpha, D_kin)",
    notes="4.4 sigma; direction errors (7, 5) deg", fits=["core", "ext"])
add(id="CatWISE_Dam23", tracer="CatWISE2020 quasars (Secrest 2021 sample)", band="IR",
    D=0.019, err=(0.002, 0.002), dir=("gal", 237.2, 41.8), x=1.7, alpha=1.26, D_kin=0.0072,
    z="mean 1.2 (same sample as Secrest 2021)",
    ref="Dam, Lewis & Brewer 2023, MNRAS 525, 231; arXiv:2212.07733", where="Sect. 2.4 (model M8, all free) and Sect. 4; x, alpha in Fig. 2 caption",
    notes="D/D_CMB = 2.68 +- 0.23 (M8), 2.46 +- 0.18 with direction fixed to CMB (M7); 5.7 sigma for D > D_CMB")
add(id="CatWISE_Wagenveld25", tracer="CatWISE2020 AGN (W1<16.5, |b|>30), N=1,567,586", band="IR",
    D=0.0151, err=(0.0016, 0.0016), dir=("eq", 141.0, -6.0), x=1.90, alpha=1.07, D_kin=0.0073,
    z="median ~1 for AGN samples (Sect. 5.2)",
    ref="Wagenveld et al. 2025, A&A 697, A112; arXiv:2503.02470", where="Table 1; x, alpha, D_kin in Sect. 3.3",
    notes="ecliptic-latitude term fitted")
add(id="Quaia_low_Mittal24corr", tracer="Quaia G<20.0 (N=755,850), |b|<40 deg masked", band="optical/IR",
    D=0.011, err=(0.005 / 1.96, 0.006 / 1.96), dir=None, x=None, alpha=0.48, D_kin=0.0048,
    z="median 1.45 (Storey-Fisher et al. 2024, ApJ 964, 69, Conclusions)",
    ref="Mittal, Oayda & Lewis 2024, MNRAS 527, 8497; Correction MNRAS 530, 4763; arXiv:2311.14938v2",
    where="Sect. 6.1 (D x 1e3 = 11 +6/-5, 95% interval); Correction (alpha mean 0.48, D_kin 0.0048, model ranking)",
    notes="1 sigma derived as 95% half-width / 1.96; direction 'consistent with the CMB within ~0.5 sigma' (no numbers); "
          "original D_kin 0.0080 used an incorrect alpha; with the corrected D_kin the free-amplitude model is preferred "
          "with a narrow prior (ln B46 = 0.7) and 'the question of its amplitude is not as decisive'",
    fits=["core", "ext"])
add(id="Quaia_high_Mittal24", tracer="Quaia G<20.5 (N=1,295,502), |b|<40 deg masked", band="optical/IR",
    D=None, err=(None, None), dir=("gal", 330.0, 60.0), alpha=0.41, D_kin=0.0043,
    z="median 1.47 (Storey-Fisher et al. 2024, Sect. on redshifts)",
    ref="Mittal, Oayda & Lewis 2024 (+Correction); arXiv:2311.14938v2", where="Sect. 6.1, 6.4; Correction",
    notes="direction drifts to (l,b)~(330,60) as the mask grows; 'CMB and inferred dipole amplitudes do not agree with each "
          "other within 2 sigma, even with the 40 deg mask'; authors 'reserve drawing further conclusions'")
add(id="Quaia_Singal24", tracer="Quaia G<20.0, |b|>30 deg, N=530,364", band="optical/IR",
    D=0.033, err=(0.005, 0.005), dir=("eq", 179.0, 17.0), x=1.3, alpha=0.48, D_kin=None,
    z="median 1.45 (Storey-Fisher et al. 2024)",
    ref="Singal 2024, MNRAS 532, L1; arXiv:2403.16581", where="Table 1 row 4",
    notes="Singal used x=1.3 and alpha~2.4 (the uncorrected Mittal value), K=6.4, p=4.2+-0.6; here K is recomputed with the "
          "corrected mean alpha=0.48 (derived); |b|>40: D=0.023+-0.006 (row 5); no selection function applied",
    fits=["ext_singal"])
# --- radio, wide-area -----------------------------------------------------------------------------
add(id="NVSS_Secrest22", tracer="NVSS S>10 mJy, N=508,144", band="radio 1.4 GHz",
    D=0.0123, err=(0.0025, 0.0025), dir=("gal", 196.0, 46.0), x=0.77, alpha=0.75, D_kin=0.0041,
    z="no redshift quoted", ref="Secrest et al. 2022; arXiv:2206.05624", where="Sect. III, Sect. II.3",
    notes="2.6 sigma")
add(id="NVSS_Siewert21", tracer="NVSS S>15 mJy, mask d, N=328,207", band="radio 1.4 GHz",
    D=0.016, err=(0.003, 0.003), dir=("eq", 138.90, -2.74), x=1.04, alpha=0.75, D_kin=0.00472,
    z="not quoted", ref="Siewert, Schmidt-Rubart & Schwarz 2021, A&A 653, A9; arXiv:2010.08366",
    where="Table 9 (d), Eq. 45 (x), Table 8 (expected d)")
add(id="NVSS_Wagenveld23", tracer="NVSS S>15 mJy, N=345,803 (Poisson, Nside 32)", band="radio 1.4 GHz",
    D=0.0140, err=(0.0029, 0.0029), dir=("eq", 146.0, -5.0), x=0.85, alpha=0.75, D_kin=0.00430,
    z="radio AGN peak z~0.8 (Sect. 2, citing Condon & Ransom 2016)",
    ref="Wagenveld, Kloeckner & Schwarz 2023, A&A 675, A72; arXiv:2305.15335", where="Table 3; x, D_kin in Sect. 4.1")
add(id="NVSS_Wagenveld25", tracer="NVSS S>15 mJy, N=352,862", band="radio 1.4 GHz",
    D=0.0139, err=(0.0029, 0.0029), dir=("eq", 151.0, -9.0), x=0.74, alpha=0.75, D_kin=0.0041,
    z="median ~1 for AGN samples (Sect. 5.2)", ref="Wagenveld et al. 2025; arXiv:2503.02470", where="Table 1; Sect. 3.1",
    notes="with 2MRS z<0.02 sources removed: 0.0120+-0.0029 towards (RA,Dec)=(152,-10)", fits=["core", "ext"])
add(id="NVSS_Boehme25", tracer="NVSS S>20 mJy, N=263,125 (negative binomial)", band="radio 1.4 GHz",
    D=0.0154, err=(0.0036, 0.0036), dir=("eq", 148.1, -9.5), x=0.88, alpha=0.73, D_kin=0.00435,
    z="not quoted", ref="Boehme et al. 2025, PRL 135, 201001; arXiv:2509.16732", where="Table III (free), Table II")
add(id="RACSlow_Wagenveld23", tracer="RACS-low S>15 mJy, N=451,003 (Poisson, Nside 32)", band="radio 887.5 MHz",
    D=0.0141, err=(0.0024, 0.0024), dir=("eq", 193.0, 4.0), x=0.82, alpha=0.75, D_kin=0.00424,
    z="not quoted", ref="Wagenveld et al. 2023; arXiv:2305.15335", where="Table 3; Sect. 4.2")
add(id="RACSlow_Wagenveld25", tracer="RACS-low S>15 mJy, N=442,046", band="radio 887.5 MHz",
    D=0.0143, err=(0.0024, 0.0024), dir=("eq", 190.0, 3.0), x=0.72, alpha=0.88, D_kin=0.0041,
    z="median ~1 for AGN samples (Sect. 5.2)", ref="Wagenveld et al. 2025; arXiv:2503.02470", where="Table 1; Sect. 3.2",
    notes="with 2MRS sources removed: 0.0134+-0.0024 towards (194, 8)", fits=["core", "ext"])
add(id="RACSlow_Boehme25", tracer="RACS-low S>20 mJy, N=330,540 (negative binomial)", band="radio 888 MHz",
    D=0.0180, err=(0.0030, 0.0028), dir=("eq", 191.1, 6.0), x=0.85, alpha=0.98, D_kin=0.00454,
    z="not quoted", ref="Boehme et al. 2025; arXiv:2509.16732", where="Table III (free), Table II")
add(id="RACSmid_Boehme25", tracer="RACS-mid S>20 mJy, N=237,069, direction fixed to CMB", band="radio 1367 MHz",
    D=0.0115, err=(0.0033, 0.0033), dir=("cmb",), x=0.88, alpha=0.81, D_kin=0.00443,
    z="not quoted", ref="Boehme et al. 2025; arXiv:2509.16732", where="Table III (CMB), Table II", fits=["ext"])
add(id="LoTSSDR2_Boehme25", tracer="LoTSS-DR2 S>5 mJy, N=376,658, direction fixed to CMB", band="radio 144 MHz",
    D=0.0139, err=(0.0042, 0.0045), dir=("cmb",), x=0.74, alpha=0.73, D_kin=0.00405,
    z="not quoted", ref="Boehme et al. 2025; arXiv:2509.16732", where="Table III (CMB, NB), Table II",
    notes="only the projected amplitude is given", fits=["core", "ext"])
add(id="VLASS_Boehme25", tracer="VLASS S>10 mJy, N=275,516, direction fixed to CMB", band="radio 3 GHz",
    D=0.0055, err=(0.0030, 0.0036), dir=("cmb",), x=0.99, alpha=0.71, D_kin=0.00456,
    z="not quoted", ref="Boehme et al. 2025; arXiv:2509.16732", where="Table III (CMB, NB), Table II",
    notes="free direction lies 60-80 deg from the CMB (declination systematics, Quick Look images)", fits=["ext"])
add(id="TGSS_Siewert21", tracer="TGSS-ADR1 S>50 mJy, mask d, N=393,447", band="radio 147 MHz",
    D=0.066, err=(0.005, 0.005), dir=("eq", 124.53, 25.66), x=0.86, alpha=0.75, D_kin=0.00434,
    z="not quoted", ref="Siewert et al. 2021; arXiv:2010.08366", where="Table 9, Eq. 45, Table 8",
    notes="flux-calibration systematics; Secrest 2022 App. A.1 obtains 0.022 after recalibration and 0.014 after a noise correction")
add(id="TGSS_Boehme25", tracer="TGSS S>100 mJy, N=234,619 (negative binomial)", band="radio 147 MHz",
    D=0.0533, err=(0.0042, 0.0040), dir=("eq", 140.7, 1.6), x=0.79, alpha=0.76, D_kin=0.00418,
    z="not quoted", ref="Boehme et al. 2025; arXiv:2509.16732", where="Table III (free), Table II", fits=["ext"])
add(id="WENSS_Siewert21", tracer="WENSS S>25 mJy, mask d, N=115,808", band="radio 325 MHz",
    D=0.032, err=(0.010, 0.010), dir=("eq", 143.34, -13.15), x=0.75, alpha=0.75, D_kin=0.00410,
    z="not quoted", ref="Siewert et al. 2021; arXiv:2010.08366", where="Table 9, Eq. 45, Table 8", fits=["ext"])
add(id="SUMSS_Siewert21", tracer="SUMSS S>18 mJy, mask d, N=99,835", band="radio 843 MHz",
    D=0.038, err=(0.009, 0.009), dir=("eq", 106.67, -9.50), x=1.00, alpha=0.75, D_kin=0.00463,
    z="not quoted", ref="Siewert et al. 2021; arXiv:2010.08366", where="Table 9, Eq. 45, Table 8", fits=["ext"])
# --- radio, deep / sparse and combined -----------------------------------------------------------
add(id="MALS_Wagenveld24", tracer="MALS DR2 S>400 microJy, 391 pointings, N=311,419", band="radio L band",
    D=0.0067, err=(0.0029, 0.0031), dir=("eq", 176.0, 20.0), x=0.69, alpha=0.75, D_kin=0.0040,
    z="star-forming galaxies, a significant fraction at z<0.5 (Sect. 6, SKADS)",
    ref="Wagenveld et al. 2024, arXiv:2408.16619 (A&A)", where="Table 2 (Poisson-linear, Crms, 400 microJy), Sect. 4.2, 5",
    notes="'consistent with the CMB dipole in terms of direction and amplitude'; cannot exclude zero at 3 sigma", fits=["ext"])
add(id="VLASS+RACS_Darling22", tracer="VLASS epoch 1 + RACS-low, number counts", band="radio",
    D=None, v_kms=(331.0, 107.0, 161.0), err=(None, None), dir=("gal", 271.0, 56.0), x=1.0, alpha=0.98, D_kin=None,
    z="not quoted", ref="Darling 2022, ApJL 931, L14; arXiv:2205.06880", where="Abstract, Table 1, Sect. 5",
    notes="velocity 331 +161/-107 km/s; D derived as K beta with K=2+x(1+alpha)", fits=["ext"])
add(id="NVSS+RACS_Wagenveld23", tracer="NVSS + RACS-low S>15 mJy, N=796,806 (multi-Poisson)", band="radio",
    D=0.0130, err=(0.0018, 0.0018), dir=("eq", 173.0, -1.0), x=0.85, alpha=0.75, D_kin=0.0043,
    z="not quoted", ref="Wagenveld et al. 2023; arXiv:2305.15335", where="Table 3; Sect. 5.3",
    notes="4.8 sigma, three times the CMB expectation; combination of entries above (not used in fits)")
add(id="LoTSS+NVSS+RACS_Boehme25", tracer="LoTSS-DR2 + NVSS + RACS-low", band="radio",
    D=None, ratio=(3.67, 0.49), err=(None, None), dir=("eq", 164.9, -10.8), D_kin=None,
    z="not quoted", ref="Boehme et al. 2025; arXiv:2509.16732", where="Table III (survey combination, NB)",
    notes="d/d_exp = 3.67 +- 0.49, 5.4 sigma; combination (not used in fits)")


# ----------------------------------------------------------------------------------------------
def finalize(s):
    if s.get("v_kms") is not None:  # velocity -> amplitude (derived)
        v, lo, hi = s["v_kms"]
        K = 2.0 + s["x"] * (1.0 + s["alpha"])
        s["D"] = K * v / C_KM_S
        s["err"] = (K * lo / C_KM_S, K * hi / C_KM_S)
        s["D_derived"] = True
    if s["D_kin"] is not None:
        s["K"] = s["D_kin"] / BETA_CMB
        s["K_source"] = "D_kin quoted / beta_CMB"
    elif s["x"] is not None and s["alpha"] is not None:
        s["K"] = 2.0 + s["x"] * (1.0 + s["alpha"])
        s["K_source"] = "2 + x(1+alpha) (derived)"
        s["D_kin_derived"] = s["K"] * BETA_CMB
    else:
        s["K"] = None
    d = s["dir"]
    s["l_b"] = None
    s["theta_cmb_deg"] = None
    if d is not None and d[0] == "gal":
        s["l_b"] = [d[1], d[2]]
    elif d is not None and d[0] == "eq":
        s["l_b"] = list(eq_to_gal(d[1], d[2]))
    if s["l_b"] is not None:
        s["theta_cmb_deg"] = angle_to_cmb(*s["l_b"])
    if s["D"] is not None and None not in s["err"]:
        s["sigma"] = 0.5 * (s["err"][0] + s["err"][1])
        if d is not None and d[0] == "cmb":
            s["D_proj"] = s["D"]
        elif s["theta_cmb_deg"] is not None:
            s["D_proj"] = s["D"] * np.cos(np.radians(s["theta_cmb_deg"]))
        else:
            s["D_proj"] = s["D"]  # direction only described as CMB-like (Mittal low)
        if s["K"] is not None:
            s["ratio_to_kin"] = s["D"] / (s["K"] * BETA_CMB)
            s["ratio_err"] = s["sigma"] / (s["K"] * BETA_CMB)
            s["beta_single_over_cmb"] = s["ratio_to_kin"]
    return s


for s in S:
    finalize(s)


def lin_fit(X, y, sig):
    """Weighted least squares y = X p; returns p, cov, chi2."""
    W = 1.0 / sig**2
    A = X.T @ (X * W[:, None])
    cov = np.linalg.inv(A)
    p = cov @ (X.T @ (W * y))
    chi2 = float(np.sum(W * (y - X @ p) ** 2))
    return p, cov, chi2


def run_fits(tag, key):
    rows = [s for s in S if tag in s["fits"] and s.get("sigma") is not None and s["K"] is not None]
    out = {"samples": [s["id"] for s in rows], "n": len(rows)}
    for which in ("amp", "proj"):
        y = np.array([s["D"] if which == "amp" else s["D_proj"] for s in rows])
        sig = np.array([s["sigma"] for s in rows])
        K = np.array([s["K"] for s in rows])
        n = len(y)
        res = {}
        p, cov, chi2 = lin_fit(K[:, None], y, sig)
        beta = float(p[0])
        res["K1_single_beta"] = {
            "beta": beta, "beta_err": float(np.sqrt(cov[0, 0])), "v_kms": beta * C_KM_S,
            "v_err_kms": float(np.sqrt(cov[0, 0])) * C_KM_S, "beta_over_beta_cmb": beta / BETA_CMB,
            "chi2": chi2, "dof": n - 1,
            "pulls": {s["id"]: float((yy - K_ * beta) / sg) for s, yy, K_, sg in zip(rows, y, K, sig)},
        }
        chi2_0 = float(np.sum(((y - K * BETA_CMB) / sig) ** 2))
        res["K0_beta_cmb"] = {"chi2": chi2_0, "dof": n,
                              "pulls": {s["id"]: float((yy - K_ * BETA_CMB) / sg) for s, yy, K_, sg in zip(rows, y, K, sig)}}
        p, cov, chi2 = lin_fit(np.ones((n, 1)), y, sig)
        res["C_constant"] = {"c": float(p[0]), "c_err": float(np.sqrt(cov[0, 0])), "chi2": chi2, "dof": n - 1}
        p, cov, chi2 = lin_fit(np.ones((n, 1)), y - K * BETA_CMB, sig)
        res["KC_kin_plus_constant"] = {"c": float(p[0]), "c_err": float(np.sqrt(cov[0, 0])), "chi2": chi2, "dof": n - 1}
        withx = [i for i, s in enumerate(rows) if s["x"] is not None]
        if len(withx) >= 3:
            xs = np.array([rows[i]["x"] for i in withx])
            Xg = np.column_stack([2.0 - 2.0 * xs, np.ones(len(withx))])
            p, cov, chi2 = lin_fit(Xg, (y - K * BETA_CMB)[withx], sig[withx])
            res["G_kin_plus_(2-2x)a_plus_b"] = {"samples": [rows[i]["id"] for i in withx], "a": float(p[0]),
                                               "a_err": float(np.sqrt(cov[0, 0])), "b": float(p[1]),
                                               "b_err": float(np.sqrt(cov[1, 1])), "chi2": chi2, "dof": len(withx) - 2}
            r = (y / (K * BETA_CMB))[withx]
            rs = (sig / (K * BETA_CMB))[withx]
            p, cov, chi2 = lin_fit(np.column_stack([np.ones(len(withx)), xs]), r, rs)
            res["R_ratio_vs_x"] = {"intercept": float(p[0]), "slope": float(p[1]),
                                   "slope_err": float(np.sqrt(cov[1, 1])), "chi2": chi2, "dof": len(withx) - 2}
        out[which] = res
    return out


cmb = {
    "planck2013_XXVII": {
        "ref": "Planck Collaboration 2014, A&A 571, A27 (Planck 2013 results XXVII); arXiv:1303.5087",
        "where": "Abstract; Sect. 5 and Table 2",
        "v_parallel_kms": 384, "stat_kms": 78, "syst_kms": 115, "direction_lb_fixed": [264, 48],
        "in_units_of_369": "1.04 +- 0.21 (stat), syst 0.31 (Sect. 5)",
        "significance": "no-velocity hypothesis rejected; estimators show velocity effects above ~4 sigma (Sect. 7)",
    },
    "ferreira_quartin_2021": {
        "ref": "Ferreira & Quartin 2021, PRL 127, 101301; arXiv:2011.08385 (pipeline: PRD 104, 063503, arXiv:2107.10846)",
        "where": "Abstract; Table 1 (aberration and Doppler separately); Table 2 (significances); Table 3 (intrinsic dipole)",
        "v_SMICA_kms": [300, -93, 111], "lb_SMICA": [[276, 33], [51, 19]],
        "v_NILC_kms": [296, -88, 111], "lb_NILC": [[280, 33], [50, 20]],
        "aberration_SMICA_kms": "300 +- 99 (stat) +- 13 (syst)", "doppler_SMICA_kms": "390 +- 140 +- 13, (l,b)=(210+-56,-2+-30)",
        "significance": "aberration 3.5-3.9 sigma, Doppler 2.6-2.9 sigma, aberration+Doppler 4.1-4.6 sigma",
        "intrinsic_dipole_95CL_mK": {"SMICA": 3.6, "NILC": 3.7, "abstract": 3.7},
        "total_dipole_mK": T_DIPOLE_MK,
    },
    "saha_2021": {
        "ref": "Saha et al. 2021, JCAP 10, 072; arXiv:2106.07666", "where": "Abstract",
        "result": "BipoSH detection consistent with v=369 km/s, (l,b)=(264,48), at 4.54, 4.97, 5.23 sigma for f_sky = 40.1, 59.1, "
                  "72.2%; Bayes factor 7-8 for the canonical value; Bayes factor ~1e-10-1e-11 against the CatWISE-implied velocity",
    },
}
# Simple Gaussian reading of the aberration/Doppler velocities (derived here, not by the authors):
# intrinsic fraction f = 1 - v_kin / v_dipole along the dipole direction.
v_fq, s_fq = 300.0, 0.5 * (93.0 + 111.0)
v_pl, s_pl = 384.0, float(np.hypot(78.0, 115.0))
cmb["derived_intrinsic_fraction"] = {
    "definition": "f = 1 - v_aberration/369.82; Gaussian errors (mean of asymmetric errors); stat and syst in quadrature for Planck 2013",
    "FQ2021_SMICA": {"f": 1 - v_fq / V_CMB, "sigma": s_fq / V_CMB, "f_95_upper": 1 - (v_fq - 1.645 * s_fq) / V_CMB},
    "Planck2013": {"f": 1 - v_pl / V_CMB, "sigma": s_pl / V_CMB, "f_95_upper": 1 - (v_pl - 1.645 * s_pl) / V_CMB},
    "authors_bound_vs_total": f"|Delta_1,int| < 3.6-3.7 mK (95%) vs total {T_DIPOLE_MK} mK: the bound does not exclude a fully intrinsic dipole",
}

fits = {
    "core": run_fits("core", "core"),
    "ext": run_fits("ext", "ext"),
}
# extended set with Singal's Quaia amplitude in place of Mittal's
for s in S:
    if "ext" in s["fits"] and s["id"] != "Quaia_low_Mittal24corr":
        s["fits"] = s["fits"] + ["ext_singal"]
fits["ext_singal"] = run_fits("ext_singal", "ext_singal")
# core without Quaia (wide-area AGN only: CatWISE, NVSS, RACS-low, LoTSS)
for s in S:
    if "core" in s["fits"] and not s["id"].startswith("Quaia"):
        s["fits"] = s["fits"] + ["core_noquaia"]
fits["core_noquaia"] = run_fits("core_noquaia", "core_noquaia")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(clean({"beta_cmb": BETA_CMB, "cmb_dipole_lb": [L_CMB, B_CMB], "cmb_aberration": cmb,
                                 "samples": S, "fits": fits}), indent=1), encoding="utf-8")

# console summary
print(f"beta_CMB = {BETA_CMB:.5e}")
print(f"{'id':28s} {'D':>8s} {'sig':>7s} {'K':>5s} {'Dkin':>7s} {'D/Dkin':>7s} {'theta':>6s} {'Dproj':>8s}  x")
for s in S:
    if s.get("sigma") is None:
        continue
    th = s["theta_cmb_deg"]
    print(f"{s['id']:28s} {s['D']:8.4f} {s['sigma']:7.4f} {s['K'] or float('nan'):5.2f} "
          f"{(s['K'] or np.nan) * BETA_CMB:7.4f} {s.get('ratio_to_kin', np.nan):7.2f} "
          f"{th if th is not None else float('nan'):6.1f} {s['D_proj']:8.4f}  {s['x']}")
for tag, f in fits.items():
    print(f"\n== {tag}: {f['samples']}")
    for which in ("amp", "proj"):
        r = f[which]
        k1 = r["K1_single_beta"]
        print(f" [{which}] K1 beta/beta_cmb={k1['beta_over_beta_cmb']:.2f}+-{k1['beta_err'] / BETA_CMB:.2f} "
              f"v={k1['v_kms']:.0f}+-{k1['v_err_kms']:.0f} chi2={k1['chi2']:.2f}/{k1['dof']}  "
              f"K0 chi2={r['K0_beta_cmb']['chi2']:.1f}/{r['K0_beta_cmb']['dof']}  "
              f"C chi2={r['C_constant']['chi2']:.2f} (c={r['C_constant']['c']:.4f})  "
              f"KC chi2={r['KC_kin_plus_constant']['chi2']:.2f} (c={r['KC_kin_plus_constant']['c']:.4f})")
        print("   pulls K1:", {k: round(v, 2) for k, v in k1["pulls"].items()})
        if "G_kin_plus_(2-2x)a_plus_b" in r:
            g = r["G_kin_plus_(2-2x)a_plus_b"]
            rr = r["R_ratio_vs_x"]
            print(f"   G: a={g['a']:.4f}+-{g['a_err']:.4f} b={g['b']:.4f}+-{g['b_err']:.4f} chi2={g['chi2']:.2f}/{g['dof']}"
                  f"   R: slope dRatio/dx={rr['slope']:.2f}+-{rr['slope_err']:.2f} chi2={rr['chi2']:.2f}/{rr['dof']}")
print(f"\nCMB derived intrinsic fraction: {cmb['derived_intrinsic_fraction']}")
print(f"wrote {OUT}")
