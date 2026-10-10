"""Redshift shape of the light-cone-ball count dipole in the Quaia slices: which assumptions set it.

The model dipole of quasar_dipole_fit.dipole for sources at observed redshift z,
    delta ln dN/dOmega dz = (2 - 2x) delta ln D_C - delta ln E,
assumes that the comoving density above a luminosity L is an isotropic function of (L, z).  It is
averaged over a flat p(z) inside each photometric slice (quaia_zslice_model.shape).  This script
splits the dipole at every z into its parts,
    d_D(z), d_E(z), d_T(z) = (3/2) int {delta ln D_C, delta ln E, delta t_L} cos psi sin psi dpsi,
(t_L = int dz/((1+z)E) the lookback time along the line of sight, in the units of the sector tables),
and recomputes the slice amplitudes and chi2_Quaia = sum ((|D_i| - A_i)/sigma_i)^2 under these changes:
  a. p(z) inside each slice: flat (reference), the Quaia photo-z histogram of the slice, and the
     true-redshift distribution n(z) P(slice | z) with Gaussian photo-z errors sigma_z (1 + z),
     sigma_z = 0.05, 0.1, 0.2, or with the catalogue errors of each source, or a fraction f = 0.2, 0.4 of
     catastrophic outliers drawn from the whole n(z).  The growth of the dipole with z is printed as
     d_E(z)/rhat(z) (for small rhat, d(z) = (rhat/2) d f/d theta of any sector quantity f).
  b. count slope x of each slice (Poisson error, one-sided estimate, common values); the kinematic
     expectation of a redshift-binned count, which gains the bin-edge term
         D_kin,i = beta [2 + x_i (1 + alpha) + ((1 + z_hi) n(z_hi) - (1 + z_lo) n(z_lo)) / N_i],
     n = dN/dz of the photo-z sample (Doppler shift of the observed redshift across the slice edges),
     applied to the measured dipole vectors (dipole_zslices/results.json).
  c. source evolution tied to the emission time t_e = t0 - t_L instead of z: the density above L is an
     isotropic function of (L, t_e), which adds gamma(z) (1 + z) <E> delta t_L, with
         gamma = d ln(dN/dz)/dz - d ln(<D_C>^2/<E>)/dz + x(z) [2 d ln <D_L>/dz + (alpha - 1)/(1 + z)]
     the evolution of the density above fixed L inferred from the Quaia dN/dz and the sky-averaged
     background of the same model (<.>: sky average at fixed z).
  d. the statistic: vector chi2 with the model dipole along the fixed axis (opposite to the CatWISE
     excess; the model dipole is negative, so it points to the excess), Rice (non-central chi, three
     components) likelihood of the amplitudes, and the best rescaling of the shape.
CatWISE (gamma p(z), x = 1.63) is evaluated with the same changes where they apply.  Reference points
(alpha_high, t0, theta_obs) = (0.38, 2.77, 7 deg) and (0.405, 4.50, 21.06 deg), alpha_low =
projected_hyperconical.ALPHA_LOW.  Catalogue: dipole_zslices/data/quaia_G20.0.fits, G < 20, Galactic
|b| > 30 deg (the selection-function cut S > 0.3 of measure_dipole.py is not applied here).
Run with HIPPOPEDE_THETA_MAX_DEG=90.  Writes json/quaia_shape_diagnostics.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pantheon_offaxis_test as T  # noqa: E402
import projected_hyperconical as PH  # noqa: E402
import quasar_dipole_fit as Q  # noqa: E402
import quaia_zslice_model as QZ  # noqa: E402

JSON = ROOT / "json"
CAT = ROOT / "dipole_zslices" / "data" / "quaia_G20.0.fits"
POINTS = [(0.38, 2.77, 7.0), (0.405, 4.50, 21.06)]
Z = np.round(np.arange(0.02, 2.1001, 0.02), 4)        # source redshifts of the model
PSI = np.linspace(0.0, np.pi, 61)
ZH = np.round(np.arange(0.0, 4.5001, 0.02), 4)        # histogram nodes of the catalogue
SIG_Z = (0.05, 0.1, 0.2)
F_OUT = (0.2, 0.4)
BETA, ALPHA = Q.BETA, Q.ALPHA_SPEC


def unitvec(l, b):
    lo, la = np.radians(l), np.radians(b)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


# ---------------------------------------------------------------- data
def load_slices():
    r = json.loads((ROOT / "dipole_zslices" / "results.json").read_text())
    rows, xz = [], []
    for s in r["slices"]:
        if not isinstance(s["z_range"], list):
            continue
        xz.append((s["median_z"], s["x"], s["alpha_colour_median_edge"]))
        if s["z_range"][1] > 1.85:
            continue
        d = s["dip"]
        rows.append({"z": tuple(s["z_range"]), "x": s["x"], "x_err": s["x_err_poisson"],
                     "x_onesided": s["x_onesided_19.8_20.0"], "D_kin": s["D_kin"],
                     "alpha_colour": s["alpha_colour_median_edge"],
                     "e": np.array(d["excess_vec"]), "sig": d["sigma_mocks_per_component"], "amp": d["excess_amp"],
                     "amp_alpha_colour": d["excess_amp_alpha_colour"]})
    xz = np.array(sorted(xz))
    return rows, xz


def catalogue():
    """Photo-z and their errors of the G < 20, |b| > 30 deg sample (None if the file is absent)."""
    if not CAT.exists():
        return None
    from astropy.io import fits
    c = fits.getdata(CAT)
    m = (np.abs(c["b"]) > 30.0) & (c["phot_g_mean_mag"] < 20.0)
    return np.asarray(c["redshift_quaia"][m], float), np.asarray(c["redshift_quaia_err"][m], float)


def density(z, grid, smooth=0.0):
    """dN/dz on grid (histogram with bin = grid step, optional Gaussian smoothing of width smooth)."""
    h = grid[1] - grid[0]
    edges = np.concatenate([grid - h / 2, [grid[-1] + h / 2]])
    n = np.histogram(z, edges)[0] / h
    if smooth > 0:
        k = np.arange(-int(4 * smooth / h), int(4 * smooth / h) + 1) * h
        w = np.exp(-0.5 * (k / smooth) ** 2)
        n = np.convolve(n, w / w.sum(), mode="same")
    return n


# ---------------------------------------------------------------- model
def components(tab, theta):
    """Dipole parts d_D, d_E, d_T on Z and the sky-averaged ln D_C, ln E, t_L."""
    zz = np.repeat(Z, len(PSI))
    cps = np.tile(np.cos(PSI), len(Z))
    n = np.stack([np.sqrt(1 - cps**2), np.zeros_like(cps), cps], axis=1)
    sky = T.Sky(zz, n)
    e = T.bilinear(tab, sky.sector_path(theta, cps), sky.zp)
    shp = (len(Z), len(PSI))
    lnD = np.log(np.trapezoid(1.0 / e, sky.zp, axis=1)).reshape(shp)
    tL = np.trapezoid(1.0 / ((1 + sky.zp) * e), sky.zp, axis=1).reshape(shp)
    lnE = np.log(e[:, -1]).reshape(shp)
    w = np.sin(PSI) / np.trapezoid(np.sin(PSI), PSI)
    out = {}
    for k, f in (("D", lnD), ("E", lnE), ("T", tL)):
        mean = np.trapezoid(f * w, PSI, axis=1)
        out["d" + k] = 1.5 * np.trapezoid((f - mean[:, None]) * np.cos(PSI) * np.sin(PSI), PSI, axis=1)
        out["mean" + k] = mean
    return out


def gamma_evol(comp, n_of_z, x_of_z, alpha=ALPHA):
    """Evolution of the comoving density above fixed L per unit z (item c), on Z."""
    lnn = np.log(np.maximum(n_of_z, 1e-12 * np.max(n_of_z)))
    lnD, lnE = comp["meanD"], comp["meanE"]
    dz = np.gradient
    return (dz(lnn, Z) - 2 * dz(lnD, Z) + dz(lnE, Z)
            + x_of_z * (2 * (dz(lnD, Z) + 1 / (1 + Z)) + (alpha - 1) / (1 + Z)))


def evol_term(comp, gam):
    return gam * (1 + Z) * np.exp(comp["meanE"]) * comp["dT"]


def slice_dipole(comp, p, x, evol=None, mass_total=None):
    """Signed slice dipole for p(z) on Z; covered fraction of the p(z) mass (mass_total: mass on all z)."""
    f = (2 - 2 * x) * comp["dD"] - comp["dE"] + (0.0 if evol is None else evol)
    ok = np.isfinite(f)
    tot = np.trapezoid(p, Z) if mass_total is None else mass_total
    cover = float(np.trapezoid(p * ok, Z) / tot)
    if cover < 0.5:
        return np.nan, cover
    parts = {k: float(np.trapezoid(np.where(ok, v, 0) * p, Z) / np.trapezoid(p * ok, Z))
             for k, v in (("D", (2 - 2 * x) * comp["dD"]), ("E", -comp["dE"]),
                          ("evol", np.zeros_like(Z) if evol is None else evol))}
    return float(np.trapezoid(np.where(ok, f, 0) * p, Z) / np.trapezoid(p * ok, Z)), cover, parts


# ---------------------------------------------------------------- statistics
def chi2_scalar(d, amp, sig):
    return float(np.sum(((np.abs(d) - amp) / sig) ** 2))


def chi2_vector(d, e, sig, u):
    return float(np.sum(np.sum((e - np.asarray(d)[:, None] * u[None, :]) ** 2, axis=1) / sig**2))


def m2lnrice(A, D, s):
    """-2 ln p(A | D, s): amplitude of a 3-vector with mean length D and Gaussian noise s per component."""
    D = np.maximum(np.abs(D), 1e-12)
    y = A * D / s**2
    lnsinh = y + np.log1p(-np.exp(-2 * y)) - np.log(2)
    lnp = 0.5 * np.log(2 / np.pi) + np.log(A / (D * s)) - (A**2 + D**2) / (2 * s**2) + lnsinh
    return float(np.sum(-2 * lnp))


def stats(d, rows, amp=None, e=None):
    amp = np.array([r["amp"] for r in rows]) if amp is None else amp
    e = np.array([r["e"] for r in rows]) if e is None else e
    sig = np.array([r["sig"] for r in rows])
    d = np.asarray(d, float)
    s_best = float(np.sum(np.abs(d) * amp / sig**2) / np.sum(d**2 / sig**2))
    return {"D": d.tolist(), "absD": np.abs(d).tolist(), "chi2_quaia": chi2_scalar(d, amp, sig),
            "trend": "growing" if abs(d[2]) > abs(d[0]) * 1.1 else ("decreasing" if abs(d[2]) < abs(d[0]) / 1.1 else "flat"),
            "chi2_vector_fixed_axis": chi2_vector(d, e, sig, U_MODEL),
            "m2lnL_rice": m2lnrice(amp, d, sig),
            "shape_rescale": s_best, "chi2_quaia_rescaled_shape": chi2_scalar(s_best * d, amp, sig)}


def constant_refs(rows, amp=None, e=None):
    amp = np.array([r["amp"] for r in rows]) if amp is None else amp
    e = np.array([r["e"] for r in rows]) if e is None else e
    sig = np.array([r["sig"] for r in rows])
    c = float(np.sum(amp / sig**2) / np.sum(1 / sig**2))
    cr = minimize_scalar(lambda c: m2lnrice(amp, np.full(3, c), sig), bounds=(1e-6, 0.06), method="bounded").x
    cv = float(np.sum((e @ U_MODEL) / sig**2) / np.sum(1 / sig**2))
    return {"const_scalar": {"amp": c, "chi2": chi2_scalar(np.full(3, c), amp, sig)},
            "const_rice": {"amp": float(cr), "m2lnL": m2lnrice(amp, np.full(3, cr), sig)},
            "const_vector_fixed_axis": {"amp": cv, "chi2": chi2_vector(np.full(3, cv), e, sig, U_MODEL)},
            "zero": {"chi2_scalar": chi2_scalar(np.zeros(3), amp, sig), "chi2_vector": chi2_vector(np.zeros(3), e, sig, U_MODEL),
                     "m2lnL_rice": m2lnrice(amp, np.zeros(3), sig)}}


# model axis in Galactic Cartesian components: the model dipole is negative along the axis opposite to the
# CatWISE excess, i.e. a positive amplitude points to the excess; U_MODEL = -direction of that axis
_, _, LB_CW = Q.excess_vector()
U_MODEL = -unitvec(*LB_CW)
V_CMB = unitvec(*Q.LB_CMB)


def kinematic_edge(rows, zc, alpha_key=None):
    """Excess vectors and amplitudes with the bin-edge term in D_kin (item b)."""
    nat = lambda z0: np.sum((zc >= z0 - 0.05) & (zc < z0 + 0.05)) / 0.1
    out = []
    for r in rows:
        lo, hi = r["z"]
        N = np.sum((zc >= lo) & (zc < hi))
        edge = ((1 + hi) * nat(hi) - (1 + lo) * nat(lo)) / N
        al = ALPHA if alpha_key is None else r[alpha_key]
        d_meas = r["e"] + r["D_kin"] * V_CMB
        dk_new = BETA * (2 + r["x"] * (1 + al) + edge)
        ex = d_meas - dk_new * V_CMB
        out.append({"edge_term": float(edge), "D_kin_old": r["D_kin"], "D_kin_new": float(dk_new),
                    "excess_vec": ex.tolist(), "excess_amp": float(np.linalg.norm(ex))})
    return out


# ---------------------------------------------------------------- main
def main():
    rows, xz = load_slices()
    cat = catalogue()
    sig = np.array([r["sig"] for r in rows])
    amp = np.array([r["amp"] for r in rows])
    out = {"theta_max_deg_env": os.environ.get("HIPPOPEDE_THETA_MAX_DEG"), "alpha_low": PH.ALPHA_LOW,
           "slices": [{"z": r["z"], "x": r["x"], "x_err": r["x_err"], "amp": r["amp"], "sigma": r["sig"]} for r in rows],
           "model_unit_vector_gal": U_MODEL.tolist(), "catwise": {"D_geo": Q.excess_vector()[1], "sigma": Q.SIG_GEO},
           "data_refs": constant_refs(rows), "points": []}
    print("data: amp", np.round(amp, 4), "sig", np.round(sig, 4), "constant refs", json.dumps(out["data_refs"]))
    x_of_z = np.interp(Z, xz[:, 0], xz[:, 1])
    if cat is not None:
        zc, ezc = cat
        n_full = density(zc, Z, smooth=0.08)
        n_fine_all = density(zc, ZH, smooth=0.0)
        out["catalogue_N"] = int(zc.size)
        # item b, data side: bin-edge term of the kinematic dipole
        kin = kinematic_edge(rows, zc)
        kin_col = kinematic_edge(rows, zc, "alpha_colour")
        out["kinematic_edge"] = {"alpha_1.26": kin, "alpha_colour": kin_col}
        amp_k = np.array([k["excess_amp"] for k in kin])
        e_k = np.array([k["excess_vec"] for k in kin])
        amp_kc = np.array([k["excess_amp"] for k in kin_col])
        out["data_refs_kinematic_edge"] = constant_refs(rows, amp_k, e_k)
        out["data_refs_kinematic_edge_alpha_colour"] = constant_refs(rows, amp_kc, np.array([k["excess_vec"] for k in kin_col]))
        print("kinematic edge:", [(round(k["edge_term"], 2), round(k["D_kin_new"], 4), round(k["excess_amp"], 4)) for k in kin])
        print("   alpha colour:", [round(k["excess_amp"], 4) for k in kin_col])
        # item a: p(z) of each slice on Z (and its total mass on all z, for the coverage)
        pz = {}
        for i, r in enumerate(rows):
            lo, hi = r["z"]
            pz[("flat", i)] = (((Z >= lo) & (Z <= hi)).astype(float), None)
            sel = (zc >= lo) & (zc < hi)
            h = density(zc[sel], Z)
            pz[("catalogue", i)] = (h, None)
            for s in SIG_Z:
                w = lambda zz: ndtr((hi - zz) / (s * (1 + zz))) - ndtr((lo - zz) / (s * (1 + zz)))
                p_all = n_fine_all * w(ZH)
                pz[(f"smear_{s:g}", i)] = (np.interp(Z, ZH, p_all), float(np.trapezoid(p_all, ZH)))
            zs, es = zc[sel], np.maximum(ezc[sel], 1e-3)
            p_err = np.zeros_like(ZH)
            for k in range(0, zs.size, 5000):
                p_err += np.exp(-0.5 * ((ZH[None, :] - zs[k:k + 5000, None]) / es[k:k + 5000, None]) ** 2).sum(0) \
                    / es[k:k + 5000].mean()   # approximate normalisation; only the shape matters
            pz[("catalogue_errors", i)] = (np.interp(Z, ZH, p_err), float(np.trapezoid(p_err, ZH)))
            for f in F_OUT:   # catastrophic photo-z outliers: a fraction f of the slice drawn from the whole n(z)
                ps = h / np.trapezoid(h, Z)
                pa = n_fine_all / np.trapezoid(n_fine_all, ZH)
                pz[(f"outliers_{f:g}", i)] = ((1 - f) * ps + f * np.interp(Z, ZH, pa),
                                              float((1 - f) + f * np.trapezoid(pa, ZH)))
        out["pz_mean_z"] = {f"{k}|{i}": float(np.trapezoid(Z * v[0], Z) / np.trapezoid(v[0], Z)) for (k, i), v in pz.items()}
    else:
        pz = {("flat", i): ((((Z >= r["z"][0]) & (Z <= r["z"][1])).astype(float)), None) for i, r in enumerate(rows)}
        n_full = None
        print("catalogue not found: only flat p(z) and items without the catalogue")

    pz_cw = Q.pz("gamma")
    pz_cw = np.interp(Z, Q.Z_Q, pz_cw, left=0.0)
    for ah, t0, th in POINTS:
        T.set_alpha_high(ah)
        tab = T.e_table(t0)
        comp = components(tab, th)
        P = {"alpha_high": ah, "t0": t0, "theta_obs": th, "variants": {}}
        P["reference_quaia_zslice_model_shape"] = [abs(QZ.shape(tab, *r["z"], r["x"], th)) for r in rows]
        rhat = PH.rhat_of_z(Z, T.RUN)
        P["components_on_Z"] = {"z": Z.tolist(), "dD": comp["dD"].tolist(), "dE": comp["dE"].tolist(),
                                "dT": comp["dT"].tolist(), "rhat_rad": np.asarray(rhat).tolist()}
        print("   z, rhat, d_E, d_E/rhat:", [(z0, round(float(np.interp(z0, Z, rhat)), 3), round(float(np.interp(z0, Z, comp["dE"])), 4),
                                             round(float(np.interp(z0, Z, comp["dE"] / rhat)), 4)) for z0 in (0.2, 0.5, 1.0, 1.5, 2.0)])

        def cw(evol=None, x=Q.X_SLOPE):
            d = slice_dipole(comp, pz_cw, x, evol)[0]
            return {"D": d, "chi2_catwise": float(((abs(d) - out["catwise"]["D_geo"]) / Q.SIG_GEO) ** 2)}

        def run(name, kind="flat", xs=None, evol=None, data=None):
            res = [slice_dipole(comp, pz[(kind, i)][0], (r["x"] if xs is None else xs[i]), evol,
                                pz[(kind, i)][1]) for i, r in enumerate(rows)]
            d = [x[0] for x in res]
            a_, e_ = (None, None) if data is None else data
            st = stats(d, rows, a_, e_) if np.all(np.isfinite(d)) else {"D": d}
            st["coverage"] = [x[1] for x in res]
            st["parts"] = [x[2] if len(x) > 2 else None for x in res]
            P["variants"][name] = st
            print(f"  {name:38s} |D| = {np.round(np.abs(d), 4)}  chi2_Q = {st.get('chi2_quaia', np.nan):6.2f}  "
                  f"vec {st.get('chi2_vector_fixed_axis', np.nan):6.2f}  rice {st.get('m2lnL_rice', np.nan):6.2f}  "
                  f"shape x{st.get('shape_rescale', np.nan):.2f} -> {st.get('chi2_quaia_rescaled_shape', np.nan):5.2f}  "
                  f"cover {np.round(st['coverage'], 2)}", flush=True)
            return st

        print(f"point alpha_high={ah}, t0={t0}, theta_obs={th}: reference shape "
              f"{np.round(P['reference_quaia_zslice_model_shape'], 4)}")
        run("baseline_flat")
        # a. redshift distribution
        if cat is not None:
            for k in ("catalogue", "smear_0.05", "smear_0.1", "smear_0.2", "catalogue_errors", "outliers_0.2", "outliers_0.4"):
                run("pz_" + k, k)
        # b. slope
        xs0 = np.array([r["x"] for r in rows]); xe = np.array([r["x_err"] for r in rows])
        run("x_minus_3sigma", xs=xs0 - 3 * xe)
        run("x_plus_3sigma", xs=xs0 + 3 * xe)
        run("x_onesided", xs=[r["x_onesided"] for r in rows])
        for xc in (1.0, 1.3, Q.X_SLOPE):
            run(f"x_all_{xc:.3g}", xs=[xc] * 3)
        run("no_dlnE_term_diagnostic", xs=None, evol=comp["dE"])     # removes -dE (diagnostic only)
        run("only_dlnE_term_diagnostic", xs=[1.0] * 3)               # (2 - 2x) = 0
        if cat is not None:
            run("kinematic_edge_data", data=(amp_k, e_k))
            run("kinematic_edge_data_alpha_colour", data=(amp_kc, None))
        # c. evolution with the emission time
        if n_full is not None:
            gam = gamma_evol(comp, n_full, x_of_z)
            gam_naive = np.gradient(np.log(np.maximum(n_full, 1e-12 * n_full.max())), Z)
            ev, ev_naive = evol_term(comp, gam), evol_term(comp, gam_naive)
            P["gamma_evol"] = gam.tolist()
            P["gamma_at"] = {str(z0): float(np.interp(z0, Z, gam)) for z0 in (0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0)}
            print("   gamma(z):", {k: round(v, 2) for k, v in P["gamma_at"].items()})
            run("evol_t_emission", evol=ev)
            run("evol_t_emission_naive_dlnn_dz", evol=ev_naive)
            run("evol_t_emission_pz_catalogue", "catalogue", evol=ev)
            run("evol_t_emission_smear_0.1", "smear_0.1", evol=ev)
            run("evol_t_emission_kin_edge", evol=ev, data=(amp_k, e_k))
            run("evol_t_emission_smear_0.1_kin_edge", "smear_0.1", evol=ev, data=(amp_k, e_k))
            run("all_catalogue_smear0.1_kinedge_noevol", "smear_0.1", data=(amp_k, e_k))
            # CatWISE: gamma from its p(z) model and x = 1.63
            x_cw = np.full_like(Z, Q.X_SLOPE)
            gam_cw = gamma_evol(comp, np.maximum(pz_cw, 1e-12), x_cw)
            P["catwise"] = {"baseline": cw(), "evol_t_emission": cw(evol_term(comp, gam_cw)),
                            "baseline_check_quasar_dipole_fit": abs(Q.dipole(tab, th, Q.X_SLOPE, Q.pz("gamma"))[0])}
        else:
            P["catwise"] = {"baseline": cw()}
        print("   CatWISE:", json.dumps(P["catwise"]))
        out["points"].append(P)
    JSON.mkdir(exist_ok=True)
    (JSON / "quaia_shape_diagnostics.json").write_text(json.dumps(out, indent=1))
    print("wrote json/quaia_shape_diagnostics.json")


if __name__ == "__main__":
    main()
