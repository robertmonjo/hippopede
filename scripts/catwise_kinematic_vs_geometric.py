"""Kinematic or geometric origin of the CatWISE dipole: dependence on the count slope x and spectral index alpha.

Data: dipole vectors of disjoint CatWISE subsamples with their shot-noise covariances from Poisson mocks
(json/catwise_cuts_dipoles.json, dipole_zslices/measure_catwise_cuts.py): (a) four W1 bins, (b) three W1-W2
terciles at W1 <= 16.4, (c) the 4 x 2 grid of W1 bins and W1-W2 halves.  Every subsample i carries the
kinematic factor K_i = D_kin/beta = 2 + x(1 + alpha) generalised to a bin, and the effective slope x_i.

Predictions for subsample i (Galactic Cartesian vectors):
  kinematic (Ellis & Baldwin):  d_i = K_i beta,  beta along the velocity;
  geometric (hippopede, observer at sector angle theta_obs):  d_i = g(x_i; theta_obs, t0, alpha_high) n_ax,
  with g = quasar_dipole_fit.dipole (signed amplitude along the model axis, p(z) gamma with mean 1.2 for every
  subsample), which is exactly linear in x, g = A + B x; no alpha.  n_ax is free.
Models:
  M0  kinematic, beta = beta_CMB along the CMB dipole (264.0, 48.3) deg                     k = 0
  M1  kinematic, beta free along the CMB direction                                         k = 1
  M1b kinematic, beta vector free                                                          k = 3
  M2  beta_CMB + geometric with the amplitude of the model, axis free:
        at the CC+SN+CatWISE minimum (alpha_high, t0, theta_obs) = (0.381, 2.80, 7.2 deg)  k = 2
        along the band of json/dipole_chi2_vs_theta_continuous_noquaia.json (t0, alpha_high fitted to
        CC+SN+CatWISE at each theta_obs), theta_obs free                                   k = 3
  M3  as M2 with beta free along the CMB direction                                         k = 3 / 4
  M2s beta_CMB + geometric x-shape (A + B x_i)/(A + B x_ref) times a free vector           k = 3
  M3s as M2s with beta free along the CMB direction                                        k = 4
  M4  beta_CMB + constant vector (an excess independent of x and alpha)                    k = 3
  L   d_i = a + (x_i - x_ref) s (any origin, linear in x)                                   k = 6
M1b, M2s and M4 have three parameters each and differ only in how the amplitude depends on x and alpha:
proportional to K_i, to A + B x_i, or constant.  chi2 = sum_i r_i^T C_i^-1 r_i over disjoint subsamples (shot
noise only); the same fits with C_i scaled by the Pearson dispersion of each subsample are also given.
Sector tables with HIPPOPEDE_THETA_MAX_DEG = 90.  Writes json/catwise_kinematic_vs_geometric.json and
figures/hippopede_catwise_cuts.(png|pdf).  --in-suffix S reads json/catwise_cuts_dipoles<S>.json (e.g. _b40, the
robustness run with |b| > 40 deg) and appends S to the outputs.

Further options (off by default; the default outputs are unchanged):
  --pz-json J        per-subsample redshift distributions (json/catwise_bins_pz.json, catwise_redshifts.py) in
                     place of the common gamma p(z): for subsample i, g_i = A_i + B_i x_i with A_i, B_i from
                     quasar_dipole_fit.dipole with the p(z) of the sources of i matched to the catalogue chosen by
                     --pz-catalogue (quaia, sdss, or sdss_faint: DR16Q matches with i > 20).  The model covers
                     0.1 <= z <= 2.1 only; --pz-treatment truncate renormalises p(z) on 0.1-2.1 (sources outside
                     are assumed to follow the modelled ones), clamp gives the mass above 2.1 (below 0.1) the dipole at z = 2.1 (0.1), which is a
                     lower bound on |g| of those sources when |g(z)| grows with z.
  --cov jk           covariance of the stacked dipole vectors of each dataset from the delete-one jackknife of
                     measure_catwise_cuts.py --jackknife (input json/catwise_cuts_dipoles_jk<S>.json, with
                     the cross-covariance between subsamples), inverse multiplied by the Hartlap factor
                     (R - 3n - 2)/(R - 1) (R regions, n subsamples); the block-diagonal version (no cross-
                     covariance) is fitted as well.  The dispersion-scaled fits are not repeated.
  --out-suffix S     suffix of json/catwise_kinematic_vs_geometric<S>.json (default: _pz_<catalogue>[_clamp]
                     and/or _jk, then the input suffix).  No figure is drawn in these modes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("HIPPOPEDE_THETA_MAX_DEG", "90")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import quasar_dipole_fit as Q  # noqa: E402

T = Q.T
JSON = ROOT / "json"
FIG = ROOT / "figures"
BETA_CMB = 1.234e-3
LB_CMB = (264.0, 48.3)
REFERENCE = {"alpha_high": 0.381, "t0": 2.80, "theta_obs": 7.2}
DATASETS = {"mag_bins": "mag_bin", "colour_bins": "colour_bin", "grid": "grid"}


def unitvec(l, b):
    lo, la = np.radians(l), np.radians(b)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def vec2lb(v):
    v = np.asarray(v) / np.linalg.norm(v)
    return [float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2])))]


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def geo_coefficients_pz(arg):
    """A_i, B_i of g_i(x) = A_i + B_i x for every subsample p(z) (dict name -> p on Q.Z_Q) at one point."""
    point, pzs = arg
    T.set_alpha_high(point["alpha_high"])
    tab = T.e_table(point["t0"])
    per = {}
    for name, p in pzs.items():
        g0, c0 = Q.dipole(tab, point["theta_obs"], 0.0, p)
        g1, _ = Q.dipole(tab, point["theta_obs"], 1.0, p)
        per[name] = {"A": g0, "B": g1 - g0, "coverage": c0}
    return {**point, "per_subsample": per}


def subsample_pz(entry, treatment):
    """p on Q.Z_Q for one subsample entry of catwise_bins_pz.json (see --pz-treatment)."""
    p = np.asarray(entry["p_ZQ"], float)
    f_in = np.trapezoid(p, Q.Z_Q)
    if treatment == "truncate":
        return p / f_in
    h = Q.Z_Q[1] - Q.Z_Q[0]
    p = p.copy()
    p[-1] += entry["f_above_2.1"] / (h / 2)
    p[0] += entry["f_below_0.1"] / (h / 2)
    return p / np.trapezoid(p, Q.Z_Q)


def geo_coefficients(point):
    """A, B of g(x) = A + B x (signed dipole along the model axis) at (alpha_high, t0, theta_obs)."""
    T.set_alpha_high(point["alpha_high"])
    tab = T.e_table(point["t0"])
    p = Q.pz("gamma")
    g0, c0 = Q.dipole(tab, point["theta_obs"], 0.0, p)
    g1, _ = Q.dipole(tab, point["theta_obs"], 1.0, p)
    g163, _ = Q.dipole(tab, point["theta_obs"], Q.X_SLOPE, p)
    return {**point, "A": g0, "B": g1 - g0, "g_x1.63": g163, "coverage": c0,
            "linearity_check": g163 - (g0 + (g1 - g0) * Q.X_SLOPE)}


class DataFull:
    """Stacked dipoles of a dataset with a full (3n x 3n) covariance (jackknife with cross-covariance)."""

    def __init__(self, rows, cov_full, factor=1.0):
        self.rows = rows
        self.d = np.array([r["d"] for r in rows])
        self.n = len(rows)
        self.Wf = factor * np.linalg.inv(cov_full)
        blocks = [cov_full[3 * i:3 * i + 3, 3 * i:3 * i + 3] / factor for i in range(self.n)]
        self.W = np.linalg.inv(np.array(blocks))  # diagonal blocks, used for the projection errors only
        self.K = np.array([r["K"] for r in rows])
        self.x = np.array([r["x_eff"] for r in rows])

    def wls(self, fixed, F):
        r0 = (self.d - fixed).reshape(-1)
        if F is None or F.shape[2] == 0:
            return np.zeros(0), float(r0 @ self.Wf @ r0)
        G = F.reshape(3 * self.n, -1)
        p = np.linalg.solve(G.T @ self.Wf @ G, G.T @ self.Wf @ r0)
        r = r0 - G @ p
        return p, float(r @ self.Wf @ r)


def jk_covariance(rows, block_diagonal=False):
    """(R-1)/R sum_r (D_(r) - <D>)(D_(r) - <D>)^T of the stacked dipole vectors D = (d_1, ..., d_n)."""
    dj = np.concatenate([np.asarray(r["d_jk"]) for r in rows], axis=1)
    R = dj.shape[0]
    dd = dj - dj.mean(0)
    cov = (R - 1) / R * dd.T @ dd
    if block_diagonal:
        cov = cov * np.kron(np.eye(len(rows)), np.ones((3, 3)))
    return cov, R


class Data:
    def __init__(self, rows, scale_dispersion=False):
        self.rows = rows
        self.d = np.array([r["d"] for r in rows])
        cov = np.array([r["cov_mocks"] for r in rows])
        if scale_dispersion:
            cov = cov * np.array([r["pearson_dispersion"] for r in rows])[:, None, None]
        self.W = np.linalg.inv(cov)
        self.K = np.array([r["K"] for r in rows])
        self.x = np.array([r["x_eff"] for r in rows])
        self.n = len(rows)

    def wls(self, fixed, F):
        """Minimise sum (d - fixed - F p)^T W (d - fixed - F p); F has shape (n, 3, k).  Returns p, chi2."""
        r0 = self.d - fixed
        if F is None or F.shape[2] == 0:
            return np.zeros(0), float(np.einsum("ni,nij,nj->", r0, self.W, r0))
        M = np.einsum("nik,nij,njl->kl", F, self.W, F)
        v = np.einsum("nik,nij,nj->k", F, self.W, r0)
        p = np.linalg.solve(M, v)
        r = r0 - np.einsum("nik,k->ni", F, p)
        return p, float(np.einsum("ni,nij,nj->", r, self.W, r))


def fit_models(data, geo_ref, geo_band, x_ref):
    v_cmb = unitvec(*LB_CMB)
    kin_cmb = data.K[:, None] * BETA_CMB * v_cmb[None, :]
    F_beta_par = (data.K[:, None] * v_cmb[None, :])[:, :, None]
    eye = np.broadcast_to(np.eye(3), (data.n, 3, 3))
    F_beta_vec = data.K[:, None, None] * eye
    out = {}

    def store(name, k, chi2, params, pred):
        out[name] = {"k": k, "chi2": chi2, "AIC": chi2 + 2 * k, "BIC": chi2 + k * np.log(3 * data.n),
                     "params": params, "pred": pred.tolist()}

    store("M0", 0, data.wls(kin_cmb, None)[1], {}, kin_cmb)
    p, c = data.wls(0 * kin_cmb, F_beta_par)
    store("M1", 1, c, {"beta": float(p[0])}, F_beta_par[:, :, 0] * p[0])
    p, c = data.wls(0 * kin_cmb, F_beta_vec)
    store("M1b", 3, c, {"beta": float(np.linalg.norm(p)), "lb": vec2lb(p), "angle_to_CMB": angle(p, v_cmb)},
          data.K[:, None] * p[None, :])
    p, c = data.wls(kin_cmb, eye)
    store("M4", 3, c, {"c": float(np.linalg.norm(p)), "lb": vec2lb(p)}, kin_cmb + p[None, :])

    def geo_fixed_amp(gc, beta_free):
        g = gc["A"] + gc["B"] * data.x

        def chi(lb):
            n = unitvec(*lb)
            fixed = g[:, None] * n[None, :]
            if beta_free:
                return data.wls(fixed, F_beta_par)
            return np.zeros(0), data.wls(kin_cmb + fixed, None)[1]

        best = None
        for l0 in range(0, 360, 30):
            for b0 in (-60, -30, 0, 30, 60):
                res = minimize(lambda q: chi(q)[1], [l0, b0], method="Nelder-Mead",
                               options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 400})
                if best is None or res.fun < best.fun:
                    best = res
        p, c = chi(best.x)
        n = unitvec(*best.x)
        beta = float(p[0]) if beta_free else BETA_CMB
        pred = data.K[:, None] * beta * v_cmb[None, :] + g[:, None] * n[None, :]
        # the model dipole is negative along n: the counts excess points to -n
        return c, {"beta": beta, "axis_lb": vec2lb(n), "excess_direction_lb": vec2lb(-n), "g": g.tolist()}, pred

    for name, beta_free, k in (("M2_ref", False, 2), ("M3_ref", True, 3)):
        c, prm, pred = geo_fixed_amp(geo_ref, beta_free)
        store(name, k, c, {**prm, "point": REFERENCE}, pred)
    for name, beta_free, k in (("M2_band", False, 3), ("M3_band", True, 4)):
        scan = []
        for gc in geo_band:
            c, prm, pred = geo_fixed_amp(gc, beta_free)
            scan.append((c, gc, prm, pred))
        best = min(scan, key=lambda s: s[0])
        store(name, k, best[0], {**best[2], "theta_obs": best[1]["theta_obs"], "t0": best[1]["t0"],
                                 "alpha_high": best[1]["alpha_high"],
                                 "scan": [{"theta_obs": s[1]["theta_obs"], "chi2": s[0]} for s in scan]}, best[3])
    shape = (geo_ref["A"] + geo_ref["B"] * data.x) / (geo_ref.get("A_full", geo_ref["A"]) +
                                                      geo_ref.get("B_full", geo_ref["B"]) * x_ref)
    F_shape = shape[:, None, None] * eye
    p, c = data.wls(kin_cmb, F_shape)
    store("M2s", 3, c, {"w": float(np.linalg.norm(p)), "lb": vec2lb(p), "shape_at_reference": True},
          kin_cmb + shape[:, None] * p[None, :])
    F = np.concatenate([F_beta_par, F_shape], axis=2)
    p, c = data.wls(0 * kin_cmb, F)
    store("M3s", 4, c, {"beta": float(p[0]), "w": float(np.linalg.norm(p[1:])), "lb": vec2lb(p[1:])},
          np.einsum("nik,k->ni", F, p))
    F = np.concatenate([eye, (data.x - x_ref)[:, None, None] * eye], axis=2)
    p, c = data.wls(0 * kin_cmb, F)
    a, s = p[:3], p[3:]
    store("L", 6, c, {"a": a.tolist(), "s": s.tolist(), "s_par_a": float(s @ a / np.linalg.norm(a))},
          np.einsum("nik,k->ni", F, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-suffix", default="")
    ap.add_argument("--pz-json", default=None, help="per-subsample p(z) (json/catwise_bins_pz.json)")
    ap.add_argument("--pz-catalogue", default="quaia", choices=("quaia", "sdss", "sdss_faint"))
    ap.add_argument("--pz-treatment", default="truncate", choices=("truncate", "clamp"))
    ap.add_argument("--cov", default="mocks", choices=("mocks", "jk"))
    ap.add_argument("--out-suffix", default=None)
    args = ap.parse_args()
    sfx = args.in_suffix
    if args.pz_json is None and args.cov == "mocks":
        return main_default(sfx)
    return main_options(args)


def main_default(sfx):
    meas = json.loads((JSON / f"catwise_cuts_dipoles{sfx}.json").read_text())
    rows = meas["subsamples"]
    full = rows[0]
    x_ref = full["x_eff"]
    band_json = json.loads((JSON / "dipole_chi2_vs_theta_continuous_noquaia.json").read_text())
    band = [{"alpha_high": r["with_cc_sn"]["alpha_high"], "t0": r["with_cc_sn"]["t0"], "theta_obs": r["theta_obs"]}
            for r in band_json["rows"] if r["theta_obs"] > 0]
    points = [REFERENCE] + band
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", "8"))) as ex:
        geo = list(ex.map(geo_coefficients, points))
    geo_ref, geo_band = geo[0], geo[1:]
    for gc in geo:
        print(f"theta_obs={gc['theta_obs']:5.2f} t0={gc['t0']:.3f} ah={gc['alpha_high']:.3f}: g = {gc['A']:+.5f} "
              f"{gc['B']:+.5f} x  (g(1.63) = {gc['g_x1.63']:+.5f}, coverage {gc['coverage']:.2f})", flush=True)

    v_cmb = unitvec(*LB_CMB)
    u_full = np.asarray(full["d"]) / np.linalg.norm(full["d"])
    results = {}
    for ds, kind in DATASETS.items():
        sub = [r for r in rows if r["kind"] == kind]
        for scale in (False, True):
            data = Data(sub, scale)
            fits = fit_models(data, geo_ref, geo_band, x_ref)
            key = ds + ("_dispersion_scaled" if scale else "")
            # projection on the full-sample direction and its linear trend in x
            proj = data.d @ u_full
            sig = np.array([np.sqrt(u_full @ np.linalg.inv(W) @ u_full) for W in data.W])
            A = np.stack([np.ones(data.n), data.x - x_ref], 1)
            C = np.linalg.inv((A / sig[:, None] ** 2).T @ A)
            lin = lambda y: C @ ((A / sig[:, None] ** 2).T @ y)  # noqa: E731
            pp = lin(proj)
            trend = {"intercept": float(pp[0]), "slope_dD_dx": float(pp[1]), "slope_sigma": float(np.sqrt(C[1, 1])),
                     "model_slopes": {name: float(lin(np.array(m["pred"]) @ u_full)[1]) for name, m in fits.items()},
                     "note": "weighted linear fit of the projection on the full-sample direction against x_eff - x_ref"}
            results[key] = {"subsamples": [r["name"] for r in sub], "x_eff": data.x.tolist(), "K": data.K.tolist(),
                            "alpha_eff": [r["alpha_eff"] for r in sub], "proj_full_dir": proj.tolist(),
                            "proj_sigma": sig.tolist(), "trend_in_x": trend, "models": fits}
            print(f"\n== {key} (n = {data.n}); projection slope dD/dx = {pp[1]:+.5f} +- {np.sqrt(C[1, 1]):.5f}")
            for name, m in fits.items():
                print(f"  {name:8s} k={m['k']} chi2={m['chi2']:8.2f} AIC={m['AIC']:8.2f} BIC={m['BIC']:8.2f} "
                      f"{ {k: v for k, v in m['params'].items() if k not in ('scan', 'g', 'point')} }")

    out = {"input": f"json/catwise_cuts_dipoles{sfx}.json", "beta_CMB": BETA_CMB, "cmb_lb": LB_CMB,
           "x_ref_full_sample": x_ref, "full_sample_direction_lb": vec2lb(u_full),
           "pz": "gamma z^2 exp(-z/0.4) on 0.1-2.1 (quasar_dipole_fit.pz), the same for every subsample",
           "theta_max_deg": os.environ["HIPPOPEDE_THETA_MAX_DEG"],
           "geo_reference": geo_ref, "geo_band": geo_band, "fits": results}
    (JSON / f"catwise_kinematic_vs_geometric{sfx}.json").write_text(json.dumps(out, indent=1))
    plot(out, rows, sfx)


def main_options(args):
    """Fits with per-subsample p(z) (--pz-json) and/or the jackknife covariance (--cov jk)."""
    in_sfx = args.in_suffix
    meas_name = f"catwise_cuts_dipoles{'_jk' if args.cov == 'jk' else ''}{in_sfx}.json"
    meas = json.loads((JSON / meas_name).read_text())
    rows = meas["subsamples"]
    full = rows[0]
    x_ref = full["x_eff"]
    band_json = json.loads((JSON / "dipole_chi2_vs_theta_continuous_noquaia.json").read_text())
    band = [{"alpha_high": r["with_cc_sn"]["alpha_high"], "t0": r["with_cc_sn"]["t0"], "theta_obs": r["theta_obs"]}
            for r in band_json["rows"] if r["theta_obs"] > 0]
    points = [REFERENCE] + band
    workers = int(os.environ.get("WORKERS", "8"))
    pz_info = None
    if args.pz_json:
        pzj = json.loads(Path(args.pz_json).read_text())
        entries = {e["name"]: e[args.pz_catalogue] for e in pzj["subsamples"]}
        pzs = {r["name"]: subsample_pz(entries[r["name"]], args.pz_treatment) for r in rows}
        pzs["__gamma__"] = Q.pz("gamma")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            geo = list(ex.map(geo_coefficients_pz, [(p, pzs) for p in points]))
        pz_info = {"json": args.pz_json, "catalogue": args.pz_catalogue, "treatment": args.pz_treatment,
                   "per_subsample_reference": {}}
        ref = geo[0]["per_subsample"]
        gam = ref["__gamma__"]
        for r in rows:
            e, c = entries[r["name"]], ref[r["name"]]
            pz_info["per_subsample_reference"][r["name"]] = {
                "x_eff": r["x_eff"], "A": c["A"], "B": c["B"], "coverage": c["coverage"],
                "g": c["A"] + c["B"] * r["x_eff"], "g_gamma": gam["A"] + gam["B"] * r["x_eff"],
                "matched_fraction": e["matched_fraction"], "mean_z": e["mean"], "median_z": e["median"],
                "f_above_2.1": e["f_above_2.1"], "f_below_0.1": e["f_below_0.1"]}
            print(f"{r['name']:24s} x={r['x_eff']:.3f} <z>={e['mean']:.3f} >2.1:{e['f_above_2.1']:.3f} "
                  f"g = {c['A']:+.5f} {c['B']:+.5f} x = {c['A'] + c['B'] * r['x_eff']:+.5f} "
                  f"(gamma {gam['A'] + gam['B'] * r['x_eff']:+.5f})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            geo = list(ex.map(geo_coefficients, points))

    def geo_view(gc, sub):
        if "per_subsample" not in gc:
            return gc
        per = gc["per_subsample"]
        return {**{k: v for k, v in gc.items() if k != "per_subsample"},
                "A": np.array([per[r["name"]]["A"] for r in sub]), "B": np.array([per[r["name"]]["B"] for r in sub]),
                "A_full": per[full["name"]]["A"], "B_full": per[full["name"]]["B"]}

    u_full = np.asarray(full["d"]) / np.linalg.norm(full["d"])
    results = {}
    for ds, kind in DATASETS.items():
        sub = [r for r in rows if r["kind"] == kind]
        if args.cov == "jk":
            variants = []
            for bd in (False, True):
                cov, R = jk_covariance(sub, bd)
                f = (R - 3 * len(sub) - 2) / (R - 1)
                variants.append((ds + ("_jk_blockdiag" if bd else "_jk"), DataFull(sub, cov, f),
                                 {"n_regions": R, "hartlap_factor": f, "block_diagonal": bd}))
        else:
            variants = [(ds, Data(sub, False), {})]
        for key, data, cinfo in variants:
            fits = fit_models(data, geo_view(geo[0], sub), [geo_view(g, sub) for g in geo[1:]], x_ref)
            proj = data.d @ u_full
            sig = np.array([np.sqrt(u_full @ np.linalg.inv(W) @ u_full) for W in data.W])
            results[key] = {"subsamples": [r["name"] for r in sub], "x_eff": data.x.tolist(), "K": data.K.tolist(),
                            "alpha_eff": [r["alpha_eff"] for r in sub], "proj_full_dir": proj.tolist(),
                            "proj_sigma": sig.tolist(), "covariance": cinfo or "Poisson mocks", "models": fits}
            print(f"\n== {key} (n = {data.n}) {cinfo}")
            for name, m in fits.items():
                prm = {k: v for k, v in m["params"].items() if k not in ("scan", "g", "point")}
                print(f"  {name:8s} k={m['k']} chi2={m['chi2']:8.2f} AIC={m['AIC']:8.2f} {prm}")

    out_sfx = args.out_suffix
    if out_sfx is None:
        out_sfx = (f"_pz_{args.pz_catalogue}" + ("_clamp" if args.pz_treatment == "clamp" else "")
                   if args.pz_json else "") + ("_jk" if args.cov == "jk" else "") + in_sfx
    out = {"input": f"json/{meas_name}", "beta_CMB": BETA_CMB, "cmb_lb": LB_CMB, "x_ref_full_sample": x_ref,
           "full_sample_direction_lb": vec2lb(u_full),
           "pz": pz_info or "gamma z^2 exp(-z/0.4) on 0.1-2.1 (quasar_dipole_fit.pz), the same for every subsample",
           "covariance": "jackknife (full and block-diagonal), Hartlap-corrected" if args.cov == "jk" else
           "Poisson mocks (block-diagonal)",
           "theta_max_deg": os.environ["HIPPOPEDE_THETA_MAX_DEG"],
           "geo_reference": geo[0], "geo_band": geo[1:], "fits": results}
    (JSON / f"catwise_kinematic_vs_geometric{out_sfx}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote json/catwise_kinematic_vs_geometric{out_sfx}.json")


def plot(out, rows, sfx=""):
    plt.rcParams.update({"font.size": 10, "mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.6), dpi=180, sharey=True)
    styles = {"M0": ("kinematic, $\\beta_{\\rm CMB}$ (M0)", "#888888", "--"),
              "M1b": ("kinematic, $\\vec\\beta$ free (M1b)", "#1f77b4", "-"),
              "M2_ref": ("$\\beta_{\\rm CMB}$ + hippopede, reference (M2)", "#d62728", "-"),
              "M3_ref": ("$\\beta$ free + hippopede, reference (M3)", "#ff7f0e", ":"),
              "M4": ("$\\beta_{\\rm CMB}$ + constant vector (M4)", "#2ca02c", "-.")}
    u_full = np.array(rows[0]["d"]) / np.linalg.norm(rows[0]["d"])
    for ax, ds, xkey, xlabel in ((axes[0], "mag_bins", "x_eff", "count slope $x$ of the W1 bin"),
                                  (axes[1], "colour_bins", "alpha_eff", "spectral index $\\alpha$ of the W1$-$W2 bin")):
        r = out["fits"][ds]
        xv = np.array(r[xkey])
        order = np.argsort(xv)
        ax.errorbar(xv, r["proj_full_dir"], yerr=r["proj_sigma"], fmt="o", color="k", ms=5, capsize=3,
                    label="CatWISE subsamples", zorder=5)
        for name, (lab, col, ls) in styles.items():
            pred = np.array(r["models"][name]["pred"]) @ u_full
            ax.plot(xv[order], pred[order], ls=ls, color=col, marker=".", lw=1.4,
                    label=f"{lab}: $\\chi^2$ = {r['models'][name]['chi2']:.1f}")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        for xi, yi, xe in zip(xv, r["proj_full_dir"], r["x_eff"]):
            if xkey == "alpha_eff":
                ax.annotate(f"$x$={xe:.2f}", (xi, yi), xytext=(6, -10), textcoords="offset points", fontsize=7)
        ax.set_ylim(0.0, 0.03)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
    axes[0].set_ylabel("dipole projected on the full-sample direction")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hippopede_catwise_cuts{sfx}.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")


if __name__ == "__main__":
    main()
