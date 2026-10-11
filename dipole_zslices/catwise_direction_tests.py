"""Directions of the CatWISE dipole in W1-W2 colour bins: significance of their difference and systematics tests.

Data: json/catwise_cuts_dipoles_jk<S>.json written by measure_catwise_cuts.py --jackknife 4 (or 8) with the
options S = "" (|b| > 30 deg), _b40, _b50 (stricter Galactic-latitude cuts), _gc50 and _b40_gc50 (pixels within
50 deg of the Galactic centre and anticentre removed), _ebv (per-pixel E(B-V) template added to the fit);
every fit has the |ecliptic latitude| template.  Each file holds, for every subsample, the dipole vector d, the
Poisson-mock covariance and the jackknife replicas d_(r) (common regions for all subsamples).

Tests on the three W1-W2 terciles (blue 0.80-0.95, middle 0.95-1.15, red >= 1.15):
  (1) common direction: d_i = a_i n with free amplitudes a_i and a common unit vector n (k = n_bins + 2), against
      free vectors (k = 3 n_bins): Delta chi2 with 2(n_bins - 1) degrees of freedom; for all three bins and for
      each pair; covariances: Poisson mocks (block diagonal) and jackknife (full, Hartlap factor
      (R - 3 n_bins - 2)/(R - 1)).
  (2) angle between bins with a 68 % interval from Gaussian draws of the jackknife covariance.
  (3) shift of each bin vector under each systematics option relative to the baseline (|b| > 30 deg).  For the
      cuts that remove pixels the samples are nested, and the covariance of the difference is approximated by
      rho (C_sub - C_base), C the Poisson-mock covariances (precise, 1000 mocks) and rho the mean jackknife/mock
      variance ratio of the two runs (the jackknife variances themselves are too noisy for the difference);
      this holds when the baseline estimator is the efficient one; where rho (C_sub - C_base) is not positive
      definite, chi2 uses its positive-eigenvalue subspace (dof = their number).  The E(B-V) run uses the same pixels and
      only the shift is given.
Writes ../json/catwise_direction_tests.json.
"""
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "..", "json")
OUT = os.path.join(JSON, "catwise_direction_tests.json")
RUNS = ["", "_b40", "_b50", "_gc50", "_b40_gc50", "_ebv"]
COLOURS = ["col_W1W2_0.8-0.95", "col_W1W2_0.95-1.15", "col_W1W2_1.15-9.0"]
LABEL = {COLOURS[0]: "blue", COLOURS[1]: "middle", COLOURS[2]: "red"}
LB_CMB, LB_GC, LB_AC = (264.0, 48.3), (0.0, 0.0), (180.0, 0.0)
RNG = np.random.default_rng(20261011)


def unitvec(l, b):
    lo, la = np.radians(l), np.radians(b)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def vec2lb(v):
    v = np.asarray(v) / np.linalg.norm(v)
    return [float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2])))]


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def covariances(rows):
    """Poisson (block diagonal) and jackknife (full, Hartlap-scaled inverse) of the stacked vectors."""
    n = len(rows)
    cp = np.zeros((3 * n, 3 * n))
    for i, r in enumerate(rows):
        cp[3 * i:3 * i + 3, 3 * i:3 * i + 3] = r["cov_mocks"]
    dj = np.concatenate([np.asarray(r["d_jk"]) for r in rows], axis=1)
    R = dj.shape[0]
    dd = dj - dj.mean(0)
    cj = (R - 1) / R * dd.T @ dd
    return {"poisson": (cp, 1.0), "jackknife": (cj, (R - 3 * n - 2) / (R - 1))}, R


def common_direction(d, cov, factor):
    """min over n of chi2 of d_i = a_i n (a_i linear); returns chi2, n, a."""
    n_b = len(d)
    W = factor * np.linalg.inv(cov)
    y = d.reshape(-1)

    def fit(lb):
        G = np.kron(np.eye(n_b), unitvec(*lb)[:, None])
        a = np.linalg.solve(G.T @ W @ G, G.T @ W @ y)
        r = y - G @ a
        return float(r @ W @ r), a

    best = None
    for l0 in range(0, 360, 30):
        for b0 in (-60, -30, 0, 30, 60):
            res = minimize(lambda q: fit(q)[0], [l0, b0], method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
    c, a = fit(best.x)
    return c, vec2lb(unitvec(*best.x)), a.tolist()


def direction_test(rows):
    d = np.array([r["d"] for r in rows])
    covs, R = covariances(rows)
    out = {"n_regions": R}
    for name, (cov, f) in covs.items():
        c, lb, a = common_direction(d, cov, f)
        dof = 2 * (len(rows) - 1)
        out[name] = {"chi2_common_direction": c, "dof": dof, "p_value": float(chi2_dist.sf(c, dof)),
                     "sigma_equiv": float(np.sqrt(chi2_dist.isf(chi2_dist.sf(c, dof), 1))) if c > 0 else 0.0,
                     "common_lb": lb, "amplitudes": a, "hartlap_factor": f}
    return out


def angle_interval(rows, cov, ndraw=20000):
    d = np.concatenate([r["d"] for r in rows])
    draws = RNG.multivariate_normal(d, cov, ndraw)
    a = np.array([angle(x[:3], x[3:]) for x in draws])
    return [float(np.percentile(a, 16)), float(np.percentile(a, 50)), float(np.percentile(a, 84))]


def main():
    out = {"runs": {}}
    base = None
    for sfx in RUNS:
        path = os.path.join(JSON, f"catwise_cuts_dipoles_jk{sfx}.json")
        meas = json.load(open(path))
        rows = {r["name"]: r for r in meas["subsamples"]}
        col = [rows[c] for c in COLOURS]
        res = {"input": os.path.basename(path), "n_pix": meas["n_pix"], "n_regions": meas["jackknife"]["n_regions"],
               "bins": {}}
        for r in col:
            d = np.array(r["d"])
            res["bins"][LABEL[r["name"]]] = {
                "N": r["N"], "x_eff": r["x_eff"], "D": r["D"], "lb": list(r["lb"]),
                "sigma_comp_mocks": r["sigma_per_component_mocks"], "sigma_comp_jk": r["sigma_per_component_jk"],
                "jk_to_mock_variance_ratio": r["jk_to_mock_variance_ratio"], "sigma_D_jk": r["sigma_D_jk"],
                "angle_to_CMB": angle(d, unitvec(*LB_CMB)), "angle_to_GC": angle(d, unitvec(*LB_GC)),
                "angle_to_anticentre": angle(d, unitvec(*LB_AC)),
                "ecl_coeff_rel_per_deg": r["ecl_coeff_rel_per_deg"], "ebv_coeff_rel": r.get("ebv_coeff_rel")}
        res["common_direction_all3"] = direction_test(col)
        res["pairs"] = {}
        for i, j in ((0, 2), (1, 2), (0, 1)):
            pair = [col[i], col[j]]
            key = f"{LABEL[pair[0]['name']]}-{LABEL[pair[1]['name']]}"
            covs, _ = covariances(pair)
            res["pairs"][key] = {"angle_deg": angle(pair[0]["d"], pair[1]["d"]),
                                 "angle_68_jk": angle_interval(pair, covs["jackknife"][0]),
                                 **direction_test(pair)}
        if sfx == "":
            base = {LABEL[r["name"]]: r for r in col}
            base_npix = meas["n_pix"]
        else:
            res["shift_vs_baseline"] = {}
            for r in col:
                b0 = base[LABEL[r["name"]]]
                dd = np.array(r["d"]) - np.array(b0["d"])
                rho = 0.5 * (r["jk_to_mock_variance_ratio"] + b0["jk_to_mock_variance_ratio"])
                cdiff = rho * (np.asarray(r["cov_mocks"]) - np.asarray(b0["cov_mocks"]))
                row = {"delta_D": r["D"] - b0["D"], "angle_deg": angle(r["d"], b0["d"]), "delta_vec": dd.tolist(),
                       "jk_mock_ratio_used": rho}
                if meas["n_pix"] < base_npix:
                    lam, vec = np.linalg.eigh(cdiff)
                    pos = lam > 0.05 * lam.max()
                    c2 = float(np.sum((vec[:, pos].T @ dd) ** 2 / lam[pos]))
                    vD = float(np.asarray(r["d"]) @ cdiff @ np.asarray(r["d"])) / r["D"] ** 2
                    row.update({"chi2_nested": c2, "dof": int(pos.sum()), "p_value": float(chi2_dist.sf(c2, pos.sum())),
                                "sigma_D_nested": float(np.sqrt(vD)) if vD > 0 else None})
                res["shift_vs_baseline"][LABEL[r["name"]]] = row
        out["runs"][sfx or "_b30"] = res
        ca = res["common_direction_all3"]
        print(f"{sfx or '_b30':10s} R={res['n_regions']:3d} " + " | ".join(
            f"{k}: D={v['D']:.4f} ({v['lb'][0]:.0f},{v['lb'][1]:.0f}) sjk={v['sigma_comp_jk']:.4f}"
            for k, v in res["bins"].items()) +
            f" || common dir chi2 Poisson {ca['poisson']['chi2_common_direction']:.1f} "
            f"(p={ca['poisson']['p_value']:.2g}), jk {ca['jackknife']['chi2_common_direction']:.1f} "
            f"(p={ca['jackknife']['p_value']:.2g}); blue-red angle {res['pairs']['blue-red']['angle_deg']:.0f} "
            f"jk p={res['pairs']['blue-red']['jackknife']['p_value']:.2g}", flush=True)
        if "shift_vs_baseline" in res:
            s = res["shift_vs_baseline"]["red"]
            print(f"{'':10s} red shift: dD={s['delta_D']:+.4f}, angle {s['angle_deg']:.0f} deg, "
                  f"chi2 {s.get('chi2_nested', np.nan):.1f} (p={s.get('p_value', np.nan):.2g}), "
                  f"sigma_D nested {s.get('sigma_D_nested') or np.nan:.4f}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
