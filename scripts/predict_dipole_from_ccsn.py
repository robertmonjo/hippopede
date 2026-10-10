"""Blind prediction of the quasar dipoles by the light-cone ball fitted to the chronometers and supernovae only.

The chi2 of the 38 chronometers and the 1579 unbinned supernovae on the grid of the overview map
(json/ball_zoom_unbinned_overview.json: alpha_high, theta_obs, t0) defines the posterior of the ball given CC + SN,
    p(alpha_high, t0, theta_obs | CC+SN) ~ prior x exp(-chi2_CCSN / 2),
with alpha_high uniform on the grid, t0 uniform in ln t0 (1.4-20) and two priors for the observer: theta_obs uniform
(0-45 deg) or weighted by the volume of its sector, sin^2(2 theta_obs).  Unfeasible observers have zero prior.
The dipole amplitudes predicted at each point (CatWISE |D| and the three Quaia slices, dipole runs with suffix _wide)
are then compared with the measured ones without refitting:
  - the CatWISE amplitude and the dipole chi2 at the best CC + SN point;
  - the predictive distribution of the CatWISE amplitude (median, 68% interval, probability of exceeding the
    measured excess);
  - the predictive likelihood of the dipole data, L = < exp(-chi2_dipoles / 2) >_posterior, as an effective
    chi2_pred = -2 ln L, against flat LCDM fitted to CC + SN, which predicts no dipole beyond the kinematic one.
For reference, LCDM with an ad hoc amplitude A common to CatWISE and Quaia, uniform on [0, A_max], has the
predictive likelihood of its marginal over A.  Run with HIPPOPEDE_THETA_MAX_DEG=90 (as the map).
Writes json/predict_dipole_from_ccsn.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_ball_unbinned as F  # noqa: E402
import plot_ball_zoom_unbinned as Z  # noqa: E402

JSON = ROOT / "json"   # numerical outputs; figures/ holds the png/pdf
DIPOLE_ALPHAS = [round(0.30 + 0.02 * k, 2) for k in range(21)]


def catwise_amplitude(ah, t0, theta):
    """|D| of the CatWISE dipole runs interpolated in alpha_high, ln t0 and theta_obs (as catwise_chi2)."""
    F.catwise_chi2(ah, t0, theta)   # loads the runs
    al = np.array(F.DIPOLE_ALPHAS)
    if not al[0] <= ah <= al[-1]:
        return np.nan
    k = min(int(np.searchsorted(al, ah, side="right")) - 1, len(al) - 2)
    w = (ah - al[k]) / (al[k + 1] - al[k])
    vals = []
    for a in (al[k], al[k + 1]):
        lt, D, g, _, _ = F._DIP[a]
        x = np.log(t0)
        n = min(int(np.searchsorted(lt, x, side="right")) - 1, len(lt) - 2) if x <= lt[-1] + 1e-12 else len(lt)
        if not 0 <= n < len(lt) - 1:
            return np.nan
        f = (x - lt[n]) / (lt[n + 1] - lt[n])
        row = (1 - f) * D[n] + f * D[n + 1]
        ok = np.isfinite(row)
        if not ok.any() or theta > g[ok].max():
            return np.nan
        vals.append(np.interp(theta, g[ok], row[ok]))
    return float((1 - w) * vals[0] + w * vals[1])


def main():
    F.set_dipole_runs(DIPOLE_ALPHAS, "_wide")
    d = json.loads((JSON / "ball_zoom_unbinned_overview.json").read_text())
    A, T, TH = (np.array(d[k], float) for k in ("alpha_grid", "t0", "theta_obs"))
    C, c_l = np.array(d["chi2_raw"], float), d["lcdm_chi2"]
    qd = json.loads((JSON / "quasar_dipole_fit_ah0p34.json").read_text())
    d_geo, s_geo = qd["D_geo"], qd["sigma"]
    qz = json.loads((JSON / "quaia_zslice_model_ah0p34.json").read_text())
    q_amp = np.array([x["excess_amp"] for x in qz["slices"]])
    q_sig = np.array([x["sigma"] for x in qz["slices"]])
    chi2_lcdm_dip = (d_geo / s_geo) ** 2 + float(np.sum((q_amp / q_sig) ** 2))

    # dipole predictions on the grid
    amp = np.full_like(C, np.nan)
    cw = np.full_like(C, np.nan)
    qw = np.full_like(C, np.nan)
    for k, ah in enumerate(A):
        for i, th in enumerate(TH):
            for j, t0 in enumerate(T):
                if np.isfinite(C[k, i, j]):
                    amp[k, i, j] = catwise_amplitude(ah, t0, th)
                    cw[k, i, j] = F.catwise_chi2(ah, t0, th)
                    qw[k, i, j] = Z.quaia_chi2(ah, t0, th)
    ok = np.isfinite(C) & np.isfinite(amp) & np.isfinite(cw) & np.isfinite(qw)

    # prior cell weights: uniform alpha_high, uniform ln t0 (geometric grid), theta_obs by trapezoid widths
    wt = np.gradient(np.log(T))
    wth = np.gradient(TH)
    priors = {"theta_uniform": np.ones_like(TH), "theta_volume": np.sin(np.radians(2 * TH)) ** 2}
    i0, j0 = np.unravel_index(np.nanargmin(np.where(np.isfinite(C), C, np.inf).min(axis=0)), C.shape[1:])
    k0 = int(np.nanargmin(C[:, i0, j0]))
    best = {"alpha_high": float(A[k0]), "t0": float(T[j0]), "theta_obs": float(TH[i0]),
            "dchi2_cc_sn": float(C[k0, i0, j0] - c_l), "catwise_amplitude": float(amp[k0, i0, j0]),
            "chi2_catwise": float(cw[k0, i0, j0]), "chi2_quaia": float(qw[k0, i0, j0])}
    print("best CC + SN point:", best, flush=True)
    out = {"lcdm_chi2_dipoles": chi2_lcdm_dip, "catwise": {"D_geo": d_geo, "sigma": s_geo}, "best_cc_sn": best, "priors": {}}
    for name, pth in priors.items():
        w = np.where(ok, np.exp(-0.5 * (C - np.nanmin(C))), 0.0) * (pth * wth)[None, :, None] * wt[None, None, :]
        w /= w.sum()
        x, ww = amp[ok], w[ok]
        o = np.argsort(x)
        cdf = np.cumsum(ww[o])
        q = lambda p: float(np.interp(p, cdf, x[o]))
        p_above = float(ww[x >= d_geo].sum())
        L_cat = float(np.sum(w[ok] * np.exp(-0.5 * cw[ok])))
        L_all = float(np.sum(w[ok] * np.exp(-0.5 * (cw[ok] + qw[ok]))))
        r = {"catwise_amplitude_median": q(0.5), "catwise_amplitude_68": [q(0.16), q(0.84)],
             "catwise_amplitude_95": [q(0.025), q(0.975)], "prob_amplitude_above_measured": p_above,
             "chi2_pred_catwise": -2 * np.log(L_cat), "chi2_pred_dipoles": -2 * np.log(L_all),
             "dchi2_pred_vs_lcdm": -2 * np.log(L_all) - chi2_lcdm_dip}
        out["priors"][name] = r
        print(name, json.dumps(r), flush=True)
    # LCDM + ad hoc common amplitude, uniform on [0, A_max]: marginal of a Gaussian in A
    Wc = np.r_[1.0 / s_geo**2, 1.0 / q_sig**2]
    Xc = np.r_[d_geo, q_amp]
    A_hat = float(np.sum(Wc * Xc) / np.sum(Wc))
    chi2_min = float(np.sum(Wc * (Xc - A_hat) ** 2))
    s_A = float(1 / np.sqrt(np.sum(Wc)))
    out["lcdm_ad_hoc"] = {}
    for a_max in (0.02, 0.05, 0.1):
        L = np.exp(-0.5 * chi2_min) * np.sqrt(2 * np.pi) * s_A / a_max   # A_hat well inside [0, A_max]
        L_cat = np.sqrt(2 * np.pi) * s_geo / a_max   # CatWISE alone, amplitude fitted exactly
        out["lcdm_ad_hoc"][str(a_max)] = {"chi2_pred_dipoles": float(-2 * np.log(L)), "dchi2_pred_vs_lcdm": float(-2 * np.log(L) - chi2_lcdm_dip),
                                          "chi2_pred_catwise": float(-2 * np.log(L_cat))}
    print("LCDM + ad hoc amplitude:", out["lcdm_ad_hoc"], flush=True)

    # Quaia predicted by every model fitted to CC + SN + CatWISE (Quaia not used in the fit)
    q_none = float(np.sum((q_amp / q_sig) ** 2))
    ag = np.linspace(d_geo - 6 * s_geo, d_geo + 6 * s_geo, 4001)   # ad hoc amplitude: posterior N(D_geo, sigma)
    pa = np.exp(-0.5 * ((ag - d_geo) / s_geo) ** 2)
    pa /= pa.sum()
    chi2_q_adhoc = np.sum(((ag[:, None] - q_amp[None, :]) / q_sig[None, :]) ** 2, axis=1)
    ijk = np.unravel_index(np.nanargmin(np.where(ok, C + cw, np.inf)), C.shape)
    b2 = {"alpha_high": float(A[ijk[0]]), "theta_obs": float(TH[ijk[1]]), "t0": float(T[ijk[2]]),
          "dchi2_cc_sn": float(C[ijk] - c_l), "chi2_catwise": float(cw[ijk]), "chi2_quaia": float(qw[ijk])}
    qp = {"lcdm_or_hyperconical_no_dipole": q_none,
          "lcdm_ad_hoc_catwise_amplitude": {"best": float(np.sum(((d_geo - q_amp) / q_sig) ** 2)),
                                            "predictive": float(-2 * np.log(np.sum(pa * np.exp(-0.5 * chi2_q_adhoc))))},
          "ball_best_cc_sn_catwise": b2, "ball_predictive": {}}
    for name, pth in priors.items():
        w = np.where(ok, np.exp(-0.5 * (C + cw - np.nanmin(np.where(ok, C + cw, np.inf)))), 0.0) \
            * (pth * wth)[None, :, None] * wt[None, None, :]
        w /= w.sum()
        qp["ball_predictive"][name] = float(-2 * np.log(np.sum(w[ok] * np.exp(-0.5 * qw[ok]))))
    out["quaia_given_cc_sn_catwise"] = qp
    print("Quaia predicted from CC + SN + CatWISE:", json.dumps(qp), flush=True)
    (JSON / "predict_dipole_from_ccsn.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
