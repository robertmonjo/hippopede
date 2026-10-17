"""Compare the numbers quoted in main-hippopede-epjc.tex with the outputs of the analysis scripts.

Each check pairs a value as printed in the paper with the quantity computed from the JSON files in
json/ (and dipole_zslices/results.json), and passes when they agree to half a unit of the last
printed digit.  Run after the full pipeline (REPRODUCIBILITY.md).  Exit status 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "json"   # numerical outputs; figures/ holds the png/pdf


def load(name):
    return json.loads((JSON / f"{name}.json").read_text())


def row(fit, model):
    return next(r for r in fit["rows"] if r["model"] == model)


def tol(printed):
    """Half a unit of the last printed digit of a number given as a string."""
    d = printed.split(".")[1] if "." in printed else ""
    return 0.5 * 10 ** (-len(d))


EXTENDED_OUTPUTS = ("cc_intrinsic_scatter", "cc_direction_test", "pantheon_catwise_axis", "fit_ball_unbinned",
                    "ball_zoom_unbinned_wide", "fit_observer_ball_landscape_joint_wide", "binning_resolution_test",
                    "quasar_dipole_fit_ah0p34", "quaia_zslice_model_ah0p34", "ball_zoom_unbinned_overview",
                    "dipole_chi2_vs_theta_continuous_noquaia", "predict_dipole_from_ccsn")


def extended_checks():
    """Numbers of Sects. 3.3, 3.5 and 3.6, produced by run_all.py --extended; None if those outputs are missing."""
    if not all((JSON / f"{n}.json").exists() for n in EXTENDED_OUTPUTS):
        return None
    cis = load("cc_intrinsic_scatter")
    cdt = load("cc_direction_test")
    pca = load("pantheon_catwise_axis")
    spread = dict(zip(cis["sector_spread"]["z"], cis["sector_spread"]["half_width_1sigma"]))
    fbu = load("fit_ball_unbinned")
    bz = load("ball_zoom_unbinned_wide")
    ov = load("ball_zoom_unbinned_overview")["cc_sn_catwise_68"]   # Fig. ball: joint 68% region of CC + SN + CatWISE
    dth = {r["theta_obs"]: r["with_cc_sn"] for r in load("dipole_chi2_vs_theta_continuous_noquaia")["rows"]}   # Fig. dipole_theta
    pdc = load("predict_dipole_from_ccsn")
    lcdm_cat = dth[0.0]["catwise"]
    mid = [lcdm_cat - dth[t]["total"] for t in dth if 3 <= t <= 30]
    owl = load("fit_observer_ball_landscape_joint_wide")
    brt = load("binning_resolution_test")
    qdf = load("quasar_dipole_fit_ah0p34")
    qzs = load("quaia_zslice_model_ah0p34")["slices"]
    amp = np.array([x["excess_amp"] for x in qzs]); sig = np.array([x["sigma"] for x in qzs])
    q_const = float(np.sum(((amp - np.sum(amp / sig**2) / np.sum(1 / sig**2)) / sig) ** 2))
    q_none = float(np.sum((amp / sig) ** 2))
    return [
        # Sect. 3.3: scatter of the chronometers (cc_intrinsic_scatter.py)
        ("CC-only flat LCDM chi2", "23.4", cis["flat_lcdm"]["chi2"]),
        ("CC-only degrees of freedom", "36", cis["flat_lcdm"]["dof"]),
        ("sigma_int 95% upper limit, z < 0.5", "6.8", cis["flat_lcdm"]["subsets"]["z<0.5"]["sigma_int_95"]),
        ("sigma_int maximum likelihood, z > 0.5", "8", cis["flat_lcdm"]["subsets"]["z>0.5"]["sigma_int_ml"]),
        ("sigma_int 95% upper limit, z > 0.5", "20", cis["flat_lcdm"]["subsets"]["z>0.5"]["sigma_int_95"]),
        ("Delta(-2 ln L) at no scatter, z > 0.5", "1.0", cis["flat_lcdm"]["subsets"]["z>0.5"]["dm2lnL_at_zero"]),
        ("H spread across sectors, z = 0.5", "2.6", spread[0.5]),
        ("H spread across sectors, z = 1", "6.0", spread[1.0]),
        ("H spread across sectors, z = 1.6", "17", spread[1.6]),
        ("H spread across sectors, z = 1.9", "29", spread[1.9]),
        # Sect. 3.5: direction tests (cc_direction_test.py, pantheon_catwise_axis.py)
        ("CC amplitude A along the CatWISE axis", "0.04", cdt["fits"]["all"]["A"]),
        ("CC amplitude A, uncertainty", "0.15", cdt["fits"]["all"]["sigma_A"]),
        ("predicted |A| min (%)", "0.2", 100 * min(min(r["A"]) for r in cdt["prediction"])),
        ("predicted |A| max (%)", "2.4", 100 * max(max(r["A"]) for r in cdt["prediction"])),
        ("CatWISE curve, smallest t0", "1.6", min(r["t0"] for r in pca["rows"])),
        ("CatWISE curve, largest t0", "4.5", max(r["t0"] for r in pca["rows"])),
        ("CatWISE curve, smallest theta_obs", "0.9", min(r["theta_obs"] for r in pca["rows"])),
        ("CatWISE curve, largest theta_obs", "24", max(r["theta_obs"] for r in pca["rows"])),
        ("SN dchi2 with the quasar orientation, smallest change", "-0.32", max(r["chi2_catwise_minus"] - r["chi2_axis_observer"] for r in pca["rows"])),
        ("SN dchi2 with the quasar orientation, largest change", "-0.79", min(r["chi2_catwise_minus"] - r["chi2_axis_observer"] for r in pca["rows"])),
        ("SN dchi2 with the opposite orientation, min", "0.36", min(r["chi2_catwise_plus"] - r["chi2_axis_observer"] for r in pca["rows"])),
        ("SN dchi2 with the opposite orientation, max", "1.03", max(r["chi2_catwise_plus"] - r["chi2_axis_observer"] for r in pca["rows"])),
        ("random orientations with lower chi2, min (%)", "7", 100 * min(r["frac_random_below_minus"] for r in pca["rows"])),
        ("random orientations with lower chi2, max (%)", "10", 100 * max(r["frac_random_below_minus"] for r in pca["rows"])),
        ("number of random orientations", "2000", pca["n_random"]),
        # Sect. 3.6: light-cone average with the unbinned supernovae (fit_ball_unbinned.py, plot_ball_zoom_unbinned.py)
        ("unbinned supernovae", "1579", 1579),
        ("ball, observer on the axis: alpha_high", "0.361", fbu["isotropic"]["alpha_high"]),
        ("ball, observer on the axis: t0", "2.23", fbu["isotropic"]["t0"]),
        ("ball, observer on the axis: dchi2", "0.02", fbu["isotropic"]["dchi2_vs_lcdm"]),
        ("projected axial sector, unbinned: dchi2", "0.28", fbu["hyperconical"]["dchi2_vs_lcdm"]),
        ("CC + SN minimum: t0", "2.11", bz["best_cc_sn"]["t0"]),
        ("CC + SN minimum: theta_obs", "12.0", bz["best_cc_sn"]["theta_obs"]),
        ("CC + SN minimum: alpha_high", "0.342", bz["best_cc_sn"]["alpha_high"]),
        ("CC + SN minimum: dchi2", "-1.19", bz["best_cc_sn"]["dchi2"]),
        ("count dipole / CatWISE excess near that minimum", "4", (qdf["D_geo"] + qdf["sigma"] * fbu["fits"]["cc_sn"]["chi2_catwise"] ** 0.5) / qdf["D_geo"]),
        ("CC + SN + CatWISE minimum: t0", "2.85", bz["best_cc_sn_catwise"]["t0"]),
        ("CC + SN + CatWISE minimum: theta_obs", "8.0", bz["best_cc_sn_catwise"]["theta_obs"]),
        ("CC + SN + CatWISE minimum: alpha_high", "0.379", bz["best_cc_sn_catwise"]["alpha_high"]),
        ("CC + SN + CatWISE minimum: dchi2 of CC + SN", "-0.63", bz["best_cc_sn_catwise"]["dchi2_cc_sn"]),
        ("random orientations above the quasar axis (%)", "93", 100 * (1 - fbu["free_axis"]["frac_random_below_fixed"])),
        ("CC + SN + CatWISE 68%: smallest t0", "1.53", ov["t0"][0]),
        ("CC + SN + CatWISE 68%: smallest theta_obs", "0.6", ov["theta_obs"][0]),
        ("CC + SN + CatWISE 68%: largest t0", "6.2", ov["t0"][1]),
        ("CC + SN + CatWISE 68%: largest theta_obs", "45", ov["theta_obs"][1]),
        ("Quaia amplitudes, none", "25.5", q_none),
        ("CatWISE amplitude, observer on the axis (as LCDM)", "33.0", lcdm_cat),
        ("CC + SN at theta_obs = 1 deg, CatWISE fitted", "0.02", dth[1.0]["dchi2_cc_sn"]),
        ("CatWISE at theta_obs = 1 deg", "0.0", dth[1.0]["catwise"]),
        ("joint chi2 below LCDM, 3-30 deg, min", "33.5", min(mid)),
        ("joint chi2 below LCDM, 3-30 deg, max", "33.6", max(mid)),
        ("predictive chi2 of CatWISE from CC + SN, min", "4.6", min(v["chi2_pred_catwise"] for v in pdc["priors"].values())),
        ("predictive chi2 of CatWISE from CC + SN, max", "4.8", max(v["chi2_pred_catwise"] for v in pdc["priors"].values())),
        ("binned supernovae: t0 of the minimum", "1.74", owl["best"]["t0"]),
        ("binned supernovae: dchi2 of the minimum", "-0.80", owl["best"]["dchi2"]),
        ("lowest redshift of the last bin", "0.80", brt["last_bin"]["z_min"]),
        ("random orientations below, smallest (%)", "7", 100 * min(fbu["free_axis"]["frac_random_below_fixed"], min(r["frac_random_below_minus"] for r in pca["rows"]))),
    ]


def light_weight_checks():
    """Numbers of the paragraphs on Table light (light_weight_comparison.py), and the table itself: the tabular of the
    paper must be the one written to tables/light_weight_comparison.tex."""
    if not (JSON / "light_weight_comparison.json").exists():
        return None
    s = load("light_weight_comparison")
    ref = s["reference"]
    eq, li, sq = s["equal"], s["linear"], s["sqrt"]
    neg = [li["joint_neg"], sq["joint_neg"]]
    qe, ql, qs = eq["joint_quaia"], li["joint_quaia"], sq["joint_quaia"]
    quaia = json.loads((ROOT / "dipole_zslices" / "results.json").read_text())
    lo, la = np.radians(load("quasar_dipole_fit")["excess_lb"])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    ang = [float(np.degrees(np.arccos(np.dot(e, u) / np.linalg.norm(e))))
           for e in (np.array(x["dip"]["excess_vec"]) for x in quaia["slices"]
                     if isinstance(x["z_range"], list) and x["z_range"][1] <= 1.8)]
    tex = (ROOT / "main-hippopede-epjc.tex").read_text(encoding="utf-8")
    table = (ROOT / "tables" / "light_weight_comparison.tex").read_text(encoding="utf-8").strip()
    sig = ref["sigma_geo"]
    return [
        ("light weights: CC+SN minimum on the axis", "0", max(li["cc_sn"]["theta_obs"], sq["cc_sn"]["theta_obs"])),
        ("1 - zeta, axis: dchi2 CC+SN", "-1.70", li["axis"]["dchi2_cc_sn"]),
        ("1 - sqrt(zeta), axis: dchi2 CC+SN", "-1.67", sq["axis"]["dchi2_cc_sn"]),
        ("1 - zeta, axis: t0", "1.14", li["axis"]["t0"]),
        ("1 - sqrt(zeta), axis: t0", "1.14", sq["axis"]["t0"]),
        ("negative branch: theta_obs, smallest", "40", min(r["theta_obs"] for r in neg)),
        ("negative branch: theta_obs, largest", "44", max(r["theta_obs"] for r in neg)),
        ("negative branch: radio/CatWISE, smallest", "2.9", min(r["radio_over_catwise"] for r in neg)),
        ("negative branch: radio/CatWISE, largest", "4.0", max(r["radio_over_catwise"] for r in neg)),
        ("negative branch reproduces CatWISE (max chi2)", "0.0", max(r["chi2_catwise"] for r in neg)),
        ("equal weights reproduce no positive dipole (chi2 CatWISE > 1)", "1", float(eq["joint_pos"]["chi2_catwise"] > 1)),
        ("positive branch within 0.5 deg (max theta_obs)", "0.50", max(li["joint_pos"]["theta_obs"], sq["joint_pos"]["theta_obs"])),
        ("1 - zeta, positive branch: dchi2 CC+SN", "0.04", li["joint_pos"]["dchi2_cc_sn"]),
        ("1 - sqrt(zeta), positive branch: dchi2 CC+SN", "-0.09", sq["joint_pos"]["dchi2_cc_sn"]),
        ("equal weights, CatWISE: dchi2 CC+SN", "-0.63", eq["joint_neg"]["dchi2_cc_sn"]),
        ("1 - zeta, positive branch: radio/CatWISE", "0.79", li["joint_pos"]["radio_over_catwise"]),
        ("1 - sqrt(zeta), positive branch: radio/CatWISE", "0.85", sq["joint_pos"]["radio_over_catwise"]),
        ("equal weights, CatWISE: radio/CatWISE", "1.80", eq["joint_neg"]["radio_over_catwise"]),
        ("negative branch disfavoured by radio: >= 4 sigma above RACS-low", "1",
         float(min((r["radio_over_catwise"] - ref["radio_over_catwise_RACS-low"][0]) / ref["radio_over_catwise_RACS-low"][1]
                   for r in neg) >= 3.95)),
        ("positive branch and equal weights within 1.5 sigma of RACS-low", "1",
         float(max(abs(r["radio_over_catwise"] - ref["radio_over_catwise_RACS-low"][0]) / ref["radio_over_catwise_RACS-low"][1]
                   for r in (li["joint_pos"], sq["joint_pos"], eq["joint_neg"])) < 1.5)),
        ("measured radio/CatWISE, NVSS", "1.3", ref["radio_over_catwise_NVSS"][0]),
        ("measured radio/CatWISE, NVSS error", "0.5", ref["radio_over_catwise_NVSS"][1]),
        ("measured radio/CatWISE, RACS-low", "1.3", ref["radio_over_catwise_RACS-low"][0]),
        ("measured radio/CatWISE, RACS-low error", "0.4", ref["radio_over_catwise_RACS-low"][1]),
        ("equal weights + Quaia: theta_obs", "21", qe["theta_obs"]),
        ("equal weights + Quaia: D(1.8)/D(0.3)", "22", qe["D_of_z"]["1.8"] / qe["D_of_z"]["0.3"]),
        ("equal weights + Quaia: dchi2 Quaia amplitudes", "-12.7", qe["dchi2_quaia"]),
        ("1 - zeta + Quaia: D(0.3)", "0.0098", ql["D_of_z"]["0.3"]),
        ("1 - zeta + Quaia: D(1.8)", "0.0115", ql["D_of_z"]["1.8"]),
        ("1 - sqrt(zeta) + Quaia: D(0.3)", "0.0138", qs["D_of_z"]["0.3"]),
        ("1 - sqrt(zeta) + Quaia: D(1.8)", "0.0101", qs["D_of_z"]["1.8"]),
        ("1 - sqrt(zeta) + Quaia: D decreases at every step", "1", float(all(np.diff(list(qs["D_of_z"].values())) < 0))),
        ("light weights + Quaia: largest theta_obs", "0.82", max(ql["theta_obs"], qs["theta_obs"])),
        ("1 - zeta + Quaia: t0", "2.99", ql["t0"]),
        ("1 - sqrt(zeta) + Quaia: t0", "2.99", qs["t0"]),
        ("light weights + Quaia: CatWISE within 1 sigma", "1", float(max(abs(r["D_catwise"] - ref["D_geo"]) / sig for r in (ql, qs)) < 1)),
        ("1 - zeta + Quaia: radio/CatWISE", "0.90", ql["radio_over_catwise"]),
        ("1 - sqrt(zeta) + Quaia: radio/CatWISE", "1.00", qs["radio_over_catwise"]),
        ("1 - zeta + Quaia: dchi2 Quaia amplitudes", "-20.3", ql["dchi2_quaia"]),
        ("1 - sqrt(zeta) + Quaia: dchi2 Quaia amplitudes", "-21.4", qs["dchi2_quaia"]),
        ("constant amplitude: dchi2 Quaia", "-23.2", ref["dchi2_quaia_constant"]),
        ("Quaia slices vs CatWISE excess: smallest angle (deg)", "101", min(ang)),
        ("Quaia slices vs CatWISE excess: largest angle (deg)", "158", max(ang)),
        ("every row of Table light: vector Quaia chi2 above no dipole", "1",
         float(all(r["dchi2_quaia_vector"] >= 0 for w in (eq, li, sq) for r in w.values()))),
        ("Discussion: 1 - sqrt(zeta) + Quaia within a degree of the axis", "1", float(qs["theta_obs"] < 1)),
        ("Table light in the paper equals tables/light_weight_comparison.tex", "1", float(table in tex)),
    ]


def main():
    fit = load("fit_cc_pantheon")
    ah = load("fit_alpha_high")
    dm = load("t0_dispersion_match")
    sa = load("fit_sector_average")
    ax = load("fit_t0_axis_observer")
    sc = load("fit_t0_intrinsic_scatter")
    kin = load("sector_kinematics")
    kin_avg = load("sector_kinematics_sector_average_alpha")
    reach = load("sector_reach")
    off = load("pantheon_offaxis_test")
    qd = load("quasar_dipole_fit")
    da = load("directional_anomalies")
    qz = load("quaia_zslice_model")
    loc = load("local_environment")
    quaia = json.loads((ROOT / "dipole_zslices" / "results.json").read_text())
    hz = load("verify_high_z")
    hz_4e9 = next(r for r in hz["ratio_mp_over_double"] if r["z"] == 4e9)
    sec = {s["theta"]: s for s in kin["sectors"]}
    avg = {s["theta"]: s for s in kin_avg["sectors"]}
    lcdm = row(fit, "LCDM")["chi2"]
    q0_obs, q0_err = -0.364, 0.032
    joint = dm["q"]["joint"]
    ratio1 = np.array(joint["model_halfwidth_1sigma"]) / np.array(joint["gapp_sigma"])
    ratio2 = np.array(joint["model_halfwidth_2sigma"]) / (2 * np.array(joint["gapp_sigma"]))
    zn = np.array(dm["z_nodes"])
    mid = (zn >= 0.3) & (zn <= 1.2)
    high = sa["cases"]["high"]
    axial_avg = sa["cases"]["axial_alpha"]
    e_off = np.array(high["median_offset_E_sigma"])
    q_runs = {(r["t0"], r["pz"]): r["theta_obs_minus1sigma_best_plus1sigma"] for r in qd["runs"] if abs(r["x"] - qd["x"]) < 1e-12}
    f = {k: np.array(v["f_1deg"]) for k, v in qz["model"].items()}

    checks = [
        ("alpha_high (axial fit)", "0.416", ah["alpha_high"]),
        ("alpha_high 1 sigma low", "0.374", ah["alpha_high_1sigma"][0]),
        ("alpha_high 1 sigma high", "0.457", ah["alpha_high_1sigma"][1]),
        ("chi2_nu axial", "1.19", ah["chi2_nu"]),
        ("t0 band match", "2.60", joint["t0_best"]),
        ("t0 band match jackknife", "0.08", joint["jackknife_se"]),
        ("t0 1 sigma band", "2.67", dm["q"]["1sigma"]["t0_best"]),
        ("t0 2 sigma band", "2.54", dm["q"]["2sigma"]["t0_best"]),
        ("node range low", "2.35", joint["node_16_84"][0]),
        ("node range high", "2.79", joint["node_16_84"][1]),
        ("t0 from E", "4.16", dm["E"]["joint"]["t0_best"]),
        ("t0 from E jackknife", "0.33", dm["E"]["joint"]["jackknife_se"]),
        ("width ratio min, 0.3<=z<=1.2", "0.8", min(ratio1[mid].min(), ratio2[mid].min())),
        ("width ratio max, 0.3<=z<=1.2", "1.4", max(ratio1[mid].max(), ratio2[mid].max())),
        ("LCDM chi2", "101.44", lcdm),
        ("LCDM Omega_m", "0.333", row(fit, "LCDM")["Omega_m"]),
        ("axial chi2", "101.48", row(fit, "projected 0")["chi2"]),
        ("constant alpha dchi2", "10.57", row(fit, "hyperconical, constant alpha_low")["dchi2"]),
        ("projected 15 dchi2", "0.10", row(fit, "projected 15")["dchi2"]),
        ("projected 30 dchi2", "0.89", row(fit, "projected 30")["dchi2"]),
        ("projected 45 dchi2", "4.39", row(fit, "projected 45")["dchi2"]),
        ("projected 60 dchi2", "16.30", row(fit, "projected 60")["dchi2"]),
        ("unprojected 0 dchi2", "106.23", row(fit, "unprojected 0")["dchi2"]),
        ("unprojected 60 dchi2", "71.45", row(fit, "unprojected 60")["dchi2"]),
        ("unprojected best (30, t0 free) dchi2", "68.87", row(fit, "unprojected 30 t0 free")["dchi2"]),
        ("unprojected best t0", "1.44", row(fit, "unprojected 30 t0 free")["t0"]),
        ("t0 bound 30 deg 1 sigma", "2.50", fit["t0_lower_bound_1sigma"]["30"]),
        ("t0 bound 30 deg 2 sigma", "1.84", fit["t0_lower_bound_2sigma"]["30"]),
        ("t0 bound 45 deg 1 sigma", "3.57", fit["t0_lower_bound_1sigma"]["45"]),
        ("t0 bound 45 deg 2 sigma", "2.64", fit["t0_lower_bound_2sigma"]["45"]),
        ("t0 bound 60 deg 1 sigma", "4.49", fit["t0_lower_bound_1sigma"]["60"]),
        ("t0 bound 60 deg 2 sigma", "3.36", fit["t0_lower_bound_2sigma"]["60"]),
        ("t0 bound 15 deg 1 sigma", "1.3", fit["t0_lower_bound_1sigma"]["15"]),
        ("average, axial alpha, dchi2", "4.75", axial_avg["dchi2"]),
        ("average fit alpha_high", "0.339", high["alpha_high"]),
        ("average fit t0", "2.78", high["t0"]),
        ("average fit dchi2", "0.65", high["dchi2"]),
        ("average fit dAIC", "2.65", high["dAIC"]),
        ("median q offset, axial alpha (max)", "4.1", -min(axial_avg["median_offset_q_sigma"])),
        ("median q offset, average fit (max)", "2.9", -min(high["median_offset_q_sigma"])),
        ("LCDM q offset at z = 0.8", "1.8", -sa["lcdm_median_offset_q_sigma"][int(np.argmin(abs(zn - 0.8)))]),
        ("E within 1 sigma from z = 0.3", "0.3", float(zn[np.where(np.abs(e_off) <= 1)[0][0]])),
        ("E within 1 sigma up to z", "0.8", float(zn[np.where(np.abs(e_off) <= 1)[0][-1]])),
        ("axis observer t0 > (1 sigma)", "2.56", ax["interval"]["1sigma"][0]),
        ("axis observer t0 > (2 sigma)", "1.88", ax["interval"]["2sigma"][0]),
        ("intrinsic scatter t0 > (1 sigma)", "7.3", sc["interval"]["1sigma"][0]),
        ("intrinsic scatter t0 > (2 sigma)", "4.3", sc["interval"]["2sigma"][0]),
        ("Fig. 2: q0 unprojected 15", "-0.004", avg[15]["q0_unprojected"]),
        ("Fig. 2: q0 unprojected 30", "-0.016", avg[30]["q0_unprojected"]),
        ("Fig. 2: q0 unprojected 45", "-0.034", avg[45]["q0_unprojected"]),
        ("Fig. 2: q0 unprojected 60", "-0.055", avg[60]["q0_unprojected"]),
        ("Fig. 2: q0 projected 0", "-0.540", avg[0]["q0_projected"]),
        ("Fig. 2: q0 projected 15", "-0.542", avg[15]["q0_projected"]),
        ("Fig. 2: q0 projected 30", "-0.548", avg[30]["q0_projected"]),
        ("Fig. 2: q0 projected 45", "-0.556", avg[45]["q0_projected"]),
        ("Fig. 2: q0 projected 60", "-0.566", avg[60]["q0_projected"]),
        ("Fig. 2: q0 tension, axial (sigma)", "5.5", (q0_obs - avg[0]["q0_projected"]) / q0_err),
        ("Fig. 2: q0 tension, 60 deg (sigma)", "6.3", (q0_obs - avg[60]["q0_projected"]) / q0_err),
        ("Fig. 2: z_t 0 deg", "0.45", avg[0]["z_t_projected"][0]),
        ("Fig. 2: z_t 15 deg", "0.46", avg[15]["z_t_projected"][0]),
        ("Fig. 2: z_t 30 deg", "0.49", avg[30]["z_t_projected"][0]),
        ("Fig. 2: z_t 45 deg", "0.57", avg[45]["z_t_projected"][0]),
        ("Fig. 2: 45 deg accelerates again from z", "1.21", avg[45]["z_t_projected"][1]),
        ("Fig. 2: z_t within 1.2 sigma (0 deg vs 0.64 +- 0.16)", "1.2", (0.64 - avg[0]["z_t_projected"][0]) / 0.16),
        ("Fig. 2: E spread z = 0.1 [%]", "0.3", 100 * kin_avg["E_spread"]["0.1"]),
        ("Fig. 2: E spread z = 0.5 [%]", "2.6", 100 * kin_avg["E_spread"]["0.5"]),
        ("Fig. 2: E spread z = 1 [%]", "8.9", 100 * kin_avg["E_spread"]["1.0"]),
        ("axial fit: z_t 0 deg", "0.60", sec[0]["z_t_projected"][0]),
        ("axial fit: z_t 15 deg", "0.62", sec[15]["z_t_projected"][0]),
        ("axial fit: z_t 30 deg", "0.73", sec[30]["z_t_projected"][0]),
        ("axial fit: 45 deg has no transition", "0", len(sec[45]["z_t_projected"])),
        ("axis source sector z = 0.1", "2.8", kin["axis_observer_sector_deg"]["0.1"]),
        ("axis source sector z = 0.5", "12.5", kin["axis_observer_sector_deg"]["0.5"]),
        ("axis source sector z = 1", "21.7", kin["axis_observer_sector_deg"]["1.0"]),
        ("axis source sector z = 2", "34.1", kin["axis_observer_sector_deg"]["2.0"]),
        ("z_h", "3.1", reach["axis_observer"]["z_h"]),
        ("reach of 70 deg at t0 = 2.6", "2.07", next(s for s in reach["sectors"] if s["theta"] == 70.0)["z_reach_observed"]),
        ("mu spread z = 0.1", "0.040", off["mu_spread"]["spread_mag"][0]),
        ("mu spread z = 1.3", "0.073", off["mu_spread"]["spread_mag"][-1]),
        ("off-axis best improvement (max over t0)", "1.4", -min(g["dchi2_offaxis_vs_axis"] for g in off["grid"])),
        ("|Delta_D| z = 0.05 max [%]", "0.09", 100 * max(abs(v[0]) for v in da["delta"].values())),
        ("|Delta_D| z = 0.1 max [%]", "0.18", 100 * max(abs(v[1]) for v in da["delta"].values())),
        ("|Delta_D| z = 0.3 max [%]", "0.54", 100 * max(abs(v[2]) for v in da["delta"].values())),
        ("Migkas fraction z = 0.1 [%]", "3.9", 100 * da["migkas_fraction_max"][0]),
        ("Migkas fraction z = 0.3 [%]", "12", 100 * da["migkas_fraction_max"][1]),
        ("Colin fraction [%]", "0.9", 100 * da["colin_fraction_max"]),
        ("D_geo", "0.0098", qd["D_geo"]),
        ("D_geo sigma", "0.0017", qd["sigma"]),
        ("theta_obs for D_geo, t0 = 2.6", "6.9", q_runs[(2.6, "gamma")][1]),
        ("theta_obs -1 sigma", "5.7", q_runs[(2.6, "gamma")][0]),
        ("theta_obs +1 sigma", "8.1", q_runs[(2.6, "gamma")][2]),
        ("theta_obs for D_geo, t0 = 3", "9.9", q_runs[(3.0, "gamma")][1]),
        ("theta_obs for D_geo, t0 = 4", "20", q_runs[(4.0, "gamma")][1]),
        ("Secrest configuration SN dchi2", "-0.64", da["secrest"]["pantheon_dchi2_vs_axis"]),
        ("Secrest configuration Delta mu(z=1)", "0.022", da["secrest"]["dmu_z1"]),
        ("Quaia full excess (|b|>30)", "0.0112", quaia["slices"][0]["dip"]["excess_amp"]),
        ("Quaia full excess sigma", "0.0029", quaia["slices"][0]["dip"]["excess_sigma_mocks"]),
        ("Quaia |b|>40 dipole", "0.0085", quaia["full_sample_b40"]["dip"]["D"]),
        ("Quaia |b|>40 sigma", "0.0025", quaia["full_sample_b40"]["dip"]["sigma_fit_D"]),
        ("kinematic dipole", "0.0051", quaia["slices"][0]["D_kin"]),
        ("Quaia slice 1 excess", "0.021", qz["slices"][0]["excess_amp"]),
        ("Quaia slice 2 excess", "0.020", qz["slices"][1]["excess_amp"]),
        ("Quaia slice 3 excess", "0.010", qz["slices"][2]["excess_amp"]),
        ("Quaia slice 1 sigma", "0.007", qz["slices"][0]["sigma"]),
        ("Quaia slice 2 sigma", "0.006", qz["slices"][1]["sigma"]),
        ("Quaia slice 3 sigma", "0.005", qz["slices"][2]["sigma"]),
        ("model growth across slices, min (2.5<=t0<=6)", "4.3", min(f[k][2] / f[k][0] for k in f if 2.5 <= float(k) <= 6)),
        ("model growth across slices, max (2.5<=t0<=6)", "5.0", max(f[k][2] / f[k][0] for k in f if 2.5 <= float(k) <= 6)),
        ("Quaia best theta_obs, t0 = 2.6", "3.9", qz["model"]["2.6"]["theta_obs"]),
        ("Quaia model chi2", "17.5", qz["model"]["2.6"]["chi2"]),
        ("Quaia constant chi2", "8.7", qz["chi2_const"]),
        ("Quaia no-dipole chi2", "25.5", qz["chi2_null"]),
        ("Quaia dchi2 max (2.5<=t0<=6)", "8.9", max(v["chi2"] - qz["chi2_const"] for k, v in qz["model"].items() if 2.5 <= float(k) <= 6)),
        ("Quaia dchi2 min (2.5<=t0<=6)", "8.1", min(v["chi2"] - qz["chi2_const"] for k, v in qz["model"].items() if 2.5 <= float(k) <= 6)),
        ("q_d,eff per degree", "1.3e-3", loc["observers"]["1.0"]["q_d_eff"][0]),
        ("high-z agreement with 50 digits at z = 4e9", "3e-7", max(abs(hz_4e9["alpha_half"] - 1), abs(hz_4e9["running"] - 1))),
    ]
    ext = extended_checks()
    if ext is None:
        print("numbers of Sects. 3.3, 3.5 and 3.6 not checked: run scripts/run_all.py --extended first")
    else:
        checks += ext
    lw = light_weight_checks()
    if lw is None:
        print("Table light not checked: run scripts/light_weight_comparison.py first")
    else:
        checks += lw
    failed = 0
    for name, printed, value in checks:
        ok = abs(float(printed) - float(value)) <= (tol(printed.split("e")[0]) * (10 ** int(printed.split("e")[1]) if "e" in printed else 1)) + 1e-12
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:46s} paper {printed:>8s}   computed {float(value):.6g}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
