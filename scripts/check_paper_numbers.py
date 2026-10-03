"""Compare the numbers quoted in main-hippopede-epjc.tex with the outputs of the analysis scripts.

Each check pairs a value as printed in the paper with the quantity computed from the JSON files in
figures/ (and dipole_zslices/results.json), and passes when they agree to half a unit of the last
printed digit.  Run after the full pipeline (REPRODUCIBILITY.md).  Exit status 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def load(name):
    return json.loads((FIG / f"{name}.json").read_text())


def row(fit, model):
    return next(r for r in fit["rows"] if r["model"] == model)


def tol(printed):
    """Half a unit of the last printed digit of a number given as a string."""
    d = printed.split(".")[1] if "." in printed else ""
    return 0.5 * 10 ** (-len(d))


def main():
    fit = load("fit_cc_pantheon")
    ah = load("fit_alpha_high")
    dm = load("t0_dispersion_match")
    sa = load("fit_sector_average")
    ax = load("fit_t0_axis_observer")
    sc = load("fit_t0_intrinsic_scatter")
    kin = load("sector_kinematics")
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
        ("q0 unprojected 15", "-0.005", sec[15]["q0_unprojected"]),
        ("q0 unprojected 30", "-0.019", sec[30]["q0_unprojected"]),
        ("q0 unprojected 45", "-0.039", sec[45]["q0_unprojected"]),
        ("q0 unprojected 60", "-0.064", sec[60]["q0_unprojected"]),
        ("q0 projected 0", "-0.540", sec[0]["q0_projected"]),
        ("q0 projected 15", "-0.543", sec[15]["q0_projected"]),
        ("q0 projected 30", "-0.549", sec[30]["q0_projected"]),
        ("q0 projected 45", "-0.558", sec[45]["q0_projected"]),
        ("q0 projected 60", "-0.570", sec[60]["q0_projected"]),
        ("q0 tension, axial (sigma)", "5.5", (q0_obs - sec[0]["q0_projected"]) / q0_err),
        ("q0 tension, 60 deg (sigma)", "6.4", (q0_obs - sec[60]["q0_projected"]) / q0_err),
        ("z_t 0 deg", "0.60", sec[0]["z_t_projected"][0]),
        ("z_t 15 deg", "0.62", sec[15]["z_t_projected"][0]),
        ("z_t 30 deg", "0.73", sec[30]["z_t_projected"][0]),
        ("30 deg accelerates again from z", "1.70", sec[30]["z_t_projected"][1]),
        ("E spread z = 0.1 [%]", "0.3", 100 * kin["E_spread"]["0.1"]),
        ("E spread z = 0.5 [%]", "2.8", 100 * kin["E_spread"]["0.5"]),
        ("E spread z = 1 [%]", "9.0", 100 * kin["E_spread"]["1.0"]),
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
        ("D_geo", "0.0099", qd["D_geo"]),
        ("D_geo sigma", "0.0017", qd["sigma"]),
        ("theta_obs for D_geo, t0 = 2.6", "6.5", q_runs[(2.6, "gamma")][1]),
        ("theta_obs -1 sigma", "5.3", q_runs[(2.6, "gamma")][0]),
        ("theta_obs +1 sigma", "7.6", q_runs[(2.6, "gamma")][2]),
        ("theta_obs for D_geo, t0 = 3", "9.2", q_runs[(3.0, "gamma")][1]),
        ("theta_obs for D_geo, t0 = 4", "18", q_runs[(4.0, "gamma")][1]),
        ("Secrest configuration SN dchi2", "-0.62", da["secrest"]["pantheon_dchi2_vs_axis"]),
        ("Secrest configuration Delta mu(z=1)", "0.021", da["secrest"]["dmu_z1"]),
        ("Quaia full excess (|b|>30)", "0.0107", quaia["slices"][0]["dip"]["excess_amp"]),
        ("Quaia full excess sigma", "0.0029", quaia["slices"][0]["dip"]["excess_sigma_mocks"]),
        ("Quaia |b|>40 dipole", "0.0085", quaia["full_sample_b40"]["dip"]["D"]),
        ("Quaia |b|>40 sigma", "0.0025", quaia["full_sample_b40"]["dip"]["sigma_fit_D"]),
        ("kinematic dipole", "0.0061", quaia["slices"][0]["D_kin"]),
        ("Quaia slice 1 excess", "0.021", qz["slices"][0]["excess_amp"]),
        ("Quaia slice 2 excess", "0.019", qz["slices"][1]["excess_amp"]),
        ("Quaia slice 3 excess", "0.011", qz["slices"][2]["excess_amp"]),
        ("Quaia slice 1 sigma", "0.007", qz["slices"][0]["sigma"]),
        ("Quaia slice 2 sigma", "0.006", qz["slices"][1]["sigma"]),
        ("Quaia slice 3 sigma", "0.005", qz["slices"][2]["sigma"]),
        ("model growth across slices, min (2.5<=t0<=6)", "4.3", min(f[k][2] / f[k][0] for k in f if 2.5 <= float(k) <= 6)),
        ("model growth across slices, max (2.5<=t0<=6)", "5.0", max(f[k][2] / f[k][0] for k in f if 2.5 <= float(k) <= 6)),
        ("Quaia best theta_obs, t0 = 2.6", "4.0", qz["model"]["2.6"]["theta_obs"]),
        ("Quaia model chi2", "16.9", qz["model"]["2.6"]["chi2"]),
        ("Quaia constant chi2", "8.6", qz["chi2_const"]),
        ("Quaia no-dipole chi2", "25.2", qz["chi2_null"]),
        ("Quaia dchi2 max (2.5<=t0<=6)", "8.4", max(v["chi2"] - qz["chi2_const"] for k, v in qz["model"].items() if 2.5 <= float(k) <= 6)),
        ("Quaia dchi2 min (2.5<=t0<=6)", "7.7", min(v["chi2"] - qz["chi2_const"] for k, v in qz["model"].items() if 2.5 <= float(k) <= 6)),
        ("axis crossing, theta_obs = 1 deg", "0.035", loc["observers"]["1.0"]["z_crossing"]),
        ("theta_obs for a crossing at z_S = 0.026", "0.74", loc["theta_obs_crossing_zS"]),
        ("q_d,eff per degree", "1.3e-3", loc["observers"]["1.0"]["q_d_eff"][0]),
        ("high-z agreement with 50 digits at z = 4e9", "3e-7", max(abs(hz_4e9["alpha_half"] - 1), abs(hz_4e9["running"] - 1))),
    ]
    failed = 0
    for name, printed, value in checks:
        ok = abs(float(printed) - float(value)) <= (tol(printed.split("e")[0]) * (10 ** int(printed.split("e")[1]) if "e" in printed else 1)) + 1e-12
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:46s} paper {printed:>8s}   computed {float(value):.6g}")
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
