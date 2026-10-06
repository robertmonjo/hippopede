"""Where does the binned/unbinned difference come from?  chi2(model) - chi2(LCDM) for the supernovae with
(a) 50 GLS bins, model evaluated at the bin redshift (as in likelihood.py, distance proxy);
(b) 50 GLS bins, model compressed exactly like the data (mu space);
(c) 55 GLS bins: the last bin (z 0.73-1.91, 60 SNe) split into 6 bins of 10;
(d) unbinned.
An overall magnitude offset is minimised analytically in every case.  Writes json/binning_resolution_test.json."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_observer_ball as OB, pantheon_offaxis_test as T, likelihood as L, compare_ball_curves as C

z, zh, mb, cov, n = T.load()
ci_full = np.linalg.inv(cov)
o = np.argsort(z)


def bins_equal(nb, split_last=None):
    idx = np.array_split(o, nb) if split_last is None else None
    if split_last is not None:
        base = np.array_split(o, nb)
        last = base[-1]
        idx = base[:-1] + np.array_split(last, split_last)
    A = np.zeros((len(z), len(idx)))
    for b, ii in enumerate(idx):
        A[ii, b] = 1
    return A


def chi2_unbinned(mu):
    r = mb - mu
    one = np.ones_like(r)
    m = (one @ ci_full @ r) / (one @ ci_full @ one)
    d = r - m
    return float(d @ ci_full @ d)


def chi2_binned_exact(mu, A):
    """GLS-compress data and model the same way; offset analytic."""
    F = A.T @ ci_full @ A
    Cb = np.linalg.inv(F)
    db = Cb @ A.T @ ci_full @ (mb - mu)          # compressed residuals of the model
    Fi = F
    one = np.ones(A.shape[1])
    m = (one @ Fi @ db) / (one @ Fi @ one)
    d = db - m
    return float(d @ Fi @ d)


def mu_of(dc_zw):
    return 5 * np.log10((1 + zh) * np.interp(z, OB.ZW, dc_zw))


def chi2_binned_proxy(dc_zw):
    m = np.interp(L.Z_SN, OB.ZW, dc_zw)
    a = (m @ L.CINV_SN @ L.D_SN) / (m @ L.CINV_SN @ m)
    r = L.D_SN - a * m
    return float(r @ L.CINV_SN @ r)


om = 0.3341
e_l = np.sqrt(om * (1 + OB.ZW) ** 3 + 1 - om)
models = {"LCDM": C.comoving(e_l)}
E, _, _ = OB.sector_table(0.4156, 30.0)
models["hyperconical"] = C.comoving(E[0].copy())
for ah, t0 in ((0.3347, 1.741), (0.3609, 2.233)):
    models[f"ball {ah}/{t0}"] = OB.model(ah, t0, 0.0)[4]

A50, A55 = bins_equal(50), bins_equal(50, split_last=6)
res = {k: (chi2_binned_proxy(v), chi2_binned_exact(mu_of(v), A50), chi2_binned_exact(mu_of(v), A55), chi2_unbinned(mu_of(v)))
       for k, v in models.items()}
ref = res["LCDM"]
print("model                 (a) bins at z_b  (b) 50 bins exact  (c) 55 bins exact  (d) unbinned")
for k, r in res.items():
    print(f"{k:22s}  {r[0] - ref[0]:+8.3f}        {r[1] - ref[1]:+8.3f}          {r[2] - ref[2]:+8.3f}          {r[3] - ref[3]:+8.3f}")
last = np.array_split(o, 50)[-1]
print(f"last of the 50 bins: {len(last)} supernovae, {z[last].min():.3f} < z < {z[last].max():.3f}")
out = {"columns": ["bins_at_zb", "bins_exact_50", "bins_exact_55", "unbinned"],
       "dchi2_vs_lcdm": {k: [r[n] - ref[n] for n in range(4)] for k, r in res.items()},
       "last_bin": {"n": int(len(last)), "z_min": float(z[last].min()), "z_max": float(z[last].max())}}
from pathlib import Path
(Path(__file__).resolve().parents[1] / "json" / "binning_resolution_test.json").write_text(json.dumps(out, indent=1))
