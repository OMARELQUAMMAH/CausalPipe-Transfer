# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Quick Fixes
==================================
REVISION NOTE: this file originally had two scripts:
  Script 1: Gradual shift detector accuracy (10 seeds) — STILL VALID, kept
            below unchanged (uses the same verified generate_stream(),
            aipw(), detect_shift_type() as causalpipe_synthetic_extended.py).
  Script 2: Retail sensitivity check — excluded the 90-95% discount bin
            and re-ran BINNED DR to test robustness of a sign reversal.
            THIS IS NOW OBSOLETE. Discrete discount bins were dropped
            entirely from the retail analysis (numerically unstable on a
            continuous treatment with a point mass at zero — see
            causalpipe_retail_fixed.py's revision note) and replaced with
            a two-part model (binary extensive-margin AIPW + spline
            intensive-margin dose-response). The corresponding robustness
            checks for that model now live in causalpipe_retail_robustness.py
            (functional-form check + extensive-margin placebo test).
            Script 2 has been removed from this file; if you need the old
            binned-sensitivity code for archival purposes, it remains in
            version history / your working folder, but it should not be
            re-run to produce numbers for the paper.

Outputs:
  gradual_detector_accuracy.txt
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import ks_2samp, ttest_ind
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join('results')
os.makedirs(OUT_DIR, exist_ok=True)

TRUE_ATE    = 5.0
N_EVENTS    = 5000
SHIFT_POINT = 2000
N_FEATURES  = 5
N_SEEDS     = 10

# ─────────────────────────────────────────────────────────────────────────────
# SHARED: Data generator and detector (identical to causalpipe_synthetic_extended.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_stream(n, shift_type, shift_point, seed=42, gradual_window=500):
    rng   = np.random.default_rng(seed)
    alpha = rng.normal(0, 0.5, N_FEATURES)
    beta  = np.ones(N_FEATURES) * 0.5
    X_list, T_list, Y_list = [], [], []

    for i in range(n):
        if shift_type == 'gradual':
            half = gradual_window / 2
            w = np.clip((i - (shift_point - half)) / gradual_window, 0, 1)
        else:
            w = 1.0 if i >= shift_point else 0.0

        if shift_type in ('covariate_shift', 'mixed', 'gradual'):
            mu = w * 4.0
        else:
            mu = 0.0
        x = rng.normal(mu, 1.0, N_FEATURES)

        logit = float(np.clip(x @ alpha, -6, 6))
        t     = int(rng.random() < 1 / (1 + np.exp(-logit)))

        if shift_type == 'mechanism_shift' and i >= shift_point:
            b = -beta
        elif shift_type == 'gradual':
            b = beta * (1 - w) + (-beta) * w
        else:
            b = beta

        y_mean = TRUE_ATE * t + float(x @ b)

        if shift_type in ('label_shift', 'mixed') and i >= shift_point:
            y_mean += 10.0
        elif shift_type == 'gradual':
            y_mean += w * 6.0

        y = y_mean + rng.normal(0, 1.0)
        X_list.append(x); T_list.append(t); Y_list.append(y)

    return (np.array(X_list), np.array(T_list), np.array(Y_list))


def detect_shift_type(X_pre, T_pre, Y_pre, X_post, T_post, Y_post, alpha=0.05):
    """Propensity-calibration detector — identical to causalpipe_synthetic_extended.py."""
    pm_diag = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(X_pre, T_pre)
    e_pre   = pm_diag.predict_proba(X_pre)[:, 1]
    e_post  = pm_diag.predict_proba(X_post)[:, 1]
    p_prop  = ks_2samp(e_pre, e_post)[1]
    cov_drift = p_prop < alpha

    XT_pre  = np.column_stack([X_pre, T_pre])
    XT_post = np.column_stack([X_post, T_post])
    om_diag = Ridge(alpha=1.0).fit(XT_pre, Y_pre)
    r_pre   = Y_pre  - om_diag.predict(XT_pre)
    r_post  = Y_post - om_diag.predict(XT_post)
    _, p_mean  = ttest_ind(r_pre, r_post)
    mean_shift = abs(r_post.mean() - r_pre.mean())
    lbl_drift  = (p_mean < alpha) and (mean_shift > 2 * r_pre.std())

    mec_drift = False
    if not cov_drift and not lbl_drift:
        p_dist    = ks_2samp(r_pre, r_post)[1]
        mec_drift = p_dist < alpha

    if cov_drift and lbl_drift:   return 'mixed'
    elif cov_drift:                return 'covariate_shift'
    elif lbl_drift:                return 'label_shift'
    elif mec_drift:                return 'mechanism_shift'
    else:                          return 'label_shift'


def aipw_batch(X, T, Y):
    """AIPW DR estimator — identical to all other scripts."""
    n  = len(X)
    XT = np.column_stack([X, T])
    pm = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(X, T)
    om = Ridge(alpha=1.0).fit(XT, Y)
    e  = np.clip(pm.predict_proba(X)[:, 1], 0.05, 0.95)
    XT1 = np.column_stack([X, np.ones(n)])
    XT0 = np.column_stack([X, np.zeros(n)])
    mu1 = om.predict(XT1)
    mu0 = om.predict(XT0)
    psi = mu1 - mu0 + T*(Y-mu1)/e - (1-T)*(Y-mu0)/(1-e)
    ate = psi.mean()
    se  = psi.std(ddof=1) / np.sqrt(n)
    return ate, ate-1.96*se, ate+1.96*se, se


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT 1: GRADUAL SHIFT DETECTOR ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def run_gradual_detector_accuracy():
    print("=" * 60)
    print("SCRIPT 1: GRADUAL SHIFT DETECTOR ACCURACY")
    print("=" * 60)
    print(f"\nTesting detector on gradual shift ({N_SEEDS} seeds)...")
    print("True shift type: gradual (mixed covariate + label + mechanism)")
    print("Detector does not have 'gradual' as a category —")
    print("it will classify as one of: covariate, label, mechanism, mixed\n")

    detections, maes_ca, maes_oracle = [], [], []

    for seed in range(N_SEEDS):
        X, T, Y = generate_stream(N_EVENTS, 'gradual', SHIFT_POINT, seed)
        sp = SHIFT_POINT

        detected = detect_shift_type(X[:sp], T[:sp], Y[:sp], X[sp:], T[sp:], Y[sp:])
        detections.append(detected)

        Xp, Tp, Yp = X[sp:], T[sp:], Y[sp:]
        XTp = np.column_stack([Xp, Tp])

        pm_src = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(X[:sp], T[:sp])
        om_src = Ridge(alpha=1.0).fit(np.column_stack([X[:sp], T[:sp]]), Y[:sp])

        if detected == 'covariate_shift':
            pm_ad = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xp, Tp)
            om_ad = om_src
        elif detected in ('label_shift', 'mechanism_shift'):
            pm_ad = pm_src
            om_ad = Ridge(alpha=1.0).fit(XTp, Yp)
        else:
            pm_ad = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xp, Tp)
            om_ad = Ridge(alpha=1.0).fit(XTp, Yp)

        n_p = len(Xp)
        e   = np.clip(pm_ad.predict_proba(Xp)[:, 1], 0.05, 0.95)
        XT1 = np.column_stack([Xp, np.ones(n_p)])
        XT0 = np.column_stack([Xp, np.zeros(n_p)])
        mu1 = om_ad.predict(XT1)
        mu0 = om_ad.predict(XT0)
        psi_ca = mu1 - mu0 + Tp*(Yp-mu1)/e - (1-Tp)*(Yp-mu0)/(1-e)
        ate_ca = psi_ca.mean()
        mae_ca = abs(ate_ca - TRUE_ATE)
        maes_ca.append(mae_ca)

        ate_or, _, _, _ = aipw_batch(Xp, Tp, Yp)
        maes_oracle.append(abs(ate_or - TRUE_ATE))

        print(f"  Seed {seed:2d}: detected={detected:<20}  "
              f"CA_MAE={mae_ca:.4f}  Oracle_MAE={abs(ate_or-TRUE_ATE):.4f}")

    counts = Counter(detections)
    print(f"\n  Detection distribution:")
    for dtype, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {dtype:<22}: {count}/{N_SEEDS} ({count/N_SEEDS*100:.0f}%)")

    most_common = counts.most_common(1)[0][0]
    most_common_pct = counts.most_common(1)[0][1] / N_SEEDS * 100
    gap = (np.mean(maes_ca) - np.mean(maes_oracle)) / max(np.mean(maes_oracle), 1e-8) * 100

    print(f"\n  CA MAE  : {np.mean(maes_ca):.4f} +/- {np.std(maes_ca):.4f}")
    print(f"  Oracle  : {np.mean(maes_oracle):.4f} +/- {np.std(maes_oracle):.4f}")
    print(f"  CA-Oracle gap: {gap:.1f}%")
    print(f"\n  Median CA MAE    : {np.median(maes_ca):.4f}")
    print(f"  Median Oracle MAE: {np.median(maes_oracle):.4f}")

    report = f"""
======================================================
GRADUAL SHIFT DETECTOR ACCURACY
======================================================
Seeds tested   : {N_SEEDS}
True type      : gradual (composite -- covariate + label + mechanism)
Detector vocab : covariate_shift, label_shift, mechanism_shift, mixed

DETECTION DISTRIBUTION:
{chr(10).join([f"  {k:<22}: {v}/{N_SEEDS} ({v/N_SEEDS*100:.0f}%)" for k, v in sorted(counts.items(), key=lambda x: -x[1])])}

Most common detection: {most_common} ({most_common_pct:.0f}% of seeds)

PERFORMANCE:
  CA MAE (mean +/- SD)  : {np.mean(maes_ca):.4f} +/- {np.std(maes_ca):.4f}
  CA MAE (median)       : {np.median(maes_ca):.4f}
  Oracle MAE (mean)     : {np.mean(maes_oracle):.4f} +/- {np.std(maes_oracle):.4f}
  Oracle MAE (median)   : {np.median(maes_oracle):.4f}
  CA-Oracle gap (mean)  : {gap:.1f}%
======================================================
"""
    print(report)
    path = os.path.join(OUT_DIR, 'gradual_detector_accuracy.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {path}")

    return {'detections': detections, 'counts': counts, 'most_common': most_common,
            'most_common_pct': most_common_pct, 'mean_mae_ca': np.mean(maes_ca),
            'median_mae_ca': np.median(maes_ca), 'mean_mae_or': np.mean(maes_oracle),
            'median_mae_or': np.median(maes_oracle), 'gap': gap}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CAUSALPIPE-TRANSFER: QUICK FIXES")
    print("(Script 2, retail bin sensitivity, removed -- superseded by")
    print(" causalpipe_retail_robustness.py; see module docstring)")
    print("=" * 60)

    grad_res = run_gradual_detector_accuracy()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"  Gradual detector: most common = {grad_res['most_common']} "
          f"({grad_res['most_common_pct']:.0f}%)")
    print("=" * 60)
