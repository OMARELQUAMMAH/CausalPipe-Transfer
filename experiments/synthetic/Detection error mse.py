# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Detection-Error MSE + Full-Adapt Comparison +
Small-Target-Sample Sweep
=======================================================================
Reproduces the §4.1.2 small-target-sample results reported in the
manuscript: how detection errors affect ATE bias, MSE, and 95% CI
coverage, compared against Full-Adapt, as the post-shift target window
size n2 shrinks. This script reports:
  1. MSE alongside MAE, for Component-Aware AND Full-Adapt, split by
     whether the detector's call was correct or a misclassification.
  2. Full-Adapt run on the EXACT SAME seeds/streams as Component-Aware,
     so the two can be compared head-to-head on the same misclassified
     cases.
  3. A small-target-sample sweep: n2 (post-shift window size) in
     {500, 1000, 2000, 3000}, each reporting detector accuracy, CA
     MAE/MSE, FA MAE/MSE, and CI coverage.

generate_stream(), aipw(), component_aware(), and detect_shift_type()
are identical to causalpipe_synthetic_extended.py — do not modify them
independently of that file; keep both in sync.

full_adapt() always retrains BOTH the propensity and outcome models on
the post-shift window, regardless of any detected shift type — logically
identical to what component_aware() does on a 'mixed'/fallback call,
called here unconditionally.

SINGLE-CLASS PROPENSITY GUARD: at small window sizes (n2=500 especially),
a truncated post-shift window can occasionally contain only one treatment
class, which crashes sklearn's LogisticRegression.fit(). Both
component_aware()'s covariate-shift branch and full_adapt() therefore use
_safe_fit_pm(), which falls back to the SOURCE propensity model if the
post-shift window doesn't have both treatment classes present, rather
than crashing. This case is also counted and reported (it is the
"single-class-propensity fallback boundary" discussed in the manuscript
alongside the CI-coverage-degradation finding).

Runtime: approximately 30-45 minutes (4 window sizes x 5 shift types x 50
seeds x 2 methods).

Output (to OUT_DIR):
  r3_major3_detection_error_results.txt
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import ks_2samp, ttest_ind

OUT_DIR = os.path.join('results')
os.makedirs(OUT_DIR, exist_ok=True)

TRUE_ATE    = 5.0
N_EVENTS    = 5000
SHIFT_POINT = 2000
N_FEATURES  = 5
N_SEEDS     = 50

SHIFT_TYPES = ['label_shift', 'covariate_shift', 'mechanism_shift', 'mixed', 'gradual']
WINDOW_SIZES = [500, 1000, 2000, 3000]   # n2: post-shift window size; 3000 = full, matches Table 3


# =============================================================================
# Identical to causalpipe_synthetic_extended.py — keep in sync
# =============================================================================

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
        prob  = 1 / (1 + np.exp(-logit))
        t     = int(rng.random() < prob)

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
        X_list.append(x)
        T_list.append(t)
        Y_list.append(y)

    return (np.array(X_list, dtype=np.float64),
            np.array(T_list, dtype=np.int32),
            np.array(Y_list, dtype=np.float64))


def aipw(X, T, Y, pm=None, om=None, clip=(0.05, 0.95)):
    n  = len(X)
    XT = np.column_stack([X, T])
    if pm is None:
        pm = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        pm.fit(X, T)
    if om is None:
        om = Ridge(alpha=1.0)
        om.fit(XT, Y)
    e   = np.clip(pm.predict_proba(X)[:, 1], clip[0], clip[1])
    XT1 = np.column_stack([X, np.ones(n)])
    XT0 = np.column_stack([X, np.zeros(n)])
    mu1 = om.predict(XT1)
    mu0 = om.predict(XT0)
    psi = (mu1 - mu0
           + T * (Y - mu1) / e
           - (1 - T) * (Y - mu0) / (1 - e))
    ate = psi.mean()
    se  = psi.std(ddof=1) / np.sqrt(n)
    return ate, ate - 1.96*se, ate + 1.96*se, se, psi, pm, om


def detect_shift_type(X_pre, T_pre, Y_pre, X_post, T_post, Y_post,
                       alpha=0.05, sd_mult=2.0):
    """Identical to causalpipe_synthetic_extended.py's detect_shift_type()."""
    pm_diag = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(X_pre, T_pre)
    e_pre   = pm_diag.predict_proba(X_pre)[:, 1]
    e_post  = pm_diag.predict_proba(X_post)[:, 1]
    p_prop  = ks_2samp(e_pre, e_post)[1]
    cov_drift = p_prop < alpha

    XT_pre  = np.column_stack([X_pre,  T_pre])
    XT_post = np.column_stack([X_post, T_post])
    om_diag = Ridge(alpha=1.0).fit(XT_pre, Y_pre)
    r_pre   = Y_pre  - om_diag.predict(XT_pre)
    r_post  = Y_post - om_diag.predict(XT_post)

    _, p_mean  = ttest_ind(r_pre, r_post)
    mean_shift = abs(r_post.mean() - r_pre.mean())
    lbl_drift  = (p_mean < alpha) and (mean_shift > sd_mult * r_pre.std())

    mec_drift = False
    if not cov_drift and not lbl_drift:
        p_dist    = ks_2samp(r_pre, r_post)[1]
        mec_drift = p_dist < alpha

    if   cov_drift and lbl_drift:  return 'mixed'
    elif cov_drift:                return 'covariate_shift'
    elif lbl_drift:                return 'label_shift'
    elif mec_drift:                return 'mechanism_shift'
    else:                          return 'label_shift'


# =============================================================================
# Single-class propensity guard (needed at small n2, e.g. n2=500)
# =============================================================================

def _safe_fit_pm(X_post, T_post, pm_fallback):
    """Fit a fresh propensity model on the post-shift window; if that window
    contains only one treatment class (can happen with a small n2 under
    covariate/mixed shift), fall back to the supplied source propensity
    model instead of crashing. Returns (pm, was_fallback: bool)."""
    if len(np.unique(T_post)) < 2:
        return pm_fallback, True
    try:
        pm = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(X_post, T_post)
        return pm, False
    except Exception:
        return pm_fallback, True


def component_aware(X, T, Y, sp, detected_type, clip=(0.05, 0.95)):
    """Returns (mae, ate, ci_lo, ci_hi, pm_ad, om_ad, pm_fallback_used)."""
    _, _, _, _, _, pm_src, om_src = aipw(X[:sp], T[:sp], Y[:sp], clip=clip)
    Xp, Tp, Yp = X[sp:], T[sp:], Y[sp:]
    XTp = np.column_stack([Xp, Tp])
    pm_fallback_used = False

    if detected_type == 'covariate_shift':
        pm_ad, pm_fallback_used = _safe_fit_pm(Xp, Tp, pm_src)
        om_ad = om_src
    elif detected_type in ('label_shift', 'mechanism_shift'):
        pm_ad = pm_src
        om_ad = Ridge(alpha=1.0).fit(XTp, Yp)
    else:  # mixed or fallback
        pm_ad, pm_fallback_used = _safe_fit_pm(Xp, Tp, pm_src)
        om_ad = Ridge(alpha=1.0).fit(XTp, Yp)

    ate, ci_lo, ci_hi, se, psi, _, _ = aipw(Xp, Tp, Yp, pm_ad, om_ad, clip=clip)
    return abs(ate - TRUE_ATE), ate, ci_lo, ci_hi, pm_ad, om_ad, pm_fallback_used


def full_adapt(X, T, Y, sp, clip=(0.05, 0.95)):
    """Always retrains BOTH propensity and outcome models on the post-shift
    window, regardless of any detected shift type."""
    _, _, _, _, _, pm_src, om_src = aipw(X[:sp], T[:sp], Y[:sp], clip=clip)
    Xp, Tp, Yp = X[sp:], T[sp:], Y[sp:]
    XTp = np.column_stack([Xp, Tp])
    pm_ad, pm_fallback_used = _safe_fit_pm(Xp, Tp, pm_src)
    om_ad = Ridge(alpha=1.0).fit(XTp, Yp)
    ate, ci_lo, ci_hi, se, psi, _, _ = aipw(Xp, Tp, Yp, pm_ad, om_ad, clip=clip)
    return abs(ate - TRUE_ATE), ate, ci_lo, ci_hi, pm_ad, om_ad, pm_fallback_used


TRUE_TYPE_FOR_ACCURACY = {
    'label_shift': 'label_shift', 'covariate_shift': 'covariate_shift',
    'mechanism_shift': 'mechanism_shift', 'mixed': 'mixed', 'gradual': 'mixed',
}


def run_window(n2, shift_types=SHIFT_TYPES, n_seeds=N_SEEDS):
    """Runs CA and FA on the SAME seeds/streams, using a post-shift window of
    size n2 for BOTH detection and adaptation/estimation."""
    results = {}
    for shift_type in shift_types:
        correct_mae_ca, correct_mse_ca = [], []
        wrong_mae_ca, wrong_mse_ca = [], []
        correct_mae_fa, correct_mse_fa = [], []
        wrong_mae_fa, wrong_mse_fa = [], []
        all_mae_ca, all_mse_ca, all_mae_fa, all_mse_fa = [], [], [], []
        n_correct = 0
        ca_coverage_hits, fa_coverage_hits = 0, 0
        ca_fallback_count, fa_fallback_count = 0, 0

        for seed in range(n_seeds):
            X, T, Y = generate_stream(N_EVENTS, shift_type, SHIFT_POINT, seed)
            X_win = np.concatenate([X[:SHIFT_POINT], X[SHIFT_POINT:SHIFT_POINT + n2]])
            T_win = np.concatenate([T[:SHIFT_POINT], T[SHIFT_POINT:SHIFT_POINT + n2]])
            Y_win = np.concatenate([Y[:SHIFT_POINT], Y[SHIFT_POINT:SHIFT_POINT + n2]])

            detected = detect_shift_type(
                X_win[:SHIFT_POINT], T_win[:SHIFT_POINT], Y_win[:SHIFT_POINT],
                X_win[SHIFT_POINT:], T_win[SHIFT_POINT:], Y_win[SHIFT_POINT:]
            )
            is_correct = (detected == TRUE_TYPE_FOR_ACCURACY[shift_type])
            if is_correct:
                n_correct += 1

            mae_ca, ate_ca, ci_lo_ca, ci_hi_ca, _, _, ca_fb = component_aware(
                X_win, T_win, Y_win, SHIFT_POINT, detected)
            mse_ca = (ate_ca - TRUE_ATE) ** 2
            mae_fa, ate_fa, ci_lo_fa, ci_hi_fa, _, _, fa_fb = full_adapt(
                X_win, T_win, Y_win, SHIFT_POINT)
            mse_fa = (ate_fa - TRUE_ATE) ** 2

            ca_fallback_count += int(ca_fb)
            fa_fallback_count += int(fa_fb)

            ca_covers = (ci_lo_ca <= TRUE_ATE <= ci_hi_ca)
            fa_covers = (ci_lo_fa <= TRUE_ATE <= ci_hi_fa)
            ca_coverage_hits += int(ca_covers)
            fa_coverage_hits += int(fa_covers)

            all_mae_ca.append(mae_ca); all_mse_ca.append(mse_ca)
            all_mae_fa.append(mae_fa); all_mse_fa.append(mse_fa)

            if is_correct:
                correct_mae_ca.append(mae_ca); correct_mse_ca.append(mse_ca)
                correct_mae_fa.append(mae_fa); correct_mse_fa.append(mse_fa)
            else:
                wrong_mae_ca.append(mae_ca); wrong_mse_ca.append(mse_ca)
                wrong_mae_fa.append(mae_fa); wrong_mse_fa.append(mse_fa)

        def m(lst):
            return (np.mean(lst), np.std(lst)) if len(lst) > 0 else (None, None)

        results[shift_type] = dict(
            n_seeds=n_seeds, n_correct=n_correct, n_wrong=n_seeds - n_correct,
            acc_pct=100.0 * n_correct / n_seeds,
            ca_mae_all=m(all_mae_ca), ca_mse_all=m(all_mse_ca),
            fa_mae_all=m(all_mae_fa), fa_mse_all=m(all_mse_fa),
            ca_mae_correct=m(correct_mae_ca), ca_mse_correct=m(correct_mse_ca),
            fa_mae_correct=m(correct_mae_fa), fa_mse_correct=m(correct_mse_fa),
            ca_mae_wrong=m(wrong_mae_ca), ca_mse_wrong=m(wrong_mse_ca),
            fa_mae_wrong=m(wrong_mae_fa), fa_mse_wrong=m(wrong_mse_fa),
            ca_coverage_pct=100.0 * ca_coverage_hits / n_seeds,
            fa_coverage_pct=100.0 * fa_coverage_hits / n_seeds,
            ca_fallback_count=ca_fallback_count, fa_fallback_count=fa_fallback_count,
        )
    return results


def fmt(pair):
    mean, sd = pair
    if mean is None:
        return "N/A (0 seeds in this bucket)"
    return f"{mean:.4f}+/-{sd:.4f}"


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    log("=" * 78)
    log("CAUSALPIPE-TRANSFER: DETECTION-ERROR MSE + FULL-ADAPT + SMALL-SAMPLE SWEEP")
    log(f"{N_SEEDS} seeds per shift type per window size")
    log("=" * 78)

    log("\n>>> SANITY CHECK: n2=3000 (full window) detection accuracy should match "
        "Table 3 (~90-100% per shift type, ~95.6% action-level overall). If it "
        "doesn't, stop and check before trusting anything else below. <<<")

    for n2 in WINDOW_SIZES:
        log("\n" + "#" * 78)
        log(f"WINDOW SIZE n2 = {n2}" + ("  (full, matches Table 3)" if n2 == 3000 else "  (small-sample)"))
        log("#" * 78)
        res = run_window(n2)
        for shift_type, r in res.items():
            log(f"\n--- {shift_type} (n2={n2}) ---")
            log(f"  Detection accuracy: {r['acc_pct']:.1f}%  ({r['n_correct']}/{r['n_seeds']} correct, "
                f"{r['n_wrong']} misclassified)")
            log(f"  ALL SEEDS      -- CA MAE={fmt(r['ca_mae_all'])}  CA MSE={fmt(r['ca_mse_all'])}")
            log(f"                    FA MAE={fmt(r['fa_mae_all'])}  FA MSE={fmt(r['fa_mse_all'])}")
            log(f"  CORRECT DETECT -- CA MAE={fmt(r['ca_mae_correct'])}  CA MSE={fmt(r['ca_mse_correct'])}")
            log(f"                    FA MAE={fmt(r['fa_mae_correct'])}  FA MSE={fmt(r['fa_mse_correct'])}")
            log(f"  MISCLASSIFIED  -- CA MAE={fmt(r['ca_mae_wrong'])}  CA MSE={fmt(r['ca_mse_wrong'])}")
            log(f"                    FA MAE={fmt(r['fa_mae_wrong'])}  FA MSE={fmt(r['fa_mse_wrong'])}")
            log(f"  95% CI coverage -- CA: {r['ca_coverage_pct']:.1f}%   FA: {r['fa_coverage_pct']:.1f}%  "
                f"(nominal target: 95.0%)")
            log(f"  Single-class-propensity fallback fired -- CA: {r['ca_fallback_count']}/{r['n_seeds']}"
                f"   FA: {r['fa_fallback_count']}/{r['n_seeds']}")

    path = os.path.join(OUT_DIR, 'r3_major3_detection_error_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
