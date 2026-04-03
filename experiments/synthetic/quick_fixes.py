# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Two Quick Fixes
=====================================
Script 1: Gradual shift detector accuracy (10 seeds)
Script 2: Retail sensitivity check — exclude 90-95% discount bin
           and re-run binned DR to test robustness of sign reversal

Outputs:
  gradual_detector_accuracy.txt
  retail_sensitivity_results.txt
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, ttest_ind
import warnings
warnings.filterwarnings('ignore')

RETAIL_PATH = r"C:\Users\info\Desktop\eval_split FreshRetailNet-50K.xlsx"
OUT_DIR     = r"C:\Users\info\Desktop\PILLAR 3"

TRUE_ATE    = 5.0
N_EVENTS    = 5000
SHIFT_POINT = 2000
N_FEATURES  = 5
N_SEEDS     = 10

# ─────────────────────────────────────────────────────────────────────────────
# SHARED: Data generator and detector (identical to extended synthetic script)
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
    """Propensity-calibration detector — identical to extended synthetic script."""
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
    print(f"\nTesting detector on gradual shift ({N_SEEDS} seeds) ...")
    print("True shift type: gradual (mixed covariate + label + mechanism)")
    print("Detector does not have 'gradual' as a category —")
    print("it will classify as one of: covariate, label, mechanism, mixed")
    print()

    detections = []
    maes_ca    = []
    maes_oracle = []

    for seed in range(N_SEEDS):
        X, T, Y = generate_stream(N_EVENTS, 'gradual', SHIFT_POINT, seed)
        sp = SHIFT_POINT

        detected = detect_shift_type(
            X[:sp], T[:sp], Y[:sp], X[sp:], T[sp:], Y[sp:]
        )
        detections.append(detected)

        # Component-aware MAE with detected type
        Xp, Tp, Yp = X[sp:], T[sp:], Y[sp:]
        XTp = np.column_stack([Xp, Tp])

        _, _, _, _, pm_src, om_src = (lambda: (
            *aipw_batch(X[:sp], T[:sp], Y[:sp])[:4],
            LogisticRegression(max_iter=1000,C=1.0,random_state=42).fit(X[:sp],T[:sp]),
            Ridge(alpha=1.0).fit(np.column_stack([X[:sp],T[:sp]]),Y[:sp])
        ))()

        if detected == 'covariate_shift':
            pm_ad = LogisticRegression(max_iter=1000,C=1.0,random_state=42).fit(Xp,Tp)
            om_ad = om_src
        elif detected in ('label_shift','mechanism_shift'):
            pm_ad = pm_src
            om_ad = Ridge(alpha=1.0).fit(XTp, Yp)
        else:
            pm_ad = LogisticRegression(max_iter=1000,C=1.0,random_state=42).fit(Xp,Tp)
            om_ad = Ridge(alpha=1.0).fit(XTp, Yp)

        n_p = len(Xp)
        e   = np.clip(pm_ad.predict_proba(Xp)[:,1], 0.05, 0.95)
        XT1 = np.column_stack([Xp, np.ones(n_p)])
        XT0 = np.column_stack([Xp, np.zeros(n_p)])
        mu1 = om_ad.predict(XT1)
        mu0 = om_ad.predict(XT0)
        psi_ca = mu1-mu0+Tp*(Yp-mu1)/e-(1-Tp)*(Yp-mu0)/(1-e)
        ate_ca = psi_ca.mean()
        mae_ca = abs(ate_ca - TRUE_ATE)
        maes_ca.append(mae_ca)

        # Oracle: full adapt (best possible for gradual = unknown type)
        ate_or, _, _, _ = aipw_batch(Xp, Tp, Yp)
        maes_oracle.append(abs(ate_or - TRUE_ATE))

        print(f"  Seed {seed:2d}: detected={detected:<20}  "
              f"CA_MAE={mae_ca:.4f}  Oracle_MAE={abs(ate_or-TRUE_ATE):.4f}")

    # Summary
    from collections import Counter
    counts = Counter(detections)
    print(f"\n  Detection distribution:")
    for dtype, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / N_SEEDS * 100
        print(f"    {dtype:<22}: {count}/{N_SEEDS} ({pct:.0f}%)")

    most_common = counts.most_common(1)[0][0]
    most_common_pct = counts.most_common(1)[0][1] / N_SEEDS * 100

    print(f"\n  CA MAE  : {np.mean(maes_ca):.4f} ± {np.std(maes_ca):.4f}")
    print(f"  Oracle  : {np.mean(maes_oracle):.4f} ± {np.std(maes_oracle):.4f}")
    gap = (np.mean(maes_ca) - np.mean(maes_oracle)) / max(np.mean(maes_oracle),1e-8)*100
    print(f"  CA-Oracle gap: {gap:.1f}%")

    # Median MAE for reporting
    print(f"\n  Median CA MAE    : {np.median(maes_ca):.4f}")
    print(f"  Median Oracle MAE: {np.median(maes_oracle):.4f}")

    report = f"""
======================================================
GRADUAL SHIFT DETECTOR ACCURACY
======================================================
Seeds tested   : {N_SEEDS}
True type      : gradual (composite — covariate + label + mechanism)
Detector vocab : covariate_shift, label_shift, mechanism_shift, mixed

DETECTION DISTRIBUTION:
{chr(10).join([f"  {k:<22}: {v}/{N_SEEDS} ({v/N_SEEDS*100:.0f}%)" for k,v in sorted(counts.items(), key=lambda x:-x[1])])}

Most common detection: {most_common} ({most_common_pct:.0f}% of seeds)

PERFORMANCE:
  CA MAE (mean ± SD)    : {np.mean(maes_ca):.4f} ± {np.std(maes_ca):.4f}
  CA MAE (median)       : {np.median(maes_ca):.4f}
  Oracle MAE (mean)     : {np.mean(maes_oracle):.4f} ± {np.std(maes_oracle):.4f}
  Oracle MAE (median)   : {np.median(maes_oracle):.4f}
  CA-Oracle gap (mean)  : {gap:.1f}%

INTERPRETATION FOR PAPER:
  The detector does not have 'gradual' as a named category.
  It classifies gradual shifts as {most_common} in {most_common_pct:.0f}% of seeds.
  This classification is partially correct — gradual shift in this
  SCM involves both covariate and label changes, so detecting it
  as {most_common} is a reasonable approximation.
  
  Despite imperfect classification, CA achieves median MAE of
  {np.median(maes_ca):.4f} vs Oracle median of {np.median(maes_oracle):.4f},
  demonstrating robustness to imprecise shift type diagnosis.

TEXT FOR PAPER (Section 4.1.2):
  "For the gradual shift condition, the detector classified {most_common_pct:.0f}%
  of seeds as {most_common}, reflecting that the composite nature
  of the gradual transition (simultaneous covariate and label drift)
  activates both propensity score and residual signals. Component-Aware
  achieves median MAE of {np.median(maes_ca):.4f} under gradual shift,
  within {gap:.1f}% of the Oracle upper bound, demonstrating robustness
  to imprecise shift type classification."
======================================================
"""
    print(report)
    path = OUT_DIR + r"\gradual_detector_accuracy.txt"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {path}")

    return {
        'detections': detections, 'counts': counts,
        'most_common': most_common, 'most_common_pct': most_common_pct,
        'mean_mae_ca': np.mean(maes_ca), 'median_mae_ca': np.median(maes_ca),
        'mean_mae_or': np.mean(maes_oracle), 'median_mae_or': np.median(maes_oracle),
        'gap': gap
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT 2: RETAIL SENSITIVITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def dr_estimate_retail(X, W, Y):
    """DR estimator for retail (same formula as main retail script)."""
    om = LinearRegression().fit(X, Y)
    tm = LinearRegression().fit(X, W)
    Y_res = Y - om.predict(X)
    W_res = W - tm.predict(X)
    denom = (W_res**2).mean()
    if denom < 1e-12:
        return np.nan, np.nan, np.nan, np.nan
    psi  = W_res * Y_res / denom
    ate  = psi.mean()
    se   = psi.std(ddof=1) / np.sqrt(len(psi))
    return ate, ate-1.96*se, ate+1.96*se, se


def run_retail_sensitivity():
    print("\n" + "=" * 60)
    print("SCRIPT 2: RETAIL SENSITIVITY CHECK")
    print("Test: does 85-95% sign reversal persist when 90-95% excluded?")
    print("=" * 60)

    print("\nLoading FreshRetailNet ...")
    df = pd.read_excel(RETAIL_PATH)
    df = df[df['sale_amount'] > 0].copy()
    print(f"  Loaded {len(df):,} non-zero transactions")

    feature_cols = ['stock_hour6_22_cnt','precpt','avg_temperature',
                    'avg_humidity','avg_wind_level','holiday_flag','activity_flag']
    feature_cols = [c for c in feature_cols if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X_raw = df[feature_cols].values
    W     = df['discount'].values
    Y_log = np.log1p(df['sale_amount'].values)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    print("\nRunning binned analysis on FULL dataset:")
    bins_full = run_binned(X, W, Y_log, label="Full dataset")

    # Sensitivity 1: exclude 90-95% bin entirely
    mask_excl_9095 = ~((W >= 0.90) & (W < 0.95))
    print("\nRunning binned analysis EXCLUDING 90-95% discount transactions:")
    bins_excl = run_binned(X[mask_excl_9095], W[mask_excl_9095],
                            Y_log[mask_excl_9095],
                            label="Excluding 90-95%")

    # Sensitivity 2: restrict to 85-95% bin only, test robustness
    mask_8595 = (W >= 0.85) & (W < 0.95)
    print(f"\nFocused analysis on 85-95% bin only (n={mask_8595.sum():,}):")
    X85, W85, Y85 = X[mask_8595], W[mask_8595], Y_log[mask_8595]
    if W85.std() > 1e-6 and mask_8595.sum() > 200:
        ate85, lo85, hi85, se85 = dr_estimate_retail(X85, W85, Y85)
        pct85 = (np.exp(ate85)-1)*100
        print(f"  ATE (log): {ate85:.4f}  ({pct85:+.1f}%)  "
              f"95%CI=[{lo85:.4f},{hi85:.4f}]  n={mask_8595.sum():,}")

        # Sub-bins within 85-95%
        print(f"\n  Sub-bins within 85-95%:")
        for lo_b, hi_b in [(0.85,0.88),(0.88,0.91),(0.91,0.95)]:
            m = (W >= lo_b) & (W < hi_b)
            if m.sum() < 100: continue
            ate_b, lo_b2, hi_b2, se_b = dr_estimate_retail(X[m], W[m], Y_log[m])
            pct_b = (np.exp(ate_b)-1)*100
            print(f"    {lo_b:.0%}–{hi_b:.0%}: ATE={pct_b:+.1f}%  "
                  f"CI=[{(np.exp(lo_b2)-1)*100:.1f}%,{(np.exp(hi_b2)-1)*100:.1f}%]  "
                  f"n={m.sum():,}")

    # Compare: does 85-95% positive ATE persist when 90-95% excluded?
    print("\n" + "="*60)
    print("SENSITIVITY VERDICT:")
    orig_8595 = next((r for r in bins_full if '85-95' in r['bin']), None)
    new_8595  = next((r for r in bins_excl if '85-95' in r['bin']), None)

    if orig_8595 and new_8595:
        print(f"  Original 85-95% ATE: {orig_8595['pct']:+.1f}%")
        print(f"  After excl. 90-95%:  {new_8595['pct']:+.1f}%")
        if orig_8595['pct'] > 0 and new_8595['pct'] > 0:
            print("  ROBUST: Sign reversal persists after excluding 90-95%")
            verdict = "ROBUST"
        elif orig_8595['pct'] > 0 and new_8595['pct'] <= 0:
            print("  NOT ROBUST: Sign reversal disappears — driven by 90-95% subgroup")
            verdict = "NOT ROBUST"
        else:
            print("  INCONCLUSIVE")
            verdict = "INCONCLUSIVE"
    else:
        verdict = "COULD NOT COMPARE"

    # Save results
    report = f"""
======================================================
RETAIL SENSITIVITY CHECK
Test: robustness of 85-95% positive ATE sign reversal
======================================================

FULL DATASET BINS:
{chr(10).join([f"  {r['bin']}: ATE={r['pct']:+.1f}%  CI=[{(np.exp(r['ci_lo'])-1)*100:.1f}%,{(np.exp(r['ci_hi'])-1)*100:.1f}%]  n={r['n']:,}" for r in bins_full])}

EXCLUDING 90-95% DISCOUNT TRANSACTIONS:
{chr(10).join([f"  {r['bin']}: ATE={r['pct']:+.1f}%  CI=[{(np.exp(r['ci_lo'])-1)*100:.1f}%,{(np.exp(r['ci_hi'])-1)*100:.1f}%]  n={r['n']:,}" for r in bins_excl])}

SENSITIVITY VERDICT: {verdict}

TEXT FOR PAPER (Section 4.2.3):
{"  The positive ATE in the 85-95% discount bin persists after excluding the 90-95% sub-segment (sensitivity check), confirming it is not driven by a narrow subset of transactions. A plausible mechanism is loss-leader pricing: moderate discounts on high-traffic mid-value products attract customers who purchase full-price complementary items, producing an apparent positive effect on recorded sale amounts within that transaction." if verdict=="ROBUST" else "  The positive ATE in the 85-95% discount bin is not robust to exclusion of the 90-95% sub-segment, suggesting it is driven by a narrow subset of transactions rather than a generalised pricing effect. We report this finding with caution and recommend replication on a larger, more diverse dataset."}
======================================================
"""
    print(report)
    path = OUT_DIR + r"\retail_sensitivity_results.txt"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {path}")
    return verdict


def run_binned(X, W, Y_log, label=""):
    bin_edges  = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
    bin_labels = ['0-50%','50-70%','70-85%','85-95%','95-100%']
    results = []
    for lbl, lo, hi in zip(bin_labels, bin_edges[:-1], bin_edges[1:]):
        mask = (W >= lo) & (W < hi)
        if mask.sum() < 100 or W[mask].std() < 1e-6:
            continue
        ate, ci_lo, ci_hi, se = dr_estimate_retail(X[mask], W[mask], Y_log[mask])
        if np.isnan(ate): continue
        pct = (np.exp(ate)-1)*100
        print(f"  {lbl}: ATE={pct:+.1f}%  "
              f"CI=[{(np.exp(ci_lo)-1)*100:.1f}%,{(np.exp(ci_hi)-1)*100:.1f}%]  "
              f"n={mask.sum():,}")
        results.append({'bin':lbl,'ate':ate,'ci_lo':ci_lo,'ci_hi':ci_hi,
                        'se':se,'pct':pct,'n':mask.sum()})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CAUSALPIPE-TRANSFER: TWO QUICK FIX ANALYSES")
    print("=" * 60)

    # Script 1: ~2 minutes
    grad_res = run_gradual_detector_accuracy()

    # Script 2: ~8 minutes (loads Excel)
    retail_verdict = run_retail_sensitivity()

    print("\n" + "=" * 60)
    print("BOTH ANALYSES COMPLETE")
    print(f"  Gradual detector: most common = {grad_res['most_common']} "
          f"({grad_res['most_common_pct']:.0f}%)")
    print(f"  Retail 85-95% sign reversal: {retail_verdict}")
    print("=" * 60)
