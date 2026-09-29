# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Retail Robustness Checks — TWO-PART MODEL
=================================================================
REVISION NOTE: this script previously tested robustness of a discrete-bin,
raw-`discount` model (linear vs quadratic DR, fixed 5-level bins). Both the
discount field and the modeling approach have since changed:

  - Discount field: true_discount = 1 - discount (see causalpipe_retail_fixed.py
    for the full explanation; the raw `discount` column is inverted).
  - Discrete bins are no longer reported anywhere: on a continuous treatment
    with a point mass at zero, bin-level DR estimates were numerically
    unstable (a 90-95% sub-segment sensitivity swing, and -100%/+4668%
    blow-ups in earlier runs). causalpipe_retail_fixed.py replaced them with
    a two-part model (binary extensive-margin AIPW + spline intensive-margin
    dose-response).

This script now checks robustness of THAT two-part specification:

  1. Functional-form check (replaces linear-vs-quadratic-vs-bins): does a
     simple LINEAR-in-W dose response on the discounted subsample broadly
     agree with the SPLINE dose response reported in causalpipe_retail_fixed.py,
     over the same 5th-95th percentile range? Large disagreement would mean
     the spline is picking up curvature the headline number should account
     for; close agreement supports using the simpler summary number.
  2. Placebo test on the FULL sample, permuting the binary "any discount"
     indicator (extensive margin) — complements the placebo test already
     run in causalpipe_retail_fixed.py, which tests the intensive-margin
     (discount-depth) effect on the discounted subsample only.
  3. Confounding demonstration (descriptive only, NOT a reported causal
     estimate): mean sale amount by true_discount level, using fixed bins
     purely to illustrate the selection-bias pattern DR adjustment corrects
     for. This is why quintile-style bins remain useful as a diagnostic
     plot even though they are no longer used to report causal effects.

Outputs (to OUT_DIR):
  retail_robustness_results.txt
  retail_robustness_figures.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, SplineTransformer
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

RETAIL_PATH = os.path.join('data', 'retail', 'eval_split FreshRetailNet-50K.xlsx')
OUT_DIR     = os.path.join('results')
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOTSTRAP    = 300
N_PERMUTATIONS = 200
N_KNOTS        = 6
SEED           = 42

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

FEATURE_COLS = ['stock_hour6_22_cnt', 'precpt', 'avg_temperature',
                 'avg_humidity', 'avg_wind_level', 'holiday_flag', 'activity_flag']


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA  (same correction as causalpipe_retail_fixed.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_retail(path):
    print("Loading FreshRetailNet dataset...")
    df = pd.read_excel(path)
    df = df[df['sale_amount'] > 0].copy()
    print(f"  Non-zero transactions: {len(df):,}")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    Y = np.log1p(df['sale_amount'].values.astype(float))
    X_raw = df[feature_cols].values.astype(float)
    X = StandardScaler().fit_transform(X_raw)

    W_true = 1.0 - df['discount'].values.astype(float)   # discount-field correction
    T = (W_true > 1e-9).astype(int)

    print(f"  true_discount: mean={W_true.mean():.2%}  "
          f"range=[{W_true.min():.2f}, {W_true.max():.2f}]")
    return X, W_true, T, Y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ESTIMATORS  (identical formulas to causalpipe_retail_fixed.py)
# ─────────────────────────────────────────────────────────────────────────────

def dr_estimate_linear(X, W, Y):
    om = LinearRegression().fit(X, Y)
    tm = LinearRegression().fit(X, W)
    Y_res = Y - om.predict(X)
    W_res = W - tm.predict(X)
    ate = (W_res * Y_res).mean() / (W_res ** 2).mean()
    psi = W_res * Y_res / (W_res ** 2).mean()
    se  = psi.std(ddof=1) / np.sqrt(len(psi))
    return ate, se


def spline_curve(X, W, Y, eval_pts, n_knots=N_KNOTS):
    om = LinearRegression().fit(X, Y)
    Y_res = Y - om.predict(X)
    spl = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
    B = spl.fit_transform(W.reshape(-1, 1))
    B_res = np.zeros_like(B)
    for j in range(B.shape[1]):
        tm = LinearRegression().fit(X, B[:, j])
        B_res[:, j] = B[:, j] - tm.predict(X)
    coef, *_ = np.linalg.lstsq(B_res, Y_res, rcond=None)
    B_eval = spl.transform(eval_pts.reshape(-1, 1))
    B_eval_centered = B_eval - B.mean(axis=0, keepdims=True)
    return B_eval_centered @ coef


def aipw_binary(X, T, Y):
    n = len(X)
    XT = np.column_stack([X, T])
    pm = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(X, T)
    om = Ridge(alpha=1.0).fit(XT, Y)
    e = np.clip(pm.predict_proba(X)[:, 1], 0.05, 0.95)
    XT1 = np.column_stack([X, np.ones(n)])
    XT0 = np.column_stack([X, np.zeros(n)])
    mu1, mu0 = om.predict(XT1), om.predict(XT0)
    psi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return psi.mean(), psi.std(ddof=1) / np.sqrt(n)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CHECK 1: LINEAR vs SPLINE FUNCTIONAL FORM  (discounted subsample)
# ─────────────────────────────────────────────────────────────────────────────

def functional_form_check(X, W, Y, n_boot=N_BOOTSTRAP):
    print("\n" + "=" * 60)
    print("CHECK 1: LINEAR vs SPLINE FUNCTIONAL FORM (discounted subsample)")
    print("=" * 60)

    p5, p95 = np.percentile(W, [5, 95])

    # Linear-in-W: delta = slope * (p95 - p5)
    ate_lin, se_lin = dr_estimate_linear(X, W, Y)
    delta_lin = ate_lin * (p95 - p5)
    delta_lin_pct = (np.exp(delta_lin) - 1) * 100

    # Spline: read off curve at p5 and p95 directly
    c = spline_curve(X, W, Y, np.array([p5, p95]))
    delta_spl = c[1] - c[0]
    delta_spl_pct = (np.exp(delta_spl) - 1) * 100

    print(f"\n  5th pct = {p5:.4f}, 95th pct = {p95:.4f}")
    print(f"  Linear-in-W,  5th->95th effect: {delta_lin_pct:+.1f}%")
    print(f"  Spline,       5th->95th effect: {delta_spl_pct:+.1f}%")

    # Bootstrap both to get comparable CIs
    boots_lin, boots_spl = [], []
    rng = np.random.default_rng(SEED)
    n = len(W)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, Wb, Yb = X[idx], W[idx], Y[idx]
        try:
            a_lin, _ = dr_estimate_linear(Xb, Wb, Yb)
            p5b, p95b = np.percentile(Wb, [5, 95])
            boots_lin.append(a_lin * (p95b - p5b))
        except Exception:
            pass
        try:
            cb = spline_curve(Xb, Wb, Yb, np.array([p5, p95]))
            if np.all(np.isfinite(cb)):
                boots_spl.append(cb[1] - cb[0])
        except Exception:
            pass

    boots_lin, boots_spl = np.array(boots_lin), np.array(boots_spl)
    ci_lin = np.percentile(boots_lin, [2.5, 97.5])
    ci_spl = np.percentile(boots_spl, [2.5, 97.5])
    overlap = not (ci_lin[1] < ci_spl[0] or ci_spl[1] < ci_lin[0])

    print(f"  Linear bootstrap 95% CI: [{(np.exp(ci_lin[0])-1)*100:.1f}%, "
          f"{(np.exp(ci_lin[1])-1)*100:.1f}%]")
    print(f"  Spline bootstrap 95% CI: [{(np.exp(ci_spl[0])-1)*100:.1f}%, "
          f"{(np.exp(ci_spl[1])-1)*100:.1f}%]")
    print(f"  CIs overlap: {'YES — linear approximation broadly consistent with spline' if overlap else 'NO — spline captures material curvature'}")

    return {
        'delta_lin_pct': delta_lin_pct, 'delta_spl_pct': delta_spl_pct,
        'ci_lin': ci_lin, 'ci_spl': ci_spl, 'boots_lin': boots_lin,
        'boots_spl': boots_spl, 'consistent': overlap
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CHECK 2: PLACEBO TEST ON THE EXTENSIVE MARGIN (full sample)
# ─────────────────────────────────────────────────────────────────────────────

def placebo_test_extensive(X, T, Y, n_permutations=N_PERMUTATIONS):
    print("\n" + "=" * 60)
    print("CHECK 2: PLACEBO TEST — extensive margin (full sample)")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    ate_real, _ = aipw_binary(X, T, Y)

    placebo_ates = []
    for _ in range(n_permutations):
        T_perm = rng.permutation(T)
        try:
            a, _ = aipw_binary(X, T_perm, Y)
            placebo_ates.append(a)
        except Exception:
            pass
    placebo_ates = np.array(placebo_ates)
    p_value = (np.sum(np.abs(placebo_ates) >= np.abs(ate_real)) + 1) / (len(placebo_ates) + 1)

    print(f"\n  Real extensive-margin ATE : {ate_real:.4f}")
    print(f"  Placebo mean              : {placebo_ates.mean():.4f}")
    print(f"  Placebo SD                : {placebo_ates.std():.4f}")
    print(f"  Permutation p-value       : {p_value:.4f}")

    return {'ate_real': ate_real, 'placebo_ates': placebo_ates, 'p_value': p_value}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONFOUNDING DEMONSTRATION  (descriptive only — not a reported estimate)
# ─────────────────────────────────────────────────────────────────────────────

def confounding_demonstration(W, Y_raw, Y_log):
    print("\n" + "=" * 60)
    print("CONFOUNDING DEMONSTRATION (descriptive only, not a causal estimate)")
    print("=" * 60)

    corr, p_corr = sp_stats.pearsonr(W, Y_log)
    print(f"\n  Raw Pearson corr (true_discount, log sale): {corr:.4f}  p={p_corr:.4f}")

    bin_edges  = [0.0, 0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
    bin_labels = ['0% (none)', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    print(f"\n  Mean sale amount by true_discount level (descriptive bins):")
    q_stats = []
    for label, lo, hi in zip(bin_labels, bin_edges[:-1], bin_edges[1:]):
        mask = (W >= lo) & (W < hi)
        n = mask.sum()
        if n == 0:
            continue
        mean_sale = Y_raw[mask].mean()
        mean_disc = W[mask].mean()
        print(f"    {label}: mean sale=${mean_sale:.2f}  mean true_discount={mean_disc:.2%}  n={n:,}")
        q_stats.append({'bin': label, 'mean_sale': mean_sale, 'mean_discount': mean_disc, 'n': n})

    return q_stats


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(form_res, placebo_res, q_stats, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A — Linear vs spline bootstrap distributions
    ax = axes[0, 0]
    ax.hist((np.exp(form_res['boots_lin']) - 1) * 100, bins=40, alpha=0.6,
            color='#2E86AB', label='Linear-in-W', density=True)
    ax.hist((np.exp(form_res['boots_spl']) - 1) * 100, bins=40, alpha=0.6,
            color='#E74C3C', label='Spline', density=True)
    ax.axvline(form_res['delta_lin_pct'], color='#2E86AB', lw=2.5, linestyle='--')
    ax.axvline(form_res['delta_spl_pct'], color='#E74C3C', lw=2.5, linestyle='--')
    ax.set_xlabel('5th->95th percentile effect (%)')
    ax.set_ylabel('Density')
    ax.set_title(f'(a) Functional-Form Check: Linear vs Spline\n'
                 f'Consistent: {form_res["consistent"]}', fontweight='bold')
    ax.legend()

    # B — Placebo test (extensive margin)
    ax = axes[0, 1]
    ax.hist(placebo_res['placebo_ates'], bins=40, color='#95A5A6', alpha=0.8,
            edgecolor='black', lw=0.5, density=True, label='Placebo (permuted T)')
    ax.axvline(placebo_res['ate_real'], color='#E74C3C', lw=2.5, linestyle='--',
               label=f'Real ATE = {placebo_res["ate_real"]:.4f}')
    ax.axvline(0, color='black', lw=1, alpha=0.4)
    ax.set_xlabel('AIPW ATE (log scale)')
    ax.set_ylabel('Density')
    ax.set_title(f'(b) Placebo Test — Extensive Margin\np={placebo_res["p_value"]:.4f}',
                 fontweight='bold')
    ax.legend(fontsize=9)

    # C — Confounding: mean sale by true_discount bin (descriptive)
    ax = axes[1, 0]
    labels = [r['bin'] for r in q_stats]
    sales  = [r['mean_sale'] for r in q_stats]
    discs  = [r['mean_discount'] * 100 for r in q_stats]
    bars = ax.bar(range(len(labels)), sales, color='#8E44AD', alpha=0.8,
                  edgecolor='black', lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20)
    ax.set_ylabel('Mean Sale Amount ($)')
    ax.set_title('(c) Confounding Structure (descriptive only)\n'
                 'not used for causal estimates', fontweight='bold')
    ax2 = ax.twinx()
    ax2.plot(range(len(labels)), discs, 'ro-', lw=2, markersize=7)
    ax2.set_ylabel('Mean true_discount (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # D — summary text panel
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"Functional form: {'consistent' if form_res['consistent'] else 'divergent'}\n"
        f"  Linear delta : {form_res['delta_lin_pct']:+.1f}%\n"
        f"  Spline delta : {form_res['delta_spl_pct']:+.1f}%\n\n"
        f"Placebo (extensive margin):\n"
        f"  Real ATE : {placebo_res['ate_real']:.4f}\n"
        f"  p-value  : {placebo_res['p_value']:.4f}\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('CausalPipe-Transfer: Retail Robustness Checks (Two-Part Model)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'retail_robustness_figures.png')
    plt.savefig(path)
    print(f"\nFigure saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_results(form_res, placebo_res, q_stats, out_dir):
    report = f"""
=======================================================================
CAUSALPIPE-TRANSFER: RETAIL ROBUSTNESS CHECKS (TWO-PART MODEL)
FreshRetailNet-50K
=======================================================================

1. FUNCTIONAL-FORM CHECK (linear-in-W vs spline, discounted subsample)
------------------------------------------------------------------------
  Linear-in-W, 5th->95th effect : {form_res['delta_lin_pct']:+.1f}%
  Spline,      5th->95th effect : {form_res['delta_spl_pct']:+.1f}%
  Bootstrap CIs overlap         : {form_res['consistent']}
  Conclusion: {"The linear approximation is broadly consistent with the spline; the headline delta reported in causalpipe_retail_fixed.py is not an artifact of functional-form choice." if form_res['consistent'] else "The spline captures material curvature beyond a linear approximation; report the spline-based headline number, not a linear one."}

2. PLACEBO TEST — EXTENSIVE MARGIN (full sample, permuted treatment)
------------------------------------------------------------------------
  Real ATE (log scale) : {placebo_res['ate_real']:.4f}
  Placebo mean         : {placebo_res['placebo_ates'].mean():.4f}
  Placebo SD           : {placebo_res['placebo_ates'].std():.4f}
  Permutation p-value  : {placebo_res['p_value']:.4f}
  (See causalpipe_retail_fixed.py Part C for the corresponding placebo
  test on the intensive margin, discounted subsample only.)

3. CONFOUNDING DEMONSTRATION (descriptive only, NOT a causal estimate)
------------------------------------------------------------------------
{chr(10).join([f"  {r['bin']}: mean sale=${r['mean_sale']:.2f}  mean true_discount={r['mean_discount']:.2%}  n={r['n']:,}" for r in q_stats])}

  This bin-level table is retained purely as a diagnostic illustration
  of the selection-bias pattern (higher discounts applied to different
  products/contexts than full-price items) that motivates doubly-robust
  adjustment. It is NOT used to report a causal effect — see
  causalpipe_retail_fixed.py for the two-part causal estimates.
=======================================================================
"""
    print(report)
    path = os.path.join(out_dir, 'retail_robustness_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Results saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CAUSALPIPE-TRANSFER: RETAIL ROBUSTNESS CHECKS (TWO-PART MODEL)")
    print("=" * 70)

    df_raw = pd.read_excel(RETAIL_PATH)
    df_raw = df_raw[df_raw['sale_amount'] > 0].copy()
    Y_raw = df_raw['sale_amount'].values.astype(float)

    X, W_true, T, Y_log, feat_cols = load_retail(RETAIL_PATH)

    mask_disc = T == 1
    form_res = functional_form_check(X[mask_disc], W_true[mask_disc], Y_log[mask_disc])
    placebo_res = placebo_test_extensive(X, T, Y_log)
    q_stats = confounding_demonstration(W_true, Y_raw, Y_log)

    save_results(form_res, placebo_res, q_stats, OUT_DIR)
    make_figures(form_res, placebo_res, q_stats, OUT_DIR)

    print("\n" + "=" * 70)
    print("RETAIL ROBUSTNESS CHECKS COMPLETE")
    print(f"  Outputs -> {OUT_DIR}/retail_robustness_results.txt / _figures.png")
    print("=" * 70)

    return form_res, placebo_res, q_stats


if __name__ == "__main__":
    form_res, placebo_res, q_stats = main()
