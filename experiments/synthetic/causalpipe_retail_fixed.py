# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Retail Analysis (FreshRetailNet-50K) — TWO-PART MODEL
============================================================================
REVISION NOTE (replaces the previous single continuous-treatment DR model):

  1. Discount field correction. The raw `discount` column is defined by the
     dataset as 1.0 = full price / 0.9 = 10% off — i.e. it is the INVERSE
     of a discount rate, not the discount rate itself. The corrected
     treatment variable used throughout is:
         true_discount = 1 - discount
     (Previous versions of this script fed the raw `discount` column
     directly into the estimator, which reversed the sign of the
     headline causal conclusion.)

  2. Discrete bins dropped entirely. `true_discount` has a massive point
     mass at 0 (most transactions are sold at full price). Feeding this
     directly into a single continuous-treatment DR estimator, or binning
     it into discrete discount levels, produces numerically unstable
     estimates (near-zero treatment variance within bins near the mass
     point produced swings as large as -100%/+4668% in earlier runs).
     Discrete bins are no longer used anywhere in this script.

  3. TWO-PART MODEL (the standard fix for a treatment with a point mass
     at zero):
       Part A — Extensive margin: binary AIPW estimate of "any discount"
                (true_discount > 0) vs "no discount" (true_discount == 0),
                using the same AIPW estimator verified on the synthetic
                benchmark.
       Part B — Intensive margin: a smooth B-spline (degree=3, 6 knots)
                dose-response curve for discount DEPTH, fit only on the
                discounted subsample (true_discount > 0), evaluated over
                the 1st-99th percentile of the observed range (no
                extrapolation into near-empty regions).
       Part C — Permutation placebo test on the discounted subsample.
       Part D — Headline summary effect: moving true_discount from its
                5th to 95th percentile WITHIN the discounted subsample,
                read directly off the spline curve, with bootstrap CI.

Outputs (to OUT_DIR):
  retail_results.txt
  retail_figures.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, SplineTransformer
import warnings
warnings.filterwarnings('ignore')

RETAIL_PATH = os.path.join('data', 'retail', 'eval_split FreshRetailNet-50K.xlsx')
OUT_DIR     = os.path.join('results')
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOTSTRAP = 300
N_PLACEBO   = 200
N_KNOTS     = 6
SEED        = 42

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

FEATURE_COLS = ['stock_hour6_22_cnt', 'precpt', 'avg_temperature',
                 'avg_humidity', 'avg_wind_level', 'holiday_flag', 'activity_flag']


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & PREPARE
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

    # Discount-field correction (Reviewer 1, point 12): raw `discount` is
    # 1.0=full price / 0.9=10% off, i.e. the inverse of a discount rate.
    W_true = 1.0 - df['discount'].values.astype(float)
    T = (W_true > 1e-9).astype(int)

    print(f"  Features used   : {feature_cols}")
    print(f"  true_discount   : mean={W_true.mean():.2%}  "
          f"range=[{W_true.min():.2f}, {W_true.max():.2f}]")
    print(f"  Discounted (T=1): {T.sum():,}   No discount (T=0): {(1 - T).sum():,}")
    return X, W_true, T, Y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PART A — EXTENSIVE MARGIN: BINARY AIPW
# ─────────────────────────────────────────────────────────────────────────────

def aipw_binary(X, T, Y):
    """AIPW DR estimator for a binary treatment (identical formula to the
    synthetic-benchmark / sepsis scripts)."""
    n = len(X)
    XT = np.column_stack([X, T])
    pm = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(X, T)
    om = Ridge(alpha=1.0).fit(XT, Y)
    e = np.clip(pm.predict_proba(X)[:, 1], 0.05, 0.95)
    XT1 = np.column_stack([X, np.ones(n)])
    XT0 = np.column_stack([X, np.zeros(n)])
    mu1, mu0 = om.predict(XT1), om.predict(XT0)
    psi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return psi.mean(), psi.std(ddof=1) / np.sqrt(n), psi


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PART B — INTENSIVE MARGIN: SPLINE DOSE-RESPONSE (discounted subsample)
# ─────────────────────────────────────────────────────────────────────────────

def spline_curve(X, W, Y, eval_pts, n_knots=N_KNOTS):
    """Partially-linear DR with a B-spline basis for the continuous
    treatment W (true_discount), residualized against X."""
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


def dr_estimate_linear(X, W, Y):
    om = LinearRegression().fit(X, Y)
    tm = LinearRegression().fit(X, W)
    Y_res = Y - om.predict(X)
    W_res = W - tm.predict(X)
    return (W_res * Y_res).mean() / (W_res ** 2).mean()


def delta_p5_p95(Xb, Wb, Yb):
    lo_w, hi_w = np.percentile(Wb, [5, 95])
    c = spline_curve(Xb, Wb, Yb, np.array([lo_w, hi_w]))
    return c[1] - c[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(eval_points, curve_pct, ci_lo_curve, ci_hi_curve,
                  placebo_ates, real_ate, ate_bin_pct, ci_lo_pct, ci_hi_pct,
                  delta_pct, d_lo_pct, d_hi_pct, out_dir):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A — Extensive margin: any discount vs none
    ax = axes[0, 0]
    ax.bar(['No discount\n(reference)', 'Any discount'],
           [0, ate_bin_pct],
           yerr=[0, (ci_hi_pct - ci_lo_pct) / 2],
           capsize=10, color=['#95A5A6', '#27AE60'], alpha=0.85,
           edgecolor='black', lw=1.2)
    ax.axhline(0, color='black', lw=1, alpha=0.4)
    ax.set_ylabel('Effect on log(1+sale) (%)')
    ax.set_title(f'(a) Extensive Margin: Any Discount vs None\n'
                 f'ATE={ate_bin_pct:+.1f}%  95% CI=[{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%]',
                 fontweight='bold')

    # B — Spline dose-response curve on discounted subsample
    ax = axes[0, 1]
    ax.plot(eval_points, curve_pct, color='#2E86AB', lw=2.5, marker='o')
    ax.fill_between(eval_points,
                     (np.exp(ci_lo_curve) - 1) * 100,
                     (np.exp(ci_hi_curve) - 1) * 100,
                     alpha=0.2, color='#2E86AB')
    ax.axhline(0, color='black', lw=1, alpha=0.4)
    ax.set_xlabel('true_discount (1 - discount)')
    ax.set_ylabel('Effect on log(1+sale) (%, rel. to mean)')
    ax.set_title('(b) Intensive Margin: Spline Dose-Response\n'
                 '(discounted subsample only, no discrete bins)', fontweight='bold')

    # C — Placebo test (discounted subsample, linear-in-W)
    ax = axes[1, 0]
    ax.hist(placebo_ates, bins=40, color='#95A5A6', alpha=0.8,
            edgecolor='black', lw=0.5, density=True, label='Placebo (shuffled W)')
    ax.axvline(real_ate, color='#E74C3C', lw=2.5, linestyle='--',
               label=f'Real ATE = {real_ate:.4f}')
    ax.axvline(0, color='black', lw=1, alpha=0.4)
    ax.set_xlabel('DR ATE (log scale, linear-in-W)')
    ax.set_ylabel('Density')
    ax.set_title('(c) Permutation Placebo Test\n(discounted subsample)', fontweight='bold')
    ax.legend(fontsize=9)

    # D — Headline summary: 5th to 95th percentile of discount depth
    ax = axes[1, 1]
    ax.bar(['Discount depth\n5th -> 95th pctile'], [delta_pct],
           yerr=[(d_hi_pct - d_lo_pct) / 2], capsize=10,
           color='#8E44AD', alpha=0.85, edgecolor='black', lw=1.2)
    ax.axhline(0, color='black', lw=1, alpha=0.4)
    ax.set_ylabel('Effect (%)')
    ax.set_title(f'(d) Headline: Discount-Depth Effect\n'
                 f'{delta_pct:+.1f}%  95% CI=[{d_lo_pct:.1f}%, {d_hi_pct:.1f}%]',
                 fontweight='bold')

    plt.suptitle('CausalPipe-Transfer: Retail Two-Part Model (Extensive + Intensive Margin)',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'retail_figures.png')
    plt.savefig(path)
    print(f"\nFigure saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CAUSALPIPE-TRANSFER: RETAIL TWO-PART MODEL")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    X_all, W_true_all, T_all, Y_all, feature_cols = load_retail(RETAIL_PATH)

    lines = []
    lines.append("=" * 78)
    lines.append("RETAIL TWO-PART MODEL: binary AIPW + spline dose-response (no discrete bins)")
    lines.append(f"n = {len(Y_all):,} total | discounted = {T_all.sum():,} | "
                 f"no discount = {(1 - T_all).sum():,}")
    lines.append("=" * 78)

    # ── Part A: extensive margin ────────────────────────────────────────
    print("\nPart A: binary AIPW (extensive margin)...")
    ate_bin, se_bin, _ = aipw_binary(X_all, T_all, Y_all)
    ate_bin_pct = (np.exp(ate_bin) - 1) * 100
    ci_lo_log, ci_hi_log = ate_bin - 1.96 * se_bin, ate_bin + 1.96 * se_bin
    ci_lo_pct, ci_hi_pct = (np.exp(ci_lo_log) - 1) * 100, (np.exp(ci_hi_log) - 1) * 100
    lines.append("\nPART A: Extensive margin (any discount vs none)")
    lines.append(f"  ATE = {ate_bin_pct:+.1f}%  95% CI=[{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%]")
    print(lines[-1])

    # ── Part B: intensive margin spline curve ───────────────────────────
    mask_disc = T_all == 1
    X, Y, W = X_all[mask_disc], Y_all[mask_disc], W_true_all[mask_disc]
    print(f"\nPart B: spline dose-response (discounted subsample, n={len(Y):,})...")
    p1, p99 = np.percentile(W, [1, 99])
    eval_points = np.linspace(p1, p99, 11)
    curve_log = spline_curve(X, W, Y, eval_points)
    curve_pct = (np.exp(curve_log) - 1) * 100

    boot_curves = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(W), size=len(W))
        try:
            c = spline_curve(X[idx], W[idx], Y[idx], eval_points)
            if np.all(np.isfinite(c)):
                boot_curves.append(c)
        except Exception:
            pass
    boot_curves = np.array(boot_curves)
    ci_lo_curve = np.percentile(boot_curves, 2.5, axis=0)
    ci_hi_curve = np.percentile(boot_curves, 97.5, axis=0)

    lines.append(f"\nPART B: Intensive margin (spline dose-response, n={len(Y):,})")
    lines.append(f"  {'true_discount':>14} {'Effect (%)':>14} {'95% CI':>22}")
    for i, w in enumerate(eval_points):
        lo_p = (np.exp(ci_lo_curve[i]) - 1) * 100
        hi_p = (np.exp(ci_hi_curve[i]) - 1) * 100
        row = f"  {w:>14.4f} {curve_pct[i]:>13.1f}%   [{lo_p:.1f}%, {hi_p:.1f}%]"
        print(row); lines.append(row)

    # ── Part C: placebo test ─────────────────────────────────────────────
    print(f"\nPart C: placebo test ({N_PLACEBO} shuffles)...")
    real_ate = dr_estimate_linear(X, W, Y)
    placebo_ates = []
    for _ in range(N_PLACEBO):
        W_shuffled = rng.permutation(W)
        try:
            a = dr_estimate_linear(X, W_shuffled, Y)
            if np.isfinite(a):
                placebo_ates.append(a)
        except Exception:
            pass
    placebo_ates = np.array(placebo_ates)
    p_value = (np.sum(np.abs(placebo_ates) >= np.abs(real_ate)) + 1) / (len(placebo_ates) + 1)
    lines.append(f"\nPART C: Permutation placebo test ({N_PLACEBO} shuffles, discounted subsample)")
    lines.append(f"  Real ATE (log, linear-in-W): {real_ate:+.4f}")
    lines.append(f"  Placebo null: mean={placebo_ates.mean():+.4f}, SD={placebo_ates.std():.4f}")
    lines.append(f"  Permutation p-value: {p_value:.4f}")
    print(lines[-3]); print(lines[-2]); print(lines[-1])

    # ── Part D: headline summary (5th -> 95th percentile) ────────────────
    print("\nPart D: headline 5th-95th percentile summary...")
    p5, p95 = np.percentile(W, [5, 95])
    delta_log = delta_p5_p95(X, W, Y)
    delta_pct = (np.exp(delta_log) - 1) * 100
    boots_delta = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(Y), size=len(Y))
        try:
            d = delta_p5_p95(X[idx], W[idx], Y[idx])
            if np.isfinite(d):
                boots_delta.append(d)
        except Exception:
            pass
    boots_delta = np.array(boots_delta)
    d_lo, d_hi = np.percentile(boots_delta, [2.5, 97.5])
    d_lo_pct, d_hi_pct = (np.exp(d_lo) - 1) * 100, (np.exp(d_hi) - 1) * 100
    lines.append(f"\nPART D: Headline summary (discount depth, 5th->95th percentile)")
    lines.append(f"  5th pct = {p5:.4f}, 95th pct = {p95:.4f}")
    lines.append(f"  Effect: {delta_pct:+.1f}%  95% CI=[{d_lo_pct:.1f}%, {d_hi_pct:.1f}%]")
    print(lines[-2]); print(lines[-1])

    lines.append("\n" + "=" * 78)
    lines.append("SUMMARY OF FINAL NUMBERS FOR THE RETAIL SECTION")
    lines.append("=" * 78)
    lines.append(f"  1. Extensive margin (any discount vs none):  {ate_bin_pct:+.1f}% "
                 f"[{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%]")
    lines.append(f"  2. Intensive margin, 5th->95th pctile:       {delta_pct:+.1f}% "
                 f"[{d_lo_pct:.1f}%, {d_hi_pct:.1f}%]  (among discounted items)")
    lines.append(f"  3. Placebo p-value (discounted subsample):   {p_value:.4f}")

    report = "\n".join(lines)
    print(report[-500:])  # tail already printed incrementally above
    out_path = os.path.join(OUT_DIR, 'retail_results.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nResults saved: {out_path}")

    make_figures(eval_points, curve_pct, ci_lo_curve, ci_hi_curve,
                 placebo_ates, real_ate, ate_bin_pct, ci_lo_pct, ci_hi_pct,
                 delta_pct, d_lo_pct, d_hi_pct, OUT_DIR)

    return {
        'ate_extensive_pct': ate_bin_pct, 'ci_extensive': (ci_lo_pct, ci_hi_pct),
        'delta_intensive_pct': delta_pct, 'ci_intensive': (d_lo_pct, d_hi_pct),
        'placebo_p_value': p_value,
    }


if __name__ == "__main__":
    results = main()
