# -*- coding: utf-8 -*-
"""
retail_final_two_part_spline_nozero_v2.py  (KBS R2 correction, 27 Oct 2026)
Version 2 = the FINAL retail script with zero-price rows dropped (the only change to the method),
plus two REPORT-ONLY additions needed for the revised text:
  PART 0  naive (unadjusted) comparison of discounted vs non-discounted rows, recomputed on the corrected sample
  PART 5  sensitivity: the same spline with product fixed effects (within-product), same bootstrap
          (check value from an independent re-run: about -4.1% for the 5th->95th contrast)
Run:  python retail_final_two_part_spline_nozero_v2.py     (about 15-30 min)
Output: results/retail_FINAL_two_part_spline_results_NOZERO.txt (run from the repository root; data path below)
 
CausalPipe-Transfer: FINAL two-part retail model (binary AIPW + spline
dose-response), replacing the quadratic curve and dropping discrete bins
entirely (Reviewer 1, points 12 and 13).
 
This is the proposed FINAL specification for the retail section:
 
  Part 1: Binary AIPW -- "any discount" (true_discount > 0) vs "no
          discount" (true_discount == 0). Same AIPW estimator already
          verified for the synthetic benchmark. (Before the R2 correction this gave +19.7% [19.2%, 20.2%];
          zero-price rows are now dropped.)
 
  Part 2: Smooth SPLINE dose-response curve (B-spline, degree=3, 6 knots --
          same specification as retail_subbin_spline.py, which is the
          reviewer's own explicit suggestion: "fit the spline you
          recommend yourselves"), fit ONCE on the full discounted
          subsample (true_discount > 0; n = 165,984 after dropping zero-price rows), evaluated at 11
          points spanning the 1st-99th percentile of the observed range
          (no extrapolation into near-empty regions). NO discrete bins --
          bins on a continuous treatment are demonstrably unstable here
          (confirmed twice now: the original 90-95% sub-segment
          sensitivity, and the -100%/+4668%/+1489% blowups in the last
          run), so the smooth curve is the only reported dose-response
          shape.
 
  Part 3: Permutation placebo test on the discounted subsample (unchanged
          from the prior run, reproduced here for completeness).
 
  Part 4: Summary effect -- moving true_discount from its 5th to 95th
          percentile WITHIN the discounted subsample, read directly off
          the spline curve, with bootstrap CI. This is the single
          headline number for "how much does discount depth matter,
          among items that are discounted at all."
 
Edit RETAIL_PATH below if needed.
"""
 
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, SplineTransformer
import warnings
warnings.filterwarnings('ignore')
 
RETAIL_PATH = os.path.join("data", "retail", "eval_split FreshRetailNet-50K.xlsx")
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "retail_FINAL_two_part_spline_results_NOZERO.txt")
N_BOOTSTRAP = 300
N_PLACEBO = 200
N_KNOTS = 6
SEED = 42
 
print("Loading FreshRetailNet dataset...")
df = pd.read_excel(RETAIL_PATH)
df = df[df['sale_amount'] > 0].copy()
print(f"  Non-zero transactions: {len(df):,}")
 
# ── ONLY CHANGE vs the previous FINAL script (KBS R1 correction) ──────────
# Drop rows whose recorded price ratio is exactly 0 (discount == 0). The
# price ratio 0 is a data error, not a 100% discount: before this change
# those rows were coded W = 1 - 0 = 1 (a "complete discount"). Thesis
# Chapters 3 and 4 already drop them. Rows with a missing discount value are
# kept, exactly as before. Nothing else in the method is changed.
n_before_drop = len(df)
df = df[df['discount'] != 0].copy()
n_zero_price = n_before_drop - len(df)
print(f"  Dropped zero-price rows (discount == 0): {n_zero_price:,}  "
      f"(expected 1,300) -> {len(df):,} rows remain (expected 334,137)")
 
FEATURE_COLS = ['stock_hour6_22_cnt', 'precpt', 'avg_temperature',
                'avg_humidity', 'avg_wind_level', 'holiday_flag', 'activity_flag']
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
 
Y_all = np.log1p(df['sale_amount'].values.astype(float))
X_raw_all = df[FEATURE_COLS].values.astype(float)
X_all = StandardScaler().fit_transform(X_raw_all)
W_true_all = 1.0 - df['discount'].values.astype(float)
rng = np.random.default_rng(SEED)
 
T_all = (W_true_all > 1e-9).astype(int)
PROD_all = pd.factorize(df['product_id'])[0] if 'product_id' in df.columns else None
n_treated, n_control = T_all.sum(), (1 - T_all).sum()
 
lines = []
lines.append("=" * 78)
lines.append("FINAL RETAIL MODEL: binary AIPW + spline dose-response (no discrete bins)")
lines.append(f"zero-price rows dropped (discount == 0): {n_zero_price:,}")
lines.append(f"n = {len(Y_all):,} total | discounted = {n_treated:,} | "
             f"no discount = {n_control:,}")
lines.append("=" * 78)
print(f"\nDiscounted: {n_treated:,}  |  No discount: {n_control:,}")
 
# ─────────────────────────────────────────────────────────────────────────
# PART 0: Naive (unadjusted) comparison -- REPORT ONLY (for the "naive vs adjusted" sentence)
# ─────────────────────────────────────────────────────────────────────────
lines.append("\nPART 0: Naive (unadjusted) comparison, discounted vs no discount")
lines.append("-" * 78)
sa = df['sale_amount'].values.astype(float)
naive_log = Y_all[T_all == 1].mean() - Y_all[T_all == 0].mean()
for r in (f"  Mean sale amount: discounted {sa[T_all == 1].mean():.4f}  vs  no discount {sa[T_all == 0].mean():.4f}",
          f"  Naive difference on the log scale: {naive_log:+.4f}  = {(np.exp(naive_log) - 1) * 100:+.1f}%"):
    print(r); lines.append(r)
 
# ─────────────────────────────────────────────────────────────────────────
# PART 1: Binary AIPW (reproduced for a self-contained final script)
# ─────────────────────────────────────────────────────────────────────────
lines.append("\nPART 1: Binary AIPW -- any discount vs none")
lines.append("-" * 78)
print("\nPart 1: binary AIPW...")
 
 
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
 
 
ate_bin, se_bin = aipw_binary(X_all, T_all, Y_all)
ate_bin_pct = (np.exp(ate_bin) - 1) * 100
ci_lo_log, ci_hi_log = ate_bin - 1.96 * se_bin, ate_bin + 1.96 * se_bin
ci_lo_pct, ci_hi_pct = (np.exp(ci_lo_log) - 1) * 100, (np.exp(ci_hi_log) - 1) * 100
row = f"  ATE = {ate_bin_pct:+.1f}%  95% CI=[{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%]"
print(row); lines.append(row)
 
# ─────────────────────────────────────────────────────────────────────────
# PART 2: Spline dose-response, discounted subsample only, no bins
# ─────────────────────────────────────────────────────────────────────────
mask_disc = T_all == 1
X, Y, W = X_all[mask_disc], Y_all[mask_disc], W_true_all[mask_disc]
print(f"\nPart 2: spline dose-response, discounted subsample n={len(Y):,}")
lines.append(f"\nPART 2: Spline dose-response (discounted subsample, n={len(Y):,})")
lines.append("-" * 78)
lines.append(f"  Observed true_discount range: [{W.min():.4f}, {W.max():.4f}], "
             f"mean={W.mean():.4f}, median={np.median(W):.4f}")
print(lines[-1])
 
p1, p99 = np.percentile(W, [1, 99])
eval_points = np.linspace(p1, p99, 11)
lines.append(f"  Evaluating curve over [{p1:.4f}, {p99:.4f}] (1st-99th percentile, "
             "no extrapolation)")
print(lines[-1])
 
 
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
    curve_log = B_eval_centered @ coef
    return curve_log
 
 
curve_log = spline_curve(X, W, Y, eval_points)
curve_pct = (np.exp(curve_log) - 1) * 100
 
boot_curves = []
n_total = len(W)
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n_total, size=n_total)
    try:
        c = spline_curve(X[idx], W[idx], Y[idx], eval_points)
        if np.all(np.isfinite(c)):
            boot_curves.append(c)
    except Exception:
        pass
boot_curves = np.array(boot_curves)
ci_lo_curve = np.percentile(boot_curves, 2.5, axis=0)
ci_hi_curve = np.percentile(boot_curves, 97.5, axis=0)
 
lines.append(f"\n  {'true_discount':>14} {'Effect (rel. to mean)':>22} {'95% CI':>26}")
for i, w in enumerate(eval_points):
    lo_p = (np.exp(ci_lo_curve[i]) - 1) * 100
    hi_p = (np.exp(ci_hi_curve[i]) - 1) * 100
    row = f"  {w:>14.4f} {curve_pct[i]:>20.1f}%   [{lo_p:.1f}%, {hi_p:.1f}%]"
    print(row); lines.append(row)
 
# ─────────────────────────────────────────────────────────────────────────
# PART 3: Placebo test, discounted subsample (reproduced for completeness)
# ─────────────────────────────────────────────────────────────────────────
lines.append(f"\nPART 3: Permutation placebo test ({N_PLACEBO} shuffles), discounted "
             "subsample")
lines.append("-" * 78)
print(f"\nPart 3: placebo test ({N_PLACEBO} shuffles)...")
 
 
def dr_estimate_linear(X, W, Y):
    om = LinearRegression().fit(X, Y)
    tm = LinearRegression().fit(X, W)
    Y_res = Y - om.predict(X)
    W_res = W - tm.predict(X)
    return (W_res * Y_res).mean() / (W_res ** 2).mean()
 
 
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
placebo_mean, placebo_sd = placebo_ates.mean(), placebo_ates.std()
p_value = (np.sum(np.abs(placebo_ates) >= np.abs(real_ate)) + 1) / (len(placebo_ates) + 1)
row1 = f"  Real ATE (log scale, linear-in-W on discounted subsample): {real_ate:+.4f}"
row2 = f"  Placebo null: mean={placebo_mean:+.4f}, SD={placebo_sd:.4f}"
row3 = f"  Permutation p-value: {p_value:.4f}"
for r in (row1, row2, row3):
    print(r); lines.append(r)
 
# ─────────────────────────────────────────────────────────────────────────
# PART 4: Headline summary -- 5th to 95th percentile, read off the spline
# ─────────────────────────────────────────────────────────────────────────
lines.append("\nPART 4: Headline summary -- effect of discount DEPTH, 5th to 95th "
             "percentile (within discounted items)")
lines.append("-" * 78)
print("\nPart 4: 5th-95th percentile summary effect...")
 
p5, p95 = np.percentile(W, [5, 95])
 
 
def delta_p5_p95(Xb, Wb, Yb):
    lo_w, hi_w = np.percentile(Wb, [5, 95])
    c = spline_curve(Xb, Wb, Yb, np.array([lo_w, hi_w]))
    return c[1] - c[0]
 
 
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
 
row = f"  5th pct = {p5:.4f}, 95th pct = {p95:.4f}"
print(row); lines.append(row)
row = f"  Effect of moving from {p5:.4f} to {p95:.4f}: {delta_pct:+.1f}%  " \
      f"95% CI=[{d_lo_pct:.1f}%, {d_hi_pct:.1f}%]"
print(row); lines.append(row)
 
# ─────────────────────────────────────────────────────────────────────────
# PART 5: Sensitivity -- same spline with PRODUCT FIXED EFFECTS (REPORT ONLY)
# Y, the covariates and the spline basis are demeaned within product before the same partialling-out
# regression; same 5th->95th contrast, same row bootstrap (300 draws).
# ─────────────────────────────────────────────────────────────────────────
fe_txt = "  product_id not in the file: skipped"
if PROD_all is not None:
    lines.append("\nPART 5: Sensitivity -- spline with product fixed effects (discounted subsample)")
    lines.append("-" * 78)
    print("\nPart 5: spline with product fixed effects...")
    g_all = pd.factorize(PROD_all[mask_disc])[0]
 
    def demean(M, g):
        G = g.max() + 1
        M = M.reshape(len(M), -1).astype(float)
        cnt = np.maximum(np.bincount(g, minlength=G), 1)
        out = np.empty_like(M)
        for j in range(M.shape[1]):
            out[:, j] = M[:, j] - (np.bincount(g, weights=M[:, j], minlength=G) / cnt)[g]
        return out
 
    def delta_fe(Xb, Wb, Yb, gb):
        lo_w, hi_w = np.percentile(Wb, [5, 95])
        spl = SplineTransformer(n_knots=N_KNOTS, degree=3, include_bias=False)
        B = spl.fit_transform(Wb.reshape(-1, 1))
        Xd, Yd, Bd = demean(Xb, gb), demean(Yb, gb)[:, 0], demean(B, gb)
        Y_res = Yd - LinearRegression().fit(Xd, Yd).predict(Xd)
        B_res = np.column_stack([Bd[:, j] - LinearRegression().fit(Xd, Bd[:, j]).predict(Xd) for j in range(Bd.shape[1])])
        coef, *_ = np.linalg.lstsq(B_res, Y_res, rcond=None)
        Be = spl.transform(np.array([[lo_w], [hi_w]]))
        return (Be[1] - Be[0]) @ coef
 
    d_fe = delta_fe(X, W, Y, g_all)
    bfe = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(Y), size=len(Y))
        try:
            d = delta_fe(X[idx], W[idx], Y[idx], pd.factorize(g_all[idx])[0])
            if np.isfinite(d):
                bfe.append(d)
        except Exception:
            pass
    fe_lo, fe_hi = np.percentile(bfe, [2.5, 97.5])
    fe_txt = (f"  With product fixed effects, 5th->95th: {(np.exp(d_fe) - 1) * 100:+.1f}%  "
              f"95% CI=[{(np.exp(fe_lo) - 1) * 100:.1f}%, {(np.exp(fe_hi) - 1) * 100:.1f}%]  "
              f"({g_all.max() + 1} products; check value about -4.1%)")
    print(fe_txt); lines.append(fe_txt)
 
lines.append("\n" + "=" * 78)
lines.append("SUMMARY OF FINAL NUMBERS FOR THE RETAIL SECTION")
lines.append("=" * 78)
lines.append(f"  1. Any discount vs none (binary AIPW):        {ate_bin_pct:+.1f}% "
             f"[{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%]")
lines.append(f"  2. Discount-depth effect, 5th->95th pctile:   {delta_pct:+.1f}% "
             f"[{d_lo_pct:.1f}%, {d_hi_pct:.1f}%]  (among discounted items)")
lines.append(f"  3. Placebo test p-value (discounted subsample): {p_value:.4f}")
lines.append("  4. Full dose-response shape: see spline curve table above (Part 2)")
lines.append(f"  5. Naive difference (log scale): {(np.exp(naive_log) - 1) * 100:+.1f}%  (Part 0)")
lines.append("  6. Sensitivity, product fixed effects:" + fe_txt.split(":", 1)[-1])
 
report = "\n".join(lines)
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\nSaved to: {OUT_FILE}")
print("\nPASTE THE FULL CONSOLE OUTPUT back -- do not summarize it yourself.")
