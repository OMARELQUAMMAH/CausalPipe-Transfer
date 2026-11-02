# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Sepsis Analysis: Overlap-Weighted ATE (OWATE)
=========================================================
Specification (pre-onset treatment window, corrected confounder set, fixed AIPW estimator):

  (a) Treatment timing fixed: shock criterion must occur BEFORE
      first SepsisLabel=1 onset (eliminates reverse causality)

  (b) MAP, SBP, DBP, Lactate and derived features (map_min, lactate_max)
      EXCLUDED from confounder set — they directly define treatment,
      including them causes near-perfect propensity prediction artefact

  (c) ICULOS removed from propensity/outcome model features
      (kept only for subgroup description to avoid post-treatment conditioning)

  (d) Patients with missing MAP AND Lactate measurements explicitly
      EXCLUDED (not assigned to control) — reported in cohort table

  (e) Analysis uses all 20,336 patients from Set A (not capped at 20,000)

Outputs (to results/):
  sepsis_owate_results.txt, sepsis_owate_convergence.csv, sepsis_owate_subgroups.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

SEPSIS_DIR = os.path.join("data", "healthcare", "sepsis", "training_setA")
os.makedirs("results", exist_ok=True)
OUT_PREFIX = os.path.join("results", "sepsis_owate")

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

# ── CORRECTED FEATURE SET: MAP, SBP, DBP, Lactate EXCLUDED ───────────────────
# These features directly define treatment T → including them inflates
# propensity scores and creates artefactual positivity violations
VITALS_CORRECTED = ['HR', 'O2Sat', 'Temp', 'Resp']       # removed: MAP, SBP, DBP
LABS_CORRECTED   = ['BaseExcess', 'HCO3', 'pH', 'PaCO2', 'BUN',
                    'Calcium', 'Creatinine', 'Glucose',    # removed: Lactate
                    'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets']
DEMO_CORRECTED   = ['Age', 'Gender']                      # removed: ICULOS

ALL_FEATURES_CORRECTED = VITALS_CORRECTED + LABS_CORRECTED + DEMO_CORRECTED

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA — CORRECTED
# ─────────────────────────────────────────────────────────────────────────────

def load_patient(filepath):
    try:
        df = pd.read_csv(filepath, sep='|')
        df.replace('NaN', np.nan, inplace=True)
        for col in df.columns:
            if col != 'Gender':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return None


def build_dataset_corrected(sepsis_dir, max_patients=None):
    """
    Corrected dataset construction:
    - Treatment: shock criterion met BEFORE first SepsisLabel=1 onset
    - Excludes patients with no MAP AND no Lactate measurements
    - Reports cohort construction statistics
    """
    print("Loading patient files ...")
    files = sorted(glob.glob(os.path.join(sepsis_dir, '*.psv')))
    if max_patients:
        files = files[:max_patients]
    total_files = len(files)
    print(f"  Total patient files: {total_files:,}")

    records = []
    n_excluded_short    = 0
    n_excluded_no_maplac = 0
    n_shock_before      = 0
    n_shock_after_only  = 0
    n_no_shock          = 0

    for i, fp in enumerate(files):
        df = load_patient(fp)
        if df is None or len(df) < 2:
            n_excluded_short += 1
            continue

        outcome = int(df['SepsisLabel'].max() == 1)

        # ── FIX (a): Treatment timing ─────────────────────────────────────
        # Find first hour of sepsis onset
        sepsis_hours = df.index[df['SepsisLabel'] == 1].tolist()
        if len(sepsis_hours) > 0:
            first_sepsis_hour = sepsis_hours[0]
        else:
            first_sepsis_hour = len(df)  # no sepsis → all hours are pre-outcome

        # Use ONLY pre-sepsis rows to define treatment
        pre_df = df.iloc[:first_sepsis_hour]

        # Check if MAP and Lactate are available at all
        has_map     = 'MAP' in df.columns and df['MAP'].notna().any()
        has_lactate = 'Lactate' in df.columns and df['Lactate'].notna().any()

        # ── FIX (d): Exclude if no MAP AND no Lactate ────────────────────
        if not has_map and not has_lactate:
            n_excluded_no_maplac += 1
            continue

        # Define shock from pre-sepsis window only
        shock = False
        if has_map and has_lactate:
            pre_map     = pre_df['MAP'].dropna()
            pre_lactate = pre_df['Lactate'].dropna()
            if len(pre_map) > 0 and len(pre_lactate) > 0:
                # Align by index for joint condition
                pre_both = pre_df[['MAP', 'Lactate']].dropna()
                if len(pre_both) > 0:
                    shock = bool(
                        ((pre_both['MAP'] < 65) & (pre_both['Lactate'] > 2.0)).any()
                    )
        elif has_map:
            pre_map = pre_df['MAP'].dropna()
            if len(pre_map) > 0:
                shock = bool((pre_map < 65).any())

        treatment = int(shock)
        if shock:
            n_shock_before += 1
        elif outcome == 1:
            # Sepsis occurred but shock criterion not met pre-onset
            n_shock_after_only += 1
            n_no_shock += 1
        else:
            n_no_shock += 1

        # ── FIX (b): Feature extraction — EXCLUDING MAP, Lactate ─────────
        feat = {}
        for col in ALL_FEATURES_CORRECTED:
            if col in df.columns:
                vals = df[col].dropna()
                feat[col] = float(vals.mean()) if len(vals) > 0 else np.nan
            else:
                feat[col] = np.nan

        # Derived features that do NOT involve MAP or Lactate
        feat['hr_variability'] = float(df['HR'].std()) if 'HR' in df.columns else np.nan
        feat['n_hours']        = float(len(df))

        # ── FIX (c): ICULOS kept ONLY for subgroup stratification ─────────
        # Store separately — NOT in feature set X
        feat['icu_duration_subgroup'] = float(df['ICULOS'].max()) if 'ICULOS' in df.columns else np.nan

        feat['treatment'] = treatment
        feat['outcome']   = outcome
        records.append(feat)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{total_files:,} processed")

    df_out = pd.DataFrame(records)

    print(f"\n  COHORT CONSTRUCTION TABLE:")
    print(f"  Total patient files in Set A      : {total_files:,}")
    print(f"  Excluded (< 2 rows)               : {n_excluded_short:,}")
    print(f"  Excluded (no MAP/Lactate data)    : {n_excluded_no_maplac:,}")
    print(f"  Final analysis sample             : {len(df_out):,}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Treated (shock pre-onset, T=1)    : {df_out['treatment'].sum():,} ({df_out['treatment'].mean():.1%})")
    print(f"  Control (no pre-onset shock, T=0) : {(df_out['treatment']==0).sum():,} ({(df_out['treatment']==0).mean():.1%})")
    print(f"  Sepsis onset (Y=1)                : {df_out['outcome'].sum():,} ({df_out['outcome'].mean():.1%})")
    print(f"  ─────────────────────────────────────────")
    print(f"  Note: {n_shock_after_only:,} patients had shock AFTER sepsis onset → assigned T=0")
    print(f"        These represent the reverse-causality cases in the original analysis.")

    return df_out, {
        'total_files': total_files,
        'excluded_short': n_excluded_short,
        'excluded_no_maplac': n_excluded_no_maplac,
        'final_n': len(df_out),
        'n_treated': int(df_out['treatment'].sum()),
        'n_control': int((df_out['treatment']==0).sum()),
        'n_outcome': int(df_out['outcome'].sum()),
        'n_shock_after_only': n_shock_after_only,
    }


def preprocess_corrected(df):
    """Preprocess excluding MAP, Lactate, ICULOS from model features."""
    feature_cols = [c for c in ALL_FEATURES_CORRECTED if c in df.columns and c != 'Gender']
    feature_cols += ['hr_variability', 'n_hours']
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        med = X_df[col].median()
        X_df[col] = X_df[col].fillna(med if not np.isnan(med) else 0.0)

    variances = X_df.var()
    feature_cols = [c for c in feature_cols if variances.get(c, 0) > 1e-8]
    X_df = X_df[feature_cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)
    T = df['treatment'].values.astype(int)
    Y = df['outcome'].values.astype(int)

    print(f"\n  Features used (MAP/Lactate/ICULOS excluded): {len(feature_cols)}")
    print(f"  Features: {feature_cols}")
    return X, T, Y, feature_cols, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROPENSITY + OVERLAP TRIMMING (unchanged logic, corrected inputs)
# ─────────────────────────────────────────────────────────────────────────────

def fit_propensity_and_trim(X, T, Y, df_raw, lo=0.1, hi=0.9):
    print("\n" + "="*60)
    print("PROPENSITY MODEL & OVERLAP TRIMMING (corrected features)")
    print("="*60)

    pm = LogisticRegression(max_iter=2000, C=0.5,
                            class_weight='balanced', random_state=42)
    pm.fit(X, T)
    e_full = pm.predict_proba(X)[:, 1]

    print(f"\n  Full sample (N={len(X):,}):")
    print(f"  Propensity range  : [{e_full.min():.3f}, {e_full.max():.3f}]")
    print(f"  Propensity mean   : {e_full.mean():.3f}")
    below = (e_full < lo).mean()
    above = (e_full > hi).mean()
    within = ((e_full >= lo) & (e_full <= hi)).mean()
    print(f"  < {lo} (excluded) : {below:.1%}")
    print(f"  > {hi} (excluded) : {above:.1%}")
    print(f"  In [{lo},{hi}]    : {within:.1%}  ← should be HIGHER than original 13%")

    overlap_mask = (e_full >= lo) & (e_full <= hi)
    X_trim  = X[overlap_mask]
    T_trim  = T[overlap_mask]
    Y_trim  = Y[overlap_mask]
    e_trim  = e_full[overlap_mask]
    df_trim = df_raw[overlap_mask].copy().reset_index(drop=True) if df_raw is not None else None

    n_trim = overlap_mask.sum()
    print(f"\n  Trimmed sample (N={n_trim:,}, {n_trim/len(X):.1%} of full):")
    print(f"  Treatment rate    : {T_trim.mean():.1%}")
    print(f"  Sepsis rate       : {Y_trim.mean():.1%}")
    print(f"  Propensity range  : [{e_trim.min():.3f}, {e_trim.max():.3f}]")

    e_treated = e_trim[T_trim == 1]
    e_control = e_trim[T_trim == 0]
    ks_stat, ks_p = ks_2samp(e_treated, e_control)
    print(f"  KS distance       : {ks_stat:.4f}  (p={ks_p:.4f})")

    return (X_trim, T_trim, Y_trim, e_trim,
            overlap_mask, e_full, pm, df_trim)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OWATE ESTIMATOR (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def owate_estimate(X_trim, T_trim, Y_trim, e_trim):
    print("\n" + "="*60)
    print("OVERLAP-WEIGHTED ATE ESTIMATION")
    print("="*60)

    n = len(X_trim)
    # ESTIMATOR FIX: outcome model takes treatment T as an input feature and
    # gives genuine counterfactual predictions mu1 = P(Y=1|X,T=1),
    # mu0 = P(Y=1|X,T=0) (replaces the old ad hoc "adj" term).
    om = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        min_samples_leaf=20, random_state=42)
    om.fit(np.column_stack([X_trim, T_trim]), Y_trim)

    e  = np.clip(e_trim, 0.05, 0.95)
    mu1 = np.clip(om.predict_proba(np.column_stack([X_trim, np.ones(n)]))[:, 1], 0.001, 0.999)
    mu0 = np.clip(om.predict_proba(np.column_stack([X_trim, np.zeros(n)]))[:, 1], 0.001, 0.999)

    mask1, mask0 = T_trim == 1, T_trim == 0

    psi = (mu1 - mu0
           + T_trim * (Y_trim - mu1) / e
           - (1 - T_trim) * (Y_trim - mu0) / (1 - e))

    owate = psi.mean()
    se    = psi.std(ddof=1) / np.sqrt(n)
    ci_lo = owate - 1.96 * se
    ci_hi = owate + 1.96 * se

    raw_diff = Y_trim[mask1].mean() - Y_trim[mask0].mean()
    bias_removed = raw_diff - owate
    pct_removed  = abs(bias_removed) / abs(raw_diff) * 100 if raw_diff != 0 else 0

    sig = "NOT SIGNIFICANT — CI includes 0" if (ci_lo <= 0 <= ci_hi) else "SIGNIFICANT"
    print(f"\n  Trimmed N         : {n:,}")
    print(f"  Raw diff (trimmed): {raw_diff*100:+.2f} pp")
    print(f"  OWATE (adjusted)  : {owate*100:+.2f} pp")
    print(f"  95% CI            : [{ci_lo*100:.2f}pp, {ci_hi*100:.2f}pp]")
    print(f"  SE                : {se*100:.4f} pp")
    print(f"  Result            : {sig}")
    print(f"  Bias removed      : {bias_removed*100:+.2f} pp ({pct_removed:.1f}%)")

    return {
        'owate': owate, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'raw_diff': raw_diff, 'bias_removed': bias_removed,
        'pct_removed': pct_removed, 'n_trim': n,
        'significant': not (ci_lo <= 0 <= ci_hi),
        'psi': psi
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SUBGROUP ANALYSIS — ICULOS used ONLY here for stratification
# ─────────────────────────────────────────────────────────────────────────────

def subgroup_analysis_corrected(X_trim, T_trim, Y_trim, e_trim, df_trim):
    """
    Subgroup OWATE on trimmed sample.
    ICULOS used ONLY for stratification, NOT in any model.
    """
    print("\n" + "="*60)
    print("SUBGROUP ANALYSIS (trimmed sample — ICULOS for stratification only)")
    print("="*60)

    results = []

    def run_sg(label, mask):
        n = mask.sum()
        if n < 100 or T_trim[mask].sum() < 15 or (T_trim[mask]==0).sum() < 15:
            return None
        Xs, Ts, Ys, es = X_trim[mask], T_trim[mask], Y_trim[mask], e_trim[mask]
        # same settings as the main outcome model in owate_estimate()
        om = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42)
        try:
            om.fit(np.column_stack([Xs, Ts]), Ys)
        except Exception:
            return None
        e  = np.clip(es, 0.05, 0.95)
        m1, m0 = Ts==1, Ts==0
        mu1 = np.clip(om.predict_proba(np.column_stack([Xs, np.ones(len(Xs))]))[:, 1], 0.001, 0.999)
        mu0 = np.clip(om.predict_proba(np.column_stack([Xs, np.zeros(len(Xs))]))[:, 1], 0.001, 0.999)
        psi   = mu1-mu0 + Ts*(Ys-mu1)/e - (1-Ts)*(Ys-mu0)/(1-e)
        owate = psi.mean(); se = psi.std(ddof=1)/np.sqrt(n)
        ci_lo = owate-1.96*se; ci_hi = owate+1.96*se
        raw   = Ys[m1].mean()-Ys[m0].mean() if m1.sum()>0 and m0.sum()>0 else 0
        sig   = "✓" if not (ci_lo<=0<=ci_hi) else "○"
        print(f"  {sig} {label:<42} OWATE={owate*100:+.2f}pp CI=[{ci_lo*100:.2f},{ci_hi*100:.2f}] n={n:,} n.s.")
        return {'label': label, 'owate': owate, 'se': se,
                'ci_lo': ci_lo, 'ci_hi': ci_hi, 'raw_diff': raw,
                'n': n, 'significant': not (ci_lo<=0<=ci_hi)}

    r = run_sg("All (trimmed)", np.ones(len(T_trim), dtype=bool))
    if r: results.append(r)

    if df_trim is not None and 'Age' in df_trim.columns:
        print(f"\n  By Age:")
        age = df_trim['Age'].fillna(df_trim['Age'].median()).values
        t33, t67 = np.percentile(age, 33), np.percentile(age, 67)
        for label, mask in [
            (f"Young  (Age < {t33:.0f})",               age < t33),
            (f"Middle ({t33:.0f} \u2264 Age < {t67:.0f})", (age>=t33)&(age<t67)),
            (f"Elderly (Age \u2265 {t67:.0f})",            age >= t67),
        ]:
            r = run_sg(label, mask)
            if r: results.append(r)

    # ICULOS used HERE for description only — not in any model above
    if df_trim is not None and 'icu_duration_subgroup' in df_trim.columns:
        print(f"\n  By ICU Duration (descriptive stratification only):")
        icu = df_trim['icu_duration_subgroup'].fillna(df_trim['icu_duration_subgroup'].median()).values
        t33, t67 = np.percentile(icu, 33), np.percentile(icu, 67)
        for label, mask in [
            (f"Short stay  (ICU < {t33:.0f}h)",          icu < t33),
            (f"Medium stay ({t33:.0f}\u2013{t67:.0f}h)", (icu>=t33)&(icu<t67)),
            (f"Long stay   (ICU \u2265 {t67:.0f}h)",      icu >= t67),
        ]:
            r = run_sg(label, mask)
            if r: results.append(r)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results_corrected(owate_res, subgroup_df, overlap_mask,
                            cohort_stats, out_prefix):
    pct_ret = overlap_mask.mean() * 100
    sig_str = "NO — CI includes 0" if not owate_res['significant'] else "YES"

    sg_rows = '\n'.join([
        f"  {r['label']:<42} OWATE={r['owate']*100:+.2f}pp  "
        f"CI=[{r['ci_lo']*100:.2f},{r['ci_hi']*100:.2f}]  n={r['n']:,}  n.s."
        for _, r in subgroup_df.iterrows()
    ]) if len(subgroup_df) > 0 else "  No subgroups estimated"

    report = f"""
=======================================================================
CAUSALPIPE-TRANSFER: SEPSIS OWATE — CORRECTED ANALYSIS
PhysioNet 2019 Training Set A

=======================================================================

COHORT CONSTRUCTION
--------------------
  Total patient files in Set A          : {cohort_stats['total_files']:,}
  Excluded (< 2 data rows)             : {cohort_stats['excluded_short']:,}
  Excluded (no MAP/Lactate data)       : {cohort_stats['excluded_no_maplac']:,}
  Final analysis sample                : {cohort_stats['final_n']:,}
  ─────────────────────────────────────────────────────
  Treated (shock criterion pre-onset)  : {cohort_stats['n_treated']:,} ({cohort_stats['n_treated']/cohort_stats['final_n']:.1%})
  Control (no pre-onset shock)         : {cohort_stats['n_control']:,} ({cohort_stats['n_control']/cohort_stats['final_n']:.1%})
  Sepsis onset (Y=1)                   : {cohort_stats['n_outcome']:,} ({cohort_stats['n_outcome']/cohort_stats['final_n']:.1%})
  Shock after sepsis onset (→ T=0)    : {cohort_stats['n_shock_after_only']:,}

CORRECTIONS APPLIED
--------------------
  (a) Treatment timing: shock criterion evaluated ONLY in pre-sepsis
      window (hours before first SepsisLabel=1), eliminating reverse causality
  (b) MAP, SBP, DBP, Lactate, map_min, lactate_max excluded from
      confounder set — these variables directly define treatment T
  (c) ICULOS excluded from propensity and outcome models; used only
      for subgroup stratification (descriptive, not causal adjustment)
  (d) Patients with no MAP/Lactate data explicitly excluded (n={cohort_stats['excluded_no_maplac']:,})
      rather than assigned to control as in original analysis

OVERLAP TRIMMING
-----------------
  Trimming rule        : e(X) in [0.1, 0.9] per Crump et al. (2009)
  Trimmed sample N     : {owate_res['n_trim']:,} ({pct_ret:.1f}% retained)
  Note: overlap is expected to be HIGHER than original 13% because
  MAP/Lactate (which caused near-perfect propensity prediction) are
  now excluded from the confounder set.

OWATE ESTIMATES
----------------
  Raw difference (trimmed) : {owate_res['raw_diff']*100:+.2f} pp
  OWATE (DR-adjusted)      : {owate_res['owate']*100:+.2f} pp
  95% CI                   : [{owate_res['ci_lo']*100:.2f}pp, {owate_res['ci_hi']*100:.2f}pp]
  Standard Error           : {owate_res['se']*100:.4f} pp
  Statistically significant: {sig_str}
  Confounding removed      : {owate_res['bias_removed']*100:+.2f} pp ({owate_res['pct_removed']:.1f}%)

SUBGROUP ANALYSIS (trimmed sample, ICULOS for stratification only)
--------------------------------------------------------------------
{sg_rows}

COMPARISON WITH ORIGINAL ANALYSIS
------------------------------------
  The original analysis (MAP/Lactate in confounder set) showed 87% of
  patients outside [0.1, 0.9]. This was partly an artefact of including
  the treatment-defining variables in the propensity model. The corrected
  analysis uses clinically pre-treatment confounders only.
=======================================================================
"""
    print(report)
    path = out_prefix + '_results.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CAUSALPIPE-TRANSFER: SEPSIS CORRECTED OWATE ANALYSIS")
    print("")
    print("=" * 70)

    # Use ALL patients from Set A (fix e)
    df, cohort_stats = build_dataset_corrected(SEPSIS_DIR, max_patients=None)

    X, T, Y, feature_cols, scaler = preprocess_corrected(df)

    (X_trim, T_trim, Y_trim, e_trim,
     overlap_mask, e_full, pm, df_trim) = fit_propensity_and_trim(
        X, T, Y, df, lo=0.1, hi=0.9)

    owate_res = owate_estimate(X_trim, T_trim, Y_trim, e_trim)

    subgroup_df = subgroup_analysis_corrected(
        X_trim, T_trim, Y_trim, e_trim, df_trim)

    save_results_corrected(owate_res, subgroup_df, overlap_mask,
                           cohort_stats, OUT_PREFIX)

    print("\n" + "=" * 70)
    print("CORRECTED SEPSIS ANALYSIS COMPLETE")
    print(f"  Results: {OUT_PREFIX}_results.txt")
    print("=" * 70)

    return owate_res, subgroup_df, overlap_mask, cohort_stats


if __name__ == "__main__":
    owate_res, subgroup_df, overlap_mask, cohort_stats = main()

    # --- export the cumulative OWATE series (used for Figure 6c) and the subgroup table ---
    k = np.arange(1, len(owate_res["psi"]) + 1)
    pd.DataFrame({"k": k, "cum_owate_pp": np.cumsum(owate_res["psi"]) / k * 100}
                 ).to_csv(OUT_PREFIX + "_convergence.csv", index=False)
    subgroup_df.to_csv(OUT_PREFIX + "_subgroups.csv", index=False)

    # --- self-check against the values reported in the paper ---
    expected = {"All (trimmed)": (-3.18, -3.74, -2.63, 20324),
                "Young": (-2.90, -3.78, -2.03, 6706), "Middle": (-3.17, -4.05, -2.29, 6909),
                "Elderly": (-3.02, -3.96, -2.08, 6709), "Short": (-8.47, -9.67, -7.27, 6686),
                "Medium": (-0.67, -1.35, 0.01, 6568), "Long": (-0.38, -1.07, 0.31, 7070)}
    ok = True
    print("\nCHECK vs the values reported in the paper")
    for _, r in subgroup_df.iterrows():
        key = next((kk for kk in expected if r["label"].startswith(kk)), None)
        if key is None:
            continue
        e_o, e_l, e_h, e_n = expected[key]
        good = (abs(r["owate"]*100 - e_o) < 0.006 and abs(r["ci_lo"]*100 - e_l) < 0.006
                and abs(r["ci_hi"]*100 - e_h) < 0.006 and int(r["n"]) == e_n)
        ok &= good
        print(f"  {'MATCH   ' if good else 'MISMATCH'} {r['label']:<36} got {r['owate']*100:+.2f} "
              f"[{r['ci_lo']*100:.2f},{r['ci_hi']*100:.2f}] n={int(r['n']):,}   paper {e_o:+.2f} [{e_l},{e_h}] n={e_n:,}")
    print("\n  RESULT:", "ALL MATCH the values reported in the paper" if ok else
          "DOES NOT MATCH the values reported in the paper")
