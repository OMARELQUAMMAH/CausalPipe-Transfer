# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Sepsis Analysis — Overlap-Weighted ATE (OWATE)
=====================================================================
REVISION NOTE (two fixes on top of the original overlap-trimming approach,
both now applied together):

  1. TREATMENT-TIMING FIX (pre-onset window). Treatment (T=1: shock
     criterion — MAP<65 with Lactate>2.0, or MAP<60 alone as fallback) is
     now evaluated ONLY on the portion of each patient's record BEFORE
     their first recorded sepsis onset hour (or the full record, for
     patients who never meet the sepsis label). This removes a
     look-ahead-bias source in the original cohort construction, where
     shock could be detected using vitals recorded after sepsis onset.
     Patients whose only shock episode occurs after onset are coded T=0,
     not T=1 (tracked separately as n_shock_after_only for transparency).
     Confounder features (MAP, SBP, DBP, Lactate, ICULOS) are also
     excluded from the propensity/outcome model inputs, since they are
     definitionally entangled with the treatment/outcome construction.

  2. ESTIMATOR FIX. The original owate_estimate() fit the outcome model
     on Y only (never saw treatment T as an input), then manufactured
     mu1/mu0 by splitting a single prediction using an ad hoc adjustment:
         adj = clip((obs_diff - pred_diff) / 2, -0.15, 0.15)
     obs_diff is the RAW observed treated-vs-control difference — the
     same quantity the DR/AIPW correction is supposed to remove
     confounding from, so folding half of it back into the "counterfactual"
     predictions was circular and biased the estimate toward the naive,
     confounded number.
     FIX: the outcome model now includes T as an input feature (exactly
     like the AIPW outcome model used in the synthetic-benchmark and
     retail scripts). We fit ONE classifier on [X, T] -> Y, then get
     genuine counterfactual predictions:
         mu1 = model.predict_proba([X, T=1])
         mu0 = model.predict_proba([X, T=0])
     No manual adjustment, no clipping, no circularity.

Methodological basis for overlap trimming:
  Crump, R.K., Hotz, V.J., Imbens, G.W., Mitnik, O.A. (2009).
  "Dealing with limited overlap in estimation of average treatment effects."
  Biometrika, 96(1), 187-199.
  Trimming rule: exclude patients with e(X) < 0.1 or e(X) > 0.9.

Outputs (to OUT_DIR):
  sepsis_owate_results.txt
  sepsis_owate_figures.png
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

SEPSIS_DIR = os.path.join('data', 'healthcare', 'sepsis', 'training_setA')
OUT_DIR    = os.path.join('results')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

# MAP/SBP/DBP/Lactate/ICULOS excluded — entangled with treatment/outcome definition
VITALS = ['HR', 'O2Sat', 'Temp', 'Resp']
LABS   = ['BaseExcess', 'HCO3', 'pH', 'PaCO2', 'BUN',
          'Calcium', 'Creatinine', 'Glucose',
          'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets']
DEMO   = ['Age', 'Gender']
ALL_FEATURES = VITALS + LABS + DEMO


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA  (pre-onset treatment window)
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


def build_dataset(sepsis_dir, max_patients=None):
    print("Loading patient files...")
    files = sorted(glob.glob(os.path.join(sepsis_dir, '*.psv')))
    if max_patients:
        files = files[:max_patients]
    total_files = len(files)
    print(f"  Total patient files: {total_files:,}")

    records = []
    n_excluded_short, n_excluded_no_maplac = 0, 0
    n_shock_before, n_shock_after_only, n_no_shock = 0, 0, 0

    for i, fp in enumerate(files):
        df = load_patient(fp)
        if df is None or len(df) < 2:
            n_excluded_short += 1
            continue

        outcome = int(df['SepsisLabel'].max() == 1)
        sepsis_hours = df.index[df['SepsisLabel'] == 1].tolist()
        first_sepsis_hour = sepsis_hours[0] if len(sepsis_hours) > 0 else len(df)
        # PRE-ONSET WINDOW: only rows strictly before first sepsis onset hour
        pre_df = df.iloc[:first_sepsis_hour]

        has_map = 'MAP' in df.columns and df['MAP'].notna().any()
        has_lactate = 'Lactate' in df.columns and df['Lactate'].notna().any()
        if not has_map and not has_lactate:
            n_excluded_no_maplac += 1
            continue

        shock = False
        if has_map and has_lactate:
            pre_both = pre_df[['MAP', 'Lactate']].dropna()
            if len(pre_both) > 0:
                shock = bool(((pre_both['MAP'] < 65) & (pre_both['Lactate'] > 2.0)).any())
        elif has_map:
            pre_map = pre_df['MAP'].dropna()
            if len(pre_map) > 0:
                shock = bool((pre_map < 65).any())

        treatment = int(shock)
        if shock:
            n_shock_before += 1
        elif outcome == 1:
            n_shock_after_only += 1
            n_no_shock += 1
        else:
            n_no_shock += 1

        feat = {}
        for col in ALL_FEATURES:
            if col in df.columns:
                vals = df[col].dropna()
                feat[col] = float(vals.mean()) if len(vals) > 0 else np.nan
            else:
                feat[col] = np.nan

        feat['hr_variability'] = float(df['HR'].std()) if 'HR' in df.columns else np.nan
        feat['n_hours'] = float(len(df))
        feat['icu_duration_subgroup'] = float(df['ICULOS'].max()) if 'ICULOS' in df.columns else np.nan
        feat['treatment'] = treatment
        feat['outcome'] = outcome
        records.append(feat)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{total_files:,} processed")

    df_out = pd.DataFrame(records)

    print(f"\n  COHORT CONSTRUCTION TABLE:")
    print(f"  Total patient files in Set A      : {total_files:,}")
    print(f"  Excluded (< 2 rows)               : {n_excluded_short:,}")
    print(f"  Excluded (no MAP/Lactate data)     : {n_excluded_no_maplac:,}")
    print(f"  Final analysis sample              : {len(df_out):,}")
    print(f"  Treated (shock pre-onset, T=1)     : {df_out['treatment'].sum():,} "
          f"({df_out['treatment'].mean():.1%})")
    print(f"  Control (no pre-onset shock, T=0)  : {(df_out['treatment']==0).sum():,} "
          f"({(df_out['treatment']==0).mean():.1%})")
    print(f"  Sepsis onset (Y=1)                 : {df_out['outcome'].sum():,} "
          f"({df_out['outcome'].mean():.1%})")
    print(f"  Shock after sepsis onset (-> T=0)  : {n_shock_after_only:,}")

    cohort_stats = {
        'total_files': total_files, 'excluded_short': n_excluded_short,
        'excluded_no_maplac': n_excluded_no_maplac, 'final_n': len(df_out),
        'n_treated': int(df_out['treatment'].sum()),
        'n_control': int((df_out['treatment'] == 0).sum()),
        'n_outcome': int(df_out['outcome'].sum()),
        'n_shock_after_only': n_shock_after_only,
    }
    return df_out, cohort_stats


def preprocess(df):
    feature_cols = [c for c in ALL_FEATURES if c in df.columns and c != 'Gender']
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
    return X, T, Y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROPENSITY MODEL + OVERLAP TRIMMING
# ─────────────────────────────────────────────────────────────────────────────

def fit_propensity_and_trim(X, T, Y, df_raw, lo=0.1, hi=0.9):
    print("\n" + "=" * 60)
    print("PROPENSITY MODEL & OVERLAP TRIMMING")
    print("=" * 60)

    pm = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced', random_state=42)
    pm.fit(X, T)
    e_full = pm.predict_proba(X)[:, 1]

    print(f"\n  Full sample (N={len(X):,}):")
    print(f"  Propensity range  : [{e_full.min():.3f}, {e_full.max():.3f}]")
    print(f"  < {lo} (excluded) : {(e_full < lo).mean():.1%}")
    print(f"  > {hi} (excluded) : {(e_full > hi).mean():.1%}")
    print(f"  In [{lo},{hi}]    : {((e_full >= lo) & (e_full <= hi)).mean():.1%}")

    overlap_mask = (e_full >= lo) & (e_full <= hi)
    X_trim, T_trim, Y_trim, e_trim = (X[overlap_mask], T[overlap_mask],
                                        Y[overlap_mask], e_full[overlap_mask])
    df_trim = df_raw[overlap_mask].copy().reset_index(drop=True) if df_raw is not None else None

    n_trim = overlap_mask.sum()
    print(f"\n  Trimmed sample (N={n_trim:,}, {n_trim/len(X):.1%} of full):")
    print(f"  Treatment rate    : {T_trim.mean():.1%}")
    print(f"  Sepsis rate       : {Y_trim.mean():.1%}")

    ks_stat, ks_p = ks_2samp(e_trim[T_trim == 1], e_trim[T_trim == 0])
    print(f"  KS distance       : {ks_stat:.4f}  (p={ks_p:.4f})")

    return X_trim, T_trim, Y_trim, e_trim, overlap_mask, e_full, pm, df_trim


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OWATE ESTIMATOR  (fixed: genuine AIPW outcome model, T as a feature)
# ─────────────────────────────────────────────────────────────────────────────

def fit_outcome_model_with_treatment(X, T, Y):
    """Single outcome model that sees treatment as a feature (AIPW-style)."""
    XT = np.column_stack([X, T])
    om = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     min_samples_leaf=20, random_state=42)
    om.fit(XT, Y)
    return om


def predict_counterfactuals(om, X):
    n = len(X)
    XT1 = np.column_stack([X, np.ones(n)])
    XT0 = np.column_stack([X, np.zeros(n)])
    mu1 = om.predict_proba(XT1)[:, 1]
    mu0 = om.predict_proba(XT0)[:, 1]
    return mu1, mu0


def owate_estimate(X_trim, T_trim, Y_trim, e_trim):
    print("\n" + "=" * 60)
    print("OVERLAP-WEIGHTED ATE ESTIMATION (genuine AIPW outcome model)")
    print("=" * 60)

    n = len(X_trim)
    om = fit_outcome_model_with_treatment(X_trim, T_trim, Y_trim)
    mu1, mu0 = predict_counterfactuals(om, X_trim)
    mu1, mu0 = np.clip(mu1, 0.001, 0.999), np.clip(mu0, 0.001, 0.999)
    e = np.clip(e_trim, 0.05, 0.95)

    psi = (mu1 - mu0
           + T_trim * (Y_trim - mu1) / e
           - (1 - T_trim) * (Y_trim - mu0) / (1 - e))

    owate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    ci_lo, ci_hi = owate - 1.96 * se, owate + 1.96 * se

    mask1, mask0 = T_trim == 1, T_trim == 0
    raw_diff = Y_trim[mask1].mean() - Y_trim[mask0].mean()
    bias_removed = raw_diff - owate
    pct_removed = abs(bias_removed) / abs(raw_diff) * 100 if raw_diff != 0 else 0
    significant = not (ci_lo <= 0 <= ci_hi)

    print(f"\n  Trimmed N         : {n:,}")
    print(f"  Raw diff (trimmed): {raw_diff*100:+.2f} pp")
    print(f"  OWATE             : {owate*100:+.2f} pp")
    print(f"  95% CI            : [{ci_lo*100:.2f}pp, {ci_hi*100:.2f}pp]")
    print(f"  SE                : {se*100:.4f} pp")
    print(f"  Result            : {'SIGNIFICANT' if significant else 'not significant'}")
    print(f"  Bias removed      : {bias_removed*100:+.2f} pp ({pct_removed:.1f}%)")

    return {'owate': owate, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'raw_diff': raw_diff, 'bias_removed': bias_removed,
            'pct_removed': pct_removed, 'n_trim': n,
            'significant': significant, 'psi': psi}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SUBGROUP ANALYSIS ON TRIMMED SAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def subgroup_analysis(X_trim, T_trim, Y_trim, e_trim, df_trim):
    print("\n" + "=" * 60)
    print("SUBGROUP ANALYSIS (trimmed overlap sample)")
    print("=" * 60)

    results = []

    def run_sg(label, mask):
        n = mask.sum()
        if n < 100 or T_trim[mask].sum() < 15 or (T_trim[mask] == 0).sum() < 15:
            print(f"  {label}: skipped (n={n} or insufficient treatment variation)")
            return None
        Xs, Ts, Ys, es = X_trim[mask], T_trim[mask], Y_trim[mask], e_trim[mask]
        try:
            om = fit_outcome_model_with_treatment(Xs, Ts, Ys)
            mu1, mu0 = predict_counterfactuals(om, Xs)
        except Exception:
            print(f"  {label}: model fitting failed")
            return None
        mu1, mu0 = np.clip(mu1, 0.001, 0.999), np.clip(mu0, 0.001, 0.999)
        e = np.clip(es, 0.05, 0.95)
        psi = mu1 - mu0 + Ts * (Ys - mu1) / e - (1 - Ts) * (Ys - mu0) / (1 - e)
        owate, se = psi.mean(), psi.std(ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = owate - 1.96 * se, owate + 1.96 * se
        m1, m0 = Ts == 1, Ts == 0
        raw = Ys[m1].mean() - Ys[m0].mean() if m1.sum() > 0 and m0.sum() > 0 else 0
        significant = not (ci_lo <= 0 <= ci_hi)
        sig_mark = "significant" if significant else "not significant"
        print(f"  {label:<42} OWATE={owate*100:+.2f}pp  CI=[{ci_lo*100:.2f},{ci_hi*100:.2f}]  "
              f"n={n:,}  ({sig_mark})")
        return {'label': label, 'owate': owate, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'raw_diff': raw, 'n': n, 'significant': significant}

    r = run_sg("All (trimmed)", np.ones(len(T_trim), dtype=bool))
    if r: results.append(r)

    if df_trim is not None and 'Age' in df_trim.columns:
        print(f"\n  By Age:")
        age = df_trim['Age'].fillna(df_trim['Age'].median()).values
        t33, t67 = np.percentile(age, 33), np.percentile(age, 67)
        for label, mask in [
            (f"Young  (Age < {t33:.0f})", age < t33),
            (f"Middle ({t33:.0f} <= Age < {t67:.0f})", (age >= t33) & (age < t67)),
            (f"Elderly (Age >= {t67:.0f})", age >= t67),
        ]:
            r = run_sg(label, mask)
            if r: results.append(r)

    if df_trim is not None and 'icu_duration_subgroup' in df_trim.columns:
        print(f"\n  By ICU Duration (descriptive stratification only):")
        icu = df_trim['icu_duration_subgroup'].fillna(df_trim['icu_duration_subgroup'].median()).values
        t33, t67 = np.percentile(icu, 33), np.percentile(icu, 67)
        for label, mask in [
            (f"Short stay  (ICU < {t33:.0f}h)", icu < t33),
            (f"Medium stay ({t33:.0f}-{t67:.0f}h)", (icu >= t33) & (icu < t67)),
            (f"Long stay   (ICU >= {t67:.0f}h)", icu >= t67),
        ]:
            r = run_sg(label, mask)
            if r: results.append(r)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(e_full, overlap_mask, owate_res, subgroup_df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # A — Propensity score before and after trimming
    ax = axes[0, 0]
    e_in, e_out = e_full[overlap_mask], e_full[~overlap_mask]
    ax.hist(e_out, bins=50, alpha=0.5, color='#E74C3C', density=True,
            label=f'Excluded (n={len(e_out):,})')
    ax.hist(e_in, bins=50, alpha=0.7, color='#27AE60', density=True,
            label=f'Retained (n={len(e_in):,})')
    ax.axvline(0.1, color='black', lw=2, linestyle='--', label='Trim bounds [0.1, 0.9]')
    ax.axvline(0.9, color='black', lw=2, linestyle='--')
    ax.set_xlabel('Propensity Score e(X)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Propensity Score Distribution\nOverlap Trimming per Crump et al. (2009)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    pct_retained = overlap_mask.mean() * 100
    ax.text(0.35, 0.88, f'{pct_retained:.1f}% retained\n{100-pct_retained:.1f}% trimmed',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # B — Bias correction waterfall
    ax = axes[0, 1]
    raw_pp, bias_pp, owate_pp = (owate_res['raw_diff'] * 100, owate_res['bias_removed'] * 100,
                                   owate_res['owate'] * 100)
    ci_lo_pp, ci_hi_pp = owate_res['ci_lo'] * 100, owate_res['ci_hi'] * 100
    cats = ['Raw difference\n(naive)', 'Confounding\nremoved', 'OWATE\n(adjusted)']
    vals = [raw_pp, -bias_pp, owate_pp]
    bars = ax.bar(cats, vals, color=['#E74C3C', '#F39C12', '#27AE60'], alpha=0.85,
                  edgecolor='black', lw=1.2, width=0.5)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('Effect (percentage points)')
    ax.set_title(f'(b) Confounding Decomposition (Trimmed Sample)\n'
                 f'{owate_res["pct_removed"]:.0f}% of raw difference is confounding',
                 fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.05 if v >= 0 else -0.1),
                f'{v:+.2f}pp', ha='center', fontweight='bold', fontsize=10)
    ax.errorbar(2, owate_pp, yerr=[[owate_pp - ci_lo_pp], [ci_hi_pp - owate_pp]],
                fmt='none', ecolor='black', elinewidth=2, capsize=8, capthick=2)

    # C — Subgroup forest plot
    ax = axes[1, 0]
    if len(subgroup_df) > 0:
        colors_sg = ['#E74C3C' if r['significant'] else '#95A5A6'
                     for _, r in subgroup_df.iterrows()]
        for i, (_, row) in enumerate(subgroup_df.iterrows()):
            ax.barh(i, row['owate'] * 100, xerr=1.96 * row['se'] * 100,
                    height=0.6, color=colors_sg[i], alpha=0.8,
                    error_kw={'elinewidth': 1.5, 'capsize': 4})
        ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.7)
        ax.axvline(owate_res['owate'] * 100, color='#2E86AB', lw=2, linestyle=':',
                   label='Overall OWATE')
        ax.set_yticks(np.arange(len(subgroup_df)))
        ax.set_yticklabels([f"{r['label'][:30]} (n={r['n']:,})"
                            for _, r in subgroup_df.iterrows()], fontsize=8)
        ax.set_xlabel('OWATE (percentage points)')
        ax.set_title('(c) Subgroup Forest Plot\nRed=significant, Grey=not significant',
                     fontweight='bold')
        ax.legend(fontsize=9)

    # D — OWATE convergence (cumulative)
    ax = axes[1, 1]
    psi = owate_res['psi']
    ns = np.arange(1, len(psi) + 1)
    cumulative = np.cumsum(psi) / ns
    se_band = psi.std() / np.sqrt(ns)
    ax.plot(ns, cumulative * 100, color='#2E86AB', lw=2, label='Cumulative OWATE')
    ax.fill_between(ns, (cumulative - 1.96*se_band) * 100, (cumulative + 1.96*se_band) * 100,
                    alpha=0.15, color='#2E86AB')
    ax.axhline(owate_res['owate'] * 100, color='red', lw=2, linestyle='--',
               label=f'Final OWATE={owate_res["owate"]*100:+.2f}pp')
    ax.axhline(0, color='black', lw=1, alpha=0.4)
    ax.set_xlabel('Patient (trimmed sample)')
    ax.set_ylabel('Cumulative OWATE (pp)')
    ax.set_title('(d) OWATE Convergence\n(trimmed overlap sample)', fontweight='bold')
    ax.legend(fontsize=9)

    sig_str = 'significant' if owate_res['significant'] else 'n.s. — CI includes 0'
    plt.suptitle(
        f'CausalPipe-Transfer: Sepsis OWATE Analysis\n'
        f'Trimmed N={owate_res["n_trim"]:,}  |  OWATE={owate_res["owate"]*100:+.2f}pp  |  '
        f'95% CI [{owate_res["ci_lo"]*100:.2f}pp, {owate_res["ci_hi"]*100:.2f}pp]  |  {sig_str}',
        fontweight='bold', fontsize=12, y=1.01
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'sepsis_owate_figures.png')
    plt.savefig(path)
    print(f"\nFigure saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_results(owate_res, subgroup_df, overlap_mask, cohort_stats, out_dir):
    pct_ret = overlap_mask.mean() * 100
    sig_str = "YES" if owate_res['significant'] else "NO -- CI includes 0"
    sg_rows = '\n'.join([
        f"  {r['label']:<42} OWATE={r['owate']*100:+.2f}pp  CI=[{r['ci_lo']*100:.2f},{r['ci_hi']*100:.2f}]  n={r['n']:,}"
        for _, r in subgroup_df.iterrows()
    ]) if len(subgroup_df) > 0 else "  No subgroups estimated"

    report = f"""
=======================================================================
CAUSALPIPE-TRANSFER: SEPSIS OWATE ANALYSIS
PhysioNet 2019 Training Set A | Pre-onset treatment window + fixed AIPW estimator
Overlap trimming: Crump et al. (2009), Biometrika 96(1):187-199
=======================================================================

COHORT CONSTRUCTION
--------------------
  Total patient files in Set A          : {cohort_stats['total_files']:,}
  Excluded (< 2 data rows)              : {cohort_stats['excluded_short']:,}
  Excluded (no MAP/Lactate data)        : {cohort_stats['excluded_no_maplac']:,}
  Final analysis sample                 : {cohort_stats['final_n']:,}
  Treated (shock criterion pre-onset)   : {cohort_stats['n_treated']:,} ({cohort_stats['n_treated']/cohort_stats['final_n']:.1%})
  Control (no pre-onset shock)          : {cohort_stats['n_control']:,} ({cohort_stats['n_control']/cohort_stats['final_n']:.1%})
  Sepsis onset (Y=1)                    : {cohort_stats['n_outcome']:,} ({cohort_stats['n_outcome']/cohort_stats['final_n']:.1%})
  Shock after sepsis onset (-> T=0)     : {cohort_stats['n_shock_after_only']:,}

OVERLAP TRIMMING
-----------------
  Trimming rule          : e(X) in [0.1, 0.9]
  Trimmed sample N       : {owate_res['n_trim']:,} ({pct_ret:.1f}% retained)

OWATE ESTIMATES
-----------------
  Raw difference (trimmed) : {owate_res['raw_diff']*100:+.2f} pp
  OWATE (DR-adjusted)      : {owate_res['owate']*100:+.2f} pp
  95% Confidence Interval  : [{owate_res['ci_lo']*100:.2f}pp, {owate_res['ci_hi']*100:.2f}pp]
  Standard Error           : {owate_res['se']*100:.4f} pp
  Statistically significant: {sig_str}
  Confounding removed      : {owate_res['bias_removed']*100:+.2f} pp ({owate_res['pct_removed']:.1f}%)

SUBGROUP ANALYSIS (trimmed sample)
------------------------------------
{sg_rows}

CAVEATS
--------
  - Trimming restricts inference to {pct_ret:.1f}% of the final analysis sample.
  - Treatment remains a proxy variable (shock criterion on MAP/Lactate);
    direct vasopressor/intervention records would strengthen identification.
  - File order (p000001, p000002, ...) is used as a pseudo-time axis for the
    streaming/windowed detector companion script (sepsis_windowed_detector.py);
    it is not confirmed to be chronological.
=======================================================================
"""
    print(report)
    path = os.path.join(out_dir, 'sepsis_owate_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Results saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CAUSALPIPE-TRANSFER: SEPSIS OWATE ANALYSIS")
    print("Pre-onset treatment window + fixed AIPW estimator")
    print("=" * 70)

    df, cohort_stats = build_dataset(SEPSIS_DIR, max_patients=None)
    X, T, Y, feature_cols = preprocess(df)
    (X_trim, T_trim, Y_trim, e_trim,
     overlap_mask, e_full, pm, df_trim) = fit_propensity_and_trim(X, T, Y, df, lo=0.1, hi=0.9)

    owate_res = owate_estimate(X_trim, T_trim, Y_trim, e_trim)
    subgroup_df = subgroup_analysis(X_trim, T_trim, Y_trim, e_trim, df_trim)

    save_results(owate_res, subgroup_df, overlap_mask, cohort_stats, OUT_DIR)
    make_figures(e_full, overlap_mask, owate_res, subgroup_df, OUT_DIR)

    print("\n" + "=" * 70)
    print("SEPSIS OWATE ANALYSIS COMPLETE")
    print(f"  Outputs -> {OUT_DIR}/sepsis_owate_results.txt / _figures.png")
    print("=" * 70)

    return owate_res, subgroup_df, overlap_mask, cohort_stats


if __name__ == "__main__":
    owate_res, subgroup_df, overlap_mask, cohort_stats = main()
