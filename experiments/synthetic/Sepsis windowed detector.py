# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Algorithm 1 Applied to Real Sequential Sepsis Windows
=============================================================================
Applies the paper's shift-detection procedure (Section 3.3 / Algorithm 1)
to real sequential windows of the sepsis cohort, to get a genuine window-
by-window count of covariate / label / mechanism / mixed shift detections,
rather than a single manuscript-level summary statistic.

IMPORTANT LIMITATION (also stated in the paper): PhysioNet 2019 Training
Set A is de-identified and contains NO real admission timestamps across
patients. File order (p000001, p000002, ...) is the only ordering
available, and it is NOT confirmed to be chronological. This script uses
file order as a pseudo-time axis. If this is not an acceptable substitute
for real time, the "streaming" framing of the sepsis case study should be
read with that caveat in mind — this is a genuine, disclosed limitation.

Reuses the exact cohort construction from causalpipe_sepsis_owate.py
(same feature set, same pre-onset treatment-timing window, same MAP/
Lactate/ICULOS exclusions), then implements Algorithm 1 verbatim:

  Step 1 (Signal 1, covariate shift): KS test on propensity scores,
          propensity model fit on the REFERENCE window only.
  Step 2 (Signal 2, label shift): Welch t-test on outcome residuals from
          the reference-window outcome model, AND |mean shift| > 2*SD.
  Step 3 (Signal 3, mechanism shift): only tested if Signals 1 and 2 both
          did NOT fire. KS test on residual distributions.
  Decision rules: exactly as in Algorithm 1's box, INCLUDING the
          "otherwise -> label_shift" fallback. Also separately logs
          whether the fallback masked a genuinely quiet window (i.e. none
          of the three signals fired at all).

Output (to OUT_DIR):
  sepsis_windowed_detector_results.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, ttest_ind
import warnings
warnings.filterwarnings('ignore')

SEPSIS_DIR = os.path.join('data', 'healthcare', 'sepsis', 'training_setA')
OUT_DIR    = os.path.join('results')
os.makedirs(OUT_DIR, exist_ok=True)

N_WINDOWS = 18
ALPHA = 0.05

# Same feature set as causalpipe_sepsis_owate.py (MAP/Lactate/ICULOS excluded)
VITALS = ['HR', 'O2Sat', 'Temp', 'Resp']
LABS   = ['BaseExcess', 'HCO3', 'pH', 'PaCO2', 'BUN',
          'Calcium', 'Creatinine', 'Glucose',
          'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets']
DEMO   = ['Age', 'Gender']
ALL_FEATURES = VITALS + LABS + DEMO


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


def build_dataset(sepsis_dir):
    """Identical cohort construction (pre-onset treatment window) to
    causalpipe_sepsis_owate.py."""
    files = sorted(glob.glob(os.path.join(sepsis_dir, '*.psv')))
    print(f"Total patient files (file-order = pseudo-time axis): {len(files):,}")

    records = []
    for i, fp in enumerate(files):
        df = load_patient(fp)
        if df is None or len(df) < 2:
            continue
        outcome = int(df['SepsisLabel'].max() == 1)
        sepsis_hours = df.index[df['SepsisLabel'] == 1].tolist()
        first_sepsis_hour = sepsis_hours[0] if len(sepsis_hours) > 0 else len(df)
        pre_df = df.iloc[:first_sepsis_hour]   # PRE-ONSET WINDOW

        has_map = 'MAP' in df.columns and df['MAP'].notna().any()
        has_lactate = 'Lactate' in df.columns and df['Lactate'].notna().any()
        if not has_map and not has_lactate:
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

        feat = {}
        for col in ALL_FEATURES:
            if col in df.columns:
                vals = df[col].dropna()
                feat[col] = float(vals.mean()) if len(vals) > 0 else np.nan
            else:
                feat[col] = np.nan
        feat['hr_variability'] = float(df['HR'].std()) if 'HR' in df.columns else np.nan
        feat['n_hours'] = float(len(df))
        feat['treatment'] = treatment
        feat['outcome'] = outcome
        feat['file_order'] = i
        records.append(feat)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{len(files):,} processed")

    return pd.DataFrame(records)


def preprocess(df):
    feature_cols = [c for c in ALL_FEATURES if c in df.columns and c != 'Gender']
    feature_cols += ['hr_variability', 'n_hours']
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
    return X, T, Y


def algorithm1(Xs, Ts, Ys, Xt, Tt, Yt, alpha=ALPHA):
    """Verbatim implementation of Algorithm 1 as specified in the manuscript."""
    # Step 1: covariate shift signal
    prop_model = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced', random_state=42)
    prop_model.fit(Xs, Ts)
    e_s = prop_model.predict_proba(Xs)[:, 1]
    e_t = prop_model.predict_proba(Xt)[:, 1]
    ks_cov_stat, ks_cov_p = ks_2samp(e_s, e_t)
    cov_drift = ks_cov_p < alpha

    # Step 2: label shift signal (residuals from reference-window outcome model)
    outcome_model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                min_samples_leaf=20, random_state=42)
    outcome_model.fit(Xs, Ys)
    pred_s = outcome_model.predict_proba(Xs)[:, 1]
    pred_t = outcome_model.predict_proba(Xt)[:, 1]
    r_s = Ys - pred_s
    r_t = Yt - pred_t
    tt_stat, tt_p = ttest_ind(r_s, r_t, equal_var=False)
    mean_shift = abs(r_t.mean() - r_s.mean())
    lbl_drift = (tt_p < alpha) and (mean_shift > 2 * r_s.std())

    # Step 3: mechanism shift signal (only if Steps 1 and 2 both silent)
    mec_drift = False
    ks_mec_p = np.nan
    if (not cov_drift) and (not lbl_drift):
        ks_mec_stat, ks_mec_p = ks_2samp(r_s, r_t)
        mec_drift = ks_mec_p < alpha

    # Decision rules (verbatim from Algorithm 1)
    none_of_three_fired = (not cov_drift) and (not lbl_drift) and (not mec_drift)
    if cov_drift and lbl_drift:
        label = 'mixed'
    elif cov_drift:
        label = 'covariate_shift'
    elif lbl_drift:
        label = 'label_shift'
    elif mec_drift:
        label = 'mechanism_shift'
    else:
        label = 'label_shift'  # conservative fallback, as literally specified

    return {
        'label': label,
        'cov_drift': cov_drift, 'ks_cov_p': ks_cov_p,
        'lbl_drift': lbl_drift, 'tt_p': tt_p, 'mean_shift': mean_shift,
        'mec_drift': mec_drift, 'ks_mec_p': ks_mec_p,
        'none_of_three_fired': none_of_three_fired,
        'is_fallback': (label == 'label_shift') and (not lbl_drift),
    }


def main():
    print("=" * 70)
    print("ALGORITHM 1 APPLIED TO REAL SEPSIS WINDOWS (verbatim implementation)")
    print("=" * 70)
    df = build_dataset(SEPSIS_DIR)
    print(f"\nFinal analysis sample: {len(df):,}")
    df = df.sort_values('file_order').reset_index(drop=True)
    X, T, Y = preprocess(df)

    n = len(df)
    window_edges = np.linspace(0, n, N_WINDOWS + 1).astype(int)
    windows = [(window_edges[i], window_edges[i + 1]) for i in range(N_WINDOWS)]
    print(f"\n{N_WINDOWS} sequential windows, ~{n // N_WINDOWS:,} patients each "
          f"(ordered by file name; see limitation note in module docstring).")

    ref_lo, ref_hi = windows[0]
    Xs, Ts, Ys = X[ref_lo:ref_hi], T[ref_lo:ref_hi], Y[ref_lo:ref_hi]
    print(f"Reference window: patients {ref_lo}-{ref_hi} (n={ref_hi - ref_lo:,})")

    results = []
    for w_idx in range(1, N_WINDOWS):
        lo, hi = windows[w_idx]
        Xt, Tt, Yt = X[lo:hi], T[lo:hi], Y[lo:hi]
        res = algorithm1(Xs, Ts, Ys, Xt, Tt, Yt)
        res['window'] = w_idx + 1
        res['n'] = hi - lo
        results.append(res)
        print(f"  Window {w_idx+1:2d}/{N_WINDOWS} (n={hi-lo:,}): "
              f"label={res['label']:<16} cov_p={res['ks_cov_p']:.4f} "
              f"lbl_p={res['tt_p']:.4f} mec_p={res['ks_mec_p']} "
              f"fallback={res['is_fallback']} none_of_three={res['none_of_three_fired']}")

    rdf = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(f"SUMMARY (windows 2..{N_WINDOWS} compared against window 1 as reference)")
    print("=" * 70)
    print(rdf['label'].value_counts())
    print(f"\nFallback-to-label_shift count (no real signal fired): "
          f"{rdf['is_fallback'].sum()} / {len(rdf)}")
    print(f"Genuinely quiet windows (none of 3 signals fired): "
          f"{rdf['none_of_three_fired'].sum()} / {len(rdf)}")

    out_path = os.path.join(OUT_DIR, 'sepsis_windowed_detector_results.csv')
    rdf.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
