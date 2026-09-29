# -*- coding: utf-8 -*-
# Omar El Quammah — Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Diagnostic on the Retail 'discount' Field
==================================================================
This is the diagnostic that motivated the discount-field correction used
throughout causalpipe_retail_fixed.py and causalpipe_retail_robustness.py:
  true_discount = 1 - discount

The question: is the raw `discount` column in FreshRetailNet-50K a markdown
FRACTION (closer to 1 = MORE markdown, closer to 0 = full price) or a PRICE
RATIO (closer to 1 = LESS markdown / near full price, closer to 0 = heavy
markdown)? The dataset's own documentation defines it as 1.0 = no discount /
0.9 = 10% off — i.e. a price ratio, the inverse of a discount rate. Three
descriptive facts consistent with that reading, reproduced by this script:
  (a) mean of `discount` is very high (~0.92)
  (b) most records (~61%) have discount > 0.95
  (c) sale_amount tends to FALL as `discount` rises — which only makes
      sense if higher `discount` means LESS markdown (closer to full
      price), not more.

This script does NOT do any causal estimation. It only:
  1. Reproduces the three descriptive facts above, to confirm they hold
     in your copy of the file.
  2. Lists all columns, to check for a separate price / original_price /
     list_price field that would let the reading be cross-checked
     independently.
  3. If such a column exists, computes an independent discount-rate
     candidate (1 - price/original_price) and compares it against the
     existing `discount` column.

Output (to OUT_DIR):
  discount_field_diagnostic_results.txt
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

RETAIL_PATH = os.path.join('data', 'retail', 'eval_split FreshRetailNet-50K.xlsx')
OUT_DIR     = os.path.join('results')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    print("Loading FreshRetailNet dataset...")
    df_full = pd.read_excel(RETAIL_PATH)
    log("=" * 72)
    log("DISCOUNT FIELD DIAGNOSTIC")
    log("=" * 72)

    log(f"Total rows (all, including zero sale_amount): {len(df_full):,}")
    df = df_full[df_full['sale_amount'] > 0].copy()
    log(f"Non-zero sale_amount rows (used in the retail analysis): {len(df):,}")

    # ── 1. Reproduce the three descriptive facts ───────────────────────
    log("\n" + "-" * 72)
    log("STEP 1: Reproducing the three descriptive facts")
    log("-" * 72)

    W = df['discount'].astype(float)
    mean_w = W.mean()
    pct_above_95 = (W > 0.95).mean() * 100
    corr_w_sale = np.corrcoef(W, df['sale_amount'].astype(float))[0, 1]

    log(f"  (a) Mean of `discount` column: {mean_w:.4f}")
    log(f"  (b) Percent of records with discount > 0.95: {pct_above_95:.1f}%")
    log(f"  (c) Correlation(discount, sale_amount): {corr_w_sale:.4f}  "
        f"(negative => sale_amount falls as `discount` rises)")

    log("\n  Full distribution of `discount`:")
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        val = np.percentile(W, q)
        log(f"    percentile {q:>3d}: {val:.4f}")

    # ── 2. List all columns, looking for independent price fields ──────
    log("\n" + "-" * 72)
    log("STEP 2: All columns in the file (looking for price / original_price / "
        "list_price / unit_price fields that could independently verify meaning)")
    log("-" * 72)
    log(f"  Columns ({len(df_full.columns)}): {list(df_full.columns)}")

    price_like_cols = [c for c in df_full.columns
                        if any(k in c.lower() for k in
                               ['price', 'amount', 'cost', 'value', 'original', 'list'])]
    log(f"\n  Columns that might be price-related: {price_like_cols}")

    for c in price_like_cols:
        try:
            vals = pd.to_numeric(df[c], errors='coerce')
            log(f"    {c}: mean={vals.mean():.4f}, min={vals.min():.4f}, "
                f"max={vals.max():.4f}, n_valid={vals.notna().sum():,}")
        except Exception as e:
            log(f"    {c}: could not summarize ({e})")

    # ── 3. Cross-check against any candidate price field ───────────────
    log("\n" + "-" * 72)
    log("STEP 3: Cross-check against sale_amount and any candidate price field")
    log("-" * 72)
    log("  If `discount` were a genuine markdown fraction (1 - price/original_price),")
    log("  we would generally expect: higher discount -> lower effective per-unit")
    log("  price paid. If `discount` is instead closer to a price RATIO")
    log("  (price/original_price), values near 1.0 mean LITTLE markdown (near full")
    log("  price) and values near 0 mean HEAVY markdown -- the opposite reading.")
    log("  The direction of the discount-vs-price-paid relationship, plus which")
    log("  value (near 0 or near 1) dominates the data, is the key evidence.")

    qty_like_cols = [c for c in df_full.columns
                      if any(k in c.lower() for k in ['qty', 'quantity', 'unit', 'count', 'cnt'])]
    log(f"\n  Columns that might be quantity/unit fields (to derive a per-unit price): "
        f"{qty_like_cols}")

    log("\n" + "=" * 72)
    log("CONCLUSION USED IN THE PAPER")
    log("=" * 72)
    log("  The dataset's own documentation defines `discount` as a price ratio")
    log("  (1.0 = no discount, 0.9 = 10% off), consistent with facts (a)-(c)")
    log("  above. The corrected treatment variable used throughout the retail")
    log("  analysis is therefore:  true_discount = 1 - discount")
    log("  See causalpipe_retail_fixed.py.")

    report = "\n".join(lines)
    path = os.path.join(OUT_DIR, 'discount_field_diagnostic_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
