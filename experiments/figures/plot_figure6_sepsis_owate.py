# -*- coding: utf-8 -*-
# Omar El Quammah -- Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Figure 6 (Sepsis OWATE analysis), KBS R1 revision.

Three panels (the old panel (a), the propensity-overlap bar chart, is dropped):
  (a) Confounding decomposition: raw -7.21 pp vs OWATE -3.18 pp [-3.74, -2.63]
  (b) Subgroup forest plot, trimmed sample (values from the sepsis analysis output)
  (c) OWATE convergence, drawn from the REAL exported series
      results/sepsis_owate_convergence.csv (all 20,324 cumulative estimates).

Run from the repository root after the sepsis script (it writes results/sepsis_owate_convergence.csv).
Saves figure6_sepsis_owate.png / .pdf next to the script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join("results", "sepsis_owate_convergence.csv")   # written by the sepsis script
GREEN, ORANGE, GREY = "#009E73", "#D55E00", "#999999"

RAW, OW, OW_LO, OW_HI = -7.21, -3.18, -3.74, -2.63
SUB = [  # label, estimate, lo, hi, n  (percentage points)
    ("All (trimmed)", -3.18, -3.74, -2.63, 20324),
    ("Young (Age < 57)", -2.90, -3.78, -2.03, 6706),
    ("Middle (57 <= Age < 72)", -3.17, -4.05, -2.29, 6909),
    ("Elderly (Age >= 72)", -3.02, -3.96, -2.08, 6709),
    ("Short stay (ICU < 31h)", -8.47, -9.67, -7.27, 6686),
    ("Medium stay (31-45h)", -0.67, -1.35, 0.01, 6568),
    ("Long stay (ICU >= 45h)", -0.38, -1.07, 0.31, 7070),
]

os.makedirs(os.path.join("results", "figures"), exist_ok=True)
d = pd.read_csv(CSV)
k, f = d["k"].values, d["cum_owate_pp"].values
final = f[-1]
def settle(tol):                      # first n after which the series never leaves final +/- tol
    out = np.where(np.abs(f - final) > tol)[0]
    return int(k[out[-1] + 1]) if len(out) else int(k[0])
n02, n015 = settle(0.2), settle(0.15)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(12.5, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.62)
axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
fig.suptitle("CausalPipe-Transfer: Sepsis OWATE Analysis\n"
             "Confounding decomposition, subgroup heterogeneity, and convergence", fontsize=12)

# (a) decomposition
axA.bar([0, 1], [RAW, OW], color=[ORANGE, GREEN], width=0.7)
axA.errorbar([1], [OW], yerr=[[OW - OW_LO], [OW_HI - OW]], color="black", capsize=6, lw=1.3)
axA.axhline(0, color="#444444", lw=0.8)
axA.text(0, RAW - 0.35, f"{RAW:.2f} pp".replace("-", "−"), ha="center", va="top", fontweight="bold")
axA.text(1, OW_LO - 0.35, f"−3.18 pp\n[−3.74, −2.63]", ha="center", va="top", fontweight="bold")
axA.set_xticks([0, 1]); axA.set_xticklabels(["Raw difference\n(unadjusted)", "OWATE\n(AIPW DR-adjusted)"])
axA.set_ylabel("Effect (percentage points)"); axA.set_ylim(-9.6, 0.6)
axA.set_title("(a) Confounding decomposition", fontsize=10.5)

# (b) forest
ys = np.arange(len(SUB))[::-1]
for y, (lab, e, lo, hi, n) in zip(ys, SUB):
    c = GREY if hi >= 0 else GREEN
    axB.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt="o", color=c, capsize=4, lw=1.4, ms=6)
axB.axvline(0, color="#bbbbbb", lw=1)
axB.axvline(OW, color="#222222", lw=1.1, ls="--")
axB.set_yticks(ys); axB.set_yticklabels([f"{s[0]}\n(n={s[4]:,})" for s in SUB], fontsize=8.5)
axB.set_xlabel("OWATE (percentage points)  |  dashed line = overall OWATE = −3.18 pp")
axB.set_title("(b) Subgroup forest plot (trimmed sample)\ngreen = significant, grey = CI includes 0", fontsize=10.5)

# (c) convergence (real series)
m = k >= 200
axC.axhspan(final - 0.2, final + 0.2, color=GREEN, alpha=0.10, lw=0, label="final ±0.2 pp")
axC.axhspan(final - 0.15, final + 0.15, color=GREEN, alpha=0.20, lw=0, label="final ±0.15 pp")
axC.plot(k[m], f[m], color=GREEN, lw=1.3, label="Cumulative OWATE")
axC.axhline(final, color="#222222", ls="--", lw=1, label=f"Final OWATE = {final:.2f} pp".replace("-", "−"))
axC.axvline(n02, color=GREY, ls=":", lw=1); axC.axvline(n015, color=GREY, ls=":", lw=1)
axC.set_xscale("log"); axC.set_xlim(200, k[-1] * 1.05)
axC.set_ylim(-5.9, -1.6)
axC.text(n02, -1.7, f" n≈{round(n02,-2):,}", fontsize=8.5, color="#444444", va="top")
axC.text(n015, -2.05, f" n≈{round(n015,-2):,}", fontsize=8.5, color="#444444", va="top")
axC.set_xlabel("Number of patients (log scale)"); axC.set_ylabel("Cumulative OWATE (percentage points)")
axC.set_title("(c) OWATE convergence over the trimmed patient stream (all 20,324 cumulative estimates)", fontsize=10.5)
axC.legend(loc="lower right", frameon=False, fontsize=8.5)
fig.text(0.5, 0.005, "Propensity model fit once on the full trimmed sample; subsamples drawn in cohort order. "
         "Series shown from n = 200; vertical dotted lines mark where the estimate stays within ±0.2 and ±0.15 pp of the final value.",
         ha="center", fontsize=7.8, style="italic", color="#444444")
fig.savefig(os.path.join("results", "figures", "figure6_sepsis_owate.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join("results", "figures", "figure6_sepsis_owate.pdf"), bbox_inches="tight")
print("settles within +/-0.2 pp at n =", n02, "| within +/-0.15 pp at n =", n015, "| final =", round(final, 4))
print("min/max of plotted series (n>=200):", round(f[m].min(), 2), round(f[m].max(), 2))
