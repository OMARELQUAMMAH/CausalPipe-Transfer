# -*- coding: utf-8 -*-
# Omar El Quammah -- Nanjing University of Information Science & Technology, 2026
"""
CausalPipe-Transfer: Figure 5 (Retail Robustness Checks), KBS R1 revision.

Every number below is copied from the output of the retail script run AFTER
dropping the 1,300 zero-price rows (retail_FINAL_two_part_spline_results_NOZERO.txt,
n = 334,137; 165,984 discounted, 168,153 not discounted). Nothing is read from
the old (pre-correction) results and nothing is drawn between points:

  (a) Part 1 and Part 4 of the output (extensive margin, 5th->95th contrast)
  (b) the 11 spline points with their 95% CIs, exactly as printed in Part 2
  (c) Part 3: real linear-in-W slope (+0.1227, log scale) against the
      permutation null (mean +0.0004, SD 0.0091, 200 shuffles). The 200
      permutation draws are not saved, so the null is drawn as a normal curve
      with the stated mean and SD (labelled as such); the real estimate is
      placed at its actual value.
  (d) Part 0 and Part 1 (naive +12.3% vs AIPW +15.6%)

Saves next to this script (figure5_retail_robustness.png / .pdf).
"""
import os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"extensive": "#0072B2", "intensive": "#009E73", "naive": "#D55E00",
         "adjusted": "#009E73", "null": "#999999", "real": "#CC79A7", "zero": "#444444"}

# ---- values from the run output -----------------------------------------
EXT, EXT_LO, EXT_HI = 15.6, 15.2, 16.1          # Part 1
INT, INT_LO, INT_HI = -25.4, -26.7, -24.9       # Part 4 (5th->95th pct, 0.0080 -> 0.3700)
P5, P95 = 0.0080, 0.3700
W   = np.array([0.0030, 0.0473, 0.0916, 0.1359, 0.1802, 0.2245,
                0.2688, 0.3131, 0.3574, 0.4017, 0.4460])
EFF = np.array([49.6, -4.1, -14.3, -9.4, -2.8, 0.8, 2.3, 2.9, 3.5, 5.0, 8.0])
LO  = np.array([48.5, -4.7, -14.5, -9.7, -3.3, 0.5, 1.9, 2.4, 2.2, 3.9, 7.0])
HI  = np.array([51.0, -3.8, -14.0, -9.1, -2.5, 1.3, 3.3, 3.5, 4.1, 5.8, 11.1])
REAL_SLOPE, NULL_MEAN, NULL_SD, P_VAL = 0.1227, 0.0004, 0.0091, 0.0050   # Part 3
NAIVE, ADJ = 12.3, 15.6                                                   # Part 0, Part 1

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) two-part model
ax = axes[0, 0]
ate = [EXT, INT]; lo = [EXT_LO, INT_LO]; hi = [EXT_HI, INT_HI]
err_lo = [ate[i] - lo[i] for i in range(2)]; err_hi = [hi[i] - ate[i] for i in range(2)]
x = np.arange(2)
ax.bar(x, ate, yerr=[err_lo, err_hi], capsize=5,
       color=[COLOR["extensive"], COLOR["intensive"]], edgecolor="white", linewidth=0.5)
ax.axhline(0, color=COLOR["zero"], linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["Extensive margin\n(any discount vs. none)",
                    "Intensive margin\n(depth, 5th→95th pct)"], fontsize=9)
ax.set_ylabel("ATE (% scale)", fontsize=10)
ax.set_ylim(-31, 21)
ax.set_title("(a) Two-part model (Table 6)", fontsize=11)
for xi, v in zip(x, ate):
    va = "bottom" if v >= 0 else "top"
    pad = err_hi[xi] + 0.6 if v >= 0 else -(err_lo[xi] + 0.6)
    ax.text(xi, v + pad, f"{v:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# (b) real spline curve with its CI band
ax2 = axes[0, 1]
ax2.axvspan(P5, P95, color=COLOR["extensive"], alpha=0.08,
            label=f"5th→95th pct span\n(headline: {INT:+.1f}%)")
ax2.fill_between(W, LO, HI, color=COLOR["intensive"], alpha=0.25, label="95% CI")
ax2.plot(W, EFF, color=COLOR["intensive"], linewidth=2, marker="o", markersize=5,
         markeredgecolor="black", label="B-spline estimate (11 evaluated points)")
ax2.axhline(0, color=COLOR["zero"], linewidth=0.8)
ax2.set_xlabel("Discount depth, W = 1 − discount", fontsize=10)
ax2.set_ylabel("Intensive-margin effect, relative to mean (%)", fontsize=10)
ax2.set_title("(b) B-spline dose-response curve (nonmonotonic)\n(Sec. 4.2.3)", fontsize=11)
ax2.legend(fontsize=7.5, loc="upper right", frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

# (c) permutation placebo test
ax3 = axes[1, 0]
xs = np.linspace(-0.045, 0.14, 800)
ax3.plot(xs, norm.pdf(xs, NULL_MEAN, NULL_SD), color=COLOR["null"], linewidth=1.5,
         label=f"Permutation null, normal approximation\n(200 shuffles; mean +{NULL_MEAN:.4f}, SD {NULL_SD})")
ax3.fill_between(xs, norm.pdf(xs, NULL_MEAN, NULL_SD), color=COLOR["null"], alpha=0.15)
ax3.axvline(REAL_SLOPE, color=COLOR["real"], linewidth=2, linestyle="--",
            label=f"Real estimate (+{REAL_SLOPE:.4f})")
ax3.set_xlabel("Linear-in-W dose-response slope (log scale)", fontsize=9.5)
ax3.set_ylabel("Density", fontsize=10)
ax3.set_title(f"(c) Permutation placebo test (p = {P_VAL:.4f})\n(Sec. 4.2.4)", fontsize=11)
ax3.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.57, 0.60), frameon=False)
ax3.spines[["top", "right"]].set_visible(False)

# (d) confounding structure
ax4 = axes[1, 1]
xd = np.arange(2); vals = [NAIVE, ADJ]
ax4.bar(xd, vals, color=[COLOR["naive"], COLOR["adjusted"]], edgecolor="white", linewidth=0.5)
ax4.set_xticks(xd)
ax4.set_xticklabels(["Naive comparison\n(unadjusted)", "DR-adjusted\n(AIPW, extensive margin)"], fontsize=9)
ax4.set_ylabel("Effect on log-scale sale amount (%)", fontsize=10)
ax4.set_title("(d) Confounding structure (Sec. 4.2.2)", fontsize=11)
for xi, v in zip(xd, vals):
    ax4.text(xi, v + 0.3, f"+{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax4.set_ylim(0, 20)
ax4.text(0.5, 18.4, "Naive comparison UNDERSTATES the causal effect by 3.3 pp\n"
         "(confounders partially mask, not inflate, the association)",
         ha="center", va="center", fontsize=8, style="italic", color="#444444")
ax4.spines[["top", "right"]].set_visible(False)

fig.suptitle("CausalPipe-Transfer: Retail Robustness Checks\n"
             "Two-part model, B-spline dose-response, placebo test, and confounding structure",
             fontsize=12, y=1.0)
fig.tight_layout()
try:
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUT_DIR = os.getcwd()
png = os.path.join(OUT_DIR, "figure5_retail_robustness.png")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
print("Saved:", png)
