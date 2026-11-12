"""Forest plot of Table 12: paired MAE difference of every arm against A0."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps

ps.apply()

# (label, delta, lo, hi, TOST p, group)  -- numbers copied from Table 12.
rows = [
    ("No MoE (A6)",                    -0.000019, -0.000511, +0.000485, 0.0001, "eq"),
    ("Dense gate, batch 40 (B2)",      +0.000075, -0.000357, +0.000517, 0.0001, "eq"),
    ("Routing at $1/4$ only (A8)",     -0.000117, -0.000562, +0.000354, 0.0001, "eq"),
    ("No load-balance loss (A4)",      +0.000118, -0.000282, +0.000506, 0.0000, "eq"),
    ("2 experts (A2)",                 -0.000173, -0.000473, +0.000124, 0.0000, "eq"),
    ("Seed 43 (A0)",                   +0.000462, -0.000153, +0.001172, 0.0566, "seed"),
    ("Dense expert (A5)",              -0.000488, -0.001231, +0.000204, 0.0828, "unres"),
    ("Seed 44 (A0)",                  -0.000497, -0.001115, +0.000087, 0.0510, "seed"),
    ("Top-1, 4 experts (A3)",          +0.001599, +0.001034, +0.002125, 0.9808, "sep"),
    ("14 epochs (A1)",                 -0.002224, -0.002788, -0.001690, 1.0000, "sep"),
]
style = {
    "eq":    (ps.C_EQUIV, "o"),
    "seed":  (ps.C_SEED,  "D"),
    "unres": (ps.C_UNRES, "s"),
    "sep":   (ps.C_SEP,   "o"),
}
S = 1e3  # plot in units of 10^-3 MAE

fig, ax = plt.subplots(figsize=(ps.TEXT_W, 2.75))
n = len(rows)
ax.axvspan(-1, 1, color=ps.C_BAND, alpha=0.10, lw=0, zorder=0)
ax.axvline(0, color="black", lw=0.6, zorder=1)
for x in (-1, 1):
    ax.axvline(x, color=ps.C_BAND, lw=0.7, ls="--", zorder=1)

for i, (lab, d, lo, hi, p, g) in enumerate(rows):
    y = n - 1 - i
    c, m = style[g]
    ax.plot([lo * S, hi * S], [y, y], color=c, lw=1.3, solid_capstyle="butt", zorder=3)
    ax.plot([d * S], [y], marker=m, ms=4.4, color=c, mec="white", mew=0.5, zorder=4)
    ax.text(3.05, y, f"{d*S:+.2f}".replace("-", "\u2212"), va="center", ha="right", fontsize=7.5)
    ax.text(3.95, y, f"{p:.4f}" if p < 0.9995 else "1.0000", va="center", ha="right", fontsize=7.5)

# group separators
for yb in (n - 5.5, n - 8.5):
    ax.axhline(yb, color="#999999", lw=0.4, ls=":", zorder=0)

ax.set_yticks(range(n))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_ylim(-0.7, n - 0.3 + 0.55)
ax.set_xlim(-3.1, 4.0)
ax.set_xticks([-3, -2, -1, 0, 1, 2])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel(r"Paired $\Delta$ real-test MAE vs. A0  ($\times10^{-3}$;  negative = arm better)", loc="left")
ax.xaxis.set_label_coords(0.0, -0.13)
ps.light_grid(ax, "x")
# column headers for the right-hand numbers
ax.text(3.05, n - 0.35, r"$\Delta$", ha="right", va="bottom", fontsize=7.5, fontweight="bold")
ax.text(3.95, n - 0.35, "TOST $p$", ha="right", va="bottom", fontsize=7.5, fontweight="bold")
ax.text(0.0, n - 0.35, r"equivalence margin $\pm1$", ha="center", va="bottom",
        fontsize=7.5, color=ps.C_BAND,
        bbox=dict(fc="white", ec="none", pad=1.0))
# hide x tick marks beyond the plotted axis region
ax.spines["bottom"].set_bounds(-3.1, 2.6)

handles = [
    Line2D([], [], color=ps.C_EQUIV, marker="o", ms=4.4, lw=1.3, label="certified equivalent"),
    Line2D([], [], color=ps.C_SEED, marker="D", ms=4.0, lw=1.3, label="seed replicate"),
    Line2D([], [], color=ps.C_UNRES, marker="s", ms=4.0, lw=1.3, label="unresolved"),
    Line2D([], [], color=ps.C_SEP, marker="o", ms=4.4, lw=1.3, label="separates"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, -0.20), ncol=4,
          frameon=False, handlelength=1.6, columnspacing=1.6)
fig.savefig("../fig_forest_equivalence.pdf")
fig.savefig("/home/claude/preview_forest.png", dpi=170)
