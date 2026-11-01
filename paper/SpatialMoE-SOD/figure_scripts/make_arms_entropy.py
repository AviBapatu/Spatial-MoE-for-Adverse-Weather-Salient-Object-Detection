"""Arm-by-arm real-test MAE (Table 4) beside routing entropy (Table 15)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps

ps.apply()

# (label, real MAE, family, entropy/logE at 1/4, 1/8, 1/16)  None = no router at that scale.
arms = [
    ("R (reported, 14 ep.)",     0.0168, "R",   (0.9997, 0.9997, 0.9982)),
    ("Ref",                      0.0195, "A",   (0.9996, 0.9997, 0.9980)),
    ("A1  14 epochs",            0.0174, "A",   (0.9996, 0.9998, 0.9972)),
    ("A2  2 experts",            0.0193, "A",   (0.9994, 0.9993, 0.9923)),
    ("A3  4 experts, top-1",     0.0212, "A",   (0.9983, 0.9766, 0.9933)),
    ("A4  no load-balance",      0.0196, "A",   (0.9996, 0.9998, 0.9969)),
    ("A5  dense expert",         0.0191, "A",   (None, None, None)),
    ("A6  no MoE",               0.0195, "A",   (None, None, None)),
    ("A7  dense gate",           0.0199, "A",   (1.0000, 1.0000, 0.9996)),
    ("A8  routing at $1/4$ only", 0.0195, "A",  (0.9996, None, None)),
    ("A9  seed 43",              0.0199, "A",   (0.9995, 0.9998, 0.9972)),
    ("A10  seed 44",             0.0191, "A",   (0.9995, 0.9997, 0.9973)),
    ("B1  batch 40",             0.0199, "B",   (1.0000, 1.0000, 0.9979)),
    ("B2  batch 40, dense gate", 0.0197, "B",   (1.0000, 1.0000, 0.9974)),
]
seed_lo, seed_hi = 0.0191, 0.0199      # Table 10, three seeds of Ref
n = len(arms)
ys = np.arange(n)[::-1]
col = {"R": ps.C_OURS, "A": ps.C_EQUIV, "B": ps.C_SEED}

fig, (a0, a1) = plt.subplots(1, 2, figsize=(ps.TEXT_W, 3.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.14})

# ---- left: MAE
a0.axvspan(seed_lo, seed_hi, color=ps.C_EQUIV, alpha=0.12, lw=0)
for (lab, mae, fam, _), y in zip(arms, ys):
    a0.plot([0.0160, mae], [y, y], color="#DDDDDD", lw=0.5, zorder=1)
    filled = fam != "B"
    a0.plot(mae, y, "o", ms=4.6, color=col[fam], mfc=col[fam] if filled else "white",
            mew=1.0, zorder=3)
    a0.text(mae + 0.00022, y, f"{mae:.4f}", va="center", ha="left", fontsize=6.8, color="#333333")
a0.set_xlim(0.0160, 0.0224)
a0.set_xticks([0.016, 0.018, 0.020, 0.022])
a0.set_xticklabels(["0.016", "0.018", "0.020", "0.022"])
a0.set_xlabel("Real-test MAE $\\downarrow$")
a0.set_yticks(ys)
a0.set_yticklabels([a[0] for a in arms])
a0.tick_params(axis="y", length=0)
a0.spines["left"].set_visible(False)
ps.light_grid(a0, "x")
a0.text(0.5 * (seed_lo + seed_hi), n - 0.35, "seed range", ha="center", va="bottom",
        fontsize=7, color=ps.C_EQUIV)

# ---- right: entropy
mk = [("o", "$1/4$"), ("s", "$1/8$"), ("^", "$1/16$")]
a1.axvline(1.0, color="black", lw=0.5, ls="--", zorder=1)
for (lab, _, fam, ent), y in zip(arms, ys):
    xs = [e for e in ent if e is not None]
    if not xs:
        a1.text(0.9795, y, "no router", va="center", ha="left", fontsize=6.8,
                color="#777777", style="italic")
        continue
    a1.plot([min(xs), max(xs)], [y, y], color="#BBBBBB", lw=0.8, zorder=1)
    for e, (m, _) in zip(ent, mk):
        if e is None:
            continue
        a1.plot(e, y, m, ms=3.8, color=col[fam], mew=0, zorder=3)
a1.set_xlim(0.975, 1.0025)
a1.set_xticks([0.98, 0.99, 1.0])
a1.set_xticklabels(["0.980", "0.990", "1.000"])
a1.set_xlabel("Routing entropy / $\\log E$   (1 = uniform)")
a1.tick_params(axis="y", length=0)
a1.spines["left"].set_visible(False)
ps.light_grid(a1, "x")
a1.annotate("A3 at $1/8$: 0.977", xy=(0.9766, ys[4]), xytext=(0.9775, ys[4] - 0.62),
            fontsize=7, va="center", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=0, shrinkB=2))

# row-group separators
for yb in (ys[0] - 0.5, ys[11] - 0.5):
    for a in (a0, a1):
        a.axhline(yb, color="#999999", lw=0.4, ls=":", zorder=0)
a0.set_ylim(-0.7, n - 0.2 + 0.4)

handles = [Line2D([], [], marker=m, ls="", color="#444444", ms=4, label=l) for m, l in mk]
a1.legend(handles=handles, title="pyramid scale", loc="lower left", frameon=False,
          ncol=3, handletextpad=0.2, columnspacing=0.9, bbox_to_anchor=(0.0, -0.30),
          title_fontsize=7.5)
fig.savefig("../fig_arms_mae_entropy.pdf")
fig.savefig("/home/claude/preview_arms.png", dpi=170)
