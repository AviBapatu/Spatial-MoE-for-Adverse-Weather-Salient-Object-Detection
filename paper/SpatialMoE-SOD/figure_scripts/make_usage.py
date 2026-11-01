"""Hard-assignment expert usage per scale on the reference arm (Table 9)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps

ps.apply()
usage = {
    "$1/4$":  [21.1, 6.3, 5.3, 5.6, 1.5, 23.8, 29.7, 6.7],
    "$1/8$":  [1.1, 6.9, 3.1, 4.2, 15.5, 0.8, 40.7, 27.7],
    "$1/16$": [9.2, 5.1, 14.1, 13.1, 4.1, 27.1, 16.3, 11.1],
}
fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_W, 1.85), sharey=True,
                         gridspec_kw={"wspace": 0.08})
x = np.arange(8)
for ax, (scale, u) in zip(axes, usage.items()):
    u = np.array(u)
    cols = [ps.C_SEP if v < 2.0 else ps.C_EQUIV for v in u]
    ax.bar(x, u, width=0.72, color=cols, lw=0)
    for xi, v in zip(x, u):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3)
    ax.axhline(12.5, color="black", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_title(f"scale {scale}", fontsize=8.5, pad=3)
    ax.set_xlabel("expert index")
    ax.set_ylim(0, 47)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("share of hard assignments (%)")
axes[2].legend(handles=[Line2D([], [], color="black", lw=0.6, ls="--", label="uniform (12.5%)"),
                        Line2D([], [], marker="s", ls="", color=ps.C_SEP, ms=5, label="dead ($<2\\%$)")],
               loc="upper right", frameon=False, handletextpad=0.2)
fig.savefig("../fig_expert_usage.pdf")
fig.savefig("/home/claude/preview_usage.png", dpi=170)
