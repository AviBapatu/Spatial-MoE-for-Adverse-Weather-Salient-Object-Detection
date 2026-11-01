"""Shared matplotlib style so every generated figure matches the manuscript.

Text width of the paper is 153 mm (6.02 in); figures are sized to it directly so
fonts print at their nominal size (8 pt, the same as the 8 pt captions).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

TEXT_W = 6.02          # inches, = 153 mm class text width

# One palette, reused everywhere (matches the pgfplots ranking figure:
# violet = ours, orange = WFANet / "unresolved", grey = other).
C_EQUIV = "#2F5FA7"    # certified equivalent / ablation family
C_SEED = "#8C8C8C"     # seed replicates
C_UNRES = "#E08A2E"    # not resolvable at this sample size
C_SEP = "#B8392B"      # separates from the reference
C_OURS = "#7B5EA7"     # reported model R
C_BAND = "#2F5FA7"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Liberation Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,     # embed TrueType, keeps text selectable and sharp
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def light_grid(ax, axis="x"):
    ax.grid(axis=axis, color="#BBBBBB", linewidth=0.35, linestyle=(0, (1, 2)))
    ax.set_axisbelow(True)
