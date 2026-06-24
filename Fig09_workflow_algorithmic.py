#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig08_workflow_algorithmic.py

Figure 8 — Algorithmic workflow for physically consistent instrument correction
STYLE: matched to Figure 1 (sharp-corner rectangles, light-grey fill, thin black borders,
simple arrows, clean typography).

Fixes requested:
1) Avoid truncation of the last (rightmost) box by ensuring everything stays within the
   axes (0–1) frame AND by saving with a small padding.
2) Reduce horizontal elongation by breaking the last box title onto two lines and by
   slightly increasing right margin.

Outputs:
- Fig08_workflow_algorithmic.png
- Fig08_workflow_algorithmic.pdf
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# -----------------------------
# Global style
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

# -----------------------------
# Helpers
# -----------------------------
def add_rect(ax, x, y, w, h, text, fc="#EFEFEF", ec="black", lw=1.2, fontsize=11):
    r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, transform=ax.transAxes)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize)
    return r

def add_arrow(ax, x0, y0, x1, y1, lw=1.2):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color="black",
                        transform=ax.transAxes)
    ax.add_patch(a)
    return a

def left_mid(r):
    x, y = r.get_x(), r.get_y()
    return (x, y + r.get_height()/2)

def right_mid(r):
    x, y = r.get_x(), r.get_y()
    return (x + r.get_width(), y + r.get_height()/2)

def top_mid(r):
    x, y = r.get_x(), r.get_y()
    return (x + r.get_width()/2, y + r.get_height())

def bottom_mid(r):
    x, y = r.get_x(), r.get_y()
    return (x + r.get_width()/2, y)

# -----------------------------
# Canvas
# -----------------------------
fig = plt.figure(figsize=(16, 4.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()

# -----------------------------
# Layout: keep all boxes safely within [0,1]
# -----------------------------
y_main = 0.56
h_box  = 0.22

# Slightly smaller width and slightly larger right margin to prevent cropping
w_box  = 0.120
gap    = 0.020
x0     = 0.030

# Ensure last box ends at <= 0.97
# total_width = x0 + 7*w_box + 6*gap
# With values above: 0.03 + 0.84 + 0.12 = 0.99 (tight but OK with pad_inches below)
xs = [x0 + i*(w_box + gap) for i in range(7)]

# -----------------------------
# Main workflow boxes (left → right)
# -----------------------------
b1 = add_rect(
    ax, xs[0], y_main, w_box, h_box,
    "Instrument model\n(continuous domain)\n\n$H(s)$"
)

b2 = add_rect(
    ax, xs[1], y_main, w_box, h_box,
    "Physical constraints\n\nCausality\nStability\nBounded inverse"
)

b3 = add_rect(
    ax, xs[2], y_main, w_box, h_box,
    "Discretization\nand digital realization\n\n$H(s)\\;\\to\\;H(z)$"
)

b4 = add_rect(
    ax, xs[3], y_main, w_box, h_box,
    "Regularized inverse\n\n$G_{\\lambda}(z)$\n($\\lambda$ controls trade-off)",
    fontsize=10
)

b5 = add_rect(
    ax, xs[4], y_main, w_box, h_box,
    "Diagnostic metrics\n\nBias $\\Delta A(\\omega)$\nNoise $|G_{\\lambda}|^2$\nUncertainty bands\nKernel diagnostics",
    fontsize=10
)

b6 = add_rect(
    ax, xs[5], y_main, w_box, h_box,
    "Quality map\nand selection\n\n$\\lambda^{*}$,\n$f_{\\mathrm{eff}}$",
    fontsize=10
)

# Last box: title forced on two lines to reduce perceived elongation
b7 = add_rect(
    ax, xs[6], y_main, w_box, h_box,
    "Corrected signal\n(digital domain)\n\n$\\hat{x}[n]$"
)

# -----------------------------
# Arrows between main boxes
# -----------------------------
for L, R in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6), (b6, b7)]:
    xL, yL = right_mid(L)
    xR, yR = left_mid(R)
    add_arrow(ax, xL, yL, xR, yR)

# -----------------------------
# Input / Output labels (kept compact)
# -----------------------------
ax.text(x0 - 0.015, y_main + h_box/2, "Input", transform=ax.transAxes,
        ha="right", va="center", fontsize=11)

ax.text(xs[6] + w_box + 0.015, y_main + h_box/2, "Output", transform=ax.transAxes,
        ha="left", va="center", fontsize=11)

# -----------------------------
# Recorded data stream y[n] (top branch)
# -----------------------------
tag_w, tag_h = 0.060, 0.075
tag_x = xs[3] + w_box/2 - tag_w/2
tag_y = y_main + h_box + 0.055

tag = add_rect(ax, tag_x, tag_y, tag_w, tag_h, "$y[n]$", fc="#FFFFFF", lw=1.2, fontsize=11)

ax.text(tag_x + tag_w/2, tag_y + tag_h + 0.010, "Recorded data stream",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

xT, yT = bottom_mid(tag)
xB, yB = top_mid(b7)
add_arrow(ax, xT, yT, xB, yB, lw=1.1)

ax.text((xT + xB)/2, (yT + yB)/2 + 0.015,
        "$G_{\\lambda^{*}}(z)$",
        transform=ax.transAxes, ha="center", va="center", fontsize=10)

# -----------------------------
# Bottom constraints line
# -----------------------------
ax.text(0.5, 0.18,
        "Constraints: causality, stability, regularization, uncertainty-aware design",
        transform=ax.transAxes, ha="center", va="center", fontsize=11)

# -----------------------------
# Save (use pad_inches to prevent right-edge truncation)
# -----------------------------
fig.savefig("Fig08_workflow_algorithmic.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
fig.savefig("Fig08_workflow_algorithmic.pdf", bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
