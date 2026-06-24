#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 — Instrument correction as a constrained inverse problem in the digital domain
Final clean version: no title, no overlaps.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ----------------------------
# Global style
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

# ----------------------------
# Helpers
# ----------------------------
def add_box(ax, x, y, w, h, text, fs=11, fc="#F2F2F2", ec="black", lw=1.1):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)
    return rect

def add_arrow(ax, p0, p1, lw=1.1, ms=12):
    ax.add_patch(FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color="black"
    ))

def mid_right(r):
    return (r.get_x() + r.get_width(), r.get_y() + r.get_height()/2)

def mid_left(r):
    return (r.get_x(), r.get_y() + r.get_height()/2)

def top_center(r):
    return (r.get_x() + r.get_width()/2, r.get_y() + r.get_height())

# ----------------------------
# Canvas
# ----------------------------
fig = plt.figure(figsize=(14.5, 6.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1.03)   # safe right margin
ax.set_ylim(0, 1)
ax.axis("off")

# ----------------------------
# Layout parameters
# ----------------------------
y_main = 0.52
h = 0.14
w = 0.12
g = 0.022
x0 = 0.03

# ----------------------------
# Main chain
# ----------------------------
b_x   = add_box(ax, x0 + (w+g)*0, y_main, w, h, "Ground motion\n$\\,x(t)$", fs=12)
b_s   = add_box(ax, x0 + (w+g)*1, y_main, w, h, "Sensor", fs=12)
b_e   = add_box(ax, x0 + (w+g)*2, y_main, w, h, "Electronics", fs=12)
b_aa  = add_box(ax, x0 + (w+g)*3, y_main, w, h, "Anti-alias\nfilter", fs=12)
b_adc = add_box(ax, x0 + (w+g)*4, y_main, w, h, "ADC\n(sampling)", fs=12)
b_dec = add_box(ax, x0 + (w+g)*5, y_main, w, h, "Decimation /\nresampling", fs=12)
b_y   = add_box(ax, x0 + (w+g)*6, y_main, w, h, "Recorded data\n$\\,y[n]$", fs=12)

main_boxes = [b_x, b_s, b_e, b_aa, b_adc, b_dec, b_y]
for i in range(len(main_boxes)-1):
    add_arrow(ax, mid_right(main_boxes[i]), mid_left(main_boxes[i+1]))

# ----------------------------
# Noise annotations
# ----------------------------
ax.text(*top_center(b_s), r"$+\,n_s(t)$", ha="center", va="bottom", fontsize=10)
ax.text(*top_center(b_e), r"$+\,n_e(t)$", ha="center", va="bottom", fontsize=10)
ax.text(*top_center(b_adc), r"$+\,n_q[n]$", ha="center", va="bottom", fontsize=10)

# ----------------------------
# Group labels
# ----------------------------
ax.text(
    (b_s.get_x() + b_aa.get_x() + w) / 2,
    y_main + h + 0.075,
    "Analog response $H(s)$ (poles–zeros, gain)",
    ha="center", fontsize=12
)

ax.text(
    (b_adc.get_x() + b_y.get_x() + w) / 2,
    y_main + h + 0.075,
    "Effective digital response $H(z)$",
    ha="center", fontsize=12
)

# ----------------------------
# Sampling / timing (shifted down to avoid overlap)
# ----------------------------
ax.text(
    b_adc.get_x() + w/2,
    y_main - 0.075,   # LOWER than before
    "Sampling: $t \\mapsto n\\Delta t$\nClock errors: $\\delta t[n]$",
    ha="center", fontsize=10
)

# ----------------------------
# Inverse branch
# ----------------------------
y_inv = 0.30
w_g = 0.22
w_xh = 0.18
h_inv = 0.13
x_inv = 0.48

b_g = add_box(ax, x_inv, y_inv, w_g, h_inv,
              "Causal minimum-phase\ninverse $\\,G(z)$",
              fs=11, fc="white", lw=1.3)

b_xh = add_box(ax, x_inv + w_g + 0.035, y_inv, w_xh, h_inv,
               "Estimated ground motion\n$\\,\\hat{x}[n]$",
               fs=11, fc="white", lw=1.3)

# Connection from y[n] to inverse (raised start to avoid text)
x_from = b_y.get_x() + w/2
y_from = b_y.get_y() + 0.01   # start slightly INSIDE the box

x_to = b_g.get_x() + w_g/2
y_to = b_g.get_y() + h_inv

ax.plot([x_from, x_from], [y_from, y_to + 0.035], color="black", linewidth=1.1)
ax.plot([x_from, x_to], [y_to + 0.035, y_to + 0.035], color="black", linewidth=1.1)
add_arrow(ax, (x_to, y_to + 0.035), (x_to, y_to))

# Arrow G(z) → x̂[n]
add_arrow(ax, mid_right(b_g), mid_left(b_xh))

# ----------------------------
# Constraints line
# ----------------------------
ax.plot(
    [x_inv, x_inv + w_g + 0.035 + w_xh],
    [y_inv - 0.035, y_inv - 0.035],
    color="black", linewidth=1.0
)
ax.text(
    x_inv + (w_g + 0.035 + w_xh)/2,
    y_inv - 0.06,
    "Constraints: causality, stability, regularization, uncertainty-aware design",
    ha="center", fontsize=10.5
)

# ----------------------------
# Save (NO TITLE)
# ----------------------------
fig.savefig("Fig01_instrument_correction_block_diagram.png",
            dpi=300, bbox_inches="tight")
fig.savefig("Fig01_instrument_correction_block_diagram.pdf",
            bbox_inches="tight")
