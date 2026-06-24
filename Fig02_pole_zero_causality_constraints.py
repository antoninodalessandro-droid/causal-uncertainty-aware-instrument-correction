#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 — Causality and stability constraints in the pole–zero domain
Refined journal-style version (GJI), with safe label margins.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Global style
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(wspace=0.35)

# ----------------------------
# Helper function
# ----------------------------
def draw_zplane(ax, poles, zeros, title, footer,
                lim=1.7, label_pos=1.33):
    """
    Draw a conceptual z-plane:
    - unit circle
    - axes
    - poles (x) and zeros (o)
    - internal axis labels placed safely within bounds
    """
    theta = np.linspace(0, 2*np.pi, 400)

    # Unit circle |z|=1
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.2)

    # Axes
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)

    # Poles and zeros
    if len(poles) > 0:
        ax.plot(np.real(poles), np.imag(poles),
                'kx', markersize=9, markeredgewidth=1.5)
    if len(zeros) > 0:
        ax.plot(np.real(zeros), np.imag(zeros),
                'ko', markersize=9, fillstyle='none', markeredgewidth=1.5)

    # Limits and aspect
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # No ticks (clean journal style)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=12)

    # Axis labels (moved inward to avoid border overlap)
    ax.text(label_pos, 0.05, r'$\Re\{z\}$', fontsize=10)
    ax.text(0.05, label_pos, r'$\Im\{z\}$', fontsize=10)

    # Footer statement (technical)
    ax.text(0, -1.35, footer, ha="center", fontsize=11)

# ----------------------------
# Panel (a): minimum-phase
# ----------------------------
poles_min = np.array([0.55 + 0.25j, 0.55 - 0.25j])
zeros_min = np.array([0.35 + 0.45j, 0.35 - 0.45j])

draw_zplane(
    axes[0],
    poles=poles_min,
    zeros=zeros_min,
    title="(a) Stable and minimum-phase system",
    footer="Causal, stable inverse admissible",
    lim=1.7,
    label_pos=1.33
)

# ----------------------------
# Panel (b): non–minimum-phase
# ----------------------------
poles_nonmin = np.array([0.6 + 0.3j, 0.6 - 0.3j])
zeros_nonmin = np.array([1.25 + 0.0j])

draw_zplane(
    axes[1],
    poles=poles_nonmin,
    zeros=zeros_nonmin,
    title="(b) Stable but non–minimum-phase system",
    footer="Causal inverse not admissible",
    lim=1.7,
    label_pos=1.33
)

# ----------------------------
# Save
# ----------------------------
fig.savefig("Fig02_pole_zero_causality_constraints.png",
            dpi=300, bbox_inches="tight")
fig.savefig("Fig02_pole_zero_causality_constraints.pdf",
            bbox_inches="tight")
