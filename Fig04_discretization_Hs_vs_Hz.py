#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 — Discretization effects: H(s) vs H(z) are not equivalent
(No zoom insets; clean two-panel figure)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

w = np.linspace(0, np.pi, 4097)
f = w / np.pi

w0 = 2.0 * np.pi * 0.12
zeta = 0.70
T = 1.0

Omega_naive = w / T
Omega_warp  = (2.0 / T) * np.tan(w / 2.0)

def H_analog(Omega):
    s = 1j * Omega
    return (w0**2) / (s**2 + 2.0*zeta*w0*s + w0**2)

Ha = H_analog(Omega_naive)
Hd = H_analog(Omega_warp)

Ha /= Ha[0]
Hd /= Hd[0]

eps_db = 20.0 * np.log10(np.abs(Hd) / np.abs(Ha))

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
plt.subplots_adjust(wspace=0.28)

ax = axes[0]
ax.plot(f, np.abs(Ha), lw=1.5, label=r"$|H_a(e^{j\omega})|$ (naïve from $H(s)$)")
ax.plot(f, np.abs(Hd), lw=1.5, label=r"$|H_d(e^{j\omega})|$ (effective $H(z)$ proxy)")
ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_ylim(1e-7, None)
ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
ax.set_ylabel("Magnitude")
ax.set_title(r"(a) $H(s)$-based vs effective digital response")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, loc="lower left")

ax = axes[1]
ax.plot(f, eps_db, lw=1.5)
ax.axhline(0.0, lw=0.9, ls="--", color="black")
ax.set_xlim(0, 1)
ax.set_ylim(-40, 5)
ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
ax.set_ylabel(r"Amplitude bias $\Delta A(\omega)$ [dB]")
ax.set_title(r"(b) Discretization-induced amplitude bias")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

fig.savefig("Fig04_discretization_Hs_vs_Hz.png", dpi=300, bbox_inches="tight")
fig.savefig("Fig04_discretization_Hs_vs_Hz.pdf", bbox_inches="tight")