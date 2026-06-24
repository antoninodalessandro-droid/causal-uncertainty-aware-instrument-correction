#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 — Uncertainty propagation (final revision)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

rng = np.random.default_rng(42)

w = np.linspace(0, np.pi, 4097)
f = w / np.pi

w0_nom = 2.0 * np.pi * 0.12
zeta_nom = 0.70
gain_nom = 1.0
T = 1.0

Omega = (2.0 / T) * np.tan(w / 2.0)

def H_analog_param(Omega, w0, zeta, gain):
    s = 1j * Omega
    return gain * (w0**2) / (s**2 + 2.0 * zeta * w0 * s + w0**2)

Ns = 300

sig_w0 = 0.05
sig_zeta = 0.05
sig_gain = 0.02

Hs = np.zeros((Ns, w.size), dtype=complex)
Gs2 = np.zeros((Ns, w.size), dtype=float)

lam = 1e-3

for k in range(Ns):
    w0 = w0_nom * (1.0 + rng.normal(0.0, sig_w0))
    zeta = zeta_nom * (1.0 + rng.normal(0.0, sig_zeta))
    gain = gain_nom * (1.0 + rng.normal(0.0, sig_gain))

    H = H_analog_param(Omega, w0, zeta, gain)
    H = H / H[0]

    Hs[k, :] = H

    Hmag2 = np.abs(H)**2
    G = np.conjugate(H) / (Hmag2 + lam)
    Gs2[k, :] = np.abs(G)**2

Habs = np.abs(Hs)
H_med = np.median(Habs, axis=0)
H_p02 = np.percentile(Habs, 2, axis=0)
H_p98 = np.percentile(Habs, 98, axis=0)

G2_med = np.median(Gs2, axis=0)
G2_p02 = np.percentile(Gs2, 2, axis=0)
G2_p98 = np.percentile(Gs2, 98, axis=0)

G2_med_norm = G2_med / G2_med[0]
G2_p02_norm = G2_p02 / G2_med[0]
G2_p98_norm = G2_p98 / G2_med[0]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
plt.subplots_adjust(wspace=0.30)

ax = axes[0]
ax.fill_between(
    f, H_p02, H_p98,
    alpha=0.25, linewidth=0.0,
    label="Uncertainty band (2–98%)"
)
ax.plot(f, H_med, lw=1.5, label="Median")

ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_ylim(1e-7, None)
ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
ax.set_ylabel(r"Effective response magnitude $|H(e^{j\omega})|$")
ax.set_title(r"(a) Propagated uncertainty in $|H(e^{j\omega})|$")

ax.minorticks_off()
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_locator(NullLocator())

ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, loc="lower left")

ax = axes[1]
ax.fill_between(
    f, G2_p02_norm, G2_p98_norm,
    alpha=0.25, linewidth=0.0,
    label="Uncertainty band (2–98%)"
)
ax.plot(f, G2_med_norm, lw=1.5, label="Median")

ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_ylim(1e-8, None)
ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
ax.set_ylabel(r"Relative noise amplification $|G_{\lambda}(e^{j\omega})|^2$")
ax.set_title(r"(b) Uncertainty in inverse-noise amplification ($\lambda = 10^{-3}$)")

ax.minorticks_off()
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_locator(NullLocator())

ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, loc="lower left")

fig.savefig("Fig06_uncertainty_propagation.png", dpi=300, bbox_inches="tight")
fig.savefig("Fig06_uncertainty_propagation.pdf", bbox_inches="tight")

plt.close(fig)