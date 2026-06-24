#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 — Self-noise amplification bounds and bias–noise trade-off
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

Omega = (2.0 / T) * np.tan(w / 2.0)

def H_analog(Omega):
    s = 1j * Omega
    return (w0**2) / (s**2 + 2.0*zeta*w0*s + w0**2)

H = H_analog(Omega)
H /= H[0]

Hmag2 = np.abs(H)**2
Hc = np.conjugate(H)

lam0 = 1e-3
G_lam0 = Hc / (Hmag2 + lam0)

S_eta = 1.0

A_lower = S_eta / np.maximum(Hmag2, 1e-12)
A_lam0 = (np.abs(G_lam0)**2) * S_eta

A_lower /= A_lower[0]
A_lam0 /= A_lam0[0]

lams = np.logspace(-6, 0, 160)
dw = w[1] - w[0]

bias_rms_db = np.zeros_like(lams)
noise_rms = np.zeros_like(lams)

for i, lam in enumerate(lams):
    Tlam = Hmag2 / (Hmag2 + lam)
    bias_db = 20.0 * np.log10(np.maximum(Tlam, 1e-30))
    bias_rms_db[i] = np.sqrt(np.mean(bias_db**2))

    Glam = Hc / (Hmag2 + lam)
    noise_rms[i] = np.sqrt(np.sum(np.abs(Glam)**2 * S_eta) * dw)

noise_rms /= noise_rms.min()

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
plt.subplots_adjust(wspace=0.32)

ax = axes[0]
ax.plot(f, A_lower, lw=1.5,
        label=r"Lower bound $\propto |H(e^{j\omega})|^{-2}$")
ax.plot(f, A_lam0, lw=1.5,
        label=rf"Regularized inverse ($\lambda={lam0:g}$)")
ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
ax.set_ylabel("Relative noise amplification")
ax.set_title("(a) Self-noise amplification bound")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, loc="lower left")

ax = axes[1]
bias_color = "tab:orange"
noise_color = "tab:blue"

ax.plot(lams, bias_rms_db, color=bias_color, lw=1.5,
        label="Bias metric (RMS, dB)")
ax.set_xscale("log")
ax.set_xlabel(r"Regularization strength $\lambda$")
ax.set_ylabel("Bias (RMS, dB)", color=bias_color)
ax.tick_params(axis="y", colors=bias_color)
ax.set_title("(b) Bias–noise trade-off")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

ax2 = ax.twinx()
ax2.plot(lams, noise_rms, color=noise_color, lw=1.5,
         label="Noise metric (RMS, normalized)")
ax2.set_ylabel("Noise (RMS, normalized)", color=noise_color)
ax2.tick_params(axis="y", colors=noise_color)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2,
          frameon=False, loc="upper center")

fig.savefig("Fig05_self_noise_bounds_and_tradeoff.png", dpi=300, bbox_inches="tight")
fig.savefig("Fig05_self_noise_bounds_and_tradeoff.pdf", bbox_inches="tight")