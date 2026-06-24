#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 — Ideal inverse vs causal regularized inverse (theoretical demonstration, no data)
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

# ----------------------------
# Utility functions
# ----------------------------
def freq_response(b, a, w):
    ejw = np.exp(-1j * w)
    num = np.zeros_like(w, dtype=complex)
    den = np.zeros_like(w, dtype=complex)
    for k, bk in enumerate(b):
        num += bk * ejw**k
    for k, ak in enumerate(a):
        den += ak * ejw**k
    return num / den

def poly_from_roots_zminus1(roots):
    coeff = np.array([1.0])
    for r in roots:
        coeff = np.convolve(coeff, np.array([1.0, -r], dtype=complex))
    return coeff

def reflect_outside_unit_circle(zeros):
    z_new = []
    for z0 in zeros:
        if np.abs(z0) > 1.0:
            z_new.append(1.0 / np.conjugate(z0))
        else:
            z_new.append(z0)
    return np.array(z_new, dtype=complex)

def stable_iir_inverse_coeffs(b, a):
    return np.array(a, dtype=complex), np.array(b, dtype=complex)

# ----------------------------
# Forward system H(z)
# ----------------------------
poles = np.array([0.65 + 0.25j, 0.65 - 0.25j])
zeros = np.array([1.35 + 0.0j, 0.30 + 0.40j, 0.30 - 0.40j])

b = poly_from_roots_zminus1(zeros)
a = poly_from_roots_zminus1(poles)

H0 = np.sum(b) / np.sum(a)
b /= H0

zeros_min = reflect_outside_unit_circle(zeros)
b_min = poly_from_roots_zminus1(zeros_min) / H0

# ----------------------------
# Frequency domain
# ----------------------------
nfft = 4096
w = np.linspace(0, np.pi, 2049)
w_norm = w / np.pi

H = freq_response(b, a, w)
H_inv_ideal = 1.0 / H

b_inv0, a_inv0 = stable_iir_inverse_coeffs(b_min, a)
G0 = freq_response(b_inv0, a_inv0, w)

lam = 2e-3
weight = 1.0 + (w_norm**6) / lam
G_reg = G0 / weight

# ----------------------------
# Time-domain kernels
# ----------------------------
Hfull = np.zeros(nfft, dtype=complex)
Hfull[:nfft//2 + 1] = np.interp(
    np.linspace(0, np.pi, nfft//2 + 1), w, H_inv_ideal
)
Hfull[nfft//2 + 1:] = np.conjugate(Hfull[1:nfft//2][::-1])
g_ideal = np.real(np.fft.fftshift(np.fft.ifft(Hfull)))

Gfull = np.zeros(nfft, dtype=complex)
Gfull[:nfft//2 + 1] = np.interp(
    np.linspace(0, np.pi, nfft//2 + 1), w, G_reg
)
Gfull[nfft//2 + 1:] = np.conjugate(Gfull[1:nfft//2][::-1])
g_reg = np.real(np.fft.ifft(Gfull))

g_ideal /= np.max(np.abs(g_ideal))
g_reg /= np.max(np.abs(g_reg))

N = 220
n0 = nfft // 2
n_ideal = np.arange(-N//2, N//2)
n_reg = np.arange(0, N)

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
plt.subplots_adjust(wspace=0.28)

# (a) Frequency domain
ax = axes[0]
ax.plot(w_norm, np.abs(H), lw=1.5, label=r"$|H(e^{j\omega})|$")
ax.plot(w_norm, np.abs(H_inv_ideal), lw=1.5,
        label=r"$|H^{-1}(e^{j\omega})|$ (ideal)")
ax.plot(w_norm, np.abs(G_reg), lw=1.5,
        label=r"$|G_{\mathrm{reg}}(e^{j\omega})|$ (causal, reg.)")
ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
ax.set_ylabel("Magnitude")
ax.set_title("(a) Frequency-domain comparison")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, loc="lower left")

# (b) Time domain
ax = axes[1]
ax.plot(n_ideal, g_ideal[n0-N//2:n0+N//2],
        lw=1.5, label="Ideal inverse kernel (non-causal)")
ax.plot(n_reg, g_reg[:N],
        lw=1.5, label="Regularized inverse kernel (causal)")
ax.set_xlim(-N//2, N)
ax.set_ylim(-1.1, 0.9)
ax.set_xlabel(r"Sample index $n$")
ax.set_ylabel("Normalized amplitude")
ax.set_title("(b) Time-domain kernels")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, loc="upper left")

# Save
fig.savefig("Fig03_ideal_vs_causal_regularized_inverse.png",
            dpi=300, bbox_inches="tight")
fig.savefig("Fig03_ideal_vs_causal_regularized_inverse.pdf",
            bbox_inches="tight")