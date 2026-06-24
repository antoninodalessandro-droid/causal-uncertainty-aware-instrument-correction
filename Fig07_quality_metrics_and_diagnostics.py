#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 — Quality metrics and stability diagnostics (COHERENT with Fig. 6)

KEEP panels (a) and (b) unchanged.
CHANGE ONLY panel (c):
- Use a wider lambda grid for the quality map (to avoid truncation of the curve).
- Keep y-axis in log scale and set y-limits to fully include the computed curve.

ADDITIONAL GRAPHICAL MODIFICATION:
- Add grid to panels (a), (b), and (c).

Outputs:
- Fig07_quality_metrics_and_diagnostics.png
- Fig07_quality_metrics_and_diagnostics.pdf
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
# Frequency grid and effective response H(e^{jω}) (as in Fig. 6)
# ----------------------------
w = np.linspace(0, np.pi, 4097)
f = w / np.pi

w0 = 2.0 * np.pi * 0.12
zeta = 0.70
T = 1.0
Omega = (2.0 / T) * np.tan(w / 2.0)

def H_analog(Omega, w0, zeta, gain=1.0):
    s = 1j * Omega
    return gain * (w0**2) / (s**2 + 2.0 * zeta * w0 * s + w0**2)

H = H_analog(Omega, w0, zeta, 1.0)
H = H / H[0]
Hc = np.conjugate(H)
Hmag2 = np.abs(H)**2

# ----------------------------
# Thresholds used in panel (a)
# ----------------------------
bias_thr_db = 1.0
noise_thr = 1e2
dw = w[1] - w[0]

def compute_for_lambda(lam):

    Tlam = Hmag2 / (Hmag2 + lam)
    bias_db = 20.0 * np.log10(np.maximum(Tlam, 1e-30))

    Glam = Hc / (Hmag2 + lam)
    noise_amp = np.abs(Glam)**2

    ok = (np.abs(bias_db) <= bias_thr_db) & (noise_amp <= noise_thr)

    if np.any(ok):
        f_eff = f[np.where(ok)[0].max()]
    else:
        f_eff = 0.0

    bias_rms = np.sqrt(np.mean(bias_db**2))
    noise_metric = np.sqrt(np.sum(noise_amp) * dw)

    return bias_db, noise_amp, f_eff, bias_rms, noise_metric, Glam

# ============================================================
# Panels (a) and (b)
# ============================================================
lams_ab = np.logspace(-6, -1, 80)

bias_rms_ab = np.zeros_like(lams_ab)
noise_ab = np.zeros_like(lams_ab)
f_eff_ab = np.zeros_like(lams_ab)

for i, lam in enumerate(lams_ab):

    _, _, f_eff_i, bias_rms_i, noise_i, _ = compute_for_lambda(lam)

    bias_rms_ab[i] = bias_rms_i
    noise_ab[i] = noise_i
    f_eff_ab[i] = f_eff_i

noise_ab_norm = noise_ab / noise_ab.min()

if np.any(f_eff_ab > 0):

    q = np.quantile(f_eff_ab, 0.90)
    cand = np.where(f_eff_ab >= q)[0]
    idx_star = cand[np.argmin(noise_ab_norm[cand])]

else:

    idx_star = np.argmin(noise_ab_norm)

lam_star = lams_ab[idx_star]

bias_db_star, noise_amp_star, f_eff_star, bias_rms_star, noise_star, G_star = compute_for_lambda(lam_star)

lam_aggr = lams_ab[0]

_, _, _, _, _, G_aggr = compute_for_lambda(lam_aggr)

# ----------------------------
# Time-domain kernels
# ----------------------------
def ifft_kernel_from_G(G_half, nfft=4096):

    Gfull = np.zeros(nfft, dtype=complex)

    Gfull[:nfft//2 + 1] = G_half
    Gfull[nfft//2 + 1:] = np.conjugate(G_half[1:nfft//2][::-1])

    return np.real(np.fft.ifft(Gfull))

nfft = 4096
w_half = np.linspace(0, np.pi, nfft//2 + 1)

G_star_half = np.interp(w_half, w, G_star)
G_aggr_half = np.interp(w_half, w, G_aggr)

g_star = ifft_kernel_from_G(G_star_half, nfft=nfft)
g_aggr = ifft_kernel_from_G(G_aggr_half, nfft=nfft)

N = 220

n = np.arange(0, N)

g_star_n = g_star[:N] / (np.max(np.abs(g_star[:N])) + 1e-12)
g_aggr_n = g_aggr[:N] / (np.max(np.abs(g_aggr[:N])) + 1e-12)

# ============================================================
# Panel (c)
# ============================================================
lams_c = np.logspace(-9, -1, 140)

bias_rms_c = np.zeros_like(lams_c)
noise_c = np.zeros_like(lams_c)
f_eff_c = np.zeros_like(lams_c)

for i, lam in enumerate(lams_c):

    _, _, f_eff_i, bias_rms_i, noise_i, _ = compute_for_lambda(lam)

    bias_rms_c[i] = bias_rms_i
    noise_c[i] = noise_i
    f_eff_c[i] = f_eff_i

noise_c_norm = noise_c / noise_c.min()

sizes_c = 40 + 240 * (f_eff_c / (f_eff_c.max() + 1e-12))

# ----------------------------
# Figure layout
# ----------------------------
fig = plt.figure(figsize=(12.6, 8.8))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.0, 1.0],
    hspace=0.35,
    wspace=0.28
)

# ============================
# (a)
# ============================
ax = fig.add_subplot(gs[0, 0])

bias_color = "tab:blue"
noise_color = "tab:orange"

ax.plot(
    f,
    bias_db_star,
    lw=1.8,
    color=bias_color,
    label=r"$\Delta A(\omega)$ (dB)"
)

ax.axhline(+bias_thr_db, lw=0.9, ls="--", color="black")
ax.axhline(-bias_thr_db, lw=0.9, ls="--", color="black")

ax.set_xlim(0, 1)
ax.set_ylim(-20, 5)

ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
ax.set_ylabel(r"Amplitude bias $\Delta A(\omega)$ [dB]")

ax.set_title(
    rf"(a) Acceptance window (example $\lambda = {lam_star:.2e}$)"
)

# ---- GRID ADDED ----
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

ax2 = ax.twinx()

ax2.plot(
    f,
    noise_amp_star,
    lw=1.8,
    color=noise_color,
    label=r"$|G_{\lambda}(e^{j\omega})|^2$"
)

ax2.set_yscale("log")

ax2.set_ylabel(
    r"Noise amplification $|G_{\lambda}(e^{j\omega})|^2$"
)

ax2.axhline(noise_thr, lw=0.9, ls="--", color="gray")

ax2.set_ylim(
    1e-2,
    max(noise_thr, np.nanmax(noise_amp_star) * 1.2)
)

if f_eff_star > 0:
    ax.axvspan(0, f_eff_star, alpha=0.10)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    lines1 + lines2,
    labels1 + labels2 + [rf"$f_{{\mathrm{{eff}}}}\approx {f_eff_star:.2f}$"],
    frameon=False,
    loc="lower left"
)

# ============================
# (b)
# ============================
axb = fig.add_subplot(gs[0, 1])

axb.plot(
    n,
    g_aggr_n,
    lw=1.6,
    label=rf"Aggressive inverse ($\lambda = {lam_aggr:.1e}$)"
)

axb.plot(
    n,
    g_star_n,
    lw=1.6,
    label=rf"Selected inverse ($\lambda = {lam_star:.1e}$)"
)

axb.set_xlim(0, 100)
axb.set_ylim(-1.1, 1.1)

axb.set_xlabel(r"Sample index $n$")
axb.set_ylabel("Normalized kernel amplitude")

axb.set_title("(b) Time-domain kernel diagnostics")

# ---- GRID ADDED ----
axb.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

axb.legend(frameon=True, loc="upper right")

# ============================
# (c)
# ============================
axc = fig.add_subplot(gs[1, :])

axc.scatter(
    bias_rms_c,
    noise_c_norm,
    s=sizes_c,
    alpha=0.85
)

axc.scatter(
    [bias_rms_star],
    [noise_star / noise_ab.min()],
    s=380,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5
)

axc.set_xlabel("Bias metric (RMS, dB)")
axc.set_ylabel("Noise metric (RMS, normalized)")

axc.set_yscale("log")

axc.set_title(
    r"(c) Quality map across regularization strength $\lambda$ (marker size $\propto f_{\mathrm{eff}}$)"
)

# ---- GRID ADDED ----
axc.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

ymin = 1.0
ymax = np.max(noise_c_norm) * 1.2

axc.set_ylim(ymin, ymax)

axc.set_xlim(0, np.max(bias_rms_c) * 1.05)

axc.text(
    bias_rms_star * 1.02,
    (noise_star / noise_ab.min()) * 1.05,
    rf"$\lambda^\star={lam_star:.1e}$, $f_{{\mathrm{{eff}}}}\approx{f_eff_star:.2f}$",
    fontsize=10
)

# ----------------------------
# Save
# ----------------------------
fig.savefig(
    "Fig07_quality_metrics_and_diagnostics.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "Fig07_quality_metrics_and_diagnostics.pdf",
    bbox_inches="tight"
)

plt.close(fig)