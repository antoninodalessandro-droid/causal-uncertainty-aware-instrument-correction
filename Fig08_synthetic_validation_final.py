#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 8 - Synthetic validation of causal regularized instrument correction.

Outputs:
    outputs_fig08/Fig08_synthetic_validation_final.png
    outputs_fig08/Fig08_synthetic_validation_final.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

OUTDIR = Path("outputs_fig08")
OUTDIR.mkdir(exist_ok=True)

fs = 100.0
dt = 1.0 / fs
n = 16384
t = np.arange(n) * dt
freq = np.fft.rfftfreq(n, dt)
fn = fs / 2.0
f_norm = freq / fn
omega = 2.0 * np.pi * freq
eps = 1e-15
rng = np.random.default_rng(20260509)

def ricker(t, t0, f0):
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a**2) * np.exp(-a**2)

x_true = (
    0.22 * np.sin(2 * np.pi * 0.025 * t)
    + 0.16 * np.sin(2 * np.pi * 0.080 * t)
    + 1.00 * ricker(t, 45.0, 1.2)
    + 0.42 * ricker(t, 72.0, 5.5)
    + 0.10 * np.sin(2 * np.pi * 16.0 * t)
    * np.exp(-0.5 * ((t - 105.0) / 5.0) ** 2)
)

x_true *= signal.windows.tukey(n, 0.08)
x_true /= np.max(np.abs(x_true))
X = np.fft.rfft(x_true)

f0_sensor = 1.0 / 120.0
f_aa = 0.72 * fn
f_fir = 0.82 * fn

sensor_low = (freq / f0_sensor) ** 2 / np.sqrt(1.0 + (freq / f0_sensor) ** 4)
sensor_low[0] = 0.0

anti_alias = 1.0 / np.sqrt(1.0 + (freq / f_aa) ** 10)
fir_decimation = 1.0 / np.sqrt(1.0 + (freq / f_fir) ** 48)

H_amp = sensor_low * anti_alias * fir_decimation
H_amp[0] = 0.0

group_delay = 0.18
phase_disp = -0.18 * (freq / fn) ** 2
H_phase = np.exp(-1j * (omega * group_delay + phase_disp))
H = H_amp * H_phase

Y_clean = H * X

white = rng.standard_normal(n)
hf = signal.lfilter([1.0, -1.85, 0.86], [1.0], rng.standard_normal(n))
hf = hf / np.std(hf)

noise_time = 0.006 * white + 0.010 * hf
N = np.fft.rfft(noise_time)

Y_noisy = Y_clean + N
y_rec = np.fft.irfft(Y_noisy, n=n)
y_rec_plot = y_rec / np.max(np.abs(y_rec))

G_ideal = np.zeros_like(H, dtype=complex)
valid = H_amp > 1e-10
G_ideal[valid] = 1.0 / H[valid]

cap = 1e4
G_ideal = np.minimum(np.abs(G_ideal), cap) * np.exp(1j * np.angle(G_ideal))

water_level = 2e-3 * np.max(H_amp**2)
G_wl = np.conj(H) / np.maximum(H_amp**2, water_level)

lam_sel = 4e-3
G_reg = np.conj(H) / (H_amp**2 + lam_sel)

def recover(G):
    return np.fft.irfft(G * Y_noisy, n=n)

x_ideal = recover(G_ideal)
x_wl = recover(G_wl)
x_reg = recover(G_reg)

def detrend_and_scale(x):
    x = signal.detrend(x, type="constant")
    mask = (t >= 20) & (t <= 120)
    alpha = np.dot(x[mask], x_true[mask]) / (np.dot(x[mask], x[mask]) + eps)
    return alpha * x

x_ideal_p = detrend_and_scale(x_ideal)
x_wl_p = detrend_and_scale(x_wl)
x_reg_p = detrend_and_scale(x_reg)

X_ideal = np.fft.rfft(x_ideal_p)
X_wl = np.fft.rfft(x_wl_p)
X_reg = np.fft.rfft(x_reg_p)

f_eff = 0.80 * fn
band = (freq >= 0.02) & (freq <= f_eff)
band_plot = (freq >= 0.015) & (freq <= 0.95 * fn)

def spectral_recovery_error_db(Xhat):
    return 20.0 * np.log10((np.abs(Xhat) + eps) / (np.abs(X) + eps))

def response_error_db(G):
    return 20.0 * np.log10(np.abs(G * H) + eps)

def spectral_bias(G):
    err = response_error_db(G)
    return np.sqrt(np.mean(err[band] ** 2))

def noise_metric(G):
    g2 = np.abs(G[band]) ** 2
    return np.sqrt(np.mean(np.log10(g2 + eps) ** 2))

def waveform_rmse(xhat):
    mask = (t >= 20) & (t <= 120)
    return np.sqrt(np.mean((xhat[mask] - x_true[mask]) ** 2)) / np.sqrt(
        np.mean(x_true[mask] ** 2)
    )

def lag_seconds(xhat):
    mask = (t >= 20) & (t <= 120)
    c = signal.correlate(xhat[mask], x_true[mask], mode="full")
    lags = signal.correlation_lags(np.sum(mask), np.sum(mask), mode="full")
    return lags[np.argmax(c)] * dt

metrics = {
    "Ideal inverse": (
        waveform_rmse(x_ideal_p),
        lag_seconds(x_ideal_p),
        spectral_bias(G_ideal),
        noise_metric(G_ideal),
    ),
    "Water-level inverse": (
        waveform_rmse(x_wl_p),
        lag_seconds(x_wl_p),
        spectral_bias(G_wl),
        noise_metric(G_wl),
    ),
    "Proposed inverse": (
        waveform_rmse(x_reg_p),
        lag_seconds(x_reg_p),
        spectral_bias(G_reg),
        noise_metric(G_reg),
    ),
}

lams = np.logspace(-7, 0, 160)
B = np.zeros_like(lams)
NN = np.zeros_like(lams)

for i, lam in enumerate(lams):
    G = np.conj(H) / (H_amp**2 + lam)
    B[i] = spectral_bias(G)
    NN[i] = noise_metric(G)

idx_sel = np.argmin(np.abs(lams - lam_sel))

def kernel_from_G(G):
    g = np.fft.irfft(G, n=n)
    return np.fft.fftshift(g)

g_ideal = kernel_from_G(G_ideal)
g_wl = kernel_from_G(G_wl)
g_reg = kernel_from_G(G_reg)
tk = (np.arange(n) - n // 2) * dt

kernel_mask = (tk >= -3.0) & (tk <= 10.0)
k_scale = np.percentile(np.abs(g_ideal[kernel_mask]), 99)
k_scale = max(k_scale, eps)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig = plt.figure(figsize=(13.6, 9.4))
gs = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.26)

legend_kw = dict(frameon=False, handlelength=2.4, borderaxespad=0.3)

ax = fig.add_subplot(gs[0, 0])
time_mask = (t >= 25) & (t <= 120)

ax.plot(t[time_mask], x_true[time_mask], lw=1.7, label="True ground motion $x[n]$")
ax.plot(
    t[time_mask],
    y_rec_plot[time_mask],
    lw=1.0,
    alpha=0.80,
    label="Recorded data $y[n]$ (normalized)",
)

ax.set_title("(a) Synthetic ground truth and recorded response")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized amplitude")
ax.legend(loc="upper right", **legend_kw)
ax.grid(True, alpha=0.25)

ax = fig.add_subplot(gs[0, 1])
zoom = (t >= 38) & (t <= 82)

ax.plot(t[zoom], x_true[zoom], lw=2.0, label="True")
ax.plot(t[zoom], x_ideal_p[zoom], lw=0.8, alpha=0.75, label="Ideal inverse")
ax.plot(t[zoom], x_wl_p[zoom], lw=1.1, alpha=0.90, label="Water-level inverse")
ax.plot(t[zoom], x_reg_p[zoom], lw=1.7, label="Proposed regularized inverse")

ax.set_title("(b) Recovered waveforms")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized amplitude")
ax.legend(loc="upper right", **legend_kw)
ax.grid(True, alpha=0.25)

ax = fig.add_subplot(gs[1, 0])

err_ideal = spectral_recovery_error_db(X_ideal)
err_wl = spectral_recovery_error_db(X_wl)
err_reg = spectral_recovery_error_db(X_reg)

ax.axvspan(
    0.02,
    f_eff,
    color="0.85",
    alpha=0.22,
    label=r"effective bandwidth $f_{\mathrm{eff}}$",
)
ax.semilogx(freq[band_plot], err_ideal[band_plot], lw=0.9, alpha=0.75, label="Ideal inverse")
ax.semilogx(freq[band_plot], err_wl[band_plot], lw=1.2, alpha=0.90, label="Water-level inverse")
ax.semilogx(freq[band_plot], err_reg[band_plot], lw=1.8, label="Proposed regularized inverse")

ax.axhline(0.0, ls="--", lw=1.0, alpha=0.65)
ax.axvline(f_eff, ls="--", lw=1.0, alpha=0.75, label="0.8 Nyquist")

ax.set_title("(c) Spectral recovery error")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"$20\log_{10}(|\hat{X}|/|X|)$ [dB]")
ax.set_ylim(-45, 25)
ax.legend(
    loc="upper left",
    ncol=2,
    columnspacing=0.9,
    handletextpad=0.5,
    **legend_kw,
)
ax.grid(True, which="both", alpha=0.25)

ax = fig.add_subplot(gs[1, 1])
plot = freq > 0

ax.axvspan(
    0.0,
    0.8,
    color="0.85",
    alpha=0.22,
    label=r"effective bandwidth $f_{\mathrm{eff}}$",
)
ax.semilogy(f_norm[plot], np.abs(G_ideal[plot]) ** 2, lw=0.9, alpha=0.75, label="Ideal inverse")
ax.semilogy(f_norm[plot], np.abs(G_wl[plot]) ** 2, lw=1.2, alpha=0.90, label="Water-level inverse")
ax.semilogy(f_norm[plot], np.abs(G_reg[plot]) ** 2, lw=1.8, label="Proposed regularized inverse")
ax.axvline(0.8, ls="--", lw=1.0, alpha=0.75, label="0.8 Nyquist")

ax.set_title("(d) Inverse noise amplification")
ax.set_xlabel(r"Normalized frequency $f/f_N$")
ax.set_ylabel(r"$|G(e^{j\omega})|^2$")
ax.set_ylim(1e-2, 1e10)
ax.legend(loc="upper left", **legend_kw)
ax.grid(True, which="both", alpha=0.25)

ax = fig.add_subplot(gs[2, 0])

ax.plot(
    tk[kernel_mask],
    g_ideal[kernel_mask] / k_scale,
    lw=0.9,
    alpha=0.75,
    label="Ideal inverse",
)
ax.plot(
    tk[kernel_mask],
    g_wl[kernel_mask] / k_scale,
    lw=1.2,
    alpha=0.90,
    label="Water-level inverse",
)
ax.plot(
    tk[kernel_mask],
    g_reg[kernel_mask] / k_scale,
    lw=2.0,
    label="Proposed regularized inverse",
)
ax.axvline(0.0, ls="--", lw=1.0, alpha=0.70)

ax.set_title("(e) Time-domain inverse kernels")
ax.set_xlabel("Lag time [s]")
ax.set_ylabel("Normalized kernel amplitude")
ax.set_ylim(-1.25, 1.25)
ax.legend(loc="upper right", **legend_kw)
ax.grid(True, alpha=0.25)

ax = fig.add_subplot(gs[2, 1])

ax.plot(B, NN, lw=1.8, label=r"$G_\lambda$ path")
ax.scatter(
    B[idx_sel],
    NN[idx_sel],
    s=75,
    marker="o",
    alpha=0.85,
    zorder=5,
    label=r"Selected $\lambda^\ast$",
)

methods = [
    ("Ideal", G_ideal, "o", (0.20, 0.035)),
    ("Water-level", G_wl, "s", (0.20, -0.015)),
    ("Proposed", G_reg, "^", (0.28, -0.085)),
]

for name, G, marker, offset in methods:
    b = spectral_bias(G)
    nm = noise_metric(G)
    ax.scatter(b, nm, s=75, marker=marker, zorder=6)
    ax.text(b + offset[0], nm + offset[1], name, va="center", fontsize=8)

ax.set_title("(f) Bias–noise diagnostic space")
ax.set_xlabel("RMS amplitude bias [dB]")
ax.set_ylabel("RMS log-noise amplification")
ax.set_ylim(0.0, 0.75)
ax.legend(loc="upper right", **legend_kw)
ax.grid(True, alpha=0.25)

txt = (
    "Performance metrics within the effective bandwidth\n"
    f"Ideal inverse: RMSE={metrics['Ideal inverse'][0]:.2f}, "
    f"lag={metrics['Ideal inverse'][1] * 1000:.0f} ms, "
    f"bias={metrics['Ideal inverse'][2]:.2f} dB\n"
    f"Water-level inverse: RMSE={metrics['Water-level inverse'][0]:.2f}, "
    f"lag={metrics['Water-level inverse'][1] * 1000:.0f} ms, "
    f"bias={metrics['Water-level inverse'][2]:.2f} dB\n"
    f"Proposed inverse: RMSE={metrics['Proposed inverse'][0]:.2f}, "
    f"lag={metrics['Proposed inverse'][1] * 1000:.0f} ms, "
    f"bias={metrics['Proposed inverse'][2]:.2f} dB"
)

fig.text(
    0.50,
    0.012,
    txt,
    ha="center",
    va="bottom",
    fontsize=8.1,
    bbox=dict(boxstyle="round,pad=0.40", facecolor="white", edgecolor="0.72", alpha=0.96),
)

png = OUTDIR / "Fig08_synthetic_validation_final.png"
pdf = OUTDIR / "Fig08_synthetic_validation_final.pdf"

fig.savefig(png, dpi=450, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("[OK] Saved:")
print(f"  {png}")
print(f"  {pdf}")

print("\nMetrics:")
for k, v in metrics.items():
    print(
        f"  {k:25s} "
        f"RMSE={v[0]:.4f}, "
        f"lag={v[1] * 1000:.1f} ms, "
        f"bias={v[2]:.3f} dB, "
        f"log-noise={v[3]:.3f}"
    )