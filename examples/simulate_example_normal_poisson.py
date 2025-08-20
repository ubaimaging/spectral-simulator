# Simulated data
# Description: Application of the method for generating
# spectral data to simulate:
# 1) Spectra of pure components and their associated
# calibration images (Normal mode -> sampling + histogram)
# 2) Study images as mixtures of components (two variants):
#    2a) Sampling + histogram (as before)
#    2b) Poisson per bin (physically more accurate for photon counting)

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math

import matplotlib.pyplot as plt
import numpy as np

from simulator.simulator import \
    distribute_photons  # distributes photons according to a reference spectrum
from simulator.simulator import \
    generate_spectra  # returns counts per bin (len = channels)
from simulator.simulator import generate_random_fractions

# -----------------------------
# Parameters
# -----------------------------
N = 2  # Number of pure components
sz = 100  # Image size (sz x sz)
channels = 30
photons_pure = int(1e4)  # photons per pixel for calibration images
photons_experiment = int(1e3)  # photons per pixel for study images
harmonics = math.floor((N + 1) / 2)

wavelength_range = np.linspace(450, 700, channels + 1)  # bin edges (B+1)
center_rng = 0.5 * (wavelength_range[0] + wavelength_range[-1])  # center of range

# Component parameters
# If you want Gaussian peaks at μ_target, set off+pha = μ_target - center_rng
mu_targets = np.linspace(500, 650, N)  # desired spectral centers
sigmas = [25] * N  # sigma for each component
phasor_shift = [0.0] * N  # use pha=0.0, keep everything in 'off'
offsets = [float(mu_targets[k] - center_rng) for k in range(N)]

rng = np.random.default_rng(123)

# -----------------------------
# 1) PURE COMPONENT SPECTRA AND CALIBRATION IMAGES
#    - Normal mode (lam=None)
# -----------------------------
pure_spectra = np.zeros((N, sz * sz, channels), dtype=int)
pure_images = np.zeros((N, channels, sz, sz), dtype=int)

for comp in range(N):
    for idx in range(sz * sz):
        # Keyword call to avoid argument mis-ordering
        counts, _, _ = generate_spectra(
            Nph=photons_pure,
            off=offsets[comp],
            pha=phasor_shift[comp],
            rang=wavelength_range,
            sigm=sigmas[comp],
            lam=None,
        )
        pure_spectra[comp, idx, :] = counts.astype(int)
    pure_images[comp] = pure_spectra[comp].T.reshape((channels, sz, sz))

# -----------------------------
# 2) STUDY IMAGE AS MIXTURE
#    2a) Sampling + histogram (as before)
# -----------------------------
# Average pure spectra across calibration images
spectral_pure = np.array(
    [np.mean(pure_images[j], axis=(1, 2)) for j in range(N)]
)  # shape (N, channels)

# Random fractions per pixel
Npix = sz * sz
fraction_N = generate_random_fractions(N, Npix)  # shape (Npix, N)

spectral = np.zeros((Npix, channels), dtype=float)

for idx in range(Npix):
    total_photons = photons_experiment
    fractions = rng.permutation(fraction_N[idx])
    photon_distribution = np.round(fractions * total_photons).astype(int)

    # Ensure at least one photon per component
    if photon_distribution.sum() == 0:
        photon_distribution[rng.integers(0, N)] = 1

    # Ensure each component has at least one photon
    for comp in range(N):
        if photon_distribution[comp] == 0:
            photon_distribution[comp] = 1

    # Re-normalize fractions
    fractions = photon_distribution / photon_distribution.sum()
    fraction_N[idx] = fractions

    combined_spectrum = np.zeros(channels, dtype=float)
    for comp in range(N):
        samples = distribute_photons(
            spectral_pure[comp], int(photon_distribution[comp])
        )
        combined_spectrum += np.histogram(samples, bins=np.arange(0, channels + 1))[0]

    spectral[idx] = combined_spectrum

spectral_image = spectral.T.reshape((channels, sz, sz))  # (C, H, W)

# -----------------------------
# 2b) Poisson per bin (recommended for photon counting)
# -----------------------------
# Normalize pure spectra to per-channel probabilities
eps = 1e-12
pure_probs = (spectral_pure + eps) / (
    spectral_pure.sum(axis=1, keepdims=True) + eps
)  # (N, C)

spectral_poisson = np.zeros((Npix, channels), dtype=int)

for idx in range(Npix):
    total_photons = photons_experiment
    fractions = fraction_N[idx]
    lam_per_bin = np.zeros(channels, dtype=float)

    # λ per channel = sum over components of (fraction * total_photons * prob_comp[channel])
    for comp in range(N):
        lam_per_bin += fractions[comp] * total_photons * pure_probs[comp]

    # Sample Poisson counts per bin directly
    spectral_poisson[idx] = rng.poisson(lam_per_bin)

spectral_image_poisson = spectral_poisson.T.reshape((channels, sz, sz))  # (C, H, W)


# -----------------------------
# Visualization
# -----------------------------
def show_mean_image(image, title="Mean Spectral Image", cmap="inferno"):
    mean_img = np.mean(image, axis=0)
    plt.figure(figsize=(5, 5))
    plt.imshow(mean_img, cmap=cmap)
    plt.title(title)
    plt.colorbar(label="Intensity")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_pixel_spectra(image, coords, title="Spectra of Selected Pixels"):
    plt.figure(figsize=(6, 4))
    for x, y in coords:
        spectrum = image[:, y, x]
        plt.plot(spectrum, label=f"Pixel ({x}, {y})")
    plt.xlabel("Spectral Channel")
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Comparison
show_mean_image(spectral_image, title="Mean Spectral Image (Sampling + Histogram)")
show_mean_image(spectral_image_poisson, title="Mean Spectral Image (Poisson per bin)")

selected_pixels = [(50, 50), (25, 25), (75, 75), (10, 90)]
show_pixel_spectra(
    spectral_image, selected_pixels, title="Pixel Spectra (Sampling + Histogram)"
)
show_pixel_spectra(
    spectral_image_poisson, selected_pixels, title="Pixel Spectra (Poisson per bin)"
)
