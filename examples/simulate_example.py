# Simulated data
# Description: Application of the method for generating spectral data to simulate:
# 1. Spectra of pure components and their associated calibration images
# 2. Study images (mixtures of spectral components)

import numpy as np
import matplotlib.pyplot as plt
import math
from simulator import (
    generate_spectra, 
    distribute_photons, 
    generate_random_fractions
)

#%% PARAMETERS
N = 1  # Number of components
sz = 100  # Image size
channels = 30
photon_total_pure = 1e4
photon_max_experimental = 1e3

photon_total_pure_array = np.full(N, int(photon_total_pure))
Nph_pure = np.full(sz * sz, photon_total_pure)
Nph_experiment = np.full(sz * sz, photon_max_experimental)
harmonics = math.floor((N + 1) / 2)

wavelength_range = np.linspace(450, 700, channels + 1)  # Spectral range

#%% GENERATION OF PURE COMPONENT SPECTRA AND IMAGES

offsets = [36.5, 36.5, 36.5, 36.5]
sigmas = [15, 35, 15, 15]
phasor_centers = np.linspace(450, 700, N + 1)

pure_spectra = np.zeros((N, sz * sz, channels))
pure_images = np.zeros((N, channels, sz, sz))
phasor_coordinates = np.zeros((N + 1, 2, harmonics))  # Not used here, just placeholder

for comp in range(N):
    for idx in range(sz * sz):
        pure_spectra[comp, idx, :] = generate_spectra(
            int(photon_total_pure), sigmas[comp], offsets[comp], phasor_centers[comp], wavelength_range
        )
    pure_images[comp] = pure_spectra[comp].T.reshape((channels, sz, sz))

#%% GENERATE STUDY IMAGE AS MIXTURE OF COMPONENTS

# Average pure spectra across each image
spectral_pure = np.array([
    np.mean(pure_images[j], axis=(1, 2)) for j in range(N)
])

# Random mixture fractions for each pixel
fraction_N = generate_random_fractions(N, len(Nph_experiment))

spectral = np.zeros((len(Nph_experiment), channels))
for idx in range(len(Nph_experiment)):
    total_photons = Nph_experiment[idx]
    fractions = np.random.permutation(fraction_N[idx])
    photon_distribution = np.round(fractions * total_photons)

    # Ensure at least one photon is assigned
    if np.sum(photon_distribution) == 0:
        photon_distribution[np.random.randint(0, N)] = 1

    fractions = photon_distribution / np.sum(photon_distribution)
    fraction_N[idx] = fractions

    combined_spectrum = np.zeros(channels)
    for comp in range(N):
        spectrum = distribute_photons(spectral_pure[comp], int(photon_distribution[comp]))
        combined_spectrum += np.histogram(spectrum, bins=np.arange(0, channels + 1))[0]

    spectral[idx] = combined_spectrum

# Reshape to hyperspectral image
spectral_image = spectral.T.reshape((channels, sz, sz))

#%% VISUALIZATION

def show_mean_image(image, cmap='inferno'):
    mean_img = np.mean(image, axis=0)
    plt.figure(figsize=(5, 5))
    plt.imshow(mean_img, cmap=cmap)
    plt.title("Mean Spectral Image")
    plt.colorbar(label="Intensity")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def show_pixel_spectra(image, coords):
    plt.figure(figsize=(6, 4))
    for (x, y) in coords:
        spectrum = image[:, y, x]
        plt.plot(spectrum, label=f"Pixel ({x}, {y})")
    plt.xlabel("Spectral Channel")
    plt.ylabel("Intensity")
    plt.title("Spectra of Selected Pixels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot mean image
show_mean_image(spectral_image)

# Plot spectra of selected pixels
selected_pixels = [(50, 50), (25, 25), (75, 75), (10, 90)]
show_pixel_spectra(spectral_image, selected_pixels)