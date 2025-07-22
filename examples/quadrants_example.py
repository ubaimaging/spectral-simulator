import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from simulator.simulator import generate_spectra

# Parameters
sz = 100
channels = 30
photon_total = 10000
rang = np.linspace(450, 700, channels + 1)

# Define 4 distinct spectra with peaks at different wavelengths
offsets = [470, 520, 580, 630]
sigmas = [10, 10, 10, 10]
phasors = [0, 0, 0, 0]

spectral_image = np.zeros((channels, sz, sz))

# Assign a spectrum to each quadrant
for i, (off, sig, pha) in enumerate(zip(offsets, sigmas, phasors)):
    spectrum = generate_spectra(photon_total, sig, off, pha, rang)
    spec_img = spectrum.reshape(-1, 1, 1).repeat(sz // 2, axis=1).repeat(sz // 2, axis=2)

    row = 0 if i < 2 else 1
    col = 0 if i % 2 == 0 else 1
    spectral_image[:, row*sz//2:(row+1)*sz//2, col*sz//2:(col+1)*sz//2] = spec_img

# Plot mean spectral image
mean_image = np.mean(spectral_image, axis=0)
plt.figure(figsize=(5, 5))
plt.imshow(mean_image, cmap="inferno")
plt.title("Mean Spectral Image")
plt.colorbar(label="Intensity")
plt.axis("off")
plt.tight_layout()
plt.show()

# Plot spectra from each quadrant
coords = [(25, 25), (25, 75), (75, 25), (75, 75)]
plt.figure(figsize=(6, 4))
for (y, x) in coords:
    spectrum = spectral_image[:, y, x]
    plt.plot(spectrum, label=f"Pixel ({x},{y})")
plt.xlabel("Spectral Channel")
plt.ylabel("Intensity")
plt.title("Spectra of 4 Quadrants")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
