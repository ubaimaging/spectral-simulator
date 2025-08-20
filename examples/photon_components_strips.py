import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from phasorpy.color import CATEGORICAL
from phasorpy.cursors import mask_from_circular_cursor, pseudo_color
from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot, plot_image

from simulator.simulator import (distribute_photons, generate_random_fractions,
                                 generate_spectra)

# -----------------------------
# Parameters
# -----------------------------
N = 2  # Number of pure components
sz = 100  # Image size (sz x sz)
channels = 30
photons_pure = int(1e4)  # photons per pixel for calibration images

# Define 5 different photon counts (increasing)
photon_levels = [int(2e2), int(5e2), int(1e3), int(2e3), int(4e3)]

wavelength_range = np.linspace(450, 700, channels + 1)
center_rng = 0.5 * (wavelength_range[0] + wavelength_range[-1])

# Component parameters - two distinct spectral components
mu_targets = [520, 620]  # Two peaks at different wavelengths
sigmas = [25, 25]
phasor_shift = [0.0, 0.0]
offsets = [float(mu_targets[k] - center_rng) for k in range(N)]

rng = np.random.default_rng(123)

# -----------------------------
# 1) Generate pure component spectra (for reference)
# -----------------------------
pure_spectra = np.zeros((N, sz * sz, channels), dtype=int)
pure_images = np.zeros((N, channels, sz, sz), dtype=int)

for comp in range(N):
    for idx in range(sz * sz):
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

# Average pure spectra
spectral_pure = np.array([np.mean(pure_images[j], axis=(1, 2)) for j in range(N)])

# -----------------------------
# 2) Create 5x5 grid patterns
# -----------------------------
strip_height = sz // 5  # Height of each horizontal strip
strip_width = sz // 5  # Width of each vertical strip

# Create photon count pattern (5 horizontal strips with increasing photon counts)
photon_pattern = np.zeros((sz, sz), dtype=int)
for y_strip in range(5):
    y_start = y_strip * strip_height
    y_end = min((y_strip + 1) * strip_height, sz)
    photon_pattern[y_start:y_end, :] = photon_levels[y_strip]

# Create component mixing pattern (5 vertical strips with different component ratios)
component_ratios = [
    [1.0, 0.0],  # Strip 0: Pure component 1
    [0.75, 0.25],  # Strip 1: 75% comp1, 25% comp2
    [0.5, 0.5],  # Strip 2: 50% each
    [0.25, 0.75],  # Strip 3: 25% comp1, 75% comp2
    [0.0, 1.0],  # Strip 4: Pure component 2
]

component_ratio_pattern = np.zeros((sz, sz, N), dtype=float)
for x_strip in range(5):
    x_start = x_strip * strip_width
    x_end = min((x_strip + 1) * strip_width, sz)
    component_ratio_pattern[:, x_start:x_end, 0] = component_ratios[x_strip][0]
    component_ratio_pattern[:, x_start:x_end, 1] = component_ratios[x_strip][1]

# -----------------------------
# 3) Generate spectral image using Poisson per bin
# -----------------------------
eps = 1e-12
pure_probs = (spectral_pure + eps) / (spectral_pure.sum(axis=1, keepdims=True) + eps)

spectral_strips = np.zeros((sz, sz, channels), dtype=int)

for y in range(sz):
    for x in range(sz):
        total_photons = photon_pattern[y, x]
        fractions = component_ratio_pattern[y, x, :]

        # Calculate lambda per bin for this pixel
        lam_per_bin = np.zeros(channels, dtype=float)
        for comp in range(N):
            lam_per_bin += fractions[comp] * total_photons * pure_probs[comp]

        # Sample Poisson counts per bin
        spectral_strips[y, x, :] = rng.poisson(lam_per_bin)

# Reshape to (channels, height, width) format
spectral_image_strips = spectral_strips.transpose(2, 0, 1)


# -----------------------------
# Visualization
# -----------------------------
def show_patterns():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Photon count pattern
    im1 = axes[0].imshow(photon_pattern, cmap="viridis")
    axes[0].set_title("Photon Count Pattern\n(5 Horizontal Strips)")
    axes[0].set_xlabel("X (Component Ratio)")
    axes[0].set_ylabel("Y (Photon Count)")

    # Add grid lines and labels for photon strips
    for i in range(1, 5):
        axes[0].axhline(y=i * strip_height - 0.5, color="white", linewidth=1, alpha=0.7)
    for i in range(5):
        y_center = i * strip_height + strip_height // 2
        axes[0].text(
            -5,
            y_center,
            f"{photon_levels[i]}",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
        )

    plt.colorbar(im1, ax=axes[0], label="Photons per pixel")

    # Component 1 ratio
    im2 = axes[1].imshow(component_ratio_pattern[:, :, 0], cmap="plasma")
    axes[1].set_title("Component 1 Fraction\n(5 Vertical Strips)")
    axes[1].set_xlabel("X (Component Ratio)")
    axes[1].set_ylabel("Y (Photon Count)")

    # Add grid lines and labels for component strips
    for i in range(1, 5):
        axes[1].axvline(x=i * strip_width - 0.5, color="white", linewidth=1, alpha=0.7)
    for i in range(5):
        x_center = i * strip_width + strip_width // 2
        axes[1].text(
            x_center,
            sz + 2,
            f"{component_ratios[i][0]:.2f}",
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
        )

    plt.colorbar(im2, ax=axes[1], label="Component 1 Fraction")

    # Component 2 ratio
    im3 = axes[2].imshow(component_ratio_pattern[:, :, 1], cmap="plasma")
    axes[2].set_title("Component 2 Fraction\n(5 Vertical Strips)")
    axes[2].set_xlabel("X (Component Ratio)")
    axes[2].set_ylabel("Y (Photon Count)")

    # Add grid lines and labels for component strips
    for i in range(1, 5):
        axes[2].axvline(x=i * strip_width - 0.5, color="white", linewidth=1, alpha=0.7)
    for i in range(5):
        x_center = i * strip_width + strip_width // 2
        axes[2].text(
            x_center,
            sz + 2,
            f"{component_ratios[i][1]:.2f}",
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
        )

    plt.colorbar(im3, ax=axes[2], label="Component 2 Fraction")

    plt.tight_layout()
    plt.show()


def show_spectral_results():
    # Mean image
    mean_img = np.mean(spectral_image_strips, axis=0)

    # Total counts per pixel
    total_counts = np.sum(spectral_image_strips, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Mean spectral image
    im1 = axes[0, 0].imshow(mean_img, cmap="inferno")
    axes[0, 0].set_title("Mean Spectral Intensity")
    axes[0, 0].set_xlabel("X (Component Ratio)")
    axes[0, 0].set_ylabel("Y (Photon Count)")
    plt.colorbar(im1, ax=axes[0, 0])

    # Total counts
    im2 = axes[0, 1].imshow(total_counts, cmap="viridis")
    axes[0, 1].set_title("Total Photon Counts")
    axes[0, 1].set_xlabel("X (Component Ratio)")
    axes[0, 1].set_ylabel("Y (Photon Count)")
    plt.colorbar(im2, ax=axes[0, 1])

    # Grid visualization showing the 25 combinations
    grid_vis = np.zeros((5, 5))
    for y_strip in range(5):
        for x_strip in range(5):
            y_center = y_strip * strip_height + strip_height // 2
            x_center = x_strip * strip_width + strip_width // 2
            grid_vis[y_strip, x_strip] = total_counts[y_center, x_center]

    im3 = axes[0, 2].imshow(grid_vis, cmap="viridis")
    axes[0, 2].set_title("25 Combinations Grid\n(Total Counts)")
    axes[0, 2].set_xlabel("Component Strip (0=Pure C1, 4=Pure C2)")
    axes[0, 2].set_ylabel("Photon Strip (0=Low, 4=High)")

    # Add text annotations
    for y in range(5):
        for x in range(5):
            axes[0, 2].text(
                x,
                y,
                f"{int(grid_vis[y, x])}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    plt.colorbar(im3, ax=axes[0, 2])

    # Sample spectra from different vertical strips (middle photon level)
    mid_y = 2 * strip_height + strip_height // 2
    axes[1, 0].plot(
        spectral_image_strips[:, mid_y, 1 * strip_width // 2],
        label="Pure Comp1",
        linewidth=2,
    )
    axes[1, 0].plot(
        spectral_image_strips[:, mid_y, 1 * strip_width + strip_width // 2],
        label="75% Comp1",
        linewidth=2,
    )
    axes[1, 0].plot(
        spectral_image_strips[:, mid_y, 2 * strip_width + strip_width // 2],
        label="50% each",
        linewidth=2,
    )
    axes[1, 0].plot(
        spectral_image_strips[:, mid_y, 3 * strip_width + strip_width // 2],
        label="75% Comp2",
        linewidth=2,
    )
    axes[1, 0].plot(
        spectral_image_strips[:, mid_y, 4 * strip_width + strip_width // 2],
        label="Pure Comp2",
        linewidth=2,
    )
    axes[1, 0].set_xlabel("Spectral Channel")
    axes[1, 0].set_ylabel("Counts")
    axes[1, 0].set_title("Spectra: Different Component Ratios\n(Middle Photon Level)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Sample spectra from different photon levels (middle component ratio)
    mid_x = 2 * strip_width + strip_width // 2
    for y_strip in range(5):
        y_center = y_strip * strip_height + strip_height // 2
        spectrum = spectral_image_strips[:, y_center, mid_x]
        axes[1, 1].plot(
            spectrum, label=f"{photon_levels[y_strip]} photons", linewidth=2
        )

    axes[1, 1].set_xlabel("Spectral Channel")
    axes[1, 1].set_ylabel("Counts")
    axes[1, 1].set_title("Spectra: Different Photon Levels\n(50/50 Component Mix)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Corner combinations comparison
    corners = [
        (0, 0, "Low photons + Pure C1"),
        (0, 4, "Low photons + Pure C2"),
        (4, 0, "High photons + Pure C1"),
        (4, 4, "High photons + Pure C2"),
    ]

    for y_strip, x_strip, label in corners:
        y_center = y_strip * strip_height + strip_height // 2
        x_center = x_strip * strip_width + strip_width // 2
        spectrum = spectral_image_strips[:, y_center, x_center]
        axes[1, 2].plot(spectrum, label=label, linewidth=2)

    axes[1, 2].set_xlabel("Spectral Channel")
    axes[1, 2].set_ylabel("Counts")
    axes[1, 2].set_title("Corner Combinations")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()


def show_phasor_analysis(spectral_image):

    # Compute phasor representation
    mean, real, imag = phasor_from_signal(spectral_image, axis=0)

    center_real = [-0.07, -0.16, -0.24, -0.34, -0.42]
    center_imag = [0.82, 0.43, 0.05, -0.33, -0.7]
    radius = 0.1

    plot = PhasorPlot(allquadrants=True)
    plot.hist2d(real, imag)
    for i in range(len(center_real)):
        plot.circle(
            center_real[i],
            center_imag[i],
            color=CATEGORICAL[i],
            radius=radius,
            linestyle="-",
            linewidth=2,
        )
    plot.show()

    masks = mask_from_circular_cursor(
        real, imag, center_real, center_imag, radius=radius
    )

    pseudo_color_image = pseudo_color(*masks, colors=CATEGORICAL)
    plot_image(pseudo_color_image)


if __name__ == "__main__":
    show_patterns()
    show_spectral_results()
    show_phasor_analysis(spectral_image_strips)

    # Print detailed statistics for all 25 combinations
    print(f"Image size: {sz}x{sz}")
    print(f"Spectral channels: {channels}")
    print(f"Component 1 peak: {mu_targets[0]} nm")
    print(f"Component 2 peak: {mu_targets[1]} nm")
    print(f"Photon levels: {photon_levels}")
    print(f"Component ratios: {component_ratios}")
    print("\n25 Combinations (Photon Level × Component Ratio):")
    print("Row = Photon Level, Column = Component Mix")
    print("Format: (expected_photons, actual_mean_counts)")

    for y_strip in range(5):
        row_str = f"Photon Level {y_strip} ({photon_levels[y_strip]}): "
        for x_strip in range(5):
            y_start, y_end = y_strip * strip_height, min(
                (y_strip + 1) * strip_height, sz
            )
            x_start, x_end = x_strip * strip_width, min((x_strip + 1) * strip_width, sz)
            region_counts = np.mean(
                np.sum(spectral_image_strips[:, y_start:y_end, x_start:x_end], axis=0)
            )
            comp_ratio = component_ratios[x_strip]
            row_str += f"({photon_levels[y_strip]}, {region_counts:.0f}) "
        print(row_str)
