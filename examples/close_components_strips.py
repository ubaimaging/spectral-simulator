import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from phasorpy.color import CATEGORICAL
from phasorpy.cursor import mask_from_circular_cursor, pseudo_color
from phasorpy.phasor import phasor_from_signal
from phasorpy.filter import phasor_filter_median, phasor_filter_pawflim
from phasorpy.plot import PhasorPlot, plot_image

# -----------------------------
# Output
# -----------------------------
output_root = 'output'
out_mixed = os.path.join(output_root, 'mixed')
out_single = os.path.join(output_root, 'single')
os.makedirs(out_mixed, exist_ok=True)
os.makedirs(out_single, exist_ok=True)

# -----------------------------
# Parameters
# -----------------------------
N = 2  # Number of pure components
sz = 100  # Image size (sz x sz)
channels = 30

# Horizontal strips: photons per pixel for each of the 5 rows
photon_levels = [int(2e2), int(5e2), int(1e3), int(2e3), int(4e3)]

# Spectral axis
wavelength_range = np.linspace(450, 700, channels + 1)  # edges
channel_centers = 0.5 * (wavelength_range[:-1] + wavelength_range[1:])
center_rng = float(channel_centers.mean())
channel_step = float(wavelength_range[1] - wavelength_range[0])

# Component spectral width (std dev, nm)
sigmas = [25.0, 25.0]

# Vertical strips: separation (nm) between component 1 and 2 centers.
# Starts below one channel step (~8.33 nm) and increases.
separation_levels = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Mixed case: fixed fractions for the two components
comp_fractions = np.array([0.5, 0.5], dtype=float)

# Single-component case: separations between consecutive strips (nm)
# len = 4 values for 5 vertical strips: sep(S0,S1), sep(S1,S2), sep(S2,S3), sep(S3,S4)
consecutive_strip_separations = [0.5, 1.0, 2.0, 5.0]

# Non-Gaussian tweaks
skew_mag = 0.15          # positive skew for even indices in single-case
ripple_amp = 0.05
ripple_period_nm = 25.0

rng = np.random.default_rng(123)

# -----------------------------
# Helpers
# -----------------------------
def component_profile(x_nm, mu_nm, sigma_nm, skew=0.0, ripple=0.0, period_nm=25.0):
    # Base Gaussian
    g = np.exp(-0.5 * ((x_nm - mu_nm) / sigma_nm) ** 2)

    # Mild skew via linear term around center (clipped to non-negative)
    if skew != 0.0:
        g = g * np.clip(1.0 + skew * (x_nm - mu_nm) / sigma_nm, 0.0, None)

    # Gentle ripple to break perfection (kept non-negative)
    if ripple != 0.0:
        g = g * (1.0 + ripple * np.sin(2.0 * np.pi * (x_nm - mu_nm) / period_nm))

    g = np.clip(g, 0.0, None)
    s = g.sum()
    if s <= 0.0:
        return np.ones_like(g) / g.size
    return g / s


def build_component_profiles():
    """Return per-strip centers and normalized profiles for both components."""
    comp_profiles = np.zeros((5, N, channels), dtype=float)
    mu1_per_strip = np.zeros(5, dtype=float)
    mu2_per_strip = np.zeros(5, dtype=float)

    for x_strip in range(5):
        sep_nm = separation_levels[x_strip]
        mu1 = center_rng - 0.5 * sep_nm
        mu2 = center_rng + 0.5 * sep_nm
        mu1_per_strip[x_strip] = mu1
        mu2_per_strip[x_strip] = mu2

        comp_profiles[x_strip, 0, :] = component_profile(
            channel_centers, mu1, sigmas[0], skew=+skew_mag, ripple=ripple_amp, period_nm=ripple_period_nm
        )
        comp_profiles[x_strip, 1, :] = component_profile(
            channel_centers, mu2, sigmas[1], skew=-skew_mag, ripple=ripple_amp, period_nm=ripple_period_nm
        )

    return comp_profiles, mu1_per_strip, mu2_per_strip


def build_patterns(sz):
    strip_height = sz // 5
    strip_width = sz // 5

    photon_pattern = np.zeros((sz, sz), dtype=int)
    for y_strip in range(5):
        y_start = y_strip * strip_height
        y_end = min((y_strip + 1) * strip_height, sz)
        photon_pattern[y_start:y_end, :] = photon_levels[y_strip]

    separation_map = np.zeros((sz, sz), dtype=float)
    for x_strip in range(5):
        x_start = x_strip * strip_width
        x_end = min((x_strip + 1) * strip_width, sz)
        separation_map[:, x_start:x_end] = separation_levels[x_strip]

    return strip_height, strip_width, photon_pattern, separation_map

def center_of_mass(x, weights):
    """Return the weighted mean (center of mass) of the spectrum."""
    return np.sum(x * weights) / np.sum(weights)

def build_single_centers_and_profiles():
    """
    For each strip, create a profile whose center of mass is at the desired nm separation
    from the previous strip. Uses skew to shift the mean as needed.
    """
    n_strips = 5
    seps = np.asarray(consecutive_strip_separations, dtype=float)
    if seps.size != n_strips - 1:
        raise ValueError(f"consecutive_strip_separations must have {n_strips-1} values")
    # The first strip's center of mass is centered around the spectral mean
    target_means = [center_rng]
    for s in seps:
        target_means.append(target_means[-1] + s)

    profiles = np.zeros((n_strips, channels), dtype=float)
    actual_means = np.zeros(n_strips, dtype=float)
    # For each strip, find the skew that puts the center of mass at the target
    for i in range(n_strips):
        # Start with a Gaussian centered at target_means[i]
        mu = target_means[i]
        # Find skew that moves the center of mass to mu
        # Try a range of skews and pick the one that gets closest
        best_skew = 0.0
        best_mean = None
        min_err = float('inf')
        for skew in np.linspace(-0.5, 0.5, 101):
            prof = component_profile(channel_centers, mu, sigmas[0], skew=skew, ripple=ripple_amp, period_nm=ripple_period_nm)
            mean = center_of_mass(channel_centers, prof)
            err = abs(mean - mu)
            if err < min_err:
                min_err = err
                best_skew = skew
                best_mean = mean
                best_prof = prof
        profiles[i, :] = best_prof
        actual_means[i] = best_mean
    return np.array(actual_means), profiles


def synthesize_spectral_image_mixed(comp_profiles):
    """Two-component mixture per strip with fixed fractions."""
    strip_height, strip_width, _, _ = build_patterns(sz)
    spectral = np.zeros((sz, sz, channels), dtype=int)

    for y_strip in range(5):
        y_start = y_strip * strip_height
        y_end = min((y_strip + 1) * strip_height, sz)
        h = y_end - y_start
        total_photons = photon_levels[y_strip]

        for x_strip in range(5):
            x_start = x_strip * strip_width
            x_end = min((x_strip + 1) * strip_width, sz)
            w = x_end - x_start

            probs = (
                comp_fractions[0] * comp_profiles[x_strip, 0, :]
                + comp_fractions[1] * comp_profiles[x_strip, 1, :]
            )
            lam_bins = total_photons * probs
            counts = rng.poisson(lam_bins.reshape(1, 1, -1), size=(h, w, channels))
            spectral[y_start:y_end, x_start:x_end, :] = counts

    return spectral.transpose(2, 0, 1)  # (channels, H, W)


def synthesize_spectral_image_single(active_profiles):
    """Render single-component strips using the provided per-strip profiles.

    Args:
      active_profiles: (5, channels) array where row i is the probability
                       profile for strip i (unique component per strip).

    Returns:
      spectral image in shape (channels, H, W)
    """
    strip_height, strip_width, _, _ = build_patterns(sz)
    spectral = np.zeros((sz, sz, channels), dtype=int)

    for y_strip in range(5):
        y_start = y_strip * strip_height
        y_end = min((y_strip + 1) * strip_height, sz)
        h = y_end - y_start
        total_photons = photon_levels[y_strip]

        for x_strip in range(5):
            x_start = x_strip * strip_width
            x_end = min((x_strip + 1) * strip_width, sz)
            w = x_end - x_start

            probs = active_profiles[x_strip, :]  # single unique component for this strip
            lam_bins = total_photons * probs
            counts = rng.poisson(lam_bins.reshape(1, 1, -1), size=(h, w, channels))
            spectral[y_start:y_end, x_start:x_end, :] = counts

    return spectral.transpose(2, 0, 1)  # (channels, H, W)


# -----------------------------
# Visualization
# -----------------------------
def show_patterns(mode, mu1_per_strip, mu2_per_strip, active_mu_per_strip=None, save_dir=output_root, show=False):
    strip_height, strip_width, photon_pattern, separation_map = build_patterns(sz)

    if mode == 'mixed':
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Photon count pattern
    im1 = axes[0].imshow(photon_pattern, cmap="viridis")
    axes[0].set_title("Photon Count Pattern\n(5 Horizontal Strips)")
    axes[0].set_xlabel("X (Vertical Strip)")
    axes[0].set_ylabel("Y (Photon Count)")
    for i in range(1, 5):
        axes[0].axhline(y=i * strip_height - 0.5, color="white", linewidth=1, alpha=0.7)
    for i in range(5):
        y_center = i * strip_height + strip_height // 2
        axes[0].text(
            -5, y_center, f"{photon_levels[i]}",
            ha="right", va="center", color="white", fontweight="bold",
        )
    plt.colorbar(im1, ax=axes[0], label="Photons per pixel")

    # Separation (nm)
    im2 = axes[1].imshow(separation_map, cmap="magma")
    axes[1].set_title("Separation Schedule (nm)\n(5 Vertical Strips)")
    axes[1].set_xlabel("X (Vertical Strip)")
    axes[1].set_ylabel("Y (Photon Count)")
    for i in range(1, 5):
        axes[1].axvline(x=i * strip_width - 0.5, color="white", linewidth=1, alpha=0.7)
    for i in range(5):
        x_center = i * strip_width + strip_width // 2
        axes[1].text(
            x_center, -5, f"{separation_levels[i]:.0f} nm",
            ha="center", va="top", color="white", fontweight="bold",
        )
    plt.colorbar(im2, ax=axes[1], label="Separation (nm)")

    if mode == 'mixed':
        # Show both component centers as maps
        mu1_map = np.zeros((sz, sz), dtype=float)
        mu2_map = np.zeros((sz, sz), dtype=float)
        for x_strip in range(5):
            x_start = x_strip * strip_width
            x_end = min((x_strip + 1) * strip_width, sz)
            mu1_map[:, x_start:x_end] = mu1_per_strip[x_strip]
            mu2_map[:, x_start:x_end] = mu2_per_strip[x_strip]

        im3 = axes[2].imshow(mu1_map, cmap="coolwarm", vmin=channel_centers.min(), vmax=channel_centers.max())
        axes[2].set_title("Component 0 Center (nm)")
        axes[2].set_xlabel("X (Vertical Strip)")
        axes[2].set_ylabel("Y (Photon Count)")
        for i in range(1, 5):
            axes[2].axvline(x=i * strip_width - 0.5, color="white", linewidth=1, alpha=0.7)
        plt.colorbar(im3, ax=axes[2], label="Wavelength (nm)")

        im4 = axes[3].imshow(mu2_map, cmap="coolwarm", vmin=channel_centers.min(), vmax=channel_centers.max())
        axes[3].set_title("Component 1 Center (nm)")
        axes[3].set_xlabel("X (Vertical Strip)")
        axes[3].set_ylabel("Y (Photon Count)")
        for i in range(1, 5):
            axes[3].axvline(x=i * strip_width - 0.5, color="white", linewidth=1, alpha=0.7)
        plt.colorbar(im4, ax=axes[3], label="Wavelength (nm)")
    else:
        # Single active component center map with labels
        active_mu_map = np.zeros((sz, sz), dtype=float)
        for x_strip in range(5):
            x_start = x_strip * strip_width
            x_end = min((x_strip + 1) * strip_width, sz)
            active_mu_map[:, x_start:x_end] = active_mu_per_strip[x_strip]

        im3 = axes[2].imshow(active_mu_map, cmap="coolwarm", vmin=channel_centers.min(), vmax=channel_centers.max())
        axes[2].set_title("Active Component Center (nm)\n(S0..S4, one per strip)")
        axes[2].set_xlabel("X (Vertical Strip)")
        axes[2].set_ylabel("Y (Photon Count)")
        for i in range(1, 5):
            axes[2].axvline(x=i * strip_width - 0.5, color="white", linewidth=1, alpha=0.7)
        for i in range(5):
            x_center = i * strip_width + strip_width // 2
            axes[2].text(x_center, 5, f"S{i}", ha="center", va="bottom", color="white", fontweight="bold")
        plt.colorbar(im3, ax=axes[2], label="Wavelength (nm)")

    plt.tight_layout()
    fname = 'patterns.png' if mode == 'mixed' else 'patterns.png'
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()


def show_spectral_results(spectral_image, mode, save_dir, show=False):
    strip_height, strip_width, _, _ = build_patterns(sz)

    mean_img = np.mean(spectral_image, axis=0)
    total_counts = np.sum(spectral_image, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im1 = axes[0, 0].imshow(mean_img, cmap="inferno")
    axes[0, 0].set_title(f"Mean Spectral Intensity ({mode})")
    axes[0, 0].set_xlabel("X (Vertical Strip)")
    axes[0, 0].set_ylabel("Y (Photon Count)")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(total_counts, cmap="viridis")
    axes[0, 1].set_title("Total Photon Counts")
    axes[0, 1].set_xlabel("X (Vertical Strip)")
    axes[0, 1].set_ylabel("Y (Photon Count)")
    plt.colorbar(im2, ax=axes[0, 1])

    grid_vis = np.zeros((5, 5))
    for y_strip in range(5):
        for x_strip in range(5):
            y_center = y_strip * strip_height + strip_height // 2
            x_center = x_strip * strip_width + strip_width // 2
            grid_vis[y_strip, x_strip] = total_counts[y_center, x_center]

    im3 = axes[0, 2].imshow(grid_vis, cmap="viridis")
    axes[0, 2].set_title("25 Combinations Grid\n(Total Counts)")
    axes[0, 2].set_xlabel("Vertical Strip (separation ↑ →)")
    axes[0, 2].set_ylabel("Photon Strip (Low → High)")
    for y in range(5):
        for x in range(5):
            axes[0, 2].text(x, y, f"{int(grid_vis[y, x])}", ha="center", va="center", color="white", fontweight="bold")
    plt.colorbar(im3, ax=axes[0, 2])

    # Spectra for the highest photon count strip (y_strip = 4)
    high_y_strip = 4
    high_y = high_y_strip * strip_height + strip_height // 2
    for x_strip in range(5):
        x_center = x_strip * strip_width + strip_width // 2
        spectrum = spectral_image[:, high_y, x_center]
        com = center_of_mass(channel_centers, spectrum)
        lbl = f"Strip {x_strip}, center={com:.1f} nm"
        axes[1, 0].plot(spectrum, label=lbl, linewidth=2)
    axes[1, 0].set_xlabel("Spectral Channel")
    axes[1, 0].set_ylabel("Counts")
    axes[1, 0].set_title(f"Spectra per Strip (highest photon row) — {mode}")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Spectra at middle strip across photon levels
    mid_x_strip = 2
    mid_x = mid_x_strip * strip_width + strip_width // 2
    for y_strip in range(5):
        y_center = y_strip * strip_height + strip_height // 2
        spectrum = spectral_image[:, y_center, mid_x]
        com = center_of_mass(channel_centers, spectrum)
        axes[1, 1].plot(spectrum, label=f"{photon_levels[y_strip]} photons, center={com:.1f} nm", linewidth=2)

    title_mid = f"sep={separation_levels[mid_x_strip]:.0f} nm"
    axes[1, 1].set_xlabel("Spectral Channel")
    axes[1, 1].set_ylabel("Counts")
    axes[1, 1].set_title(f"Photon Levels @ {title_mid}")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Corner combinations comparison
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    for y_strip, x_strip in corners:
        y_center = y_strip * strip_height + strip_height // 2
        x_center = x_strip * strip_width + strip_width // 2
        spectrum = spectral_image[:, y_center, x_center]
        com = center_of_mass(channel_centers, spectrum)
        label = f"{'Low' if y_strip==0 else 'High'} photons, sep={separation_levels[x_strip]:.0f} nm, center={com:.1f} nm"
        axes[1, 2].plot(spectrum, label=label, linewidth=2)

    axes[1, 2].set_xlabel("Spectral Channel")
    axes[1, 2].set_ylabel("Counts")
    axes[1, 2].set_title("Corner Combinations")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.subplots_adjust(hspace=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_results.png'), dpi=150)
    if show:
        plt.show()


def show_phasor_analysis(spectral_image, save_dir, single=False, show=False):
    mean, real, imag = phasor_from_signal(spectral_image, axis=0, harmonic=[1, 2])

    # These cursors are placeholders; adjust as needed for your data
    if single:
        center_real = [-0.817, -0.817, -0.82, -0.82, -0.815]
        center_imag = [0.085, 0.075, 0.053, 0.015, -0.09]
        radius = 0.005
    else:
        center_real = [-0.32, -0.7, -0.8]
        center_imag = [0.02, 0.07, 0.08]
        radius = 0.02

    # Plot first harmonic phasor
    plot = PhasorPlot(allquadrants=True)
    plot.hist2d(real[0], imag[0], cmap='turbo', bins=200)
    plot.cursor(center_real, center_imag, radius=radius, color=CATEGORICAL[:len(center_real)], linestyle="-", linewidth=1)
    # Save phasor histogram (PhasorPlot may provide .save; fallback using matplotlib if needed)
    plt.savefig(os.path.join(save_dir, 'phasor_plot_no_filter.png'), dpi=150)
    if show:
        plot.show()

    masks = mask_from_circular_cursor(real[0], imag[0], center_real, center_imag, radius=radius)
    pseudo_color_image = pseudo_color(*masks, colors=CATEGORICAL)
    plot_image(pseudo_color_image, title='Pseudo Color Image (No Filter)', show=False)
    plt.savefig(os.path.join(save_dir, 'pseudo_color_no_filter.png'), dpi=150)
    if show:
        plt.show()

    # Filtered phasor (median)
    size=5
    repeat=3
    mean_median_filt, real_median_filt, imag_median_filt = phasor_filter_median(mean, real, imag, size=size, repeat=repeat)
    plot_median_filt = PhasorPlot(allquadrants=True, title=f'Filtered Phasor (Median {size}x{size}, {repeat} repeats)', xlim=(-0.9, -0.7), ylim=(-0.4, 0.2))
    plot_median_filt.hist2d(real_median_filt[0], imag_median_filt[0], cmap='turbo', bins=200)
    plot_median_filt.cursor(center_real, center_imag, radius=radius, color=CATEGORICAL[:len(center_real)], linestyle="-", linewidth=1.5)

    # Save filtered phasor histogram
    plt.savefig(os.path.join(save_dir, f'phasor_plot_median_filtered_{size}x{size}_{repeat}repeats.png'), dpi=150)
    if show:
        plot_median_filt.show()

    masks = mask_from_circular_cursor(real_median_filt[0], imag_median_filt[0], center_real, center_imag, radius=radius)
    pseudo_color_image = pseudo_color(*masks, colors=CATEGORICAL)
    plot_image(pseudo_color_image, title=f'Pseudo Color Image Median Filtered {size}x{size}, {repeat} repeats', show=False)
    plt.savefig(os.path.join(save_dir, f'pseudo_color_median_filtered_{size}x{size}_{repeat}repeats.png'), dpi=150)
    if show:
        plt.show()

    # Pawflim filtering
    sigma=5
    levels=2
    mean_pawflim_filt, real_pawflim_filt, imag_pawflim_filt = phasor_filter_pawflim(mean, real, imag, sigma=sigma, levels=levels)
    plot_pawflim_filt = PhasorPlot(allquadrants=True, title=f'Filtered Phasor (Pawflim) sigma={sigma}, {levels} levels', xlim=(-0.9, -0.7), ylim=(-0.4, 0.2))
    plot_pawflim_filt.hist2d(real_pawflim_filt[0], imag_pawflim_filt[0], cmap='turbo', bins=200)
    plot_pawflim_filt.cursor(center_real, center_imag, radius=radius, color=CATEGORICAL[:len(center_real)], linestyle="-", linewidth=1.5)

    plt.savefig(os.path.join(save_dir, f'phasor_plot_pawflim_filtered_sigma{sigma}_{levels}levels.png'), dpi=150)
    if show:
        plot_pawflim_filt.show()

    masks = mask_from_circular_cursor(real_pawflim_filt[0], imag_pawflim_filt[0], center_real, center_imag, radius=radius)
    pseudo_color_image = pseudo_color(*masks, colors=CATEGORICAL)
    plot_image(pseudo_color_image, title=f'Pseudo Color Image Pawflim Filtered sigma={sigma}, {levels} levels', show=False)
    plt.savefig(os.path.join(save_dir, f'pseudo_color_pawflim_filtered_sigma{sigma}_{levels}levels.png'), dpi=150)
    if show:
        plt.show()

def print_stats(label, spectral_image):
    strip_height, strip_width, _, _ = build_patterns(sz)
    print(f"\n--- {label} ---")
    print(f"Image size: {sz}x{sz}")
    print(f"Spectral channels: {channels}")
    print(f"Channel step: {channel_step:.2f} nm")
    print(f"Separation levels (nm): {separation_levels}")

    print("25 Combinations (Photon Level × Vertical Strip):")
    for y_strip in range(5):
        row_str = f"Photon Level {y_strip} ({photon_levels[y_strip]}): "
        for x_strip in range(5):
            y_start, y_end = y_strip * strip_height, min((y_strip + 1) * strip_height, sz)
            x_start, x_end = x_strip * strip_width, min((x_strip + 1) * strip_width, sz)
            region_counts = np.mean(np.sum(spectral_image[:, y_start:y_end, x_start:x_end], axis=0))
            row_str += f"({photon_levels[y_strip]}, {region_counts:.0f}) "
        print(row_str)


if __name__ == "__main__":
    # Build base patterns and component profiles (mixed case)
    strip_height, strip_width, photon_pattern, separation_map = build_patterns(sz)
    comp_profiles, mu1_per_strip, mu2_per_strip = build_component_profiles()

    # Single-case: one unique component per strip with consecutive separations
    mu_per_strip, single_profiles = build_single_centers_and_profiles()

    # Synthesize both datasets
    spectral_mixed = synthesize_spectral_image_mixed(comp_profiles)
    spectral_single = synthesize_spectral_image_single(single_profiles)

    # Mixed: plots and saves
    show_mixed = False
    show_patterns(mode='mixed', mu1_per_strip=mu1_per_strip, mu2_per_strip=mu2_per_strip, save_dir=out_mixed, show=show_mixed)
    show_spectral_results(spectral_mixed, mode='mixed', save_dir=out_mixed, show=show_mixed)
    show_phasor_analysis(spectral_mixed, save_dir=out_mixed, show=show_mixed)
    print_stats("MIXED (two components per strip, 50/50)", spectral_mixed)

    # Single: plots and saves (pass mode != 'single' to avoid strip_component_ids labeling)
    show_single = False
    show_patterns(mode='single', mu1_per_strip=mu1_per_strip, mu2_per_strip=mu2_per_strip,
                  active_mu_per_strip=mu_per_strip, save_dir=out_single, show=show_single)
    show_spectral_results(spectral_single, mode='single-unique', save_dir=out_single, show=show_single)
    show_phasor_analysis(spectral_single, save_dir=out_single, single=True, show=show_single)
    print_stats("SINGLE (one unique component per strip, consecutive separations)", spectral_single)

    print(f"\nSaved figures to:\n- {out_mixed}\n- {out_single}")