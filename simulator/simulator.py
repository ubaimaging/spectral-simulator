import numpy
from numpy.typing import ArrayLike, NDArray

from scipy.stats import gaussian_kde

from typing import Any, Tuple, Optional


def generate_spectra(
    Nph: int,
    off: float,
    pha: float,
    rang: ArrayLike,
    sigm: float = 1.0,
    lam: Optional[float] = None,
    seed: Optional[int] = None,
    clip_to_range: bool = False,
) -> Tuple[NDArray[numpy.int64], NDArray[numpy.float64], int]:
    """
    Generate a spectral histogram (counts per bin) over the wavelength range `rang`.

    Modes
    -----
    - Normal mode (lam is None): draw Nph samples from Normal(mean, sigm),
      where mean = center(rang) + off + pha.
    - Poisson mode (lam is not None): draw Nph samples from Poisson(lam) and
      shift by off + pha. (Note: this treats Poisson as a sampler over 'wavelength-like'
      integers; for physically accurate photon counting per bin, prefer drawing Poisson
      counts per bin from an expected spectrum.)

    Parameters
    ----------
    Nph : int
        Number of samples (photons) to draw before histogramming.
    off : float
        Shift relative to the center of `rang` (in wavelength units).
    pha : float
        Additional shift term (documented as phasor adjustment; here applied linearly).
    rang : array-like
        Monotonic array of bin edges (length B+1).
    sigm : float, default=1.0
        Standard deviation in Normal mode (must be > 0).
    lam : float or None, default=None
        Poisson mean in Poisson mode (must be > 0 if provided).
    seed : int or None
        Random seed for reproducibility.
    clip_to_range : bool, default=False
        If True, clip samples to [rang[0], rang[-1]] before histogramming;
        otherwise, samples outside the range are dropped by numpy.histogram.

    Returns
    -------
    spectra : (B,) int64 ndarray
        Counts per bin.
    bin_edges : (B+1,) float64 ndarray
        The bin edges actually used (np.asarray(rang, float)).
    lost : int
        Number of samples that fell outside the range (0 if clip_to_range=True).
    """
    if Nph <= 0:
        raise ValueError("Nph must be a positive integer.")
    rang = numpy.asarray(rang, dtype=float)
    if rang.ndim != 1 or rang.size < 2:
        raise ValueError("'rang' must be a 1D array of at least 2 edges.")
    if not numpy.all(numpy.diff(rang) > 0):
        raise ValueError("'rang' must be strictly increasing.")

    rng = numpy.random.default_rng(seed)
    mean = (rang[0] + rang[-1]) / 2.0 + off + pha

    if lam is None:
        if sigm <= 0:
            raise ValueError("In Normal mode, 'sigm' must be > 0.")
        samples = rng.normal(loc=mean, scale=sigm, size=Nph)
    else:
        if lam <= 0:
            raise ValueError("In Poisson mode, 'lam' must be > 0.")
        # Poisson returns integers around lam; shift to wavelength-like values
        samples = rng.poisson(lam=lam, size=Nph).astype(float) + (off + pha)

    if clip_to_range:
        samples = numpy.clip(samples, rang[0], rang[-1])
        lost = 0
    else:
        # Count how many fall outside before histogramming (for bookkeeping)
        lost = int(numpy.sum((samples < rang[0]) | (samples > rang[-1])))

    spectra, edges = numpy.histogram(samples, bins=rang)
    return spectra.astype(numpy.int64), edges.astype(numpy.float64), lost



def distribute_photons(
        spectral_curve: ArrayLike,
        num_photons: int,
        /,
 ) -> NDArray[Any]:
    """
    Distributes a specific number of photons based on a given spectral density curve.

    Parameters
    ----------
    spectral_curve : ndarray
        Probability density curve used to model the photon distribution.
    num_photons : int
        Number of photons to distribute according to the given spectral curve.

    Returns
    -------
    photon_distribution : ndarray
        Array containing the photon distribution across the available wavelength range.
    """

    # Validate inputs
    if not isinstance(spectral_curve, numpy.ndarray):
        raise TypeError("spectral_curve must be a NumPy array.")
    if not isinstance(num_photons, int) or num_photons < 1:
        raise ValueError("num_photons must be an integer greater than or equal to 1.")

    # Define the wavelength range
    wavelengths = numpy.arange(len(spectral_curve))

    # Kernel density estimation (KDE) with spectral curve as weights
    kde = gaussian_kde(wavelengths, weights=spectral_curve)

    # Generate 1000 evaluation points across the wavelength range
    eval_points = numpy.linspace(wavelengths.min(), wavelengths.max(), 1000)

    # Evaluate the smoothed density over these points
    smoothed_density = kde(eval_points)

    # Normalize the density to ensure it sums to 1 for use as probabilities
    smoothed_density /= smoothed_density.sum()

    # Sample photons based on the smoothed density
    photon_distribution = numpy.random.choice(
        eval_points, size=num_photons, p=smoothed_density
    )
    return photon_distribution


def generate_random_fractions(
        N: int,
        n: int = 1
) -> NDArray[Any]:
    """
    Generates a matrix of random fractions, where each row contains N fractions 
    that sum to 1.

    Parameters
    ----------
    N : int
        Number of fractions in which the unit is partitioned.
    n : int, optional
        Number of samples to generate. Default is 1.

    Returns
    -------
    fractions : ndarray
        A matrix of size (n, N) containing random fractions, where each row sums to 1.
    """
    
    if not (isinstance(N, int) and N >= 1 and isinstance(n, int) and n >= 1):
        raise ValueError("Both N and n must be integers greater than or equal to 1.")
    
    # Generate random numbers and normalize them so that each row sums to 1
    fractions = numpy.random.rand(n, N)
    fractions /= fractions.sum(axis=1, keepdims=True)
    return fractions
