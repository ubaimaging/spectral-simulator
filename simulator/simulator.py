import numpy
from numpy.typing import ArrayLike, NDArray

from scipy.stats import gaussian_kde

from typing import Any


def generate_spectra(
        Nph: int,
        sigm: float,
        off: float,
        pha: float,
        rang: ArrayLike,
        /,
) -> NDArray[Any]:
    """
    Generates a spectral profile with a Gaussian distribution, given a number of photons 
    distributed over a wavelength range.

    Parameters
    ----------
    Nph : int
        Number of photons that the spectrum should contain.
    sigm : float
        Standard deviation of the distribution.
    off : float
        Shift of the distribution mean, relative to the center of the 'rang' vector.
    pha : float
        Adjustment factor for the phasor position.
    rang : list or ndarray
        Range of wavelengths over which the signal is represented.

    Returns
    -------
    spectra : ndarray
        A vector containing the spectral profile with the Nph photons distributed
        within the spectral range defined by 'rang'.
    """
    # Parameter validation
    if Nph <= 0:
        raise ValueError("The number of photons 'Nph' must be a positive integer.")
    
    if sigm < 0:
        raise ValueError("The 'sigm' parameter must be a positive value.")
    
    if len(rang) < 2:
        raise ValueError("The 'rang' parameter must be an array with at least two elements.")

    # Generate photons with a normal distribution
    wavelengths = numpy.random.randn(Nph) * sigm + pha + off
    
    # Create the histogram of the spectrum over the given range
    spectra, _ = numpy.histogram(wavelengths, bins=rang)
    return spectra



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
