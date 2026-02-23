import numpy as np
from optics import circular_pupil
from optics.propagation import mtf_from_pupil_autocorr

def mtf_circular_analytic(nu: np.ndarray) -> np.ndarray:
    """
    Analytic MTF of a diffraction-limited circular pupil for normalized spatial frequency nu in [0,1]:
    MTF(nu) = (2/pi) [ acos(nu) - nu*sqrt(1-nu^2) ].
    """
    nu = np.clip(nu, 0.0, 1.0)
    return (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu**2))

def test_mtf_matches_circular_analytic_midband_autocorr():
    n = 512
    radius = 0.42
    pupil = circular_pupil(n, radius=radius)

    rho, mtf = mtf_from_pupil_autocorr(pupil)

    # IMPORTANT:
    # Autocorrelation output is parameterized by pupil SHIFT (lag) coordinates, not frequency.
    # For a circular pupil of radius r (in normalized pupil coords), the autocorr support
    # radius equals the pupil DIAMETER (2r). That maps to normalized frequency nu in [0,1] as:
    #   nu = shift / (2r)
    cutoff_shift = 2.0 * radius
    nu = np.clip(rho / cutoff_shift, 0.0, 1.0)

    mtf_a = mtf_circular_analytic(nu)

    # Compare mid-band where discretization is most reliable
    band = (nu > 0.05) & (nu < 0.75)
    err = float(np.mean(np.abs(mtf[band] - mtf_a[band])))

    # With correct axis mapping, this should be reasonably close.
    assert err < 0.08
