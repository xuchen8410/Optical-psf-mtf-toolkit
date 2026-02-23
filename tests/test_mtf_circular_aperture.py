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

    # Map rho to normalized frequency nu in [0,1].
    # For our grid, the OTF support (cutoff) scales ~ with pupil radius.
    # We estimate cutoff from where MTF first becomes small in the tail (robust to discretization).
    # Autocorr-based MTF should approach ~0 near cutoff.
    tail = slice(int(0.70 * len(mtf)), len(mtf))
    cutoff = rho[int(np.argmin(mtf[tail]) + 0.70 * len(mtf))]
    assert cutoff > 0.05

    nu = np.clip(rho / cutoff, 0.0, 1.0)
    mtf_a = mtf_circular_analytic(nu)

    # Compare mid-band (most reliable numerically)
    band = (nu > 0.05) & (nu < 0.75)
    err = float(np.mean(np.abs(mtf[band] - mtf_a[band])))

    # Autocorr definition aligns with analytic MTF; should be reasonably close.
    assert err < 0.07
