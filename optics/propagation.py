from __future__ import annotations
import numpy as np

def _fft2c(a: np.ndarray) -> np.ndarray:
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))

def _ifft2c(a: np.ndarray) -> np.ndarray:
    """Centered 2D IFFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))

def fraunhofer_psf(pupil_field: np.ndarray) -> np.ndarray:
    """
    Fraunhofer diffraction (far-field). PSF ∝ |FT{pupil}|^2.
    Returns intensity PSF (not normalized).
    """
    e_img = _fft2c(pupil_field)
    psf = (np.abs(e_img) ** 2).astype(np.float64)
    return psf

def otf_from_psf(psf: np.ndarray) -> np.ndarray:
    """
    OTF is FT of PSF (in incoherent imaging). Returns complex OTF (centered).
    """
    return _fft2c(psf)

def mtf_radial(psf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial average of MTF from PSF:
      MTF = |OTF| / |OTF(0,0)|
    Returns (rho_bins, mtf_radial).
    rho is normalized spatial frequency on grid.
    """
    otf = otf_from_psf(psf)
    mtf2d = np.abs(otf)
    mtf2d /= (mtf2d[mtf2d.shape[0] // 2, mtf2d.shape[1] // 2] + 1e-15)

    n = psf.shape[0]
    y, x = np.indices((n, n))
    xn = (x - n / 2) / n
    yn = (y - n / 2) / n
    rho = np.sqrt(xn**2 + yn**2)

    nbins = n // 2
    rho_max = 0.5 * np.sqrt(2)  # corner in normalized coords
    rbin = np.floor(rho * nbins / rho_max).astype(int)
    rbin = np.clip(rbin, 0, nbins - 1)

    sums = np.bincount(rbin.ravel(), weights=mtf2d.ravel(), minlength=nbins)
    counts = np.bincount(rbin.ravel(), minlength=nbins)
    mtf_r = sums / np.maximum(counts, 1)

    rho_bins = (np.arange(nbins) + 0.5) * rho_max / nbins
    return rho_bins, mtf_r
