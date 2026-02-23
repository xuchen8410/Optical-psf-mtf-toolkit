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


def mtf_from_pupil_autocorr(pupil_amp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial MTF via pupil intensity autocorrelation (definition-consistent with analytic circular MTF).
    For a clear circular pupil with uniform amplitude, this aligns well with the closed-form MTF.

    pupil_amp: real amplitude mask (0..1)
    Returns (rho_bins, mtf_radial) on the same rho definition used in mtf_radial().
    """
    # Use pupil intensity (incoherent OTF depends on pupil intensity autocorrelation)
    P = pupil_amp.astype(np.float64)
    # Autocorrelation via Fourier domain: autocorr = IFFT(|FFT(P)|^2)
    A = _ifft2c(np.abs(_fft2c(P)) ** 2).real
    # Normalize OTF peak to 1
    A /= (A[A.shape[0] // 2, A.shape[1] // 2] + 1e-15)

    # Radial average of A (this is MTF for a real, symmetric pupil)
    n = P.shape[0]
    y, x = np.indices((n, n))
    xn = (x - n / 2) / n
    yn = (y - n / 2) / n
    rho = np.sqrt(xn**2 + yn**2)

    nbins = n // 2
    rho_max = 0.5 * np.sqrt(2)
    rbin = np.floor(rho * nbins / rho_max).astype(int)
    rbin = np.clip(rbin, 0, nbins - 1)

    sums = np.bincount(rbin.ravel(), weights=A.ravel(), minlength=nbins)
    counts = np.bincount(rbin.ravel(), minlength=nbins)
    mtf_r = sums / np.maximum(counts, 1)

    rho_bins = (np.arange(nbins) + 0.5) * rho_max / nbins
    return rho_bins, mtf_r
