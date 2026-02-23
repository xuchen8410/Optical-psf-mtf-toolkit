from __future__ import annotations
import numpy as np

def circular_pupil(n: int, radius: float = 0.45) -> np.ndarray:
    """
    Circular pupil mask on an NxN grid in normalized coords [-0.5, 0.5).
    radius is normalized to half-width of grid (0.5). Typical radius < 0.5.
    """
    y, x = np.indices((n, n))
    xn = (x - n / 2) / n
    yn = (y - n / 2) / n
    r = np.sqrt(xn**2 + yn**2)
    return (r <= radius).astype(np.float64)

def zernike_defocus(n: int) -> np.ndarray:
    """
    Zernike defocus (Noll index 4 shape) on unit disk.
    Z(r) = 2 r^2 - 1 inside unit disk, 0 outside.
    This is a shape function; scale to waves externally.
    """
    y, x = np.indices((n, n))
    xn = (x - n / 2) / (n / 2)
    yn = (y - n / 2) / (n / 2)
    r = np.sqrt(xn**2 + yn**2)
    z = 2 * r**2 - 1
    z[r > 1.0] = 0.0
    return z

def apply_phase(pupil_amp: np.ndarray, phase_waves: np.ndarray) -> np.ndarray:
    """
    pupil_amp: amplitude mask (0..1)
    phase_waves: phase in waves (OPD / lambda)
    return complex pupil field.
    """
    if pupil_amp.shape != phase_waves.shape:
        raise ValueError("pupil_amp and phase_waves must have the same shape.")
    return pupil_amp * np.exp(1j * 2 * np.pi * phase_waves)
