from __future__ import annotations
import numpy as np

def normalize_energy(psf: np.ndarray) -> np.ndarray:
    s = float(np.sum(psf))
    if s <= 0:
        raise ValueError("PSF has non-positive total energy.")
    return psf / s

def strehl_ratio(psf_aberr: np.ndarray, psf_ideal: np.ndarray) -> float:
    """
    Strehl = peak(aberrated) / peak(ideal), after both PSFs are energy-normalized.
    """
    a = normalize_energy(psf_aberr)
    i = normalize_energy(psf_ideal)
    return float(np.max(a) / (np.max(i) + 1e-15))
