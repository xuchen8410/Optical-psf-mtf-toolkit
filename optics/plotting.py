from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def save_psf_image(psf: np.ndarray, path: str, log: bool = True) -> None:
    psf_plot = psf.astype(np.float64, copy=True)
    if log:
        psf_plot = np.log10(psf_plot + 1e-12)

    plt.figure()
    plt.imshow(psf_plot, origin="lower")
    plt.colorbar()
    plt.title("PSF (log10)" if log else "PSF")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def save_mtf_curve(rho: np.ndarray, mtf: np.ndarray, path: str) -> None:
    plt.figure()
    plt.plot(rho, mtf)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Normalized spatial frequency (arb.)")
    plt.ylabel("MTF")
    plt.title("Radial MTF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
