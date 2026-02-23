import numpy as np
from optics import circular_pupil, apply_phase
from optics.propagation import fraunhofer_psf

def test_psf_energy_scales_with_pupil_power_invariant_to_global_phase():
    n = 256
    pupil = circular_pupil(n, radius=0.40)

    field = apply_phase(pupil, np.zeros((n, n)))
    psf = fraunhofer_psf(field)

    pupil_power = np.sum(np.abs(field) ** 2)
    psf_power = np.sum(psf)

    # Apply a uniform global phase offset (should not change intensity PSF)
    field2 = apply_phase(pupil, 0.123 * np.ones((n, n)))
    psf2 = fraunhofer_psf(field2)
    psf_power2 = np.sum(psf2)

    r1 = psf_power / (pupil_power + 1e-15)
    r2 = psf_power2 / (pupil_power + 1e-15)

    assert np.isfinite(r1) and np.isfinite(r2)
    assert abs(r1 - r2) / r1 < 1e-9
