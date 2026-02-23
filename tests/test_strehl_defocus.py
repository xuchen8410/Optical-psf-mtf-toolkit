import numpy as np
from optics import circular_pupil, zernike_defocus, apply_phase
from optics.propagation import fraunhofer_psf
from optics.metrics import normalize_energy, strehl_ratio

def test_strehl_decreases_monotonically_with_defocus():
    n = 384
    pupil = circular_pupil(n, radius=0.42)
    zdef = zernike_defocus(n)

    pupil_ideal = apply_phase(pupil, np.zeros((n, n)))
    psf_ideal = normalize_energy(fraunhofer_psf(pupil_ideal))

    amps = [0.0, 0.05, 0.10, 0.20]  # waves
    strehl = []
    for a in amps:
        field = apply_phase(pupil, a * zdef)
        psf = normalize_energy(fraunhofer_psf(field))
        strehl.append(strehl_ratio(psf, psf_ideal))

    assert abs(strehl[0] - 1.0) < 1e-2
    assert strehl[0] >= strehl[1] >= strehl[2] >= strehl[3]
