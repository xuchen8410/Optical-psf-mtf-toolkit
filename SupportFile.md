## 📘 Theory Overview

This repository focuses on **optical imaging performance analysis** using principles from wave optics and Fourier optics.  
It provides computational tools to understand how physical optical imperfections translate into measurable image quality.

---

## 1. Image Formation in Optical Systems

An optical imaging system forms an image by transmitting and transforming incoming light waves through lenses or mirrors.  
While geometric optics describes ray paths, real imaging performance is fundamentally governed by **wave behavior**, including diffraction and phase variations.

Because light behaves as a wave, a single point object does not form a perfect point in the image plane. Instead, energy spreads due to diffraction and aberrations.

This response defines the system’s fundamental imaging limit.

---

## 2. Point Spread Function (PSF)

The **Point Spread Function (PSF)** describes how an optical system images a single point source.

- It represents the spatial distribution of light intensity in the image plane.
- All extended images can be modeled as a superposition of shifted PSFs.
- Optical aberrations modify the PSF shape and reduce image sharpness.

The PSF is therefore the most direct spatial-domain description of optical performance.

---

## 3. Optical Transfer Function (OTF) and MTF

Analyzing imaging behavior in the frequency domain provides deeper insight.

The **Optical Transfer Function (OTF)** is the Fourier transform of the PSF and characterizes how spatial frequencies propagate through the system.

Its magnitude is the:

**Modulation Transfer Function (MTF)**

which measures contrast preservation at different spatial frequencies:

- **Low frequencies** → large image structures
- **High frequencies** → fine image detail

Higher MTF values indicate better preservation of image contrast and resolution.

---

## 4. Wavefront Errors and Image Quality

Imaging degradation is often caused by phase errors in the optical wavefront arising from:

- design residual aberrations
- manufacturing tolerances
- alignment errors
- thermal or structural deformation
- environmental disturbances

Wavefront deviations alter diffraction behavior and directly impact PSF and MTF performance.

The imaging relationship can be summarized as:

This chain connects physical optical errors to observable image performance.

---

## 5. Computational Approach

This project implements numerical tools based on **Fourier optics** to:

- model wavefront-driven image formation
- compute PSF and MTF metrics
- evaluate imaging performance quantitatively
- support reproducible optical analysis workflows

The goal is to bridge theoretical optics and practical computational evaluation in a transparent and extensible framework.

---

## 🎯 Scope

The toolkit is intended for:

- optical engineers
- computational imaging researchers
- students learning imaging physics
- system engineers evaluating optical performance

No prior specialization in optical design software is required.
