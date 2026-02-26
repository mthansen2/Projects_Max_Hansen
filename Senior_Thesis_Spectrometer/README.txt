# Senior Thesis Spectrometer Software (Ti:Sapphire Laser)

Undergraduate Applied Physics senior thesis project focused on developing software for a low-cost spectrometer to measure and visualize the output spectrum of a Ti:Sapphire laser.

## Overview

This project builds on prior spectrometer hardware work by improving the software pipeline used to convert USB camera image data into a usable spectrometer output (intensity vs. wavelength). The goal was to create a practical and robust interface for undergraduate lab use.

The software processes camera pixel intensities and converts pixel position to wavelength through calibration, enabling real-time spectral visualization.

## What I Worked On

- Developed spectrometer software for USB camera-based spectral acquisition
- Processed camera image data into intensity vs. wavelength output
- Implemented calibration workflow using known laser wavelengths
- Integrated functionality into an existing lab interface/workflow
- Tested and compared results against a commercial Ocean Optics spectrometer
- Evaluated Ti:Sapphire laser operation in:
  - Continuous wave (CW) mode
  - Mode-locked / pulsed operation

## Technical Highlights

- **Optics / Spectrometer concepts**
  - Diffraction grating geometry
  - Wavelength mapping from camera pixel position
  - Calibration using known reference wavelengths
- **Software / Instrumentation**
  - LabVIEW-based implementation
  - USB camera image acquisition
  - Pixel-row intensity extraction and summation
  - Exposure / integration handling
  - Error handling and interface integration
- **Validation**
  - Comparison against Ocean Optics spectrometer output
  - Demonstrated close spectral agreement in testing

## Key Results (from thesis)

- Calibrated the system across the Ti:Sapphire region of interest (~750–850 nm)
- Demonstrated software operation for both CW and pulsed Ti:Sapphire output
- Compared normalized spectra with Ocean Optics and achieved close agreement (reported in thesis)
- Thesis notes replication within roughly **0.3 nm** in comparison testing

## Repository Contents

- `KMAX Spectrometer/` — project files related to the spectrometer software/workflow
- `Max_Hansen_Senior_Thesis.pdf` — full thesis writeup
- `README.md` — project summary (this file)

## Thesis Summary

The project focused on creating a usable software interface for a previously constructed spectrometer system. The software captures and processes USB camera image data, extracts pixel intensities from selected rows, sums intensity information, and converts pixel location to wavelength after calibration. This enables visualization of spectral output from the Ti:Sapphire laser for student use and experimentation.

## Future Improvements (ideas)

- Improve mechanical slit alignment stability to reduce spectral distortion
- Expand calibration workflow and automation
- Improve sensitivity/noise handling and exposure tuning
- Port / reimplement portions of the pipeline in Python for broader reproducibility
- Add data logging and export tools for lab experiments

## Author

**Max Hansen**  
Applied Physics (Undergraduate Senior Thesis, 2024)

---

If you are viewing this project as part of my portfolio repo, the PDF contains the full methodology, calibration process, testing, and results.