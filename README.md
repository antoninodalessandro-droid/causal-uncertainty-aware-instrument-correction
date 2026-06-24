README

Code repository accompanying the manuscript:

Causal and Uncertainty-Aware Instrument Correction in Broadband Seismology

Geophysical Journal International (GJI)

Author:
Antonino D'Alessandro
Istituto Nazionale di Geofisica e Vulcanologia (INGV)
Rome, Italy

---

DESCRIPTION

This repository contains the Python scripts used to generate all figures,
synthetic experiments, diagnostic metrics, and numerical results presented
in the manuscript:

D'Alessandro Antonino
Causal and Uncertainty-Aware Instrument Correction in Broadband Seismology

Geophysical Journal International (GJI).

The study develops a physically consistent framework for instrument
correction in broadband seismology based on causal inverse filtering,
regularization, uncertainty propagation, and digital-system constraints.

No observational seismic waveform data were used in this work.
All figures, numerical examples, and synthetic validation experiments are
generated directly by the scripts contained in this repository.

---

REPOSITORY CONTENTS

Fig01_instrument_correction_block_diagram.py
Figure 1 - Instrument correction as a constrained inverse problem
in the digital domain.

Fig02_pole_zero_causality_constraints.py
Figure 2 - Causality and stability constraints in the pole-zero domain.

Fig03_ideal_vs_causal_regularized_inverse.py
Figure 3 - Ideal inverse versus causal regularized inverse.

Fig04_discretization_Hs_vs_Hz.py
Figure 4 - Discretization effects and differences between H(s) and H(z).

Fig05_self_noise_bounds_and_tradeoff.py
Figure 5 - Self-noise amplification bounds and bias-noise trade-off.

Fig06_uncertainty_propagation.py
Figure 6 - Propagation of instrument-response uncertainty.

Fig07_quality_metrics_and_diagnostics.py
Figure 7 - Quality metrics and stability diagnostics.

Fig08_synthetic_validation_final.py
Figure 8 - Synthetic validation of the proposed framework.

Fig09_workflow_algorithmic.py
Figure 9 - Algorithmic workflow for physically consistent
instrument correction.

---

SOFTWARE REQUIREMENTS

Python 3.x

Required packages:

* NumPy
* SciPy
* Matplotlib

Installation example:

pip install numpy scipy matplotlib

---

REPRODUCIBILITY

Each script is fully self-contained and can be executed independently.

Running a script automatically generates the corresponding figure in
PNG and PDF formats.

Example:

python Fig05_self_noise_bounds_and_tradeoff.py

The synthetic signals, instrument responses, uncertainty realizations,
Monte Carlo simulations, and diagnostic metrics are generated internally
by the scripts.

No external datasets are required.

---

DATA AVAILABILITY

No observational seismic waveform data were used in this study.

All input parameters, synthetic datasets, numerical experiments,
diagnostic metrics, and figure outputs are generated directly by the
Python scripts contained in this repository.

The repository therefore contains all software and reproducible materials
necessary to regenerate the figures and numerical results presented in
the manuscript.

---

LICENSE

This repository is distributed for scientific and educational purposes.

Please cite the associated manuscript when using or adapting the code.

---

CONTACT

Antonino D'Alessandro
Istituto Nazionale di Geofisica e Vulcanologia (INGV)
Rome, Italy

Email: [antonino.dalessandro@ingv.it](mailto:antonino.dalessandro@ingv.it)

---
