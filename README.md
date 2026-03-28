## Analog Neuromorphic Prediction Machines for Real-Time Sensor Processing

Real-time, low-power sensor processing is rapidly becoming the space for commercial success of neuromorphic computing. This can be attributed to the growing interest in state-space models and neural dynamical systems for continuous sensor signal analysis at high refresh rates, which can present prohibitive power scaling challenges within traditional digital systems. This paper proposes an Analog Neuromorphic Prediction Machine (ANPM) that aligns this challenge with the strengths of neuromorphic computing by combining multi-bit compute-in-memory devices with analog circuits for low-power sensor signal prediction. Two core dynamical systems are selected for our circuit implementation: an oscillator and an integrator, chosen for their natural alignment with the temporal and spectral structure of physical sensor signals. We show that a 128-neuron ANPM with these two underlying dynamical systems is capable of computing Short-Time Fourier Transform and predicting into a horizon of 500 ms for periodic, quasi-periodic, and chaotic 0.25- to 5-Hz signals of length 10 s. We also provide underlying circuitry for the ANPM with the following features: (1) a ferrodiode-based synaptic weighting circuit; (2) ensembles of oscillator and integrator circuits that can execute inference up to 200 Hz and 10 kHz; (3) a per-neuron per-inference energy of 1.64 µJ (200 Hz) and 0.29 µJ (10 kHz) with a total power consumption of < 1 W for systems with up to 3000 neural units (200 Hz) or 345 neural units (10 kHz); (4) an area reduction factor of 40x over SRAM-based digital systems; and (5) a per-layer latency of 3.18 ms (200 Hz) and 63.87 µs (10 kHz) that allows the system to run in real-time on continuous sensor streams. In total, we present the first end-to-end theoretical work that integrates a ferrodiode into a viable neuromorphic sensor processing system.

Basic modules and parameters in `uh_li.py` and `bp_li.py` adapted from:

```
@inproceedings{higuchi2024understanding,
  title        = {Understanding the Convergence in Balanced Resonate-and-Fire Neurons},
  author       = {Higuchi, Saya and and Boht{\'e}, Sander and Otte, Sebastian},
  year         = 2024,
  booktitle    = {Austrian Symposium on AI, Robotics, and Vision (AIRoV)},
  publisher    = {innsbruck university press},
  series       = {Proceedings of Austrian Symposium on AI, Robotics, and Vision 2024},
  volume       = 1,
  pages        = {437--445},
  pdf = 	 {https://ulb-dok.uibk.ac.at/ulbtirolfodok/download/pdf/12691798},
}
```
