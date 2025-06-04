# Comparison and Limitations of Single-Photon Level Sources in Quantum Lidar

## Description

This module is a simple Python package used to compute the performance of different single-photon-level sources for quantum lidar applications. The sources studied in this work are:
1) Single-photon sources based on quantum dots
2) Entangled photon-pair sources based on spontaneous parametric down-conversion (SPDC)
3) Attenuated pulsed lasers

The goal of this work is to provide a fair comparison of the performance of lidar systems for stealth applications. The lidar systems modeled in this work use a source and a detector operating in the single-photon regime. No joint measurement is performed.
To ensure a fair comparison between the different sources, time correlation is employ for all. For the entangled-photon source, the detection of an idler photon is used to trigger the start of the time-correlation window, and coincidence detection is performed. For the single-photon sources and the attenuated pulsed laser, the triggering electrical signal is used for coincidence detection.
In order to perform a fair comparison, one must fix the optical power leaving the system and one of the following:

1) The non-vacuum probability: the probability that at least one photon is emitted and leaves the lidar system.
2) The multi-photon probability: the probability that more than one photon is emitted and leaves the lidar system


##### Why those parameters ?

A quantum lidar system with single-photon-level sources allows for enhanced stealthiness, which is defined as the ability to detect an adversary without being detected. This application is considered the primary use case for quantum lidar systems operating in the single-photon regime.

If the target possesses a classical photodetector, then it cannot detect the quantum lidar. However, if it possesses a single-photon detector, it might be able to detect it.

By matching the non-vacuum probability, one can compare different sources when they are equally detectable by an adversary with a non-number-resolving detector. By matching the multi-photon probability, one can compare the sources when they are equally detectable by an adversary with a number-resolving detector.

The metrics used are the signal-to-noise ratio (SNR) as well as the receiver operating characteristic (ROC) curves.

The details are presented in the paper [add reference]. The paper should be read before diving into the code and the tutorials.

## Quick Usage Preview
### Calculating the SNR When the Multi-Photon Probability Is Matched Across Sources
```python
from Sources import SetupParameters, PulsedLaser, EntangledPhotonSPDC, SinglePhoton

param_laser = SetupParameters(
	fock_space_dim=5,
	output_power=2e6,
	multi_photon_probability=0.05,
	no_vacuum_probability=None,
	sp_collection=None,
	sp_p1=None,
	sp_p2=None,
	spdc_eps_heralding=None,
	spdc_eps_collection=None,
	atmosphere=1,
	target_distance=1,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.5,
	background=400,
	detector_dark=200,
	timing_window=0.5e-9,
)

laser = PulsedLaser(param_laser)

snr_laser = laser.signal_to_noise_rate()
```

### Producing ROC Curves for the Various Sources
```python

signal_rate_laser = laser.signal_rate()
noise_rate_laser = laser.noise_rate()
trigger_rate_laser = laser.trigger_rate()

acquisition_time = 1
precision = 20
range_interval = 50

true_positive_laser, false_positive_laser = RocAnalysis(
	signal_rate=signal_rate_laser,
	noise_rate=noise_rate_laser,
	trigger_rate=trigger_rate_laser,
	threshold_limit=trigger_rate_laser/100,
	range_interval=range_interval,
	timing_window=param["timing_window"],
	acquisition_time=acquisition_time,
	precision=precision,
).compute_p_d_p_fa()



```

## Important files

The file ```Sources.py``` contains the implementation of the three sources considered. The file  ```Analysis.py``` contains algorithms to calculate the ROC curves and the range limitation of a given system. Furthermore, this file allows simulation of a typical histogram detection.

## Tutorial

Tutorials on using this module are available and should be followed in order:

1) [Setting parameters](Tutorials/Setting%20parameters.ipynb)
2) [ROC Curves](Tutorials/ROC%20Curves.ipynb)
3) [Histogram simulation](Tutorials/Histogram%20simulation.ipynb)
4) [Range limitation](Tutorials/Range%20limitation.ipynb)

## Paper

TODO: ADD LINK TO PAPER AND ADD THE FACT THAT PEOPLE CAN REPRODUCE THE GRAPH

## Citation

[...]

## Authors

This work was carried out by Anthony Drouin (IQC/UWaterloo), Dr. Jean-Philippe Bourgoin (Single Quantum System), Dr. Sara Hosseini (IQC/UWaterloo/NRC), Prof. François Sfigakis (IQC/UWaterloo), Prof. Jonathan Baugh (IQC/UWaterloo), and Prof. Michael E. Reimer (IQC/UWaterloo/Single Quantum System). Prof. Michael E. Reimer is the corresponding author for this work (michael.reimer@uwaterloo.ca).

The implementation of the theoretical model was done by Anthony Drouin (anthony.drouin@uwaterloo.ca).

## License

[Apache License 2.0](LICENSE)
