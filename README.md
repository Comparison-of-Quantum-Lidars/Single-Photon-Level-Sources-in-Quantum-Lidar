# Comparison and Limitations of Single-Photon Level Sources in Quantum LiDAR

## Description

This module is a simple Python package that is used to simulate the performance of different single-photon level 
sources for quantum LiDAR applications. The source studied in this work are the following:
1) Single-photon sources based on quantum dots
2) Entangled photon pair sources based on spontaneous parametric down-conversion
3) Attenuated Pulsed Laser Sources

The goal of this work is to compare fairly the performance of lidars systems based on these sources. The goal is to
provide a fair and complete comparison of quantum illumination and classical illumination in the context of LiDAR.
The LiDAR system that are modeled in this work are using a source and a detector operating in the quantum regime. No
joint-measurement is performed in the lidar architectures studied since those systems requires either a prior knowledge
of the target, or needs a sweep of the delay line which drastically reduces the speed of the system. Since robust
Quantum Memory and Non-Destructive measurement device are not yet available, we are using lidar system where the
coincidence measurement is post-processed on a _classical_ computer.

To ensure a fair comparison between the different sources, time-correlation is performed for all. For the Entangled-photon source,
the detection of an idler photon is used to trigger the start of the time-correlation window and coincidence is performed.
For the single-photon sources and the attenuated pulsed laser, the triggering electrical signal is used to do the coincidence.

In order to perform a fair comparison, one need to fix some parameters across the different sources. Obviously, the
optical power leaving the system is the same for all sources. From there, one of the following parameters can be fixed:
1) The Non-Vacuum Probability: This is the probability that at least one photon is emitted and is leaving the lidar system.
2) The Multi-Photon Probability: This is the probability that more than one photon is emitted and is leaving the lidar system.

##### Why those parameters ?

Quantum lidar is a broad term that includes multiple architecture. Technically, as long as one part of a lidar uses the
quantization of light, it can be considered a quantum lidar. For many applications, using a classical source with a
single-photon detector is the best option: this approach enhances detection without compromising output power,
allowing for long-range applications while maintaining a high precision.

When transitioning to quantum sources, one loses the ability to operate at long range, since the probability of a photon
being reflected within the solid angle of the detector decreases as $1/L^2$ where $L$ is the distance lidar-target.
The reason someone might consider going to a quantum source is to increase stealthiness, which is the ability to detect
without being detected. If the target possesses a _classical_ photodetector, than it cannot detect the quantum lidar.
However, if it possesses a single-photon detector, it might be able to detect the quantum lidar.

By matching the non-vacuum probability, one can compare the different sources when they are all as detectable for an
adversary with a non-number resolving detector. By matching the multi-photon probability, one can compare the different
sources when they are all as detectable for an adversary with a number-resolving detector.

## Quick Usage Preview
### Calculating the SNR when the non-vacuum probability is matched across sources
```
from Sources import SetupParameters, PulsedLaser, EntangledPhotonSPDC, SinglePhoton

param_laser = SetupParameters(
	fock_space_dim=5,
	output_power=2e6,
	trigger_rate=None,
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

snr_laser = laser.signal_to_noise_ratio()
```



## Tutorial

## About

## Thanks

## License

## Citation

