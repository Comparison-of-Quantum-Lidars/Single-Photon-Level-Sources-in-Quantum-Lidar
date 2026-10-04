# Comparison and Limitations of Single-Photon Level Sources in Quantum Lidar

## Description

This module is a simple Python package used to compute the performance of different single-photon-level sources for quantum lidar applications. The sources studied in this work are:
1) An array of single-photon sources based on quantum dots
2) Correlated photon pair source based on spontaneous parametric down-conversion (SPDC)
3) Attenuated pulsed laser

The goal of this work is to provide a fair comparison of the performance of lidar systems for stealth applications. The lidar systems modeled in this work use a source and a detector operating in the single-photon regime.
To ensure a fair comparison between the different sources, time correlation is employed for all. For the correlated photon pair source, the detection of an idler photon is used to trigger the start of the time-correlation window, and coincidence detection is performed. For the single-photon sources and the attenuated pulsed laser, the triggering electrical signal is used for coincidence detection.

Two parameters are of interest to ensure a fair comparison for each source: the optical power leaving the lidar system ($P_{out}$) in photons per second and how the optical power is distributed. While the former is straightforward to apply (i.e. all source must be equalized in term of optical power), the latter is remarkably subtle to determine and requires careful consideration. Indeed, once the optical power is fixed, one may instead choose to equalize the source triggering rate, the mean photon number per pulse, the probability of emitting at least one photon per pulse, or other quantities. Consequently, there are many ways to distribute the same optical power, each leading to different performance because the sources exhibit fundamentally different photon-number statistics. To address this, we revert back to the main application of quantum lidar with sources at the single-photon level: covert ranging. Therefore, in addition to equalizing the optical power, we equalize the average photon number per non-vacuum pulse ($\mathcal{N}_{nv}$) for each source. This quantity corresponds to the mean photon number conditioned on a non-vacuum emission and can be interpreted as the mean photon number measured by an adversary equipped with an ideal detector. This choice anchors the comparison in the primary application of quantum lidar. Under this criterion, all sources present the adversary with the same optical power and the same mean photon number per detected pulse, thereby equalizing their detectability. In this package, we also offer the option to equalize the non-vacuum probability (i.e., the probability of emitting at least one photon per pulse), the multi-photon probability (i.e., the probability of emitting two or more photons per pulse), and the source triggering rate.

The details are presented in the paper [reference to be added soon]. We recommend that interested readers read the paper before diving into the code and tutorials.

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

## Manuscript

This work is part of the manuscript *Comparison and limitations of quantum lidar in the single-photon regime*, published in *Optica Quantum*, which is accessible [here](https://doi.org/10.1364/OPTICAQ.579366). We encourage readers to familiarize themselves with the content of the manuscript before using this package.

While we have made every effort to make the package as easy to use as possible, we do not claim that it follows all best practices for code formatting and presentation. However, we believe that science benefits from open and accessible research, which is why we are making our code publicly available.

Every figure from the manuscript can be reproduced using the provided scripts, with the exception of Fig. 1 and Fig. 2, which are schematics. The scripts used to reproduce the figures are located in the [Figures](Figures) folder.

### Figures 3, 4, and 5

To reproduce Figures 3, 4, and 5, the user must first run [Figure 3_4_5 data generator.py](Figures/Figure%203_4_5%20data%20generator.py). This script generates a `.npy` file containing the raw data used for the figures, including the SNR and triggering rate for different collection efficiencies and numbers of single-photon sources (SPS).

Generating this file may take some time, as it requires a large number of calculations. Once the `.npy` file has been generated, [Figure 3 - SNR vs number non vacuum pulse.py](Figures/Figure%203%20-%20SNR%20vs%20number%20non%20vacuum%20pulse.py) can be run to reproduce Figure 3.

Figures 4 and 5 can be reproduced using [Figure 4 & 5 - ROC curves.py](Figures/Figure%204%20%26%205%20-%20ROC%20curves.py). Before running the script, the user must specify the number of SPS using the `number_sps` variable:

- `number_sps = 10` reproduces Figure 4.
- `number_sps = 1` reproduces Figure 5.

### Figures 6 and 7

Figures 6 and 7 can be reproduced using [Figure 6 & 7 Range limitation.py](Figures/Figure%206%20%26%207%20Range%20limitation.py).

Before running the script, the user must set `number_sps_array` to the desired number of SPS:

- `number_sps_array = 10` reproduces Figure 6.
- `number_sps_array = 1` reproduces Figure 7.

On the first run, the `.compute()` command at line 60 can be uncommented to generate the required `.npy` file. Generating this file may take some time. Once the `.npy` file has been generated, we recommend commenting out `.compute()` again and loading the saved dataset instead. This avoids repeating the computationally intensive calculation each time the script is run.

### Supplementary Figures

A similar procedure is required for [Figure S2](Figures/Figure%20S2%20-%20Distance%20vs%20apperture.py) and [Figure S3](Figures/Figure%20S3%20-%20Distance%20vs%20acquisition%20time.py). The first time the scripts are run, the required `.npy` files must be generated using `.compute()`. Once the data have been generated, the calculation can be commented out and the saved datasets can be loaded for faster execution.

[Figure S1](Figures/Figure%20S1%20-%20ROC%20curves%20%26%20Histogram%20simulation.py) can be run directly without first generating a dataset.

## Citation

[...]

## Authors

This work was carried out by Anthony Drouin (IQC/UWaterloo), Dr. Jean-Philippe Bourgoin (Single Quantum System), Dr. Sara Hosseini (IQC/UWaterloo/NRC), Prof. François Sfigakis (IQC/UWaterloo), Prof. Jonathan Baugh (IQC/UWaterloo), and Prof. Michael E. Reimer (IQC/UWaterloo/Single Quantum System). Anthony Drouin is the corresponding author for this work (anthony.drouin@uwaterloo.ca).

## License

[Apache License 2.0](LICENSE)
