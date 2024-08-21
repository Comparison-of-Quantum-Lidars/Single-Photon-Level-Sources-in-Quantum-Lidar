import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=20,
	output_power=2e6,
	trigger_rate=None,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=0.57,
	sp_p1=0.99,
	sp_p2=1e-3,
	spdc_eps_heralding=0.57*0.5,
	spdc_eps_collection=0.57,
	atmosphere=1,
	target_distance=1,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.5,
	background=203719,
	detector_dark=200,
	timing_window=0.5e-9,
)

distance = np.linspace(5, 500, 500)

# -*- MATCH MULTI-PHOTON PROBABILITY -*-
param_laser = deepcopy(param)
param_sps = deepcopy(param)
param_eps = deepcopy(param)

sps = SinglePhoton(param_sps)
# no_vacuum_probability = sps.no_vacuum_probability
# param_laser["no_vacuum_probability"] = no_vacuum_probability
# param_eps["no_vacuum_probability"] = no_vacuum_probability

multi_photon_probability = sps.multi_photon_probability
param_laser["multi_photon_probability"] = multi_photon_probability
param_eps["multi_photon_probability"] = multi_photon_probability

snr_laser = np.zeros_like(distance)
snr_sps = np.zeros_like(distance)
snr_eps = np.zeros_like(distance)

for idx, d in enumerate(tqdm(distance)):
	param_laser["target_distance"] = d
	param_sps["target_distance"] = d
	param_eps["target_distance"] = d

	laser = PulsedLaser(param_laser)
	sps = SinglePhoton(param_sps)
	eps = EntangledPhotonSPDC(param_eps)

	snr_laser[idx] = laser.signal_to_noise_rate()
	snr_sps[idx] = sps.signal_to_noise_rate()
	snr_eps[idx] = eps.signal_to_noise_rate()

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.plot(distance, snr_laser, "-", label="Laser", linewidth=2.5, color="blue")
plt.plot(distance, snr_sps, "--", label="Single Photon Source", linewidth=2.5, color="red")
plt.plot(distance, snr_eps, "-.", label="Entangled Photon Source", linewidth=2.5, color="green")
plt.xlabel("Distance [m]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.legend(fontsize=22, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.show()