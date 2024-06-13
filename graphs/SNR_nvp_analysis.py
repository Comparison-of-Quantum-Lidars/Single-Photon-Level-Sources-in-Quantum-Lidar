import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

param = SetupParameters(
	fock_space_dim=25,
	output_power=2e6,
	trigger_rate=None,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=None,
	sp_p1=0.99,
	sp_p2=1e-3,
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

### Pulsed Laser ###

param_laser = deepcopy(param)

no_vacuum_probability = np.linspace(0.001, 0.99, 60)
#no_vacuum_probability = np.linspace(0.001, 0.49, 60)

snr_laser = []
mpp_laser = []
no_vacuum_probability_laser = no_vacuum_probability
trigger_rate_laser = []

for nvp in tqdm(no_vacuum_probability):
	param_laser.no_vacuum_probability = nvp
	snr = PulsedLaser(param_laser).signal_to_noise_rate()
	snr_laser.append(snr)
	mpp_laser.append(PulsedLaser(param_laser).multi_photon_probability)
	trigger_rate_laser.append(PulsedLaser(param_laser).trigger_rate)

### Single Photon Source ###

param_sps = deepcopy(param)

collection_efficiency = np.linspace(0.01, 1, 60)

snr_sps = []
no_vacuum_probability_sps = []
mpp_sps = []

for ce in tqdm(collection_efficiency):
	param_sps.sp_collection = ce
	sps = SinglePhoton(param_sps)
	assert sps.no_vacuum_probability == param_sps.sp_p1 * sps.extr_efficiency + param_sps.sp_p2* sps.extr_efficiency*(2-sps.extr_efficiency)
	snr = sps.signal_to_noise_rate()
	snr_sps.append(snr)
	no_vacuum_probability_sps.append(sps.no_vacuum_probability)
	mpp_sps.append(sps.multi_photon_probability)



### Entangled Photon Source ###

param_eps = deepcopy(param)

snr_eps = []


no_vacuum_probability_eps = []
mpp_eps = []

trigger_rate_eps = []

for ce in tqdm(collection_efficiency):
	param_eps.spdc_eps_collection = ce
	param_eps.spdc_eps_heralding = ce * param["detection_efficiency"]
	extr_efficiency = ce * param["atmosphere"]
	param_eps.no_vacuum_probability = param_sps["sp_p1"] * extr_efficiency + param_sps["sp_p2"] * extr_efficiency*(2-extr_efficiency)
	eps = EntangledPhotonSPDC(param_eps)
	snr = eps.signal_to_noise_rate()
	snr_eps.append(snr)
	no_vacuum_probability_eps.append(eps.no_vacuum_probability)
	mpp_eps.append(eps.multi_photon_probability)
	trigger_rate_eps.append(eps.trigger_rate)


### GRAPHS ###


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig = plt.figure(figsize=(16.1, 10))

plt.plot(no_vacuum_probability_laser, trigger_rate_laser, "-", color="blue", label="Pulsed Laser", linewidth=3)
plt.plot(no_vacuum_probability_eps, trigger_rate_eps, "-.", color="green", label="Entangled Photon Source", linewidth=3)
plt.legend(frameon=False, fontsize=22)
plt.xlabel("No Vacuum Probability [-]", fontsize=22)
plt.ylabel("Trigger Rate [Hz]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.show()

fig = plt.figure(figsize=(16.1, 10))

plt.plot(no_vacuum_probability_laser, mpp_laser, "-", color="blue", label="Pulsed Laser", linewidth=3)
plt.plot(no_vacuum_probability_sps, mpp_sps, "--", color="red", label="Single Photon Source", linewidth=3)
plt.plot(no_vacuum_probability_eps, mpp_eps, "-.", color="green", label="Entangled Photon Source", linewidth=3)
plt.legend(frameon=False, fontsize=22)
plt.xlabel("No Vacuum Probability [-]", fontsize=22)
plt.ylabel("Multi-Photon Probability [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.show()

fig = plt.figure(figsize=(16.1, 10))

plt.plot(no_vacuum_probability_laser, snr_laser, "-", color="blue", label="Pulsed Laser", linewidth=3)
plt.plot(no_vacuum_probability_sps, snr_sps, "--", color="red", label="Single Photon Source", linewidth=3)
plt.plot(no_vacuum_probability_eps, snr_eps, "-.", color="green", label="Entangled Photon Source", linewidth=3)
plt.legend(frameon=False, fontsize=22)
plt.xlabel("No Vacuum Probability [-]", fontsize=22)
plt.ylabel("Signal-to-Noise Ratio [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.xlim([0, 0.5])
plt.ylim([0, 75])
plt.show()


### SWEEP ONLY NON-VACUUM PROBABILITY OF THE LASER AND EPS WITHOUT AFFECTING COLLECTION EFFICIENCY ###

param = SetupParameters(
	fock_space_dim=25,
	output_power=2e6,
	trigger_rate=None,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=None,
	sp_p1=None,
	sp_p2=None,
	spdc_eps_heralding=0.05,
	spdc_eps_collection=0.05,
	atmosphere=1,
	target_distance=1,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=1,
	background=400,
	detector_dark=200,
	timing_window=0.5e-9,
)

non_vacuum_prob = np.linspace(0.001, 0.99, 60)

snr_laser = []
mpp_laser = []
average_photon_laser = []
snr_eps = []
mpp_eps = []
average_photon_eps = []

for nvp in tqdm(non_vacuum_prob):
	param["no_vacuum_probability"] = nvp
	laser = PulsedLaser(param)
	eps = EntangledPhotonSPDC(param)
	snr_laser.append(laser.signal_to_noise_rate())
	mpp_laser.append(laser.multi_photon_probability)
	average_photon_laser.append(laser.compute_alpha()**2)
	snr_eps.append(eps.signal_to_noise_rate())
	mpp_eps.append(eps.multi_photon_probability)
	average_photon_eps.append(eps.epsilon)


plt.plot(non_vacuum_prob, snr_laser, "-", color="blue", label="Pulsed Laser", linewidth=3)
plt.plot(non_vacuum_prob, snr_eps, "-.", color="green", label="Entangled Photon Source", linewidth=3)
plt.legend(frameon=False, fontsize=22)
plt.xlabel("No Vacuum Probability [-]", fontsize=22)
plt.ylabel("Signal-to-Noise Ratio [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.show()

plt.plot(non_vacuum_prob, mpp_laser, "-", color="blue", label="Pulsed Laser", linewidth=3)
plt.plot(non_vacuum_prob, mpp_eps, "-.", color="green", label="Entangled Photon Source", linewidth=3)
plt.legend(frameon=False, fontsize=22)
plt.xlabel("No Vacuum Probability [-]", fontsize=22)
plt.ylabel("Multi-Photon Probability [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.show()

plt.plot(average_photon_laser, non_vacuum_prob, "-", color="blue", label="Pulsed Laser", linewidth=3)
plt.plot(average_photon_eps,non_vacuum_prob, "-.", color="green", label="Entangled Photon Source", linewidth=3)
plt.legend(frameon=False, fontsize=22)
plt.ylabel("No Vacuum Probability [-]", fontsize=22)
plt.xlabel("Average photon per pulse [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.show()