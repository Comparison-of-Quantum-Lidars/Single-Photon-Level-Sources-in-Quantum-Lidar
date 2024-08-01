import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=35,
	output_power=400000,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=None,
	sp_p1=0.99,
	sp_p2=5e-3,
	spdc_eps_heralding=None,
	spdc_eps_collection=None,
	atmosphere=0.5,
	target_distance=2,
	receiver_diameter=0.05,
	target_albedo=0.5,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.7,
	background=25,
	detector_dark=25,
	timing_window=0.5e-9,
)

collection_efficiency = np.array([0.2, 0.57, 0.8, 1])
# *** Single Photon Source ***

param_sps = deepcopy(param)
non_vacuum_probability_sps = np.zeros_like(collection_efficiency)
snr_sps = np.zeros_like(collection_efficiency)
average_photon_per_pulse_sps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(collection_efficiency):
	param_sps["sp_collection"] = ce
	extr_eff = ce * param["optics_transmitter"]
	param_sps["no_vacuum_probability"] = param_sps["sp_p1"] * extr_eff + (
				param_sps["sp_p2"] * extr_eff * (2 - extr_eff))
	sps = SinglePhoton(param_sps)
	non_vacuum_probability_sps[idx] = sps.no_vacuum_probability
	snr_sps[idx] = sps.signal_to_noise_rate()
	average_photon_per_pulse_sps[idx] = sps.average_photon_per_pulse

non_vacuum_probability = np.linspace(0.0001, 0.8, 96)
non_vacuum_probability = np.concatenate((non_vacuum_probability, non_vacuum_probability_sps))
non_vacuum_probability = np.sort(non_vacuum_probability)

# *** Pulsed Laser ***

param_laser = deepcopy(param)
snr_laser = np.zeros_like(non_vacuum_probability)
average_photon_per_pulse_laser = np.zeros_like(non_vacuum_probability)

for idx, nvp in enumerate(tqdm(non_vacuum_probability)):
	param_laser["no_vacuum_probability"] = nvp
	laser = PulsedLaser(param_laser)
	snr_laser[idx] = laser.signal_to_noise_rate()
	average_photon_per_pulse_laser[idx] = laser.average_photon_per_pulse

# *** Entangled Photon Source ***

param_eps = deepcopy(param)
snr_entangled = np.zeros((non_vacuum_probability.shape[0], collection_efficiency.shape[0]))
average_photon_per_pulse_entangled = np.zeros((non_vacuum_probability.shape[0], collection_efficiency.shape[0]))

last_state = np.zeros((non_vacuum_probability.shape[0], collection_efficiency.shape[0]))


def adjustable_fock_space(average_photon_per_pulse):
	return round(average_photon_per_pulse) + 48


for idx_nvp, ce in enumerate(collection_efficiency):
	param_eps["spdc_eps_collection"] = ce
	param_eps["spdc_eps_heralding"] = ce * param_eps["detection_efficiency"]

	for idx_ce, nvp in enumerate(tqdm(non_vacuum_probability)):
		param_eps["no_vacuum_probability"] = nvp
		eps = EntangledPhotonSPDC(param_eps)
		average_photon_per_pulse_entangled[idx_ce, idx_nvp] = eps.average_photon_per_pulse
		param_eps["fock_space_dim"] = adjustable_fock_space(average_photon_per_pulse_entangled[idx_ce, idx_nvp])
		snr_entangled[idx_ce, idx_nvp] = eps.signal_to_noise_rate()
		last_state[idx_ce, idx_nvp] = eps.prob_of_last_element_fock_space

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig = plt.figure(figsize=(16.2, 10))

plt.semilogy(non_vacuum_probability, snr_entangled[:, 0], "-.", label="Entangled Photon Source", color="green",
             linewidth=3, zorder=1)
plt.semilogy(non_vacuum_probability, snr_entangled[:, 1], "-.", color="green", linewidth=3, zorder=1)
plt.semilogy(non_vacuum_probability, snr_entangled[:, 2], "-.", color="green", linewidth=3, zorder=1)
plt.semilogy(non_vacuum_probability, snr_entangled[:, 3], "-.", color="green", linewidth=3, zorder=1)
plt.semilogy(non_vacuum_probability, snr_laser, "-", label="Pulsed Laser", color="blue", linewidth=3, zorder=1)
plt.scatter(non_vacuum_probability_sps, snr_sps, label="Single Photon Source", color="red", s=100, zorder=2)
plt.text(non_vacuum_probability_sps[0] - 0.01, snr_sps[0] - 3.5, "20%", fontsize=22, fontweight="bold")
plt.text(non_vacuum_probability_sps[1] - 0.01, snr_sps[1] - 8, "57%", fontsize=22, fontweight="bold")
plt.text(non_vacuum_probability_sps[2] - 0.04, snr_sps[2] - 11, "80%", fontsize=22, fontweight="bold")
plt.text(non_vacuum_probability_sps[3] - 0.053, snr_sps[3] - 15, "100%", fontsize=22, fontweight="bold")
plt.text(-0.055, snr_entangled[0, 0], "20%", fontsize=22, fontweight="bold")
plt.text(-0.055, snr_entangled[0, 1] - 5, "57%", fontsize=22, fontweight="bold")
plt.text(-0.055, snr_entangled[0, 2] - 5, "80%", fontsize=22, fontweight="bold")
plt.text(-0.055, snr_entangled[0, 3] + 2, "100%", fontsize=22, fontweight="bold")
plt.text(-0.055, snr_entangled[0, 3] + 30, r"$\eta_{signal}$", fontsize=28, fontweight="bold")
plt.xlim([-0.065, 0.81])
plt.ylim([0.1, None])
plt.xlabel("Non-Vacuum Probability", fontsize=22)
plt.ylabel("SNR", fontsize=22)
plt.legend(frameon=False, fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.show()
