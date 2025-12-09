import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=50,
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
	background=250000,
	detector_dark=25,
	timing_window=0.5e-9,
)

collection_efficiency = np.array([0.2, 0.57, 0.8, 1])

# *** Single Photon Source ***

param_sps = deepcopy(param)
snr_sps = np.zeros_like(collection_efficiency)
average_photon_per_pulse_sps = np.zeros_like(collection_efficiency)
multi_photon_probability_sps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(collection_efficiency):
	param_sps["sp_collection"] = ce
	param_sps["multi_photon_probability"] = param_sps["sp_p2"] * (ce * param["optics_transmitter"]) ** 2
	sps = SinglePhoton(param_sps)
	snr_sps[idx] = sps.signal_to_noise_rate()
	average_photon_per_pulse_sps[idx] = sps.average_photon_per_pulse
	multi_photon_probability_sps[idx] = sps.multi_photon_probability

multi_photon_probability = np.linspace(0.001, 0.5, 96)
multi_photon_probability = np.concatenate((multi_photon_probability, multi_photon_probability_sps))
multi_photon_probability = np.sort(multi_photon_probability)

# *** LASER ***

param_laser = deepcopy(param)
snr_laser = np.zeros_like(multi_photon_probability)
average_photon_per_pulse_laser = np.zeros_like(multi_photon_probability)

for idx, mpp in enumerate(tqdm(multi_photon_probability)):
	param_laser["multi_photon_probability"] = mpp
	laser = PulsedLaser(param_laser)
	snr_laser[idx] = laser.signal_to_noise_rate()
	average_photon_per_pulse_laser[idx] = laser.average_photon_per_pulse

# *** ENTANGLED PHOTON SOURCE ***

param_eps = deepcopy(param)
snr_eps = np.zeros((multi_photon_probability.shape[0], collection_efficiency.shape[0]))
average_photon_per_pulse_eps = np.zeros((multi_photon_probability.shape[0], collection_efficiency.shape[0]))

last_state = np.zeros((multi_photon_probability.shape[0], collection_efficiency.shape[0]))

for idx_ce, ce in enumerate(collection_efficiency):
	param_eps["spdc_eps_collection"] = ce
	param_eps["spdc_eps_heralding"] = ce * param_eps["detection_efficiency"]
	for idx_mpp, mpp in enumerate(tqdm(multi_photon_probability)):
		param_eps["multi_photon_probability"] = mpp
		eps = EntangledPhotonSPDC(param_eps)
		snr_eps[idx_mpp, idx_ce] = eps.signal_to_noise_rate()
		average_photon_per_pulse_eps[idx_mpp, idx_ce] = eps.epsilon
		last_state[idx_mpp, idx_ce] = eps.prob_of_last_element_fock_space

# INTERSECTION


#*** GRAPHS ***

idx = np.argmin(np.abs(snr_laser - snr_sps[0]))
match_mpp_laser_02 = multi_photon_probability[idx]
match_snr_laser_02 = snr_laser[idx]

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

#intersection between laser and SPS for each collection

idx = np.argmin(np.abs(snr_laser - snr_sps[0]))
match_mpp_laser_02 = multi_photon_probability[idx]
match_snr_laser_02 = snr_laser[idx]

idx = np.argmin(np.abs(snr_laser - snr_sps[1]))
match_mpp_laser_057 = multi_photon_probability[idx]
match_snr_laser_057 = snr_laser[idx]

idx = np.argmin(np.abs(snr_laser - snr_sps[2]))
match_mpp_laser_08 = multi_photon_probability[idx]
match_snr_laser_08 = snr_laser[idx]

idx = np.argmin(np.abs(snr_laser - snr_sps[3]))
match_mpp_laser_1 = multi_photon_probability[idx]
match_snr_laser_1 = snr_laser[idx]

fig = plt.figure(figsize=(16.2, 10))

plt.semilogy(multi_photon_probability, snr_laser, "-", label="Pulsed Laser", linewidth=3, color="blue", zorder=2)
plt.semilogy(multi_photon_probability, snr_eps[:, 0], "-.", label="Entangled Photon Source", linewidth=3, color="green",
             zorder=2)
plt.text(-0.0285, snr_sps[0], "20%", fontsize=20, fontweight="bold")
plt.semilogy(multi_photon_probability, snr_eps[:, 1], "-.", linewidth=3, color="green", zorder=2)
plt.text(-0.0285, snr_sps[1], "57%", fontsize=20, fontweight="bold")
plt.semilogy(multi_photon_probability, snr_eps[:, 2], "-.", linewidth=3, color="green", zorder=2)
plt.text(-0.0285, snr_sps[2], "80%", fontsize=20, fontweight="bold")
plt.semilogy(multi_photon_probability, snr_eps[:, 3], "-.", linewidth=3, color="green", zorder=2)
plt.text(-0.0285, snr_sps[3] + 5, "100%", fontsize=20, fontweight="bold")
plt.text(-0.0285, snr_sps[3] + 25, r"$\eta_{signal}$", fontsize=25, fontweight="bold")
plt.scatter(multi_photon_probability_sps, snr_sps, label="Single Photon Source", linewidths=3, color="red", s=50,
            zorder=3)

plt.hlines(match_snr_laser_02, multi_photon_probability_sps[0], match_mpp_laser_02, "k", linestyle="--", linewidth=3,
           zorder=1, alpha=0.75)
plt.vlines(match_mpp_laser_02, 0, match_snr_laser_02, "k", linestyle="--", linewidth=3, zorder=1, alpha=0.75)
str_title = r"$P_{multi}^{laser}: $" + f"{match_mpp_laser_02:.2f}"
plt.text(match_mpp_laser_02 + 0.005, 2, str_title, fontsize=18)

plt.hlines(match_snr_laser_057, multi_photon_probability_sps[1], match_mpp_laser_057, "k", linestyle="--", linewidth=3,
           zorder=1, alpha=0.75)
plt.vlines(match_mpp_laser_057, 0, match_snr_laser_057, "k", linestyle="--", linewidth=3, zorder=1, alpha=0.75)
str_title = r"$P_{multi}^{laser}: $" + f"{match_mpp_laser_057:.2f}"
plt.text(match_mpp_laser_057 + 0.002, 2, str_title, fontsize=18)

plt.hlines(match_snr_laser_08, multi_photon_probability_sps[2], match_mpp_laser_08, "k", linestyle="--", linewidth=3,
           zorder=1, alpha=0.75)
plt.vlines(match_mpp_laser_08, 0, match_snr_laser_08, "k", linestyle="--", linewidth=3, zorder=1, alpha=0.75)
str_title = r"$P_{multi}^{laser}: $" + f"{match_mpp_laser_08:.2f}"
plt.text(match_mpp_laser_08 + 0.001, 2, str_title, fontsize=18)

plt.hlines(match_snr_laser_1, multi_photon_probability_sps[3], match_mpp_laser_1, "k", linestyle="--", linewidth=3,
           zorder=1, alpha=0.75)
plt.vlines(match_mpp_laser_1, 0, match_snr_laser_1, "k", linestyle="--", linewidth=3, zorder=1, alpha=0.75)
str_title = r"$P_{multi}^{laser}: $" + f"{match_mpp_laser_1:.2f}"
plt.text(match_mpp_laser_1 + 0.004, 2, str_title, fontsize=18)

plt.xlabel("Multi-Photon Probability", fontsize=22)
plt.ylabel("SNR", fontsize=22)
plt.legend(frameon=False, fontsize=22, bbox_to_anchor=(1, 0.5))
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.xlim([-0.035, 0.4])
plt.ylim([1.4, None])
plt.show()
