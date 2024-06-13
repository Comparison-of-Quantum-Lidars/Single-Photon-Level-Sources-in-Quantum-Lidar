import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=35,
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

collection_efficiency = np.array([0.4, 0.57, 0.8, 1])

# *** Single Photon Source ***

param_sps = deepcopy(param)
snr_sps = np.zeros_like(collection_efficiency)
average_photon_per_pulse_sps = np.zeros_like(collection_efficiency)
multi_photon_probability_sps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(collection_efficiency):
	param_sps["sp_collection"] = ce
	param_sps["multi_photon_probability"] = param_sps["sp_p2"] * ce**2
	sps = SinglePhoton(param_sps)
	snr_sps[idx] = sps.signal_to_noise_rate()
	average_photon_per_pulse_sps[idx] = sps.average_photon_per_pulse
	multi_photon_probability_sps[idx] = sps.multi_photon_probability

multi_photon_probability = np.linspace(0.001, 0.5, 30)
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

for idx_ce, ce in enumerate(collection_efficiency):
	param_eps["spdc_eps_collection"] = ce
	param_eps["spdc_eps_heralding"] = ce * param_eps["detection_efficiency"]
	for idx_mpp, mpp in enumerate(tqdm(multi_photon_probability)):
		param_eps["multi_photon_probability"] = mpp
		eps = EntangledPhotonSPDC(param_eps)
		snr_eps[idx_mpp, idx_ce] = eps.signal_to_noise_rate()
		average_photon_per_pulse_eps[idx_mpp, idx_ce] = eps.signal_to_noise_rate()

plt.plot(multi_photon_probability, average_photon_per_pulse_eps[:,0], label="1")
plt.plot(multi_photon_probability, average_photon_per_pulse_eps[:,-1], label="2")
plt.legend()
plt.show()

# *** GRAPHS ***

#np.savez("match_mpp", multi_photon_probability=multi_photon_probability,
#       multi_photon_probability_sps=multi_photon_probability_sps,
#       snr_laser=snr_laser, snr_sps=snr_sps, snr_eps=snr_eps)

#print(np.load("match_mpp.npz")["multi_photon_probability"])

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")


plt.plot(multi_photon_probability, snr_laser, "-", label="Pulsed Laser", linewidth=3, color="blue")
plt.scatter(multi_photon_probability_sps, snr_sps, label="Single Photon Source", linewidths=3, color="red")
plt.plot(multi_photon_probability, snr_eps[:, 0], "-.", label="Entangled Photon Source", linewidth=3, color="green")
plt.plot(multi_photon_probability, snr_eps[:, 1], "-.", label="0.57", linewidth=3, color="green")
plt.plot(multi_photon_probability, snr_eps[:, 2], "-.", label="0.8", linewidth=3, color="green")
plt.plot(multi_photon_probability, snr_eps[:, 3], "-.", label="1", linewidth=3, color="green")
plt.xlabel("Non-Vacuum Probability [-]", fontsize=22)
plt.ylabel("Log(SNR) [-]", fontsize=22)
plt.legend(frameon=False, fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.show()

