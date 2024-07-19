import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from Analysis import *
from copy import deepcopy

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=40,
	output_power=2e6,
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

collection_efficiency = np.array([0.2, 0.57, 0.8, 1])

multi_photon_probability = np.zeros_like(collection_efficiency)

# *** Single Photon Source ***

param_sps = deepcopy(param)

signal_sps = np.zeros_like(collection_efficiency)
noise_sps = np.zeros_like(collection_efficiency)
trigger_rate_sps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(tqdm(collection_efficiency)):
	param_sps["sp_collection"] = ce
	param_sps["multi_photon_probability"] = param["sp_p2"] * (ce**2)
	sps = SinglePhoton(param_sps)
	signal_sps[idx] = sps.signal_rate()
	noise_sps[idx] = sps.noise_rate()
	trigger_rate_sps[idx] = sps.trigger_rate
	multi_photon_probability[idx] = param_sps["multi_photon_probability"]

# *** Pulsed Laser ***

param_laser = deepcopy(param)
signal_laser = np.zeros_like(collection_efficiency)
noise_laser = np.zeros_like(collection_efficiency)
trigger_rate_laser = np.zeros_like(collection_efficiency)

for idx, mpp in enumerate(tqdm(multi_photon_probability)):
	param_laser["multi_photon_probability"] = mpp
	laser = PulsedLaser(param_laser)
	signal_laser[idx] = laser.signal_rate()
	noise_laser[idx] = laser.noise_rate()
	trigger_rate_laser[idx] = laser.trigger_rate

# *** EPS ***

param_eps = deepcopy(param)
signal_eps = np.zeros_like(collection_efficiency)
noise_eps = np.zeros_like(collection_efficiency)
trigger_rate_eps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(tqdm(collection_efficiency)):
	param_eps["spdc_eps_collection"] = ce
	param_eps["spdc_eps_heralding"] = ce * param_eps["detection_efficiency"]
	param_eps["multi_photon_probability"] = multi_photon_probability[idx]
	eps = EntangledPhotonSPDC(param_eps)
	signal_eps[idx] = eps.signal_rate()
	noise_eps[idx] = eps.noise_rate()
	trigger_rate_eps[idx] = eps.trigger_rate


# *** ROC Curves ***

range_distance = 50

true_positive_sps_02, false_positive_sps_02 = RocAnalysis(
	signal_rate=signal_sps[0],
	noise_rate=noise_sps[0],
	trigger_rate=trigger_rate_sps[0],
	threshold_limit=trigger_rate_sps[0]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_laser_02, false_positive_laser_02 = RocAnalysis(
	signal_rate=signal_laser[0],
	noise_rate=noise_laser[0],
	trigger_rate=trigger_rate_laser[0],
	threshold_limit=trigger_rate_laser[0]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_02, false_positive_eps_02 = RocAnalysis(
	signal_rate=signal_eps[0],
	noise_rate=noise_eps[0],
	trigger_rate=trigger_rate_eps[0],
	threshold_limit=trigger_rate_eps[0]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()


true_positive_sps_057, false_positive_sps_057 = RocAnalysis(
	signal_rate=signal_sps[1],
	noise_rate=noise_sps[1],
	trigger_rate=trigger_rate_sps[1],
	threshold_limit=trigger_rate_sps[1]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_laser_057, false_positive_laser_057 = RocAnalysis(
	signal_rate=signal_laser[1],
	noise_rate=noise_laser[1],
	trigger_rate=trigger_rate_laser[1],
	threshold_limit=trigger_rate_laser[1]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_057, false_positive_eps_057 = RocAnalysis(
	signal_rate=signal_eps[1],
	noise_rate=noise_eps[1],
	trigger_rate=trigger_rate_eps[1],
	threshold_limit=trigger_rate_eps[1]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_sps_08, false_positive_sps_08 = RocAnalysis(
	signal_rate=signal_sps[2],
	noise_rate=noise_sps[2],
	trigger_rate=trigger_rate_sps[2],
	threshold_limit=trigger_rate_sps[2]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_laser_08, false_positive_laser_08 = RocAnalysis(
	signal_rate=signal_laser[2],
	noise_rate=noise_laser[2],
	trigger_rate=trigger_rate_laser[2],
	threshold_limit=trigger_rate_laser[2]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_08, false_positive_eps_08 = RocAnalysis(
	signal_rate=signal_eps[2],
	noise_rate=noise_eps[2],
	trigger_rate=trigger_rate_eps[2],
	threshold_limit=trigger_rate_eps[2]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_sps_1, false_positive_sps_1 = RocAnalysis(
	signal_rate=signal_sps[3],
	noise_rate=noise_sps[3],
	trigger_rate=trigger_rate_sps[3],
	threshold_limit=trigger_rate_sps[3]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_laser_1, false_positive_laser_1 = RocAnalysis(
	signal_rate=signal_laser[3],
	noise_rate=noise_laser[3],
	trigger_rate=trigger_rate_laser[3],
	threshold_limit=trigger_rate_laser[3]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_1, false_positive_eps_1 = RocAnalysis(
	signal_rate=signal_eps[3],
	noise_rate=noise_eps[3],
	trigger_rate=trigger_rate_eps[3],
	threshold_limit=trigger_rate_eps[3]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

# 2x2 subplots

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 0.2

axs[0, 0].plot(false_positive_sps_02, true_positive_sps_02, "--", label="Single Photon Source", color="red", linewidth=2.5)
axs[0, 0].plot(false_positive_laser_02, true_positive_laser_02, "-", label="Pulsed Laser", color="blue", linewidth=2.5)
axs[0, 0].plot(false_positive_eps_02, true_positive_eps_02, "-.", label="Entangled Photon Source", color="green", linewidth=2.5)
str_title = r"$\eta_{signal}$ = 20%, $P_{multi}$= " + f"{multi_photon_probability[0]*100:.3f}%"
axs[0, 0].set_title(str_title, fontsize=18)
axs[0, 0].tick_params(axis='both', which='major', labelsize=22)


# 0.57

axs[0, 1].plot(false_positive_sps_057, true_positive_sps_057, "--", label="Single Photon Source", color="red", linewidth=2.5)
axs[0, 1].plot(false_positive_laser_057, true_positive_laser_057, "-", label="Pulsed Laser", color="blue", linewidth=2.5)
axs[0, 1].plot(false_positive_eps_057, true_positive_eps_057, "-.", label="Entangled Photon Source", color="green", linewidth=2.5)
str_title = r"$\eta_{signal}$ = 57%, $P_{multi}$= " + f"{multi_photon_probability[1]*100:.2f}%"
axs[0, 1].set_title(str_title, fontsize=18)
axs[0, 1].tick_params(axis='both', which='major', labelsize=22)

# 0.8

axs[1, 0].plot(false_positive_sps_08, true_positive_sps_08, "--", label="Single Photon Source", color="red", linewidth=2.5)
axs[1, 0].plot(false_positive_laser_08, true_positive_laser_08, "-", label="Pulsed Laser", color="blue", linewidth=2.5)
axs[1, 0].plot(false_positive_eps_08, true_positive_eps_08, "-.", label="Entangled Photon Source", color="green", linewidth=2.5)
str_title = r"$\eta_{signal}$ = 80%, $P_{multi}$= " + f"{multi_photon_probability[2]*100:.2f}%"
axs[1, 0].set_title(str_title, fontsize=18)
axs[1, 0].tick_params(axis='both', which='major', labelsize=22)

# 1

axs[1, 1].plot(false_positive_sps_1, true_positive_sps_1, "--", label="Single Photon Source", color="red", linewidth=2.5)
axs[1, 1].plot(false_positive_laser_1, true_positive_laser_1, "-", label="Pulsed Laser", color="blue", linewidth=2.5)
axs[1, 1].plot(false_positive_eps_1, true_positive_eps_1, "-.", label="Entangled Photon Source", color="green", linewidth=2.5)
str_title = r"$\eta_{signal}$ = 100%, $P_{multi}$= " + f"{multi_photon_probability[3]*100:.2f}%"
axs[1, 1].set_title(str_title, fontsize=18)
axs[1, 1].tick_params(axis='both', which='major', labelsize=22)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

fig.text(0.5, 0.04, 'False Positive', ha='center', fontsize=24)
fig.text(0.05, 0.5, 'True Positive', va='center', rotation='vertical', fontsize=24)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fontsize=20, ncol=3)

plt.show()



