import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from Analysis import *



# *** SETUP ***

param = SetupParameters(
	fock_space_dim=45,
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
	target_distance=2,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.5,
	background=400,
	detector_dark=200,
	timing_window=0.5e-9,
)

collection_efficiency = np.array([0.2, 0.57, 0.8])

non_vacuum_probability = np.zeros_like(collection_efficiency)

# *** Single Photon Source ***

param_sps = deepcopy(param)
signal_sps = np.zeros_like(collection_efficiency)
noise_sps = np.zeros_like(collection_efficiency)
trigger_rate_sps = np.zeros_like(collection_efficiency)
average_photon_per_pulse_sps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(tqdm(collection_efficiency)):
	param_sps["sp_collection"] = ce
	param_sps["no_vacuum_probability"] = param_sps["sp_p1"] * ce + (param_sps["sp_p2"] * ce * (2 - ce))
	sps = SinglePhoton(param_sps)
	non_vacuum_probability[idx] = sps.no_vacuum_probability
	signal_sps[idx] = sps.signal_rate()
	noise_sps[idx] = sps.noise_rate()
	trigger_rate_sps[idx] = sps.trigger_rate
	average_photon_per_pulse_sps[idx] = sps.average_photon_per_pulse

# *** Pulsed Laser ***

param_laser = deepcopy(param)
signal_laser = np.zeros_like(collection_efficiency)
noise_laser = np.zeros_like(collection_efficiency)
trigger_rate_laser = np.zeros_like(collection_efficiency)

for idx, nvp in enumerate(tqdm(non_vacuum_probability)):
	param_laser["no_vacuum_probability"] = nvp
	laser = PulsedLaser(param_laser)
	signal_laser[idx] = laser.signal_rate()
	noise_laser[idx] = laser.noise_rate()
	trigger_rate_laser[idx] = laser.trigger_rate

# *** Entangled Photon Source ***

param_eps = deepcopy(param)
signal_eps = np.zeros_like(collection_efficiency)
noise_eps = np.zeros_like(collection_efficiency)
trigger_rate_eps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(tqdm(collection_efficiency)):
	param_eps["spdc_eps_heralding"] = ce * param_eps["detection_efficiency"]
	param_eps["spdc_eps_collection"] = ce
	param_eps["no_vacuum_probability"] = non_vacuum_probability[idx]
	eps = EntangledPhotonSPDC(param_eps)
	signal_eps[idx] = eps.signal_rate()
	noise_eps[idx] = eps.noise_rate()
	trigger_rate_eps[idx] = eps.trigger_rate

# *** ROC Curves ***

range_distance=50

# 0.2

true_positive_sps_020, false_positive_sps_020 = RocAnalysis(
	signal_rate=signal_sps[0],
	noise_rate=noise_sps[0],
	trigger_rate=trigger_rate_sps[0],
	threshold_limit=trigger_rate_sps[0]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_laser_020, false_positive_laser_020 = RocAnalysis(
	signal_rate=signal_laser[0],
	noise_rate=noise_laser[0],
	trigger_rate=trigger_rate_laser[0],
	threshold_limit=trigger_rate_laser[0]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_020, false_positive_eps_020 = RocAnalysis(
	signal_rate=signal_eps[0],
	noise_rate=noise_eps[0],
	trigger_rate=trigger_rate_eps[0],
	threshold_limit=trigger_rate_eps[0]/1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

# 0.57

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

# 0.8

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

# *** PLOT ***

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig, axs = plt.subplots(1, 3, figsize=(10, 10))

axs[0].plot(false_positive_sps_020, true_positive_sps_020, "--", color="red", linewidth=2.5, zorder=2)
axs[0].plot(false_positive_laser_020, true_positive_laser_020, "-", label="Pulsed Laser", color="blue", linewidth=2.5, zorder=1)
axs[0].plot(false_positive_eps_020, true_positive_eps_020, "-.", label="Entangled Photon Source", color="green", linewidth=2.5, zorder=3)
str_title = r"$\eta_{signal} = 0.2$, $P_{nv} = $" + f"{non_vacuum_probability[0] * 100:.2f}%"
axs[0].set_title(str_title, fontsize=21)
axs[0].tick_params(axis='both', which='major', labelsize=22)

axs[1].plot(false_positive_sps_057, true_positive_sps_057, "--", color="red", linewidth=2.5, zorder=2)
axs[1].plot(false_positive_laser_057, true_positive_laser_057, "-", color="blue", linewidth=2.5, zorder=1)
axs[1].plot(false_positive_eps_057, true_positive_eps_057, "-.", color="green", linewidth=2.5, zorder=3)
str_title = r"$\eta_{signal} = 0.57$, $P_{nv} = $" + f"{non_vacuum_probability[1] * 100:.2f}%"
axs[1].set_title(str_title, fontsize=21)
axs[1].tick_params(axis='both', which='major', labelsize=22)

axs[2].plot(false_positive_sps_08, true_positive_sps_08, "--", color="red", linewidth=2.5, zorder=2)
axs[2].plot(false_positive_laser_08, true_positive_laser_08, "-", color="blue", linewidth=2.5, zorder=1)
axs[2].plot(false_positive_eps_08, true_positive_eps_08, "-.", color="green", linewidth=2.5, zorder=3)
str_title = r"$\eta_{signal} = 0.8$, $P_{nv} = $" + f"{non_vacuum_probability[2] * 100:.2f}%"
axs[2].set_title(str_title, fontsize=22)
axs[2].tick_params(axis='both', which='major', labelsize=22)

fig.text(0.5, 0.04, 'False Positive [-]', ha='center', fontsize=25)
fig.text(0.04, 0.5, 'True Positive [-]', va='center', rotation='vertical', fontsize=25)

plt.show()









