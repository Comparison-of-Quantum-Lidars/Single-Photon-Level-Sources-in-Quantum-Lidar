import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from Analysis import *

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=None,
	output_power=400000,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	number_nv_pulse=None,
	number_mp_pulse=None,
	sp_collection=None,
	sp_p1=0.99,
	sp_p2=5e-3,
	spdc_eps_heralding=None,
	spdc_eps_collection=None,
	atmosphere=0.5,
	target_distance=3,
	receiver_diameter=0.05,
	target_albedo=0.5,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.7,
	background=250000,
	detector_dark=25,
	timing_window=0.5e-9,
)

collection_efficiency = [0.2, 0.57, 0.8, 1]
nv_pulse_number_nnrd = 0.5e5

# *** Pulsed Laser ***

param_laser = deepcopy(param)
signal_laser = np.zeros_like(nv_pulse_number_nnrd)
noise_laser = np.zeros_like(nv_pulse_number_nnrd)
trigger_rate_laser = np.zeros_like(nv_pulse_number_nnrd)

print("Pulsed Laser")
for idx, nvp in enumerate(nv_pulse_number):
	param_laser["number_nv_pulse"] = nvp
	laser = PulsedLaser(param_laser)
	signal_laser[idx] = laser.signal_rate()
	noise_laser[idx] = laser.noise_rate()
	trigger_rate_laser[idx] = laser.trigger_rate
	print(laser.compute_number_nv_pulse(), 4e5 / laser.compute_number_nv_pulse(), laser.trigger_rate)
	print(laser.no_vacuum_probability)

# *** Entangled Photon Source ***

param_eps = deepcopy(param)
signal_eps = np.zeros_like(nv_pulse_number)
noise_eps = np.zeros_like(nv_pulse_number)
trigger_rate_eps = np.zeros_like(nv_pulse_number)

print("Entangled Photon Source")
for idx, nvp in enumerate(nv_pulse_number):
	param_eps["spdc_eps_heralding"] = collection_efficiency * param_eps["detection_efficiency"]
	param_eps["spdc_eps_collection"] = collection_efficiency
	param_eps["number_nv_pulse"] = nvp
	eps = EntangledPhotonSPDC(param_eps)
	signal_eps[idx] = eps.signal_rate()
	noise_eps[idx] = eps.noise_rate()
	trigger_rate_eps[idx] = eps.trigger_rate
	print(eps.compute_number_nv_pulse(), 4e5 / eps.compute_number_nv_pulse())



# *** ROC Curves ***

# Maximum distance considered. Increasing the distance results in a higher number of bins which increases the
# probability of a false positive detection.

range_distance = 50

# Collection efficiency: 0.2

true_positive_laser_020, false_positive_laser_020 = RocAnalysis(
	signal_rate=signal_laser[0],
	noise_rate=noise_laser[0],
	trigger_rate=trigger_rate_laser[0],
	threshold_limit=trigger_rate_laser[0] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_020, false_positive_eps_020 = RocAnalysis(
	signal_rate=signal_eps[0],
	noise_rate=noise_eps[0],
	trigger_rate=trigger_rate_eps[0],
	threshold_limit=trigger_rate_eps[0] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

# Collection efficiency: 0.57


true_positive_laser_057, false_positive_laser_057 = RocAnalysis(
	signal_rate=signal_laser[1],
	noise_rate=noise_laser[1],
	trigger_rate=trigger_rate_laser[1],
	threshold_limit=trigger_rate_laser[1] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_057, false_positive_eps_057 = RocAnalysis(
	signal_rate=signal_eps[1],
	noise_rate=noise_eps[1],
	trigger_rate=trigger_rate_eps[1],
	threshold_limit=trigger_rate_eps[1] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

print(f"Signal Rate laser: {signal_laser[1]}, Noise Rate laser: {noise_laser[1]}, Trigger Rate laser: {trigger_rate_laser[1]}")
print(f"Signal Rate eps: {signal_eps[1]}, Noise Rate eps: {noise_eps[1]}, Trigger Rate eps: {trigger_rate_eps[1]}")

# Collection efficiency: 0.8


true_positive_laser_08, false_positive_laser_08 = RocAnalysis(
	signal_rate=signal_laser[2],
	noise_rate=noise_laser[2],
	trigger_rate=trigger_rate_laser[2],
	threshold_limit=trigger_rate_laser[2] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_08, false_positive_eps_08 = RocAnalysis(
	signal_rate=signal_eps[2],
	noise_rate=noise_eps[2],
	trigger_rate=trigger_rate_eps[2],
	threshold_limit=trigger_rate_eps[2] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

# Collection efficiency: 1.0

true_positive_sps_100, false_positive_sps_100 = RocAnalysis(
	signal_rate=signal_sps,
	noise_rate=noise_sps,
	trigger_rate=trigger_rate_sps,
	threshold_limit=trigger_rate_sps / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_laser_100, false_positive_laser_100 = RocAnalysis(
	signal_rate=signal_laser[3],
	noise_rate=noise_laser[3],
	trigger_rate=trigger_rate_laser[3],
	threshold_limit=trigger_rate_laser[3] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

true_positive_eps_100, false_positive_eps_100 = RocAnalysis(
	signal_rate=signal_eps[3],
	noise_rate=noise_eps[3],
	trigger_rate=trigger_rate_eps[3],
	threshold_limit=trigger_rate_eps[3] / 1000,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

# *** PLOT ***

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(false_positive_laser_020, true_positive_laser_020, "-", label="Pulsed Laser", color="blue", linewidth=2.5,
            zorder=1)
axs[0, 0].plot(false_positive_eps_020, true_positive_eps_020, "-.", label="Entangled Photon Source", color="green",
            linewidth=2.5, zorder=3)
str_title = r"$N_{nv} = $" + f"{nv_pulse_number[0]:.2f}"
axs[0, 0].set_title(str_title, fontsize=21)
axs[0, 0].tick_params(axis='both', which='major', labelsize=22)

axs[0, 1].plot(false_positive_laser_057, true_positive_laser_057, "-", color="blue", linewidth=2.5, zorder=1)
axs[0, 1].plot(false_positive_eps_057, true_positive_eps_057, "-.", color="green", linewidth=2.5, zorder=3)
str_title = r"$N_{nv} = $" + f"{nv_pulse_number[1]:.2f}"
axs[0, 1].set_title(str_title, fontsize=21)
axs[0, 1].tick_params(axis='both', which='major', labelsize=22)

axs[1, 0].plot(false_positive_laser_08, true_positive_laser_08, "-", color="blue", linewidth=2.5, zorder=1)
axs[1, 0].plot(false_positive_eps_08, true_positive_eps_08, "-.", color="green", linewidth=2.5, zorder=3)
str_title = r"$N_{nv} = $" + f"{nv_pulse_number[2]:.2f}"
axs[1, 0].set_title(str_title, fontsize=22)
axs[1, 0].tick_params(axis='both', which='major', labelsize=22)

axs[1, 1].plot(false_positive_sps_100, true_positive_sps_100, "--", label="Single Photon Source", color="red", linewidth=2.5, zorder=2)
axs[1, 1].plot(false_positive_laser_100, true_positive_laser_100, "-", color="blue", linewidth=2.5, zorder=1)
axs[1, 1].plot(false_positive_eps_100, true_positive_eps_100, "-.", color="green", linewidth=2.5, zorder=3)
str_title = r"$N_{nv} = $" + f"{nv_pulse_number[3]:.2f}"
axs[1, 1].set_title(str_title, fontsize=22)
axs[1, 1].tick_params(axis='both', which='major', labelsize=22)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

fig.text(0.5, 0.04, 'False Positive', ha='center', fontsize=25)
fig.text(0.04, 0.5, 'True Positive', va='center', rotation='vertical', fontsize=25)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fontsize=20, ncol=3)

plt.show()
