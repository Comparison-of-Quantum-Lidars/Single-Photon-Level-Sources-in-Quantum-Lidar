import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
from Analysis import RocAnalysis
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=25,
	output_power=2e6,
	trigger_rate=None,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=1,
	sp_p1=0.99,
	sp_p2=1e-3,
	spdc_eps_heralding=1,
	spdc_eps_collection=1,
	atmosphere=1,
	target_distance=1,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=1,
	optics_receiver=1,
	detection_efficiency=1,
	background=400,
	detector_dark=200,
	timing_window=0.5e-9,
)

match_multi_photon_probability = False
range_interval = None
distance = np.linspace(1, 250, 250)
acquisition_time = np.array([1, 60, 3600])
target_false = 0.2
target_true = 0.8
colored_marker = True



def distance_at_target(distance, false_positive_array, target_percent):
	idx = np.argmin(np.abs(false_positive_array - target_percent))
	return distance[idx]

def atmospheric_loss(distance, attenuation=0.02/1000):
	return np.exp(-attenuation*distance)


# -*- MATCH MULTI-PHOTON PROBABILITY -*-
# SETUP
param_laser = deepcopy(param)
param_sps = deepcopy(param)
param_eps = deepcopy(param)

sps = SinglePhoton(param_sps)

if match_multi_photon_probability:
	multi_photon_probability = sps.multi_photon_probability
	param_laser["multi_photon_probability"] = multi_photon_probability
	param_eps["multi_photon_probability"] = multi_photon_probability
else:
	no_vacuum_probability = sps.no_vacuum_probability
	param_laser["no_vacuum_probability"] = no_vacuum_probability
	param_eps["no_vacuum_probability"] = no_vacuum_probability



laser = PulsedLaser(param_laser)
sps = SinglePhoton(param_sps)
eps = EntangledPhotonSPDC(param_eps)

snr_laser = []
snr_sps = []
snr_eps = []

noise_laser_all = []
noise_sps_all = []
noise_eps_all = []

target_false_value_laser = {
	acquisition_time[0]: np.zeros_like(distance),
	acquisition_time[1]: np.zeros_like(distance),
	acquisition_time[2]: np.zeros_like(distance)
}

target_false_value_sps = {
	acquisition_time[0]: np.zeros_like(distance),
	acquisition_time[1]: np.zeros_like(distance),
	acquisition_time[2]: np.zeros_like(distance)
}

target_false_value_eps = {
	acquisition_time[0]: np.zeros_like(distance),
	acquisition_time[1]: np.zeros_like(distance),
	acquisition_time[2]: np.zeros_like(distance)
}

last_element = []
for idx, d in enumerate(tqdm(distance)):
	param_laser["target_distance"] = d
	param_sps["target_distance"] = d
	param_eps["target_distance"] = d

	param_laser["atmosphere"] = atmospheric_loss(d)
	param_sps["atmosphere"] = atmospheric_loss(d)
	param_eps["atmosphere"] = atmospheric_loss(d)

	laser = PulsedLaser(param_laser)
	sps = SinglePhoton(param_sps)
	eps = EntangledPhotonSPDC(param_eps)

	signal_laser = laser.signal_rate()
	noise_laser = laser.noise_rate()
	trigger_rate_laser = laser.trigger_rate
	snr_laser_current = (signal_laser-noise_laser)/noise_laser
	snr_laser.append(snr_laser_current)

	signal_sps = sps.signal_rate()
	noise_sps = sps.noise_rate()
	trigger_rate_sps = sps.trigger_rate
	snr_sps_current = (signal_sps-noise_sps)/noise_sps
	snr_sps.append(snr_sps_current)

	signal_eps = eps.signal_rate()
	noise_eps = eps.noise_rate()
	trigger_rate_eps = eps.trigger_rate
	snr_eps_current = (signal_eps-noise_eps)/noise_eps
	snr_eps.append(snr_eps_current)

	last_element.append(eps.prob_of_last_element_fock_space)
	noise_laser_all.append(noise_laser)
	noise_sps_all.append(noise_sps)
	noise_eps_all.append(noise_eps)

	for at in acquisition_time:
		roc_laser = RocAnalysis(
			signal_rate=signal_laser,
			noise_rate=noise_laser,
			trigger_rate=trigger_rate_laser,
			threshold_limit=trigger_rate_laser/100,
			range_interval=d if range_interval is None else range_interval,
			timing_window=param["timing_window"],
			acquisition_time=at
		)

		roc_sps = RocAnalysis(
			signal_rate=signal_sps,
			noise_rate=noise_sps,
			trigger_rate=trigger_rate_sps,
			threshold_limit=trigger_rate_sps/100,
			range_interval=d if range_interval is None else range_interval,
			timing_window=param["timing_window"],
			acquisition_time=at
		)

		roc_eps = RocAnalysis(
			signal_rate=signal_eps,
			noise_rate=noise_eps,
			trigger_rate=trigger_rate_eps,
			threshold_limit=trigger_rate_eps/100,
			range_interval=d if range_interval is None else range_interval,
			timing_window=param["timing_window"],
			acquisition_time=at
		)

		true_positive_laser, false_positive_laser = roc_laser.compute_p_d_p_fa()
		true_positive_sps, false_positive_sps = roc_sps.compute_p_d_p_fa()
		true_positive_eps, false_positive_eps = roc_eps.compute_p_d_p_fa()

		target_false_value_laser[at][idx] = np.interp(target_false, false_positive_laser, true_positive_laser)
		target_false_value_sps[at][idx] = np.interp(target_false, false_positive_sps, true_positive_sps)
		target_false_value_eps[at][idx] = np.interp(target_false, false_positive_eps, true_positive_eps)


plt.plot(distance, last_element)
plt.show()

plt.plot(distance, noise_laser_all, label="Laser")
plt.plot(distance, noise_sps_all, label="SPS")
plt.plot(distance, noise_eps_all, label="EPS")
plt.legend()
plt.show()

snr_laser = np.array(snr_laser)
snr_sps = np.array(snr_sps)
snr_eps = np.array(snr_eps)

distance_cutoff_laser = {
	acquisition_time[0]: distance_at_target(distance, target_false_value_laser[acquisition_time[0]], target_true),
	acquisition_time[1]: distance_at_target(distance, target_false_value_laser[acquisition_time[1]], target_true),
	acquisition_time[2]: distance_at_target(distance, target_false_value_laser[acquisition_time[2]], target_true)
}
distance_cutoff_laser["snr_at0"] = snr_laser[np.where(distance == distance_cutoff_laser[acquisition_time[0]])]
distance_cutoff_laser["snr_at1"] = snr_laser[np.where(distance == distance_cutoff_laser[acquisition_time[1]])]
distance_cutoff_laser["snr_at2"] = snr_laser[np.where(distance == distance_cutoff_laser[acquisition_time[2]])]


distance_cutoff_sps = {
	acquisition_time[0]: distance_at_target(distance, target_false_value_sps[acquisition_time[0]], target_true),
	acquisition_time[1]: distance_at_target(distance, target_false_value_sps[acquisition_time[1]], target_true),
	acquisition_time[2]: distance_at_target(distance, target_false_value_sps[acquisition_time[2]], target_true)
}
distance_cutoff_sps["snr_at0"] = snr_sps[np.where(distance == distance_cutoff_sps[acquisition_time[0]])]
distance_cutoff_sps["snr_at1"] = snr_sps[np.where(distance == distance_cutoff_sps[acquisition_time[1]])]
distance_cutoff_sps["snr_at2"] = snr_sps[np.where(distance == distance_cutoff_sps[acquisition_time[2]])]


distance_cutoff_eps = {
	acquisition_time[0]: distance_at_target(distance, target_false_value_eps[acquisition_time[0]], target_true),
	acquisition_time[1]: distance_at_target(distance, target_false_value_eps[acquisition_time[1]], target_true),
	acquisition_time[2]: distance_at_target(distance, target_false_value_eps[acquisition_time[2]], target_true)
}
distance_cutoff_eps["snr_at0"] = snr_eps[np.where(distance == distance_cutoff_eps[acquisition_time[0]])]
distance_cutoff_eps["snr_at1"] = snr_eps[np.where(distance == distance_cutoff_eps[acquisition_time[1]])]
distance_cutoff_eps["snr_at2"] = snr_eps[np.where(distance == distance_cutoff_eps[acquisition_time[2]])]







# *** PLOTTING ***

fig = plt.figure(figsize=(16.2, 10))

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.semilogy(distance, snr_laser, "-", label="Pulsed Laser", linewidth=3, color="blue", zorder=1)
plt.semilogy(distance, snr_sps, "--", label="Single Photon", linewidth=3, color="red", zorder=1)
plt.semilogy(distance, snr_eps, "-.", label="Entangled Photon", linewidth=3, color="green", zorder=1)

plt.scatter(0, 0, label=f"{target_true*100}% true detection\n{target_false*100}% false detection:", alpha=0)

plt.scatter(distance_cutoff_laser[acquisition_time[0]], distance_cutoff_laser["snr_at0"], color="k", marker="o", s=100, label="1s", zorder=1)
plt.scatter(distance_cutoff_laser[acquisition_time[1]], distance_cutoff_laser["snr_at1"], color="k", marker="s", s=100, label="1min", zorder=1)
plt.scatter(distance_cutoff_laser[acquisition_time[2]], distance_cutoff_laser["snr_at2"], color="k", marker="^", s=100, label="1h", zorder=1)

str_maker_laser = "blue" if colored_marker else "k"
plt.scatter(distance_cutoff_laser[acquisition_time[0]], distance_cutoff_laser["snr_at0"], color=str_maker_laser, marker="o", s=100, zorder=2)
plt.scatter(distance_cutoff_laser[acquisition_time[1]], distance_cutoff_laser["snr_at1"], color=str_maker_laser, marker="s", s=100, zorder=2)
plt.scatter(distance_cutoff_laser[acquisition_time[2]], distance_cutoff_laser["snr_at2"], color=str_maker_laser, marker="^", s=100, zorder=2)

str_maker_sps = "red" if colored_marker else "k"
plt.scatter(distance_cutoff_sps[acquisition_time[0]], distance_cutoff_sps["snr_at0"], color=str_maker_sps, marker="o", s=100, zorder=2)
plt.scatter(distance_cutoff_sps[acquisition_time[1]], distance_cutoff_sps["snr_at1"], color=str_maker_sps, marker="s", s=100, zorder=2)
plt.scatter(distance_cutoff_sps[acquisition_time[2]], distance_cutoff_sps["snr_at2"], color=str_maker_sps, marker="^", s=100, zorder=2)

str_maker_eps = "green" if colored_marker else "k"
plt.scatter(distance_cutoff_eps[acquisition_time[0]], distance_cutoff_eps["snr_at0"], color=str_maker_eps, marker="o", s=100, zorder=2)
plt.scatter(distance_cutoff_eps[acquisition_time[1]], distance_cutoff_eps["snr_at1"], color=str_maker_eps, marker="s", s=100, zorder=2)
plt.scatter(distance_cutoff_eps[acquisition_time[2]], distance_cutoff_eps["snr_at2"], color=str_maker_eps, marker="^", s=100, zorder=2)

plt.xlabel("Distance [m]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.legend(fontsize=22, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=22)

plt.show()

