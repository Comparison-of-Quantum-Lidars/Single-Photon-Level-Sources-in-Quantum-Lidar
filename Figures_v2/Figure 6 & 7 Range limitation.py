import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
from Analysis import RangeLimitation, RocAnalysis
import matplotlib.pyplot as plt
from copy import deepcopy

param = SetupParameters(
	fock_space_dim=9,
	output_power=400000,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	number_nv_pulse=None,
	sp_collection=0.8,
	sp_p1=0.99,
	sp_p2=5e-3,
	spdc_eps_heralding=0.8*0.7,
	spdc_eps_collection=0.8,
	atmosphere=0.5,
	target_distance=None,
	receiver_diameter=0.05,
	target_albedo=0.5,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.7,
	background=250000,
	detector_dark=25,
	timing_window=0.5e-9,
)

number_sps_array = 1
sps = SinglePhoton(param, number_sps=number_sps_array)
number_nv_pulse_sps = sps.number_nv_pulse
param["number_nv_pulse"] = number_nv_pulse_sps

range_interval = None
distance = np.linspace(0.5, 50, 2000)
acquisition_time = np.array([1, 60, 3600])
target_false_positive = 0.2
target_true_positive = 0.8
colored_marker = True
precision = 25
threshold_limit_factor = 100

results = RangeLimitation(
	params=param,
	parameter_to_match="number_nv_pulse",
	range_interval=None,
	distance=distance,
	acquisition_time=acquisition_time,
	target_false_positive=target_false_positive,
	target_true_positive=target_true_positive,
	precision_roc=precision,
	threshold_limit_factor_roc=threshold_limit_factor,
	number_nv_pulse_for_match=number_nv_pulse_sps,
	number_sps_array=number_sps_array
)

results = np.load("Range_limitation_NRD_ce_80_2025_07_05_10sps_35_fock_dim.npy", allow_pickle=True).item()

#np.save("Range_limitation_NRD_ce_80_2025_07_05_1sps_4_fock_dim.npy", results)


distance = results["distance"]
snr_laser = results["snr_laser"]
snr_sps = results["snr_sps"]
snr_eps = results["snr_eps"]
distance_cutoff_laser = results["distance_cutoff_laser"]
distance_cutoff_sps = results["distance_cutoff_sps"]
distance_cutoff_eps = results["distance_cutoff_eps"]

# *** PLOTTING ***

fig = plt.figure(figsize=(16.2, 10))

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.semilogy(distance, snr_laser, "-.", label="Pulsed Laser", linewidth=3, color="blue", zorder=2)
plt.semilogy(distance, snr_sps, "--", label="Single Photon", linewidth=3, color="red", zorder=3)
plt.semilogy(distance, snr_eps, "-", label="Entangled Photon", linewidth=3, color="green", zorder=1)

plt.scatter(0, 0, label=f"{target_true_positive * 100}% true detection\n{target_false_positive * 100}% false detection:", alpha=0)

plt.scatter(distance_cutoff_laser[acquisition_time[0]], distance_cutoff_laser["snr_at0"], color="k", marker="o", s=100,
            label="1s", zorder=1)
plt.scatter(distance_cutoff_laser[acquisition_time[1]], distance_cutoff_laser["snr_at1"], color="k", marker="s", s=100,
            label="1min", zorder=1)
plt.scatter(distance_cutoff_laser[acquisition_time[2]], distance_cutoff_laser["snr_at2"], color="k", marker="^", s=100,
            label="1h", zorder=1)

str_maker_laser = "blue" if colored_marker else "k"
plt.scatter(distance_cutoff_laser[acquisition_time[0]], distance_cutoff_laser["snr_at0"], color=str_maker_laser,
            marker="o", s=100, zorder=5)
plt.scatter(distance_cutoff_laser[acquisition_time[1]], distance_cutoff_laser["snr_at1"], color=str_maker_laser,
            marker="s", s=100, zorder=5)
plt.scatter(distance_cutoff_laser[acquisition_time[2]], distance_cutoff_laser["snr_at2"], color=str_maker_laser,
            marker="^", s=100, zorder=5)

str_maker_sps = "red" if colored_marker else "k"
plt.scatter(distance_cutoff_sps[acquisition_time[0]], distance_cutoff_sps["snr_at0"], color=str_maker_sps, marker="o",
            s=100, zorder=6)
plt.scatter(distance_cutoff_sps[acquisition_time[1]], distance_cutoff_sps["snr_at1"], color=str_maker_sps, marker="s",
            s=100, zorder=6)
plt.scatter(distance_cutoff_sps[acquisition_time[2]], distance_cutoff_sps["snr_at2"], color=str_maker_sps, marker="^",
            s=100, zorder=6)

str_maker_eps = "green" if colored_marker else "k"
plt.scatter(distance_cutoff_eps[acquisition_time[0]], distance_cutoff_eps["snr_at0"], color=str_maker_eps, marker="o",
            s=100, zorder=4)
plt.scatter(distance_cutoff_eps[acquisition_time[1]], distance_cutoff_eps["snr_at1"], color=str_maker_eps, marker="s",
            s=100, zorder=4)
plt.scatter(distance_cutoff_eps[acquisition_time[2]], distance_cutoff_eps["snr_at2"], color=str_maker_eps, marker="^",
            s=100, zorder=4)

plt.xlabel("Distance [m]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.legend(fontsize=22, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.xlim([0, 50])
plt.show()


exit()
distance = 2.739
at = 1
range_interval = distance

param_test = deepcopy(param)
param_test["number_nv_pulse"] = None
param_test["target_distance"] = distance

sps = SinglePhoton(param_test, number_sps=4)
param_test["number_nv_pulse"] = sps.number_nv_pulse

laser = PulsedLaser(param_test)
eps = EntangledPhotonSPDC(param_test)

signal_laser = laser.signal_rate()
signal_sps = sps.signal_rate()
signal_eps = eps.signal_rate()

noise_laser = laser.noise_rate()
noise_sps = sps.noise_rate()
noise_eps = eps.noise_rate()

trigger_rate_laser = laser.trigger_rate
trigger_rate_sps = sps.trigger_rate
trigger_rate_eps = eps.trigger_rate

tp_laser, fp_laser = RocAnalysis(
	signal_rate=signal_laser,
	noise_rate=noise_laser,
	trigger_rate=trigger_rate_laser,
	threshold_limit=trigger_rate_laser / 100,
	range_interval=range_interval,
	timing_window=param["timing_window"],
	acquisition_time=at,
).compute_p_d_p_fa()

tp_sps, fp_sps = RocAnalysis(
	signal_rate=signal_sps,
	noise_rate=noise_sps,
	trigger_rate=trigger_rate_sps,
	threshold_limit=trigger_rate_sps / 100,
	range_interval=range_interval,
	timing_window=param["timing_window"],
	acquisition_time=at,
).compute_p_d_p_fa()

tp_eps, fp_eps = RocAnalysis(
	signal_rate=signal_eps,
	noise_rate=noise_eps,
	trigger_rate=trigger_rate_eps,
	threshold_limit=trigger_rate_eps / 100,
	range_interval=range_interval,
	timing_window=param["timing_window"],
	acquisition_time=at,
).compute_p_d_p_fa()

plt.plot(fp_laser, tp_laser, "-", label="Pulsed Laser", color="blue", linewidth=2.5, zorder=1)
plt.plot(fp_sps, tp_sps, "--", label="Single Photon", color="red", linewidth=2.5, zorder=2)
plt.plot(fp_eps, tp_eps, "-.", label="Entangled Photon", color="green", linewidth=2.5, zorder=3)
plt.xlabel("False Positive Rate", fontsize=22)
plt.ylabel("True Positive Rate", fontsize=22)
plt.title("ROC Curves for Different Sources at 14.18 m", fontsize=22)
plt.legend(fontsize=22, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()