import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
from Analysis import RangeLimitation
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


param = SetupParameters(
	fock_space_dim=5,
	output_power=400000,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=0.57,
	sp_p1=0.99,
	sp_p2=5e-3,
	spdc_eps_heralding=0.57*0.7,
	spdc_eps_collection=0.57,
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


match_multi_photon_probability = True
range_interval = None
distance = np.linspace(0.5, 30, 300)
acquisition_time = np.array([1, 60, 3600])
target_false_positive = 0.2
target_true_positive = 0.8
colored_marker = True
precision = 25
threshold_limit_factor = 100

results = RangeLimitation(
	params=param,
	match_multi_photon_probability=match_multi_photon_probability,
	range_interval=range_interval,
	distance=distance,
	acquisition_time=acquisition_time,
	target_false_positive=target_false_positive,
	target_true_positive=target_true_positive,
	precision_roc=precision,
	threshold_limit_factor_roc=threshold_limit_factor,
).compute()

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

plt.semilogy(distance, snr_laser, "-", label="Pulsed Laser", linewidth=3, color="blue", zorder=1)
plt.semilogy(distance, snr_sps, "--", label="Single Photon", linewidth=3, color="red", zorder=1)
plt.semilogy(distance, snr_eps, "-.", label="Entangled Photon", linewidth=3, color="green", zorder=1)

plt.scatter(0, 0, label=f"{target_true_positive * 100}% true detection\n{target_false_positive * 100}% false detection:", alpha=0)

plt.scatter(distance_cutoff_laser[acquisition_time[0]], distance_cutoff_laser["snr_at0"], color="k", marker="o", s=100,
            label="1s", zorder=1)
plt.scatter(distance_cutoff_laser[acquisition_time[1]], distance_cutoff_laser["snr_at1"], color="k", marker="s", s=100,
            label="1min", zorder=1)
plt.scatter(distance_cutoff_laser[acquisition_time[2]], distance_cutoff_laser["snr_at2"], color="k", marker="^", s=100,
            label="1h", zorder=1)

str_maker_laser = "blue" if colored_marker else "k"
plt.scatter(distance_cutoff_laser[acquisition_time[0]], distance_cutoff_laser["snr_at0"], color=str_maker_laser,
            marker="o", s=100, zorder=2)
plt.scatter(distance_cutoff_laser[acquisition_time[1]], distance_cutoff_laser["snr_at1"], color=str_maker_laser,
            marker="s", s=100, zorder=2)
plt.scatter(distance_cutoff_laser[acquisition_time[2]], distance_cutoff_laser["snr_at2"], color=str_maker_laser,
            marker="^", s=100, zorder=2)

str_maker_sps = "red" if colored_marker else "k"
plt.scatter(distance_cutoff_sps[acquisition_time[0]], distance_cutoff_sps["snr_at0"], color=str_maker_sps, marker="o",
            s=100, zorder=2)
plt.scatter(distance_cutoff_sps[acquisition_time[1]], distance_cutoff_sps["snr_at1"], color=str_maker_sps, marker="s",
            s=100, zorder=2)
plt.scatter(distance_cutoff_sps[acquisition_time[2]], distance_cutoff_sps["snr_at2"], color=str_maker_sps, marker="^",
            s=100, zorder=2)

str_maker_eps = "green" if colored_marker else "k"
plt.scatter(distance_cutoff_eps[acquisition_time[0]], distance_cutoff_eps["snr_at0"], color=str_maker_eps, marker="o",
            s=100, zorder=2)
plt.scatter(distance_cutoff_eps[acquisition_time[1]], distance_cutoff_eps["snr_at1"], color=str_maker_eps, marker="s",
            s=100, zorder=2)
plt.scatter(distance_cutoff_eps[acquisition_time[2]], distance_cutoff_eps["snr_at2"], color=str_maker_eps, marker="^",
            s=100, zorder=2)

plt.xlabel("Distance [m]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.legend(fontsize=22, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.xlim([0, 30])
plt.show()