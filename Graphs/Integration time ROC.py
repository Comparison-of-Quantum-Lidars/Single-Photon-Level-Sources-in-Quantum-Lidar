import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from Analysis import *
from copy import deepcopy

scenario = "ROC_Comparison_Distance_all_source"

if scenario == "histogram_roc_combined":

	param = SetupParameters(
		fock_space_dim=40,
		output_power=2e6,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.57,
		sp_p1=0.99,
		sp_p2=1e-3,
		spdc_eps_heralding=0.285,
		spdc_eps_collection=0.57,
		atmosphere=1,
		target_distance=15,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.8,
		optics_receiver=0.5,
		detection_efficiency=0.5,
		background=400,
		detector_dark=200,
		timing_window=0.5e-9,
	)

	acquisition_time = [0.1, 1, 10]

	range_distance = 25

	param_eps = deepcopy(param)
	param_eps["spdc_eps_heralding"] = 0.285
	param_eps["spdc_eps_collection"] = 0.57
	param_eps["multi_photon_probability"] = param_eps["sp_p2"] * (param_eps["sp_collection"]**2)

	eps = EntangledPhotonSPDC(param_eps)
	signal_eps = eps.signal_rate()
	noise_eps = eps.noise_rate()
	snr_eps = eps.signal_to_noise_rate()
	print(f"SNR of the Entangled Source: {snr_eps:.2f}")
	trigger_rate_eps = eps.compute_effective_trigger_rate()
	print(f"Effective Trigger Rate of the Entangled Source: {trigger_rate_eps:.2f}")

	count_eps_all = {}
	bin_edges_distance_eps_all = {}

	true_positive_eps = {}
	false_positive_eps = {}

	for at in acquisition_time:
		count_eps, _, bin_edges_distance_eps = HistogramAnalysis(param_eps, signal_eps, noise_eps, acquisition_time=at, range_distance=range_distance, effective_trigger_rate=trigger_rate_eps).histogram_simulation()
		count_eps_all[at] = count_eps
		bin_edges_distance_eps_all[at] = bin_edges_distance_eps
		true_positive_rate_eps, false_positive_rate_eps = RocAnalysis(
			signal_rate=signal_eps,
			noise_rate=noise_eps,
			trigger_rate=trigger_rate_eps,
			threshold_limit=trigger_rate_eps/100,
			timing_window=param_eps["timing_window"],
			range_interval=range_distance,
			acquisition_time=at,
		).compute_p_d_p_fa()
		true_positive_eps[at] = true_positive_rate_eps
		false_positive_eps[at] = false_positive_rate_eps

	# 3x2 subplots

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig, axs = plt.subplots(2, 3, figsize=(35, 20))

	axs[0, 0].plot(false_positive_eps[acquisition_time[0]], true_positive_eps[acquisition_time[0]], "-", linewidth=2, color="k")
	axs[0, 0].tick_params(axis='both', which='major', labelsize=20)
	axs[0, 0].set_title("0.1s", fontsize=22, fontweight="bold")

	axs[1, 0].plot(bin_edges_distance_eps_all[acquisition_time[0]], count_eps_all[acquisition_time[0]], "-", color="k", linewidth=2)
	axs[1, 0].set_xlim([0, 25])
	axs[1, 0].tick_params(axis='both', which='major', labelsize=20)

	axs[0, 1].plot(false_positive_eps[acquisition_time[1]], true_positive_eps[acquisition_time[1]], "-", linewidth=2, color="k")
	axs[0, 1].tick_params(axis='both', which='major', labelsize=20)
	axs[0, 1].set_title("1s", fontsize=22, fontweight="bold")

	axs[1, 1].plot(bin_edges_distance_eps_all[acquisition_time[1]], count_eps_all[acquisition_time[1]], "-", color="k", linewidth=2)
	axs[1, 1].set_xlim([0, 25])
	axs[1, 1].tick_params(axis='both', which='major', labelsize=20)

	axs[0, 2].plot(false_positive_eps[acquisition_time[2]], true_positive_eps[acquisition_time[2]], "-", color="k", linewidth=2)
	axs[0, 2].tick_params(axis='both', which='major', labelsize=20)
	axs[0, 2].set_title("10s", fontsize=22, fontweight="bold")

	axs[1, 2].plot(bin_edges_distance_eps_all[acquisition_time[2]], count_eps_all[acquisition_time[2]], "-", color="k", linewidth=2)
	axs[1, 2].set_xlim([0, 25])
	axs[1, 2].tick_params(axis='both', which='major', labelsize=20)

	plt.subplots_adjust(wspace=0.4, hspace=0.3)

	# label for first row of subplots
	fig.text(0.53, 0.5, "False Positive", ha='center', fontsize=22, fontweight="bold")
	fig.text(0.53, 0.08, "Distance [m]", ha='center', fontsize=22, fontweight="bold")

	fig.text(0.08, 0.7, "True Positive", va='center', rotation='vertical', fontsize=22, fontweight="bold")
	fig.text(0.08, 0.3, "Counts", va='center', rotation='vertical', fontsize=22, fontweight="bold")

	plt.show()

if scenario == "ROC_Comparison_Acquisition_Time_all_Sources":
	param = SetupParameters(
		fock_space_dim=40,
		output_power=2e6,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.57,
		sp_p1=0.99,
		sp_p2=1e-3,
		spdc_eps_heralding=0.285,
		spdc_eps_collection=0.57,
		atmosphere=1,
		target_distance=15,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.8,
		optics_receiver=0.5,
		detection_efficiency=0.5,
		background=400,
		detector_dark=200,
		timing_window=0.5e-9,
	)

	acquisition_time = [0.1, 1, 10]

	range_distance = 50

	# *** Single Photon Source ***
	param_sps = deepcopy(param)
	param_sps["multi_photon_probability"]=param_sps["sp_p2"] * (param_sps["sp_collection"]**2)
	sps = SinglePhoton(param_sps)
	signal_sps = sps.signal_rate()
	noise_sps = sps.noise_rate()
	trigger_rate_sps = sps.compute_effective_trigger_rate()

	# *** Pulsed Laser ***
	param_laser = deepcopy(param)
	param_laser["multi_photon_probability"]=param_laser["sp_p2"] * (param_laser["sp_collection"]**2)
	laser = PulsedLaser(param_laser)
	signal_laser = laser.signal_rate()
	noise_laser = laser.noise_rate()
	trigger_rate_laser = laser.compute_effective_trigger_rate()

	# *** Entangled Source ***
	param_eps = deepcopy(param)
	param_eps["spdc_eps_heralding"] = 0.285
	param_eps["spdc_eps_collection"] = 0.57
	param_eps["multi_photon_probability"] = param_eps["sp_p2"] * (param_eps["sp_collection"]**2)
	eps = EntangledPhotonSPDC(param_eps)
	signal_eps = eps.signal_rate()
	noise_eps = eps.noise_rate()
	trigger_rate_eps = eps.compute_effective_trigger_rate()

	# *** ROC Curves ***

	# acquisition time = 0.1s

	true_positive_sps_01, false_positive_sps_01 = RocAnalysis(
		signal_rate=signal_sps,
		noise_rate=noise_sps,
		trigger_rate=trigger_rate_sps,
		threshold_limit=trigger_rate_sps/100,
		timing_window=param_sps["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[0],
	).compute_p_d_p_fa()

	true_positive_laser_01, false_positive_laser_01 = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/100,
		timing_window=param_laser["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[0],
	).compute_p_d_p_fa()

	true_positive_eps_01, false_positive_eps_01 = RocAnalysis(
		signal_rate=signal_eps,
		noise_rate=noise_eps,
		trigger_rate=trigger_rate_eps,
		threshold_limit=trigger_rate_eps/100,
		timing_window=param_eps["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[0],
	).compute_p_d_p_fa()

	# acquisition time = 1s

	true_positive_sps_1, false_positive_sps_1 = RocAnalysis(
		signal_rate=signal_sps,
		noise_rate=noise_sps,
		trigger_rate=trigger_rate_sps,
		threshold_limit=trigger_rate_sps/100,
		timing_window=param_sps["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[1],
	).compute_p_d_p_fa()

	true_positive_laser_1, false_positive_laser_1 = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/100,
		timing_window=param_laser["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[1],
	).compute_p_d_p_fa()

	true_positive_eps_1, false_positive_eps_1 = RocAnalysis(
		signal_rate=signal_eps,
		noise_rate=noise_eps,
		trigger_rate=trigger_rate_eps,
		threshold_limit=trigger_rate_eps/100,
		timing_window=param_eps["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[1],
	).compute_p_d_p_fa()

	# acquisition time = 10s

	true_positive_sps_10, false_positive_sps_10 = RocAnalysis(
		signal_rate=signal_sps,
		noise_rate=noise_sps,
		trigger_rate=trigger_rate_sps,
		threshold_limit=trigger_rate_sps/100,
		timing_window=param_sps["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[2],
	).compute_p_d_p_fa()

	true_positive_laser_10, false_positive_laser_10 = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/100,
		timing_window=param_laser["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[2],
	).compute_p_d_p_fa()

	true_positive_eps_10, false_positive_eps_10 = RocAnalysis(
		signal_rate=signal_eps,
		noise_rate=noise_eps,
		trigger_rate=trigger_rate_eps,
		threshold_limit=trigger_rate_eps/100,
		timing_window=param_eps["timing_window"],
		range_interval=range_distance,
		acquisition_time=acquisition_time[2],
	).compute_p_d_p_fa()

	# *** Plotting ***

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig, axs = plt.subplots(1, 3)

	axs[0].plot(false_positive_sps_01, true_positive_sps_01, "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
	axs[0].plot(false_positive_laser_01, true_positive_laser_01, "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
	axs[0].plot(false_positive_eps_01, true_positive_eps_01, "-.", linewidth=2, color="green", label="Entangled Source", zorder=2)
	axs[0].tick_params(axis='both', which='major', labelsize=20)
	axs[0].set_title("a) 0.1s", fontsize=22, fontweight="bold", loc="left")
	axs[0].set_aspect('equal', adjustable='box')

	axs[1].plot(false_positive_sps_1, true_positive_sps_1, "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
	axs[1].plot(false_positive_laser_1, true_positive_laser_1, "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
	axs[1].plot(false_positive_eps_1, true_positive_eps_1, "-.", linewidth=2, color="green", label="Entangled Source", zorder=2)
	axs[1].tick_params(axis='both', which='major', labelsize=20)
	axs[1].set_title("b) 1s", fontsize=22, fontweight="bold", loc="left")
	axs[1].set_aspect('equal', adjustable='box')

	axs[2].plot(false_positive_sps_10, true_positive_sps_10, "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
	axs[2].plot(false_positive_laser_10, true_positive_laser_10, "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
	axs[2].plot(false_positive_eps_10, true_positive_eps_10, "-.", linewidth=2, color="green", label="Entangled Source", zorder=2)
	axs[2].tick_params(axis='both', which='major', labelsize=20)
	axs[2].set_title("c) 10s", fontsize=22, fontweight="bold", loc="left")
	axs[2].set_aspect('equal', adjustable='box')

	fig.text(0.53, 0.25, "False Positive", ha='center', fontsize=22, fontweight="bold")
	fig.text(0.08, 0.5, "True Positive", va='center', rotation='vertical', fontsize=22, fontweight="bold")

	handles, labels = axs[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.85), fontsize=20, ncol=3)

	plt.show()

if scenario=="ROC_Comparison_Acquisition_Time_and_Noise_all_Sources_match_mpp":

	# noise source: 100, 1000, 10 000
	# acquisition time: 0.1, 1, 10
	# 3x3 subplots

	result_laser = {}
	result_eps = {}
	result_sps = {}

	# param = SetupParameters(
	# 	fock_space_dim=15,
	# 	output_power=1.3e6,
	# 	trigger_rate=None,
	# 	multi_photon_probability=None,
	# 	no_vacuum_probability=None,
	# 	sp_collection=0.33,
	# 	sp_p1=0.99,
	# 	sp_p2=1e-3,
	# 	spdc_eps_heralding=0.33 * 0.5,
	# 	spdc_eps_collection=0.33,
	# 	atmosphere=1,
	# 	target_distance=0.32,
	# 	receiver_diameter=0.03,
	# 	target_albedo=0.2,
	# 	optics_transmitter=0.5,
	# 	optics_receiver=0.5,
	# 	detection_efficiency=0.5,
	# 	background=0,
	# 	detector_dark=200,
	# 	timing_window=2e-9,
	# )

	param = SetupParameters(
		fock_space_dim=40,
		output_power=2e6,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.57,
		sp_p1=0.99,
		sp_p2=1e-3,
		spdc_eps_heralding=0.285,
		spdc_eps_collection=0.57,
		atmosphere=1,
		target_distance=15,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.8,
		optics_receiver=0.5,
		detection_efficiency=0.5,
		background=400,
		detector_dark=200,
		timing_window=0.5e-9,
	)

	param_laser = deepcopy(param)
	param_sps = deepcopy(param)
	param_eps = deepcopy(param)

	sps = SinglePhoton(param_sps)
	multi_photon_probability = sps.multi_photon_probability
	param_sps["multi_photon_probability"] = multi_photon_probability

	param_laser["multi_photon_probability"] = multi_photon_probability
	param_eps["multi_photon_probability"] = multi_photon_probability


	jamming_photon = [200, 2000, 20000]
	acquisition_time = [0.1, 1, 10]

	range_distance = 50

	for jp in tqdm(jamming_photon):
		param_laser["background"] = jp
		param_eps["background"] = jp
		param_sps["background"] = jp
		for at in acquisition_time:
			laser = PulsedLaser(param_laser)
			eps = EntangledPhotonSPDC(param_eps)
			sps = SinglePhoton(param_sps)

			signal_laser = laser.signal_rate()
			noise_laser = laser.noise_rate()
			trigger_rate_laser = laser.trigger_rate

			signal_eps = eps.signal_rate()
			noise_eps = eps.noise_rate()
			trigger_rate_eps = eps.trigger_rate

			signal_sps = sps.signal_rate()
			noise_sps = sps.noise_rate()
			trigger_rate_sps = sps.trigger_rate

			true_positive_laser, false_positive_laser = RocAnalysis(
				signal_rate=signal_laser,
				noise_rate=noise_laser,
				trigger_rate=trigger_rate_laser,
				threshold_limit=trigger_rate_laser / 100,
				timing_window=param_laser["timing_window"],
				range_interval=range_distance,
				acquisition_time=at,
			).compute_p_d_p_fa()

			true_positive_eps, false_positive_eps = RocAnalysis(
				signal_rate=signal_eps,
				noise_rate=noise_eps,
				trigger_rate=trigger_rate_eps,
				threshold_limit=trigger_rate_eps / 100,
				timing_window=param_eps["timing_window"],
				range_interval=range_distance,
				acquisition_time=at,
			).compute_p_d_p_fa()

			true_positive_sps, false_positive_sps = RocAnalysis(
				signal_rate=signal_sps,
				noise_rate=noise_sps,
				trigger_rate=trigger_rate_sps,
				threshold_limit=trigger_rate_sps / 100,
				timing_window=param_sps["timing_window"],
				range_interval=range_distance,
				acquisition_time=at,
			).compute_p_d_p_fa()

			result_laser[(jp, at)] = (true_positive_laser, false_positive_laser)
			result_eps[(jp, at)] = (true_positive_eps, false_positive_eps)
			result_sps[(jp, at)] = (true_positive_sps, false_positive_sps)

	# 3x3 subplots

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig, axs = plt.subplots(3, 3, figsize=(35, 20))

	for i, jp in enumerate(jamming_photon):
		for j, at in enumerate(acquisition_time):
			axs[i, j].plot(result_laser[(jp, at)][1], result_laser[(jp, at)][0], "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
			axs[i, j].plot(result_eps[(jp, at)][1], result_eps[(jp, at)][0], "-.", linewidth=2, color="green", label="Entangled Source", zorder=3)
			axs[i, j].plot(result_sps[(jp, at)][1], result_sps[(jp, at)][0], "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
			axs[i, j].tick_params(axis='both', which='major', labelsize=20)
			axs[i, j].set_aspect('equal', adjustable='box')

	fig.text(0.53, 0.05, "False Positive", ha='center', fontsize=25, fontweight="bold")
	fig.text(0.08, 0.5, "True Positive", va='center', rotation='vertical', fontsize=25, fontweight="bold")

	handles, labels = axs[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=20, ncol=3)

	plt.show()

if scenario=="ROC_Comparison_Acquisition_Time_and_Noise_all_Sources_match_nvp":

	# noise source: 100, 1000, 10 000
	# acquisition time: 0.1, 1, 10
	# 3x3 subplots

	result_laser = {}
	result_eps = {}
	result_sps = {}

	# param = SetupParameters(
	# 	fock_space_dim=15,
	# 	output_power=1.3e6,
	# 	trigger_rate=None,
	# 	multi_photon_probability=None,
	# 	no_vacuum_probability=None,
	# 	sp_collection=0.33,
	# 	sp_p1=0.99,
	# 	sp_p2=1e-3,
	# 	spdc_eps_heralding=0.33 * 0.5,
	# 	spdc_eps_collection=0.33,
	# 	atmosphere=1,
	# 	target_distance=0.32,
	# 	receiver_diameter=0.03,
	# 	target_albedo=0.2,
	# 	optics_transmitter=0.5,
	# 	optics_receiver=0.5,
	# 	detection_efficiency=0.5,
	# 	background=0,
	# 	detector_dark=200,
	# 	timing_window=2e-9,
	# )

	param = SetupParameters(
		fock_space_dim=40,
		output_power=2e6,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.57,
		sp_p1=0.99,
		sp_p2=1e-3,
		spdc_eps_heralding=0.285,
		spdc_eps_collection=0.57,
		atmosphere=1,
		target_distance=15,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.8,
		optics_receiver=0.5,
		detection_efficiency=0.5,
		background=400,
		detector_dark=200,
		timing_window=0.5e-9,
	)

	param_laser = deepcopy(param)
	param_sps = deepcopy(param)
	param_eps = deepcopy(param)

	sps = SinglePhoton(param_sps)
	no_vacuum_probability = sps.no_vacuum_probability
	param_sps["no_vacuum_probability"] = no_vacuum_probability
	param_laser["no_vacuum_probability"] = no_vacuum_probability
	param_eps["no_vacuum_probability"] = no_vacuum_probability


	jamming_photon = [200, 2000, 20000]
	acquisition_time = [0.1, 1, 10]

	range_distance = 50

	for jp in tqdm(jamming_photon):
		param_laser["background"] = jp
		param_eps["background"] = jp
		param_sps["background"] = jp
		for at in acquisition_time:
			laser = PulsedLaser(param_laser)
			eps = EntangledPhotonSPDC(param_eps)
			sps = SinglePhoton(param_sps)

			signal_laser = laser.signal_rate()
			noise_laser = laser.noise_rate()
			trigger_rate_laser = laser.trigger_rate

			signal_eps = eps.signal_rate()
			noise_eps = eps.noise_rate()
			trigger_rate_eps = eps.trigger_rate

			signal_sps = sps.signal_rate()
			noise_sps = sps.noise_rate()
			trigger_rate_sps = sps.trigger_rate

			true_positive_laser, false_positive_laser = RocAnalysis(
				signal_rate=signal_laser,
				noise_rate=noise_laser,
				trigger_rate=trigger_rate_laser,
				threshold_limit=trigger_rate_laser / 100,
				timing_window=param_laser["timing_window"],
				range_interval=range_distance,
				acquisition_time=at,
			).compute_p_d_p_fa()

			true_positive_eps, false_positive_eps = RocAnalysis(
				signal_rate=signal_eps,
				noise_rate=noise_eps,
				trigger_rate=trigger_rate_eps,
				threshold_limit=trigger_rate_eps / 100,
				timing_window=param_eps["timing_window"],
				range_interval=range_distance,
				acquisition_time=at,
			).compute_p_d_p_fa()

			true_positive_sps, false_positive_sps = RocAnalysis(
				signal_rate=signal_sps,
				noise_rate=noise_sps,
				trigger_rate=trigger_rate_sps,
				threshold_limit=trigger_rate_sps / 100,
				timing_window=param_sps["timing_window"],
				range_interval=range_distance,
				acquisition_time=at,
			).compute_p_d_p_fa()

			result_laser[(jp, at)] = (true_positive_laser, false_positive_laser)
			result_eps[(jp, at)] = (true_positive_eps, false_positive_eps)
			result_sps[(jp, at)] = (true_positive_sps, false_positive_sps)

	# 3x3 subplots

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig, axs = plt.subplots(3, 3, figsize=(35, 20))

	for i, jp in enumerate(jamming_photon):
		for j, at in enumerate(acquisition_time):
			axs[i, j].plot(result_laser[(jp, at)][1], result_laser[(jp, at)][0], "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
			axs[i, j].plot(result_eps[(jp, at)][1], result_eps[(jp, at)][0], "-.", linewidth=2, color="green", label="Entangled Source", zorder=3)
			axs[i, j].plot(result_sps[(jp, at)][1], result_sps[(jp, at)][0], "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
			axs[i, j].tick_params(axis='both', which='major', labelsize=20)
			axs[i, j].set_aspect('equal', adjustable='box')

	fig.text(0.53, 0.05, "False Positive", ha='center', fontsize=25, fontweight="bold")
	fig.text(0.08, 0.5, "True Positive", va='center', rotation='vertical', fontsize=25, fontweight="bold")

	handles, labels = axs[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=20, ncol=3)


	plt.show()

if scenario=="ROC_Comparison_Distance_all_source":

	param = SetupParameters(
		fock_space_dim=40,
		output_power=2e6,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.57,
		sp_p1=0.99,
		sp_p2=1e-3,
		spdc_eps_heralding=0.57*0.5,
		spdc_eps_collection=0.2,
		atmosphere=1,
		target_distance=1,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.8,
		optics_receiver=0.5,
		detection_efficiency=0.5,
		background=400,
		detector_dark=200,
		timing_window=0.5e-9,
	)

	range_distance = 100
	distance = [5, 15, 25, 45]

	param_laser = deepcopy(param)
	param_sps = deepcopy(param)
	param_eps = deepcopy(param)

	sps = SinglePhoton(param_sps)
	non_vacuum_probability = sps.no_vacuum_probability
	param_sps["no_vacuum_probability"] = non_vacuum_probability
	param_laser["no_vacuum_probability"] = non_vacuum_probability
	param_eps["no_vacuum_probability"] = non_vacuum_probability

	laser = PulsedLaser(param_laser)
	eps = EntangledPhotonSPDC(param_eps)
	sps = SinglePhoton(param_sps)

	result_laser = {}
	result_eps = {}
	result_sps = {}

	for d in tqdm(distance):
		param_laser["target_distance"] = d
		param_eps["target_distance"] = d
		param_sps["target_distance"] = d
		laser = PulsedLaser(param_laser)
		eps = EntangledPhotonSPDC(param_eps)
		sps = SinglePhoton(param_sps)

		signal_laser = laser.signal_rate()
		noise_laser = laser.noise_rate()
		trigger_rate_laser = laser.trigger_rate

		signal_eps = eps.signal_rate()
		noise_eps = eps.noise_rate()
		trigger_rate_eps = eps.trigger_rate

		signal_sps = sps.signal_rate()
		noise_sps = sps.noise_rate()
		trigger_rate_sps = sps.trigger_rate

		true_positive_laser, false_positive_laser = RocAnalysis(
			signal_rate=signal_laser,
			noise_rate=noise_laser,
			trigger_rate=trigger_rate_laser,
			threshold_limit=trigger_rate_laser / 100,
			timing_window=param_laser["timing_window"],
			range_interval=range_distance,
			acquisition_time=1,
		).compute_p_d_p_fa()

		true_positive_eps, false_positive_eps = RocAnalysis(
			signal_rate=signal_eps,
			noise_rate=noise_eps,
			trigger_rate=trigger_rate_eps,
			threshold_limit=trigger_rate_eps / 100,
			timing_window=param_eps["timing_window"],
			range_interval=range_distance,
			acquisition_time=1,
		).compute_p_d_p_fa()

		true_positive_sps, false_positive_sps = RocAnalysis(
			signal_rate=signal_sps,
			noise_rate=noise_sps,
			trigger_rate=trigger_rate_sps,
			threshold_limit=trigger_rate_sps / 100,
			timing_window=param_sps["timing_window"],
			range_interval=range_distance,
			acquisition_time=1,
		).compute_p_d_p_fa()

		result_laser[d] = (true_positive_laser, false_positive_laser)
		result_eps[d] = (true_positive_eps, false_positive_eps)
		result_sps[d] = (true_positive_sps, false_positive_sps)

	# 2x2 subplots

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig, axs = plt.subplots(2, 2, figsize=(35, 20))

	axs[0, 0].plot(result_laser[distance[0]][1], result_laser[distance[0]][0], "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
	axs[0, 0].plot(result_eps[distance[0]][1], result_eps[distance[0]][0], "-.", linewidth=2, color="green", label="Entangled Source", zorder=3)
	axs[0, 0].plot(result_sps[distance[0]][1], result_sps[distance[0]][0], "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
	axs[0, 0].tick_params(axis='both', which='major', labelsize=20)
	axs[0, 0].set_title("a) 5m", fontsize=22, fontweight="bold", loc="left")
	axs[0, 0].set_aspect('equal', adjustable='box')

	axs[0, 1].plot(result_laser[distance[1]][1], result_laser[distance[1]][0], "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
	axs[0, 1].plot(result_eps[distance[1]][1], result_eps[distance[1]][0], "-.", linewidth=2, color="green", label="Entangled Source", zorder=3)
	axs[0, 1].plot(result_sps[distance[1]][1], result_sps[distance[1]][0], "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
	axs[0, 1].tick_params(axis='both', which='major', labelsize=20)
	axs[0, 1].set_title("b) 15m", fontsize=22, fontweight="bold", loc="left")
	axs[0, 1].set_aspect('equal', adjustable='box')

	axs[1, 0].plot(result_laser[distance[2]][1], result_laser[distance[2]][0], "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
	axs[1, 0].plot(result_eps[distance[2]][1], result_eps[distance[2]][0], "-.", linewidth=2, color="green", label="Entangled Source", zorder=3)
	axs[1, 0].plot(result_sps[distance[2]][1], result_sps[distance[2]][0], "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
	axs[1, 0].tick_params(axis='both', which='major', labelsize=20)
	axs[1, 0].set_title("c) 25m", fontsize=22, fontweight="bold", loc="left")
	axs[1, 0].set_aspect('equal', adjustable='box')

	axs[1, 1].plot(result_laser[distance[3]][1], result_laser[distance[3]][0], "-", linewidth=2, color="blue", label="Pulsed Laser", zorder=1)
	axs[1, 1].plot(result_eps[distance[3]][1], result_eps[distance[3]][0], "-.", linewidth=2, color="green", label="Entangled Source", zorder=3)
	axs[1, 1].plot(result_sps[distance[3]][1], result_sps[distance[3]][0], "--", linewidth=2, color="red", label="Single Photon Source", zorder=2)
	axs[1, 1].tick_params(axis='both', which='major', labelsize=20)
	axs[1, 1].set_title("d) 45m", fontsize=22, fontweight="bold", loc="left")
	axs[1, 1].set_aspect('equal', adjustable='box')

	fig.text(0.53, 0.05, "False Positive", ha='center', fontsize=25, fontweight="bold")
	fig.text(0.08, 0.5, "True Positive", va='center', rotation='vertical', fontsize=25, fontweight="bold")

	handles, labels = axs[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=20, ncol=3)

	plt.show()

