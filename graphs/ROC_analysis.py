import numpy as np
import matplotlib.pyplot as plt
from Sources import *
from copy import copy, deepcopy
from Analysis import *

scenario = "matched_no_vacuum_probability"

if scenario == "matched_trigger_rate_laser_sps":


	param = SetupParameters(
		fock_space_dim=10,
		output_power=2e7,
		trigger_rate=2e7/((0.9999+2*1e-4)*0.2),
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.2,
		sp_p1=0.9999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
		channel_efficiency=1,
		target_distance=40,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.5,
		optics_receiver=0.5,
		detection_efficiency=0.9,
		background=100,
		detector_dark=50,
		timing_window=0.5e-9,
	)

	range_interval = 50

	### PULSED LASER ###
	signal_laser = PulsedLaser(param).signal_rate()
	noise_laser = PulsedLaser(param).noise_rate()
	trigger_rate_laser = PulsedLaser(param).compute_effective_trigger_rate()

	true_positive_laser, false_positive_laser = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### SINGLE PHOTON ###
	param_sps = deepcopy(param)
	param_sps["trigger_rate"] = None
	signal_sp = SinglePhoton(param_sps).signal_rate()
	noise_sp = SinglePhoton(param_sps).noise_rate()
	trigger_rate_sp = SinglePhoton(param_sps).compute_effective_trigger_rate()

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal_sp,
		noise_rate=noise_sp,
		trigger_rate=trigger_rate_sp,
		threshold_limit=trigger_rate_sp/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled Source ###
	# param_eps = deepcopy(param)
	# signal_eps = EntangledPhotonSPDC(param_eps).signal_rate()
	# noise_eps = EntangledPhotonSPDC(param_eps).noise_rate()
	# trigger_rate_eps = EntangledPhotonSPDC(param_eps).compute_effective_trigger_rate()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))


	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	#plt.title("ROC curve with matched output power", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()

if scenario == "matched_trigger_rate_laser_eps":

	param = SetupParameters(
		fock_space_dim=10,
		output_power=2e7,
		trigger_rate=2e6,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=None,
		sp_p1=None,
		sp_p2=None,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
		channel_efficiency=1,
		target_distance=40,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.5,
		optics_receiver=0.5,
		detection_efficiency=0.9,
		background=100,
		detector_dark=50,
		timing_window=0.5e-9,
	)

	range_interval = 50

	### PULSED LASER ###
	signal_laser = PulsedLaser(param).signal_rate()
	noise_laser = PulsedLaser(param).noise_rate()
	trigger_rate_laser = PulsedLaser(param).compute_effective_trigger_rate()

	true_positive_laser, false_positive_laser = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled Source ###
	param_eps = deepcopy(param)
	signal_eps = EntangledPhotonSPDC(param_eps).signal_rate()
	noise_eps = EntangledPhotonSPDC(param_eps).noise_rate()
	trigger_rate_eps = EntangledPhotonSPDC(param_eps).compute_effective_trigger_rate()

	true_positive_eps, false_positive_eps = RocAnalysis(
		signal_rate=signal_eps,
		noise_rate=noise_eps,
		trigger_rate=trigger_rate_eps,
		threshold_limit=trigger_rate_eps/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_eps, true_positive_eps, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title("ROC curve with matched triggering rate", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()


if scenario == "matched_multi_photon":

	### Match Multi-photon probability with source & entangled###

	param = SetupParameters(
		fock_space_dim=10,
		output_power=2e7,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.2,
		sp_p1=0.9999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
		channel_efficiency=1,
		target_distance=40,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.5,
		optics_receiver=0.5,
		detection_efficiency=0.9,
		background=100,
		detector_dark=50,
		timing_window=0.5e-9,
	)

	param["multi_photon_probability"] = param["sp_p2"] * (param["sp_collection"]*param["channel_efficiency"])**2

	range_interval = 50

	### Pulsed Laser ###

	signal = PulsedLaser(param).signal_rate()
	noise = PulsedLaser(param).noise_rate()
	trigger_rate = PulsedLaser(param).compute_effective_trigger_rate()

	true_positive_laser_sps, false_positive_laser_sps = RocAnalysis(
		signal_rate=signal,
		noise_rate=noise,
		trigger_rate=trigger_rate,
		threshold_limit=trigger_rate/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Single photon ###

	signal = SinglePhoton(param).signal_rate()
	noise = SinglePhoton(param).noise_rate()
	trigger_rate = SinglePhoton(param).compute_effective_trigger_rate()

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal,
		noise_rate=noise,
		trigger_rate=trigger_rate,
		threshold_limit=trigger_rate/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled ###

	signal = EntangledPhotonSPDC(param, adjust_eps_rate=True).signal_rate()
	noise = EntangledPhotonSPDC(param, adjust_eps_rate=True).noise_rate()
	trigger_rate = EntangledPhotonSPDC(param, adjust_eps_rate=True).compute_effective_trigger_rate()


	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal,
		noise_rate=noise,
		trigger_rate=trigger_rate,
		threshold_limit=trigger_rate/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()




	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser_sps, true_positive_laser_sps, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title("ROC curve with matched multi-photon probability", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()

if scenario == "matched_no_vacuum_probability":
	param = SetupParameters(
		fock_space_dim=10,
		output_power=2e7,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.2,
		sp_p1=0.9999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
		channel_efficiency=1,
		target_distance=40,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.5,
		optics_receiver=0.5,
		detection_efficiency=0.9,
		background=100,
		detector_dark=50,
		timing_window=0.5e-9,
	)
	total_loss = param["sp_collection"]*param["channel_efficiency"]
	param["no_vacuum_probability"] = param["sp_p1"] * total_loss + param["sp_p2"] * total_loss * (2-total_loss)

	range_interval = 50

	### Pulsed Laser ###

	signal = PulsedLaser(param).signal_rate()
	noise = PulsedLaser(param).noise_rate()
	trigger_rate = PulsedLaser(param).compute_effective_trigger_rate()

	true_positive_laser_sps, false_positive_laser_sps = RocAnalysis(
		signal_rate=signal,
		noise_rate=noise,
		trigger_rate=trigger_rate,
		threshold_limit=trigger_rate/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	print(signal/noise)

	### Single photon ###

	signal = SinglePhoton(param).signal_rate()
	noise = SinglePhoton(param).noise_rate()
	trigger_rate = SinglePhoton(param).compute_effective_trigger_rate()

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal,
		noise_rate=noise,
		trigger_rate=trigger_rate,
		threshold_limit=trigger_rate/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled ###

	signal = EntangledPhotonSPDC(param).signal_rate()
	noise = EntangledPhotonSPDC(param).noise_rate()
	trigger_rate = EntangledPhotonSPDC(param).compute_effective_trigger_rate()

	print(signal/noise)

	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal,
		noise_rate=noise,
		trigger_rate=trigger_rate,
		threshold_limit=trigger_rate/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser_sps, true_positive_laser_sps, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title("ROC curve with matched no-vacuum probability", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()

if scenario == "effect_channel_efficiency_laser":

	param = SetupParameters(
		fock_space_dim=10,
		output_power=2e7,
		trigger_rate=2e7,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.2,
		sp_p1=0.9999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
		channel_efficiency=None,
		target_distance=40,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.5,
		optics_receiver=0.5,
		detection_efficiency=0.9,
		background=100,
		detector_dark=50,
		timing_window=0.5e-9,
	)

	range_interval = 50

	param_channel_1 = deepcopy(param)
	param_channel_1["channel_efficiency"] = 1

	param_channel_08 = deepcopy(param)
	param_channel_08["channel_efficiency"] = 0.8

	param_channel_01 = deepcopy(param)
	param_channel_01["channel_efficiency"] = 0.1

	signal_laser_channel_1 = PulsedLaser(param_channel_1).signal_rate()
	noise_laser_channel_1 = PulsedLaser(param_channel_1).noise_rate()
	trigger_rate_laser_channel_1 = PulsedLaser(param_channel_1).compute_effective_trigger_rate()

	true_positive_laser_channel_1, false_positive_laser_channel_1 = RocAnalysis(
		signal_rate=signal_laser_channel_1,
		noise_rate=noise_laser_channel_1,
		trigger_rate=trigger_rate_laser_channel_1,
		threshold_limit=trigger_rate_laser_channel_1/1000,
		range_interval=range_interval,
		timing_window=param_channel_1["timing_window"]
	).compute_p_d_p_fa()

	signal_laser_channel_08 = PulsedLaser(param_channel_08).signal_rate()
	noise_laser_channel_08 = PulsedLaser(param_channel_08).noise_rate()
	trigger_rate_laser_channel_08 = PulsedLaser(param_channel_08).compute_effective_trigger_rate()

	true_positive_laser_channel_08, false_positive_laser_channel_08 = RocAnalysis(
		signal_rate=signal_laser_channel_08,
		noise_rate=noise_laser_channel_08,
		trigger_rate=trigger_rate_laser_channel_08,
		threshold_limit=trigger_rate_laser_channel_08/1000,
		range_interval=range_interval,
		timing_window=param_channel_08["timing_window"]
	).compute_p_d_p_fa()

	signal_laser_channel_01 = PulsedLaser(param_channel_01).signal_rate()
	noise_laser_channel_01 = PulsedLaser(param_channel_01).noise_rate()
	trigger_rate_laser_channel_01 = PulsedLaser(param_channel_01).compute_effective_trigger_rate()

	true_positive_laser_channel_01, false_positive_laser_channel_01 = RocAnalysis(
		signal_rate=signal_laser_channel_01,
		noise_rate=noise_laser_channel_01,
		trigger_rate=trigger_rate_laser_channel_01,
		threshold_limit=trigger_rate_laser_channel_01/1000,
		range_interval=range_interval,
		timing_window=param_channel_01["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser_channel_1, true_positive_laser_channel_1, "-", label="Channel efficiency = 1", linewidth=2.5, color="blue")
	plt.plot(false_positive_laser_channel_08, true_positive_laser_channel_08, "--", label="Channel efficiency = 0.8", linewidth=2.5, color="blue")
	plt.plot(false_positive_laser_channel_01, true_positive_laser_channel_01, "-.", label="Channel efficiency = 0.1", linewidth=2.5, color="blue")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title("ROC curve of laser for different channel efficiency", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()

