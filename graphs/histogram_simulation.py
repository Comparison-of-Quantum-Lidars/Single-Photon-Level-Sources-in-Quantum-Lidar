import numpy as np
import matplotlib.pyplot as plt
from Sources import PulsedLaser, SinglePhoton, EntangledPhotonSPDC, SetupParameters
import random
from tqdm import tqdm
from scipy.stats import binom
from Analysis import HistogramAnalysis, HistogramAnalysisFromAdversaryPerspective


scenario = "histogram_target"

if scenario == "histogram_target":

	param = SetupParameters(
		fock_space_dim=15,
		output_power=2e7,
		trigger_rate=None,
		multi_photon_probability=None,
		no_vacuum_probability=None,
		sp_collection=0.2,
		sp_p1=0.9999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
		channel_efficiency=0.5,
		target_distance=10,
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
	timing_window = param["timing_window"]
	acquisition_time = 5
	acquisition_rate = 5e6

	signal_pulsed = PulsedLaser(param).signal_rate()
	noise_pulsed = PulsedLaser(param).noise_rate()
	counts_pulsed, bin_edges_pulsed, bin_edges_distance_pulsed = HistogramAnalysis(param, signal_pulsed, noise_pulsed, acquisition_time=acquisition_time, acquisition_rate=acquisition_rate, effective_trigger_rate=PulsedLaser(param).compute_effective_trigger_rate()).histogram_simulation()
	bin_edges_pulsed = bin_edges_pulsed / 1e-9
	snr_pulsed = PulsedLaser(param).signal_to_noise_rate()


	signal_sps = SinglePhoton(param).signal_rate()
	noise_sps = SinglePhoton(param).noise_rate()
	counts_sps, bin_edges_sps, bin_edges_distance_sps = HistogramAnalysis(param, signal_sps, noise_sps, acquisition_time=acquisition_time, acquisition_rate=acquisition_rate, effective_trigger_rate=SinglePhoton(param).compute_effective_trigger_rate()).histogram_simulation()
	bin_edges_sps = bin_edges_sps / 1e-9
	snr_sps = SinglePhoton(param).signal_to_noise_rate()

	signal_entangled = EntangledPhotonSPDC(param).signal_rate()
	noise_entangled = EntangledPhotonSPDC(param).noise_rate()
	counts_entangled, bin_edges_entangled, bin_edges_distance_entangled = HistogramAnalysis(param, signal_entangled, noise_entangled, acquisition_time=acquisition_time, acquisition_rate=acquisition_rate, effective_trigger_rate=EntangledPhotonSPDC(param).compute_effective_trigger_rate()).histogram_simulation()
	bin_edges_entangled = bin_edges_entangled / 1e-9
	snr_entangled = EntangledPhotonSPDC(param).signal_to_noise_rate()
	print(signal_entangled, noise_entangled, snr_entangled)


	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))
	plt.plot(bin_edges_sps[:-1], counts_sps[:, 0], "-", color="blue", label=f"Single Photon - SNR : {snr_sps:.2f}", linewidth=2.5)
	plt.plot(bin_edges_pulsed[:-1], counts_pulsed[:, 0], "-", color="red", label=f"Pulsed Laser - SNR : {snr_pulsed:.2f}", linewidth=2.5)
	plt.plot(bin_edges_entangled[:-1], counts_entangled[:, 0], "-", color="green", label=f"Entangled Photon - SNR : {snr_entangled:.2f}", linewidth=2.5)
	plt.xlabel('Time of Flight [ns]', fontsize=22)
	plt.ylabel('Coincidence Counts [-]', fontsize=22)
	plt.tick_params(axis='both', which='major', labelsize=22)
	plt.legend(frameon=False, fontsize=22)
	plt.show()

	fig = plt.figure(figsize=(16.1, 10))
	plt.plot(bin_edges_sps[:-1], counts_sps[:, 0]-min(counts_sps[:, 0]), "-", color="blue", label=f"Single Photon - SNR : {snr_sps:.2f}", linewidth=2.5)
	plt.plot(bin_edges_pulsed[:-1], counts_pulsed[:, 0]-min(counts_pulsed[:, 0]), "-", color="red", label=f"Pulsed Laser - SNR : {snr_pulsed:.2f}", linewidth=2.5)
	plt.plot(bin_edges_entangled[:-1], counts_entangled[:, 0]-min(counts_entangled[:, 0]), "-", color="green", label=f"Entangled Photon - SNR : {snr_entangled:.2f}", linewidth=2.5)
	plt.xlabel('Time of Flight [ns]', fontsize=22)
	plt.ylabel("Normalized Coincidence Counts [-]", fontsize=22)
	plt.tick_params(axis='both', which='major', labelsize=22)
	plt.legend(frameon=False, fontsize=22)
	plt.show()

	fig = plt.figure(figsize=(16.1, 10))
	plt.bar(bin_edges_distance_sps[:-1], counts_sps[:, 0], width=timing_window * 299792458, align="edge", edgecolor="black")
	plt.xlabel('Distance [m]', fontsize=22)
	plt.ylabel('Coincidence Counts [-]', fontsize=22)
	plt.tick_params(axis='both', which='major', labelsize=22)
	plt.show()

	fig = plt.figure(figsize=(16.1, 10))
	plt.bar(bin_edges_distance_entangled[:-1], counts_entangled[:, 0], width=timing_window * 299792458, align="edge", edgecolor="black")
	plt.xlabel('Distance [m]', fontsize=22)
	plt.ylabel('Coincidence Counts [-]', fontsize=22)
	plt.tick_params(axis='both', which='major', labelsize=22)
	plt.show()

	fig = plt.figure(figsize=(16.1, 10))
	plt.bar(bin_edges_distance_pulsed[:-1], counts_pulsed[:, 0], width=timing_window * 299792458, align="edge", edgecolor="black")
	plt.xlabel('Distance [m]', fontsize=22)
	plt.ylabel('Coincidence Counts [-]', fontsize=22)
	plt.tick_params(axis='both', which='major', labelsize=22)
	plt.show()

if scenario == "adversary":

	param_lidar = SetupParameters(
		fock_space_dim=5,
		output_power=5e6,
		laser_rate=5e6,
		sp_collection=0.2,
		sp_p1=0.9999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.5,
		spdc_eps_collection=0.2,
		spdc_emission=0.1,
		target_distance=10,
		receiver_diameter=0.5,
		target_albedo=0.2,
		optics_transmitter=0.5,
		optics_receiver=0.5,
		detection_efficiency=0.5,
		background=400,
		detector_dark=200,
		timing_window=0.5e-9,
	)
	param_adversary = SetupParameters(
		fock_space_dim=5,
		output_power=None,
		laser_rate=1e8,
		sp_collection=None,
		sp_p1=None,
		sp_p2=None,
		spdc_eps_heralding=None,
		spdc_eps_collection=None,
		spdc_emission=None,
		target_distance=None,
		receiver_diameter=None,
		target_albedo=None,
		optics_transmitter=None,
		optics_receiver=0.5,
		detection_efficiency=0.5,
		background=400,
		detector_dark=200,
		timing_window=0.5e-9,
	)

	acquisition_time = 5
	source = "pulsed"
	jitter_std_dev = 0.5e-9
	counts_pulsed, bin_edges_pulsed, bin_edges_distance_pulsed = HistogramAnalysisFromAdversaryPerspective(param_lidar, param_adversary, acquisition_time, source, jitter_std_dev).histogram_simulation_adversary()

	fig = plt.figure(figsize=(16.1, 10))
	plt.plot(bin_edges_pulsed[:-1], counts_pulsed[:, 0], "-", color="red", label=f"Pulsed Laser - SNR : {snr_pulsed:.2f}", linewidth=2.5)
	plt.xlabel('Time of Flight [ns]', fontsize=22)
	plt.ylabel('Coincidence Counts [-]', fontsize=22)
	plt.tick_params(axis='both', which='major', labelsize=22)
	plt.legend(frameon=False, fontsize=22)
	plt.show()
