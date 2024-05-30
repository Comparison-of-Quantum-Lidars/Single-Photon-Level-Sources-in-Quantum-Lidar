import numpy as np
import matplotlib.pyplot as plt
from Sources import *
from copy import copy, deepcopy
from Analysis import *

# scenario: matched_trigger_rate/matched_detectability_pout/matched_detectability_average_photon_number/matched_detectability_pmulti/matched_multi_photon
scenario = "matched_trigger_rate"

if scenario == "matched_trigger_rate":


	param = SetupParameters(
		fock_space_dim=10,
		output_power=2e7,
		trigger_rate=2e7/((0.9999+2*1e-4)*0.2),
		multi_photon_probability=None,
		sp_collection=0.2,
		sp_p1=0.9999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
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
	trigger_rate_laser = PulsedLaser(param).effective_trigger_rate()

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
	trigger_rate_sp = SinglePhoton(param_sps).effective_trigger_rate()

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal_sp,
		noise_rate=noise_sp,
		trigger_rate=trigger_rate_sp,
		threshold_limit=trigger_rate_sp/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	print(signal_sp, noise_sp, trigger_rate_sp, signal_sp/noise_sp)

	### Entangled Source ###
	signal_spdc = EntangledPhotonSPDC(param, adjust_eps_rate=True).signal_rate()
	noise_spdc = EntangledPhotonSPDC(param, adjust_eps_rate=True).noise_rate()
	trigger_rate_spdc = EntangledPhotonSPDC(param, adjust_eps_rate=True).effective_trigger_rate()

	print(signal_spdc, noise_spdc, trigger_rate_spdc, signal_spdc/noise_spdc)

	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal_spdc,
		noise_rate=noise_spdc,
		trigger_rate=trigger_rate_spdc,
		threshold_limit=trigger_rate_spdc/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))


	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	#plt.title("ROC curve with matched output power", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()

if scenario == "matched_detectability_pout":

### ROC curve with matched matched trigger rate (as seen by an adversary with a non-number resolving detector) ###

	param = SetupParameters(
		fock_space_dim=10,
		output_power=None,
		trigger_rate=None,
		multi_photon_probability=None,
		sp_collection=0.2,
		sp_p1=0.99,
		sp_p2=1e-2,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
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

	#A_R/P_0 = 0.99 & A_R = 1e6 & P_0 = 1.0101e6
	param.output_power = 1.0101e6
	param_laser = deepcopy(param)
	param_laser.multi_photon_probability = 0.0002
	param_sps = deepcopy(param)
	param_sps.multi_photon_probability = 0.0526
	param_sps.sp_p1, param_sps.sp_p2 = 1-0.0526, 0.0526
	param_eps = deepcopy(param)
	param_eps.multi_photon_probability = 0.0023

	range_interval = 50

	### PULSED LASER ###
	signal_laser = PulsedLaser(param_laser).signal_rate()
	noise_laser = PulsedLaser(param_laser).noise_rate()
	trigger_rate_laser = PulsedLaser(param_laser).effective_trigger_rate()

	alpha = PulsedLaser(param_laser).compute_alpha()
	detectability = (1-np.exp(-alpha**2))/(alpha**2)
	print(f"A_R/Pout laser : {detectability}")

	true_positive_laser, false_positive_laser = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### SINGLE PHOTON ###
	signal_sp = SinglePhoton(param_sps).signal_rate()
	noise_sp = SinglePhoton(param_sps).noise_rate()
	trigger_rate_sp = SinglePhoton(param_sps).effective_trigger_rate()

	trigger_rate = SinglePhoton(param_sps).effective_trigger_rate()
	pg = 1-0.0526
	pgg = 0.0526
	eta=0.2
	detectability=(1-(1-pg-pgg)-(1-eta)*pg-((1-eta)**2)*pgg)/((pg+2*pgg)*eta)
	print(f"A_R/Pout single photon : {detectability}")


	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal_sp,
		noise_rate=noise_sp,
		trigger_rate=trigger_rate_sp,
		threshold_limit=trigger_rate_sp/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled Source ###
	signal_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).signal_rate()
	noise_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).noise_rate()
	trigger_rate_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).effective_trigger_rate()

	epsilon = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).epsilon

	detectability = (1-(1/((epsilon+1)-epsilon*(1-eta))))/(epsilon*eta)
	print(f"A_R/Pout entangled : {detectability}")

	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal_spdc,
		noise_rate=noise_spdc,
		trigger_rate=trigger_rate_spdc,
		threshold_limit=trigger_rate_spdc/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title(f"A_R=1e6, Pout=1,01e6", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()

	# A_R/P_0 = 0.9 & A_R = 1e6 & P_0 = 1.11e5
	param.output_power = 1.11111e6
	param_laser = deepcopy(param)
	param_laser.multi_photon_probability = 0.02
	param_sps = deepcopy(param)
	param_sps.multi_photon_probability = 1
	param_sps.sp_p1, param_sps.sp_p2 = 1 - 1, 1
	param_eps = deepcopy(param)
	param_eps.multi_photon_probability = 0.1276

	range_interval = 50

	### PULSED LASER ###
	signal_laser = PulsedLaser(param_laser).signal_rate()
	noise_laser = PulsedLaser(param_laser).noise_rate()
	trigger_rate_laser = PulsedLaser(param_laser).effective_trigger_rate()

	true_positive_laser, false_positive_laser = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser / 1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### SINGLE PHOTON ###
	signal_sp = SinglePhoton(param_sps).signal_rate()
	noise_sp = SinglePhoton(param_sps).noise_rate()
	trigger_rate_sp = SinglePhoton(param_sps).effective_trigger_rate()

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal_sp,
		noise_rate=noise_sp,
		trigger_rate=trigger_rate_sp,
		threshold_limit=trigger_rate_sp / 1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled Source ###
	signal_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).signal_rate()
	noise_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).noise_rate()
	trigger_rate_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).effective_trigger_rate()

	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal_spdc,
		noise_rate=noise_spdc,
		trigger_rate=trigger_rate_spdc,
		threshold_limit=trigger_rate_spdc / 1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title(f"A_R=1e6, Pout=1,111e6", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()

	# A_R/P_0 = 0.5 & A_R = 1e6 & P_0 = 2e6
	param.output_power = 2e6
	param_laser = deepcopy(param)
	param_laser.multi_photon_probability = 0.4730
	param_eps = deepcopy(param)
	param_eps.multi_photon_probability = 0.6944

	range_interval = 50

	### PULSED LASER ###
	signal_laser = PulsedLaser(param_laser).signal_rate()
	noise_laser = PulsedLaser(param_laser).noise_rate()
	trigger_rate_laser = PulsedLaser(param_laser).effective_trigger_rate()

	true_positive_laser, false_positive_laser = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser / 1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### SINGLE PHOTON ###
	signal_sp = SinglePhoton(param_sps).signal_rate()
	noise_sp = SinglePhoton(param_sps).noise_rate()
	trigger_rate_sp = SinglePhoton(param_sps).effective_trigger_rate()

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal_sp,
		noise_rate=noise_sp,
		trigger_rate=trigger_rate_sp,
		threshold_limit=trigger_rate_sp / 1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled Source ###
	signal_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).signal_rate()
	noise_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).noise_rate()
	trigger_rate_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).effective_trigger_rate()

	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal_spdc,
		noise_rate=noise_spdc,
		trigger_rate=trigger_rate_spdc,
		threshold_limit=trigger_rate_spdc / 1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title(f"A_R=1e6, Pout=2e6", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()


if scenario == "matched_detectability_average_photon_number":

### ROC curve with matched matched trigger rate (as seen by an adversary with a number resolving detector) ###

	param = SetupParameters(
		fock_space_dim=10,
		output_power=None,
		trigger_rate=None,
		multi_photon_probability=None,
		sp_collection=0.2,
		sp_p1=None,
		sp_p2=None,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
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

	#A_R/P_0 = 0.99 & A_R = 1e6 & P_0 = 1.0101e6
	param_laser = deepcopy(param)
	param_laser.output_power = 79.09888e6
	param_laser.multi_photon_probability=0.264
	param_sps = deepcopy(param)
	param_sps.output_power = 50100491
	param_sps.multi_photon_probability = 0
	param_sps.sp_p1, param_sps.sp_p2 = 1, 0
	param_eps = deepcopy(param)
	param_eps.output_power = 59999600
	param_eps.multi_photon_probability = 0.25

	range_interval = 50

	### PULSED LASER ###
	signal_laser = PulsedLaser(param_laser).signal_rate()
	noise_laser = PulsedLaser(param_laser).noise_rate()
	trigger_rate_laser = PulsedLaser(param_laser).effective_trigger_rate()

	alpha = PulsedLaser(param_laser).compute_alpha()
	detectability = (param_laser.output_power/(1))*(1-np.exp(-alpha**2))/(alpha**2)
	print(f"A_R laser : {detectability}")

	true_positive_laser, false_positive_laser = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### SINGLE PHOTON ###
	signal_sp = SinglePhoton(param_sps).signal_rate()
	noise_sp = SinglePhoton(param_sps).noise_rate()
	trigger_rate_sp = SinglePhoton(param_sps).effective_trigger_rate()

	trigger_rate = SinglePhoton(param_sps).effective_trigger_rate()
	pg = 1
	pgg = 0
	eta = 0.2

	detectability = param_sps.output_power*(1-(1-pg-pgg)-(1-eta)*pg-((1-eta)**2)*pgg)/((pg+2*pgg)*eta)
	print(f"A_R/Pout single photon : {detectability}")

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal_sp,
		noise_rate=noise_sp,
		trigger_rate=trigger_rate_sp,
		threshold_limit=trigger_rate_sp/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled Source ###
	signal_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).signal_rate()
	noise_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).noise_rate()
	trigger_rate_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).effective_trigger_rate()

	epsilon = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).epsilon
	detectability = param_eps.output_power*(1-(1/((epsilon+1)-epsilon*(1-eta))))/(epsilon*eta)
	print(f"A_R/Pout entangled : {detectability}")

	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal_spdc,
		noise_rate=noise_spdc,
		trigger_rate=trigger_rate_spdc,
		threshold_limit=trigger_rate_spdc/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title(r"A_R=5e7, $\mu=1$", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()


if scenario == "matched_detectability_pmulti":

### ROC curve with matched matched trigger rate (as seen by an adversary with a number resolving detector) ###

	param = SetupParameters(
		fock_space_dim=10,
		output_power=None,
		trigger_rate=None,
		multi_photon_probability=None,
		sp_collection=0.2,
		sp_p1=None,
		sp_p2=None,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
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

	#A_R/P_0 = 0.99 & A_R = 1e6 & P_0 = 1.0101e6
	param_laser = deepcopy(param)
	param_laser.output_power = 31521163
	param_laser.multi_photon_probability=0.8
	param_sps = deepcopy(param)
	param_sps.output_power = 10976012
	param_sps.multi_photon_probability = 0.8
	param_sps.sp_p1, param_sps.sp_p2 = 0.2, 0.8
	param_eps = deepcopy(param)
	param_eps.output_power = 26944575
	param_eps.multi_photon_probability = 0.8

	range_interval = 50

	### PULSED LASER ###

	signal_laser = PulsedLaser(param_laser).signal_rate()
	noise_laser = PulsedLaser(param_laser).noise_rate()
	trigger_rate_laser = PulsedLaser(param_laser).effective_trigger_rate()

	alpha = PulsedLaser(param_laser).compute_alpha()
	detectability = (param_laser.output_power/(1))*(1-np.exp(-alpha**2))/(alpha**2)
	print(f"A_R laser : {detectability}")

	true_positive_laser, false_positive_laser = RocAnalysis(
		signal_rate=signal_laser,
		noise_rate=noise_laser,
		trigger_rate=trigger_rate_laser,
		threshold_limit=trigger_rate_laser/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### SINGLE PHOTON ###
	signal_sp = SinglePhoton(param_sps).signal_rate()
	noise_sp = SinglePhoton(param_sps).noise_rate()
	trigger_rate_sp = SinglePhoton(param_sps).effective_trigger_rate()

	trigger_rate = SinglePhoton(param_sps).effective_trigger_rate()
	pg = 0.2
	pgg = 0.8
	eta = 0.2

	detectability = param_sps.output_power*(1-(1-pg-pgg)-(1-eta)*pg-((1-eta)**2)*pgg)/((pg+2*pgg)*eta)
	print(f"A_R/Pout single photon : {detectability}")

	true_positive_sps, false_positive_sps = RocAnalysis(
		signal_rate=signal_sp,
		noise_rate=noise_sp,
		trigger_rate=trigger_rate_sp,
		threshold_limit=trigger_rate_sp/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	### Entangled Source ###
	signal_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).signal_rate()
	noise_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).noise_rate()
	trigger_rate_spdc = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).effective_trigger_rate()

	epsilon = EntangledPhotonSPDC(param_eps, adjust_eps_rate=True).epsilon
	detectability = param_eps.output_power*(1-(1/((epsilon+1)-epsilon*(1-eta))))/(epsilon*eta)
	print(f"A_R/Pout entangled : {detectability}")

	true_positive_spdc, false_positive_spdc = RocAnalysis(
		signal_rate=signal_spdc,
		noise_rate=noise_spdc,
		trigger_rate=trigger_rate_spdc,
		threshold_limit=trigger_rate_spdc/1000,
		range_interval=range_interval,
		timing_window=param["timing_window"]
	).compute_p_d_p_fa()

	plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

	fig = plt.figure(figsize=(16.1, 10))

	plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
	plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
	plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
	plt.xlabel("False positive", fontsize=22)
	plt.ylabel("True positive", fontsize=22)
	plt.title(r"A_R=1e7, $P_{multi}=0.8$", fontsize=22, fontweight="bold")
	plt.legend(fontsize=22, frameon=False)
	plt.tick_params(labelsize=22)
	plt.show()


if scenario == "matched_multi_photon":

	### Match Multi-photon probability with source & entangled###

	param = SetupParameters(
		fock_space_dim=10,
		output_power=2e7,
		trigger_rate=None,
		multi_photon_probability=1e-4,
		sp_collection=0.2,
		sp_p1=0.999,
		sp_p2=1e-4,
		spdc_eps_heralding=0.18,
		spdc_eps_collection=0.2,
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
	range_interval=50

	### Pulsed Laser ###


	signal = PulsedLaser(param).signal_rate()
	noise = PulsedLaser(param).noise_rate()
	trigger_rate = PulsedLaser(param).effective_trigger_rate()

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
	trigger_rate = SinglePhoton(param).effective_trigger_rate()

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
	trigger_rate = EntangledPhotonSPDC(param, adjust_eps_rate=True).effective_trigger_rate()


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

