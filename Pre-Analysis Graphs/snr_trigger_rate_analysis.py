# import numpy as np
# from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from copy import deepcopy
#
# param = SetupParameters(
# 	fock_space_dim=40,
# 	output_power=2e7,
# 	multi_photon_probability=None,
# 	no_vacuum_probability=None,
# 	sp_collection=0.2,
# 	sp_p1=0.99,
# 	sp_p2=1e-3,
# 	spdc_eps_heralding=1,
# 	spdc_eps_collection=1,
# 	atmosphere=1,
# 	target_distance=1,
# 	receiver_diameter=0.05,
# 	target_albedo=0.2,
# 	optics_transmitter=0.8,
# 	optics_receiver=0.5,
# 	detection_efficiency=1,
# 	background=400,
# 	detector_dark=200,
# 	timing_window=0.5e-9,
# )
#
#
# param_laser = deepcopy(param)
# param_eps = deepcopy(param)
#
# trigger_rate = np.linspace(1e6, 1e7, 50)
#
# snr_laser = []
# snr_eps = []
#
# for tr in tqdm(trigger_rate):
# 	param_laser.trigger_rate = tr
# 	param_eps.trigger_rate = tr
# 	snr_laser.append(PulsedLaser(param_laser).signal_to_noise_rate())
# 	eps = EntangledPhotonSPDC(param_eps)
# 	assert np.isclose(eps.compute_effective_trigger_rate(), tr), f"{eps.compute_effective_trigger_rate()} != {tr}"
# 	snr_eps.append(eps.signal_to_noise_rate())
#
# plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")
#
# fig = plt.figure(figsize=(16.1, 10))
#
# plt.plot(trigger_rate, snr_laser, "-", label="Pulsed Laser", color="blue", linewidth=2.5)
# plt.plot(trigger_rate, snr_eps, "-.", label="Entangled Photon", color="green", linewidth=2.5)
# plt.xlabel("Trigger Rate [Hz]", fontsize=22)
# plt.ylabel("Signal-to-Noise Ratio", fontsize=22)
# plt.xscale("log")
# plt.legend(frameon=False, fontsize=22)
# plt.tick_params(axis='both', which='major', labelsize=22)
#
# plt.show()