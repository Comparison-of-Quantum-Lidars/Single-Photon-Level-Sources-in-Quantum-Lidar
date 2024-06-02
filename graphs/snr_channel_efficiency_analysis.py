import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

param = SetupParameters(
	fock_space_dim=5,
	output_power=2e6,
	trigger_rate=None,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=1,
	sp_p1=0.99,
	sp_p2=1e-3,
	spdc_eps_heralding=1,
	spdc_eps_collection=1,
	channel_efficiency=None,
	target_distance=1,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=1,
	background=400,
	detector_dark=200,
	timing_window=0.5e-9,
)


##### MATCHING THE MULTI-PHOTON PROBABILITY #####

param_match_multi_photon = deepcopy(param)

channel_efficiency = np.linspace(0.05, 1, 6000)

multi_photon_probability = []
snr_laser = []
snr_sps = []
snr_eps = []

average_photon_eps = []
average_photon_sps = []
average_photon_laser = []

for ce in tqdm(channel_efficiency):
	param_match_multi_photon.channel_efficiency = ce
	param_match_multi_photon.multi_photon_probability = param_match_multi_photon["sp_p2"] * ((param_match_multi_photon.channel_efficiency * param_match_multi_photon["sp_collection"])**2)
	laser = PulsedLaser(param_match_multi_photon).signal_to_noise_rate()
	sps = SinglePhoton(param_match_multi_photon).signal_to_noise_rate()
	eps = EntangledPhotonSPDC(param_match_multi_photon).signal_to_noise_rate()
	multi_photon_probability.append(param_match_multi_photon.multi_photon_probability)
	snr_laser.append(laser)
	snr_sps.append(sps)
	snr_eps.append(eps)
	average_photon_eps.append(EntangledPhotonSPDC(param_match_multi_photon).epsilon)
	average_photon_sps.append(SinglePhoton(param_match_multi_photon).sp_p1 + 2*SinglePhoton(param_match_multi_photon).sp_p2)
	average_photon_laser.append((PulsedLaser(param_match_multi_photon).compute_alpha())**2)

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.plot(channel_efficiency, snr_laser, "-", label="Pulsed Laser", linewidth=3, color="blue")
plt.plot(channel_efficiency, snr_sps, "--", label="Single Photon Source", linewidth=3, color="red")
plt.plot(channel_efficiency, snr_eps, "-.", label="Entangled Photon Source", linewidth=3, color="green")
plt.xlabel("Channel Efficiency [-]", fontsize=22)
plt.ylabel("Signal to Noise Ratio [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.legend(frameon=False, fontsize=22)
plt.show()

plt.plot(channel_efficiency, multi_photon_probability, "-", label="Multi-Photon Probability", linewidth=3, color="black")
plt.xlabel("Channel Efficiency [-]", fontsize=22)
plt.ylabel("Multi-Photon Probability [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
#plt.legend(frameon=False, fontsize=22)
plt.show()

plt.plot(channel_efficiency, average_photon_laser, "-", label="Pulsed Laser", linewidth=3, color="blue")
plt.plot(channel_efficiency, average_photon_sps, "--", label="Single Photon Source", linewidth=3, color="red")
plt.plot(channel_efficiency, average_photon_eps, "-.", label="Entangled Photon Source", linewidth=3, color="green")
plt.xlabel("Channel Efficiency [-]", fontsize=22)
plt.ylabel("Average Photon Number [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.legend(frameon=False, fontsize=22)
plt.show()

### Match No-Vacuum Probability ###

param_match_no_vacuum = deepcopy(param)

channel_efficiency = np.linspace(0.05, 1, 6000)
snr_laser = []
snr_sps = []
snr_eps = []
no_vacuum_probability = []

average_photon_eps = []
average_photon_sps = []
average_photon_laser = []


for ce in tqdm(channel_efficiency):
	param_match_no_vacuum.channel_efficiency = ce
	param_match_no_vacuum.no_vacuum_probability = param_match_no_vacuum["sp_p1"] * ce * param_match_no_vacuum["sp_collection"] + param_match_no_vacuum["sp_p2"] * ce * param_match_no_vacuum["sp_collection"] * (2 - (ce * param_match_no_vacuum["sp_collection"]))
	laser = PulsedLaser(param_match_no_vacuum).signal_to_noise_rate()
	sps = SinglePhoton(param_match_no_vacuum).signal_to_noise_rate()
	eps = EntangledPhotonSPDC(param_match_no_vacuum).signal_to_noise_rate()
	no_vacuum_probability.append(param_match_no_vacuum.no_vacuum_probability)
	snr_laser.append(laser)
	snr_sps.append(sps)
	snr_eps.append(eps)
	average_photon_eps.append(EntangledPhotonSPDC(param_match_no_vacuum).epsilon)
	average_photon_sps.append(SinglePhoton(param_match_no_vacuum).sp_p1 + 2*SinglePhoton(param_match_no_vacuum).sp_p2)
	average_photon_laser.append((PulsedLaser(param_match_no_vacuum).compute_alpha())**2)

plt.plot(channel_efficiency, no_vacuum_probability, "-", label="No Vacuum Probability", linewidth=3, color="black")
plt.xlabel("Channel Efficiency [-]", fontsize=22)
plt.ylabel("No Vacuum Probability [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
#plt.legend(frameon=False, fontsize=22)
plt.show()


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.plot(channel_efficiency, snr_laser, "-", label="Pulsed Laser", linewidth=3, color="blue")
plt.plot(channel_efficiency, snr_sps, "--", label="Single Photon Source", linewidth=3, color="red")
plt.plot(channel_efficiency, snr_eps, "-.", label="Entangled Photon Source", linewidth=3, color="green")
plt.xlabel("Channel Efficiency [-]", fontsize=22)
plt.ylabel("Signal to Noise Ratio [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.xlim([0, 0.7])
plt.legend(frameon=False, fontsize=22)
plt.show()

plt.plot(channel_efficiency, average_photon_laser, "-", label="Pulsed Laser", linewidth=3, color="blue")
plt.plot(channel_efficiency, average_photon_sps, "--", label="Single Photon Source", linewidth=3, color="red")
plt.plot(channel_efficiency, average_photon_eps, "-.", label="Entangled Photon Source", linewidth=3, color="green")
plt.xlabel("Channel Efficiency [-]", fontsize=22)
plt.ylabel("Average Photon Number [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=22)
plt.xlim([0, 0.7])
plt.legend(frameon=False, fontsize=22)
plt.show()