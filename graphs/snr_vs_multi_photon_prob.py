from Sources import EntangledPhotonSPDC, SinglePhoton, PulsedLaser, SetupParameters
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

param = SetupParameters(
	fock_space_dim=8,
	output_power=2e6,
	trigger_rate=None,
	multi_photon_probability=None,
	sp_collection=0.2,
	sp_p1=0.9999,
	sp_p2=1e-4,
	spdc_eps_heralding=0.05,
	spdc_eps_collection=0.2,
	target_distance=1,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.5,
	background=400,
	detector_dark=200,
	timing_window=0.5e-9,
)

param_sps = deepcopy(param)

multi_photon_probability = np.linspace(0.005, 0.8, 1500)



snr_laser = []
trigger_rate_laser = []
snr_sps = []
trigger_rate_sps = []
snr_eps = []
trigger_rate_eps = []

for mpp in tqdm(multi_photon_probability):
	param["multi_photon_probability"] = mpp
	param_sps["multi_photon_probability"] = mpp
	param_sps["sp_p1"] = 1-mpp
	param_sps["sp_p2"] = mpp
	laser = PulsedLaser(param)
	sps = SinglePhoton(param_sps)
	eps = EntangledPhotonSPDC(param, adjust_eps_rate=False)
	snr_laser.append(laser.signal_to_noise_rate())
	snr_sps.append(sps.signal_to_noise_rate())
	snr_eps.append(eps.signal_to_noise_rate())
	trigger_rate_laser.append(laser.effective_trigger_rate())
	trigger_rate_sps.append(sps.effective_trigger_rate())
	trigger_rate_eps.append(eps.effective_trigger_rate())


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.plot(multi_photon_probability, snr_laser, "-", label="Pulsed Laser", linewidth=2.5)
plt.plot(multi_photon_probability, snr_sps, "--", label="Single Photon", linewidth=2.5)
plt.plot(multi_photon_probability, snr_eps, "-.", label="Entangled Photon SPDC", linewidth=2.5)
plt.xlabel("Multi-photon probability [-]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.legend(frameon=False, fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

plt.plot(trigger_rate_laser, snr_laser, "--", label="Pulsed Laser", linewidth=2.5)
plt.plot(trigger_rate_sps, snr_sps, "-", label="Single Photon", linewidth=2.5, color="red")
plt.plot(trigger_rate_eps, snr_eps, "-.", label="Entangled Photon SPDC", linewidth=2.5)
plt.xlabel("Effective Trigger Rate [Hz]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.legend(frameon=False, fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()


