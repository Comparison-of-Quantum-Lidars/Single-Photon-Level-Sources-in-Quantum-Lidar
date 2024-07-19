from Sources import EntangledPhotonSPDC, SinglePhoton, PulsedLaser, SetupParameters
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

param = SetupParameters(
	fock_space_dim=8,
	output_power=2e6,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	sp_collection=0.1,
	sp_p1=0.99,
	sp_p2=1e-3,
	spdc_eps_heralding=1,
	spdc_eps_collection=1,
	atmosphere=1,
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

multi_photon_probability = np.linspace(0.005, 0.8, 1500)

snr_laser = []
snr_eps = []

for mpp in tqdm(multi_photon_probability):
	param.multi_photon_probability = mpp
	snr = PulsedLaser(param).signal_to_noise_rate()
	snr_laser.append(snr)
	snr = EntangledPhotonSPDC(param).signal_to_noise_rate()
	snr_eps.append(snr)


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.plot(multi_photon_probability, snr_laser, "-", label="Pulsed Laser", linewidth=2.5, color="blue")
plt.plot(multi_photon_probability, snr_eps, "-.", label="Entangled Photon SPDC", linewidth=2.5, color="green")
plt.xlabel("Multi-photon probability [-]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.legend(frameon=False, fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()



