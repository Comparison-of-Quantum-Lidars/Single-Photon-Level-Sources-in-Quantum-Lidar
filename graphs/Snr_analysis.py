import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm

param = SetupParameters(
	fock_space_dim=5,
	output_power=2e6,
	laser_rate=5e6,
	sp_collection=0.1,
	sp_p1=0.99,
	sp_p2=1e-3,
	spdc_eps_heralding=0.5,
	spdc_eps_collection=0.2,
	spdc_emission=0.1,
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

### SNR SPS as a function of collection efficiency ###

collection_efficiency_sps = np.linspace(0.05, 1, 1000)
snr_all_sps = []
for eff in tqdm(collection_efficiency_sps):
	param.sp_collection = eff
	signal = SinglePhoton(param).signal_rate()
	noise = SinglePhoton(param).noise_rate()
	snr = (signal)/noise
	snr_all_sps.append(snr)


### SNR Pulsed Laser as a function of alpha ###
alpha = np.linspace(0.05, 1.15, 6000)
multi_photon_prob = 1-(np.exp(-alpha**2)*(1+alpha**2))
snr_all_laser = []

for a in tqdm(alpha):
	output_power = param["laser_rate"]*(a**2)
	param.output_power = output_power
	signal = PulsedLaser(param).signal_rate()
	noise = PulsedLaser(param).noise_rate()
	snr = signal/noise
	snr_all_laser.append(snr)

### SNR Entangled Source as a function of eps_heralding ###
#eps_heralding = np.linspace(0.001, 1, 3000)
#snr_all_spdc = []

#for eps in tqdm(eps_heralding):
#	param.spdc_eps_heralding = eps
#	signal = EntangledPhotonSPDC(param).signal_rate()
#	noise = EntangledPhotonSPDC(param).noise_rate()
#	snr = signal/noise
#	snr_all_spdc.append(snr)


### GRAPHS ###

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

# 1) SNR SPS as a function of collection efficiency

idx = (np.abs(collection_efficiency_sps - 0.05)).argmin()
eff_5 = collection_efficiency_sps[idx]
snr_5 = snr_all_sps[idx]

idx = (np.abs(collection_efficiency_sps - 0.2)).argmin()
eff_20 = collection_efficiency_sps[idx]
snr_20 = snr_all_sps[idx]

idx = (np.abs(collection_efficiency_sps - 1)).argmin()
eff_100 = collection_efficiency_sps[idx]
snr_100 = snr_all_sps[idx]

# intersection between snr_100 and snr_all and find multi-photon probability
idx = (np.abs(np.array(snr_all_laser) - np.array(snr_100))).argmin()
multi_100 = multi_photon_prob[idx]
intersection_snr_100 = snr_all_laser[idx]

plt.figure(figsize=(16.1, 10))
plt.semilogy(multi_photon_prob, snr_all_laser, "-", linewidth=2, label="Pulsed Laser")
plt.hlines(snr_5, 0, 0.38, colors="b", linestyles="--", label="Single Photon Source", linewidth=3)
plt.text(0.1, snr_5+0.1, "Collection efficiency of SPS=5%", fontsize=16)
plt.hlines(snr_20, 0, 0.38, colors="b", linestyles="--", linewidth=3)
plt.text(0.1, snr_20+0.5, "Collection efficiency of SPS=20%", fontsize=16)
plt.hlines(snr_100, 0, 0.38, colors="b", linestyles="--", linewidth=3)
plt.text(0.1, snr_100+2, "Collection efficiency of SPS=100%", fontsize=16)
plt.vlines(multi_100, 1e-2, intersection_snr_100, colors="k", linestyles="-.", linewidth=2)
plt.text(0.265, 25.5, f"Probability of multi-photons={multi_100:.3f}\nSPS : 1 photon/pulse", fontsize=14)
plt.xlabel("Multi-Photon probability [-]", fontsize=22)
plt.ylabel("Log(SNR) [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=25)
#plt.title("Pulsed laser source", fontsize=22, fontweight="bold")
plt.legend(fontsize=18, frameon=False, loc="lower right")
plt.ylim([0.8, 60])
plt.show()


# 2) SNR Pulsed Laser as a function of the multi-photon probability

idx = (np.abs(multi_photon_prob - 0.001)).argmin()
mutli_0001 = multi_photon_prob[idx]
snr_0001 = snr_all_laser[idx]

idx = (np.abs(multi_photon_prob - 0.01)).argmin()
mutli_001 = multi_photon_prob[idx]
snr_001 = snr_all_laser[idx]

idx = (np.abs(multi_photon_prob - 0.1)).argmin()
mutli_01 = multi_photon_prob[idx]
snr_01 = snr_all_laser[idx]

idx = (np.abs(multi_photon_prob - multi_100)).argmin()
mutli_1 = multi_photon_prob[idx]
snr_1 = snr_all_laser[idx]

# put text above the horizontal line
plt.figure(figsize=(16.1, 10))
plt.plot(collection_efficiency_sps, snr_all_sps, "-", linewidth=2, label="Single Photon Source")
plt.hlines(snr_0001, 0, 1, colors="k", linestyles="--", label="Pulsed Laser", linewidth=2)
plt.text(0.6, snr_0001+1, "Multi-photon probability of laser=0.001", fontsize=16)
plt.hlines(snr_001, 0, 1, colors="k", linestyles="--", linewidth=2)
plt.text(0.6, snr_001+1, "Multi-photon probability of laser=0.01", fontsize=16)
plt.hlines(snr_01, 0, 1, colors="k", linestyles="--", linewidth=2)
plt.text(0.6, snr_01+1, "Multi-photon probability of laser=0.1", fontsize=16)
plt.hlines(snr_1, 0, 1, colors="k", linestyles="--", linewidth=2)
plt.text(0, snr_1-4, f"Multi-photon probability of laser={multi_100:.3f}\nSPS : 1 photon/pulse", fontsize=16)
plt.xlabel("Collection efficiency [-]", fontsize=22)
plt.ylabel("SNR [-]", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
#plt.title("Single photon source", fontsize=22, fontweight="bold")
plt.legend(fontsize=22, frameon=False, loc="best", bbox_to_anchor=(0.4, 0.5))
plt.show()




