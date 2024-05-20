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

collection_efficiency = np.linspace(0.05, 1, 6000)
snr_all_entangled = []

for collec_eff in tqdm(collection_efficiency):
	param.spdc_eps_heralding = collec_eff * param["detection_efficiency"]
	param.spdc_eps_collection = collec_eff
	snr = EntangledPhotonSPDC(param).signal_to_noise_rate()
	snr_all_entangled.append(snr)


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


### Multi-Graph ###

# Comparaison with single photon source

idx = (np.abs(collection_efficiency_sps - 0.05)).argmin()
snr_005_sps = snr_all_sps[idx]
idx = (np.abs(collection_efficiency_sps - 0.2)).argmin()
snr_02_sps = snr_all_sps[idx]
idx = (np.abs(collection_efficiency_sps - 1)).argmin()
snr_1_sps = snr_all_sps[idx]

# Comparaison with multi-photon prob

idx = (np.abs(multi_photon_prob - 0.001)).argmin()
snr_0001_laser = snr_all_laser[idx]
idx = (np.abs(multi_photon_prob - 0.01)).argmin()
snr_001_laser = snr_all_laser[idx]
idx = (np.abs(multi_photon_prob - 0.1)).argmin()
snr_01_laser = snr_all_laser[idx]
idx = (np.abs(multi_photon_prob - multi_100)).argmin()
snr_1_laser = snr_all_laser[idx]

# Comparaison with entangled source
idx = (np.abs(collection_efficiency - 0.05)).argmin()
snr_005_entangled = snr_all_entangled[idx]

idx = (np.abs(collection_efficiency - 0.2)).argmin()
snr_02_entangled = snr_all_entangled[idx]

idx = (np.abs(collection_efficiency - 1)).argmin()
snr_1_entangled = snr_all_entangled[idx]

# 1) SNR of entangled photon source as a subplot

fig, ax = plt.subplots(1, 2, figsize=(16.1, 30))

ax[0].plot(collection_efficiency, snr_all_entangled, "-", linewidth=2, label="Entangled Photon Source")
ax[0].hlines(snr_005_sps, 0, 1, colors="b", linestyles="--", label="Single Photon Source", linewidth=3)
ax[0].text(0.25, snr_005_sps+0.5, "Collection efficiency of SPS=5%", fontsize=16)
ax[0].hlines(snr_02_sps, 0, 1, colors="b", linestyles="--", linewidth=3)
ax[0].text(0.25, snr_02_sps+0.5, "Collection efficiency of SPS=20%", fontsize=16)
ax[0].hlines(snr_1_sps, 0, 1, colors="b", linestyles="--", linewidth=3)
ax[0].text(0, snr_1_sps-2, "Collection efficiency of SPS=100%", fontsize=16)
#ax[0].set_xlabel("Collection efficiency [-]", fontsize=22)
#ax[0].set_ylabel("SNR [-]", fontsize=22)
ax[0].tick_params(axis='both', which='major', labelsize=22)
ax[0].legend(fontsize=18, frameon=False, loc="upper left")

# Intersection with multi-photon probability

max_snr_entangled = max(snr_all_entangled)
idx = (np.abs(np.array(snr_all_laser) - max_snr_entangled)).argmin()
multi_prob_match_entangled = multi_photon_prob[idx]
intersection_snr_entangled = snr_all_laser[idx]


ax[1].plot(collection_efficiency, snr_all_entangled, "-", linewidth=2, label="Entangled Photon Source")
ax[1].hlines(snr_0001_laser, 0, 1, colors="r", linestyles="--", label="Pulsed Laser", linewidth=3)
ax[1].text(0.12, snr_0001_laser+1, "Multi-photon probability of laser=0.001", fontsize=16)
ax[1].hlines(snr_001_laser, 0, 1, colors="r", linestyles="--", linewidth=3)
ax[1].text(0.18, snr_001_laser+1, "Multi-photon probability of laser=0.01", fontsize=16)
ax[1].hlines(snr_01_laser, 0, 1, colors="r", linestyles="--", linewidth=3)
ax[1].text(0.5, snr_01_laser-5, "Multi-photon probability\nof laser=0.1", fontsize=16)
ax[1].hlines(intersection_snr_entangled, 0, 1, colors="r", linestyles="--", linewidth=3)
ax[1].text(0, intersection_snr_entangled-4, f"Multi-photon probability of laser={multi_prob_match_entangled:.2f}\n", fontsize=16)
#ax[1].hlines(snr_1_laser, 0, 1, colors="r", linestyles="--", linewidth=3)
#ax[1].text(0, snr_1_laser-7, f"Multi-photon probability\nof laser={multi_100:.3f}\nSPS : 1 photon/pulse", fontsize=16)
#ax[1].set_xlabel("Collection efficiency [-]", fontsize=22)
#ax[1].set_ylabel("SNR [-]", fontsize=22)
ax[1].tick_params(axis='both', which='major', labelsize=22)
ax[1].legend(fontsize=18, frameon=False, loc="center left", bbox_to_anchor=(0, 0.75))

fig.text(0.5, 0.04, 'Collection efficiency [-]', ha='center', fontsize=22)
fig.text(0.04, 0.5, 'SNR [-]', va='center', rotation='vertical', fontsize=22)

plt.show()

# 2) SNR of single photon source as a subplot

snr_max_sps = max(snr_all_sps)
idx = (np.abs(np.array(snr_all_entangled) - snr_max_sps)).argmin()
eff_entangled = collection_efficiency[idx]
snr_entangled_match = snr_all_entangled[idx]


fig, ax = plt.subplots(1, 2, figsize=(16.1, 30))

ax[0].plot(collection_efficiency_sps, snr_all_sps, "-", linewidth=2, label="Single Photon Source")
ax[0].hlines(snr_005_entangled, 0, 1, colors="g", linestyles="--", label="Entangled Photon Source", linewidth=3)
ax[0].text(0.4, snr_005_entangled+1, "Collection efficiency of\nentangled source=5%", fontsize=16)
ax[0].hlines(snr_02_entangled, 0, 1, colors="g", linestyles="--", linewidth=3)
ax[0].text(0.4, snr_02_entangled+1, "Collection efficiency of\nentangled source=20%", fontsize=16)
ax[0].hlines(snr_entangled_match, 0, 1, colors="g", linestyles="--", linewidth=3)
ax[0].text(0, snr_entangled_match-4, f"Collection efficiency of\nentangled source={eff_entangled*100:.2f}%", fontsize=16)
#ax[0].hlines(snr_1_entangled, 0, 1, colors="g", linestyles="--", linewidth=3)
#ax[0].text(0, snr_1_entangled-5, "Collection efficiency of\nentangled source=100%", fontsize=16)
#ax[0].set_xlabel("Collection efficiency [-]", fontsize=22)
#ax[0].set_ylabel("SNR [-]", fontsize=22)
ax[0].tick_params(axis='both', which='major', labelsize=22)
ax[0].legend(fontsize=18, frameon=False, loc="center left", bbox_to_anchor=(0, 0.8))

ax[1].plot(collection_efficiency_sps, snr_all_sps, "-", linewidth=2, label="Single Photon Source")
ax[1].hlines(snr_0001_laser, 0, 1, colors="r", linestyles="--", label="Pulsed Laser", linewidth=3)
ax[1].text(0.12, snr_0001_laser+1, "Multi-photon probability of laser=0.001", fontsize=16)
ax[1].hlines(snr_001_laser, 0, 1, colors="r", linestyles="--", linewidth=3)
ax[1].text(0.19, snr_001_laser+0.5, "Multi-photon probability of laser=0.01", fontsize=16)
ax[1].hlines(snr_01_laser, 0, 1, colors="r", linestyles="--", linewidth=3)
ax[1].text(0.5, snr_01_laser-4, "Multi-photon probability\nof laser=0.1", fontsize=16)
ax[1].hlines(snr_1_laser, 0, 1, colors="r", linestyles="--", linewidth=3)
ax[1].text(0, snr_1_laser-5, f"Multi-photon probability\nof laser={multi_100:.3f}\nSPS : 1 photon/pulse", fontsize=16)
#ax[1].set_xlabel("Collection efficiency [-]", fontsize=22)
#ax[1].set_ylabel("SNR [-]", fontsize=22)
ax[1].tick_params(axis='both', which='major', labelsize=22)
ax[1].legend(fontsize=18, frameon=False, loc="center left", bbox_to_anchor=(0, 0.7))

fig.text(0.5, 0.04, 'Collection efficiency [-]', ha='center', fontsize=22)
fig.text(0.04, 0.5, 'SNR [-]', va='center', rotation='vertical', fontsize=22)

plt.show()

# 3) SNR of pulsed laser source as a subplot

fig, ax = plt.subplots(1, 2, figsize=(16.1, 30))

ax[0].semilogy(multi_photon_prob, snr_all_laser, "-", linewidth=2, label="Pulsed Laser")
ax[0].hlines(snr_005_sps, 0, 0.38, colors="b", linestyles="--", label="Single Photon Source", linewidth=3)
ax[0].text(0.01, snr_005_sps+0.2, "Collection efficiency\nof SPS=5%", fontsize=16)
ax[0].hlines(snr_20, 0, 0.38, colors="b", linestyles="--", linewidth=3)
ax[0].text(0.0435, snr_20+1, "Collection efficiency\nof SPS=20%", fontsize=16)
ax[0].hlines(snr_100, 0, 0.38, colors="b", linestyles="--", linewidth=3)
ax[0].text(-0.016, snr_100+2, "Collection efficiency of SPS=100%", fontsize=16)
ax[0].text(0.265, 28.5, f"Probability of\nmulti-photons\n={multi_100:.3f}", fontsize=14)
#ax[0].set_xlabel("Multi-Photon probability [-]", fontsize=22)
#ax[0].set_ylabel("Log(SNR) [-]", fontsize=22)
ax[0].tick_params(axis='both', which='major', labelsize=25)
ax[0].legend(fontsize=18, frameon=False, loc="lower right")
ax[0].set_ylim([0.8, 60])

ax[1].semilogy(multi_photon_prob, snr_all_laser, "-", linewidth=2, label="Pulsed Laser")
ax[1].hlines(snr_005_entangled, 0, 0.38, colors="g", linestyles="--", label="Entangled Photon Source", linewidth=3)
ax[1].text(0.03, snr_005_entangled+0.2, "Collection efficiency of\nentangled source=5%", fontsize=16)
ax[1].hlines(snr_02_entangled, 0, 0.38, colors="g", linestyles="--", linewidth=3)
ax[1].text(0.05, snr_02_entangled+1, "Collection efficiency of\nentangled source=20%", fontsize=16)
ax[1].hlines(snr_1_entangled, 0, 0.38, colors="g", linestyles="--", linewidth=3)
ax[1].text(-0.01, snr_1_entangled-13, "Collection efficiency of\nentangled source=100%", fontsize=16)
#ax[1].set_xlabel("Multi-Photon probability [-]", fontsize=22)
#ax[1].set_ylabel("Log(SNR) [-]", fontsize=22)
ax[1].tick_params(axis='both', which='major', labelsize=25)
ax[1].legend(fontsize=18, frameon=False, loc="lower right")
ax[1].set_ylim([0.8, 60])

fig.text(0.5, 0.04, 'Multi-Photon probability [-]', ha='center', fontsize=22)
fig.text(0.04, 0.5, 'Log(SNR) [-]', va='center', rotation='vertical', fontsize=22)

plt.show()




