import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=35,
	output_power=400000,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	number_nv_pulse=None,
	sp_collection=None,
	sp_p1=0.99,
	sp_p2=5e-3,
	spdc_eps_heralding=None,
	spdc_eps_collection=None,
	atmosphere=0.5,
	target_distance=2,
	receiver_diameter=0.05,
	target_albedo=0.5,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.7,
	background=250000,
	detector_dark=25,
	timing_window=0.5e-9,
)

collection_efficiency = np.array([0.35, 0.57, 0.8, 1])
#collection_efficiency = np.array([0.57, 0.8, 1])
# *** Single Photon Source ***

param_sps = deepcopy(param)
number_nv_pulse_sps = np.zeros_like(collection_efficiency)
snr_sps = np.zeros_like(collection_efficiency)
signal_sps = np.zeros_like(collection_efficiency)
noise_sps = np.zeros_like(collection_efficiency)
trigger_rate_sps = np.zeros_like(collection_efficiency)
average_photon_per_pulse_sps = np.zeros_like(collection_efficiency)

for idx, ce in enumerate(collection_efficiency):
	param_sps["sp_collection"] = ce
	extr_eff = ce * param["optics_transmitter"]
	sps = SinglePhoton(param_sps)
	number_nv_pulse_sps[idx] = sps.number_nv_pulse
	snr_sps[idx] = sps.signal_to_noise_rate()
	average_photon_per_pulse_sps[idx] = sps.average_photon_per_pulse
	signal_sps[idx] = sps.signal_rate()
	noise_sps[idx] = sps.noise_rate()
	trigger_rate_sps[idx] = sps.trigger_rate

mini = 50000
maxi = 395000

non_vacuum_number = np.linspace(mini, maxi, 100)
non_vacuum_number = np.concatenate((non_vacuum_number, number_nv_pulse_sps))
other = np.array([75000, 100000, 150000, 200000, 250000, 300000, 350000, 375000])
non_vacuum_number = np.concatenate((non_vacuum_number, other))
non_vacuum_number = np.sort(non_vacuum_number)[::-1]

# *** Pulsed Laser ***

param_laser = deepcopy(param)
snr_laser = np.zeros_like(non_vacuum_number)
signal_laser = np.zeros_like(non_vacuum_number)
noise_laser = np.zeros_like(non_vacuum_number)
trigger_rate_laser = np.zeros_like(non_vacuum_number)
average_photon_per_pulse_laser = np.zeros_like(non_vacuum_number)

for idx, nvp in enumerate(tqdm(non_vacuum_number)):
	param_laser["number_nv_pulse"] = nvp
	laser = PulsedLaser(param_laser)
	snr_laser[idx] = laser.signal_to_noise_rate()
	average_photon_per_pulse_laser[idx] = laser.average_photon_per_pulse
	signal_laser[idx] = laser.signal_rate()
	noise_laser[idx] = laser.noise_rate()
	trigger_rate_laser[idx] = laser.trigger_rate

# *** Entangled Photon Source ***

param_eps = deepcopy(param)
snr_entangled = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
signal_sps = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
noise_sps = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
trigger_rate_sps = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
average_photon_per_pulse_entangled = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))

last_state = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))

# Assert that the fock space dimension is large enough to accommodate the average photon per pulse.
# This allows for a more progressive increase in the fock space dimension as the average photon per pulse increases.
# This adhoc function allows that the vector |n> does not truncate information.


def adjustable_fock_space(average_photon_per_pulse):
	return round(average_photon_per_pulse * 3.5) + 10


for idx_nvp, ce in enumerate(collection_efficiency):
	param_eps["spdc_eps_collection"] = ce
	param_eps["spdc_eps_heralding"] = ce * param_eps["detection_efficiency"]

	for idx_ce, nvp in enumerate(tqdm(non_vacuum_number)):
		param_eps["number_nv_pulse"] = nvp
		eps = EntangledPhotonSPDC(param_eps)
		average_photon_per_pulse_entangled[idx_ce, idx_nvp] = eps.average_photon_per_pulse
		param_eps["fock_space_dim"] = adjustable_fock_space(average_photon_per_pulse_entangled[idx_ce, idx_nvp])
		eps = EntangledPhotonSPDC(param_eps)
		print(param_eps["fock_space_dim"], eps.prob_of_last_element_fock_space)
		snr_entangled[idx_ce, idx_nvp] = eps.signal_to_noise_rate()
		last_state[idx_ce, idx_nvp] = eps.prob_of_last_element_fock_space
		signal_sps[idx_ce, idx_nvp] = eps.signal_rate()
		noise_sps[idx_ce, idx_nvp] = eps.noise_rate()
		trigger_rate_sps[idx_ce, idx_nvp] = eps.trigger_rate
		#print(eps.prob_of_last_element_fock_space)


result = {
	"sps": {
		"0.35": {
			"signal": signal_sps[0],
			"noise": noise_sps[0],
			"snr": snr_sps[0],
			"trigger_rate": trigger_rate_sps[0],
			"average_photon_per_pulse": average_photon_per_pulse_sps[0],
		},
		"0.57": {
			"signal": signal_sps[1],
			"noise": noise_sps[1],
			"snr": snr_sps[1],
			"trigger_rate": trigger_rate_sps[1],
			"average_photon_per_pulse": average_photon_per_pulse_sps[1],
		},
		"0.8": {
			"signal": signal_sps[2],
			"noise": noise_sps[2],
			"snr": snr_sps[2],
			"trigger_rate": trigger_rate_sps[2],
			"average_photon_per_pulse": average_photon_per_pulse_sps[2],
		},
		"1": {
			"signal": signal_sps[3],
			"noise": noise_sps[3],
			"snr": snr_sps[3],
			"trigger_rate": trigger_rate_sps[3],
			"average_photon_per_pulse": average_photon_per_pulse_sps[3],
		},
	},
	"laser": {
		"signal": signal_laser,
		"noise": noise_laser,
		"snr": snr_laser,
		"trigger_rate": trigger_rate_laser,
		"average_photon_per_pulse": average_photon_per_pulse_laser,
	},
	"eps": {
		"0.35": {
			"signal": signal_sps[:, 0],
			"noise": noise_sps[:, 0],
			"snr": snr_entangled[:, 0],
			"trigger_rate": trigger_rate_sps[:, 0],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 0],
			"last_state": last_state[:, 0],
		},
		"0.57": {
			"signal": signal_sps[:, 1],
			"noise": noise_sps[:, 1],
			"snr": snr_entangled[:, 1],
			"trigger_rate": trigger_rate_sps[:, 1],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 1],
			"last_state": last_state[:, 1],
		},
		"0.8": {
			"signal": signal_sps[:, 2],
			"noise": noise_sps[:, 2],
			"snr": snr_entangled[:, 2],
			"trigger_rate": trigger_rate_sps[:, 2],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 2],
			"last_state": last_state[:, 2],
		},
		"1": {
			"signal": signal_sps[:, 3],
			"noise": noise_sps[:, 3],
			"snr": snr_entangled[:, 3],
			"trigger_rate": trigger_rate_sps[:, 3],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 3],
			"last_state": last_state[:, 3],
		},
	},
	"param": param,
	"non_vacuum_number": non_vacuum_number,
	"collection_efficiency": collection_efficiency,
}

np.save("2m_snr_matching_non_vacuum_pulse_2025_06_22.npy", result)


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig, ax1 = plt.subplots(figsize=(16.2, 10))

ax1.semilogy(non_vacuum_number, snr_entangled[:, 0], "-.", label="Entangled Photon Source", color="green",
             linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_entangled[:, 1], "-.", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_entangled[:, 2], "-.", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_entangled[:, 3], "-.", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_laser, "-", label="Pulsed Laser", color="blue", linewidth=3, zorder=1)
ax1.scatter(number_nv_pulse_sps, snr_sps, label="Single Photon Source", color="red", s=100, zorder=2)

ax1.axvspan(50000, 100000, color="lightblue", alpha=0.35, label="Optimal regime against a NNRD", zorder=0)
ax1.axvspan(350000, 400000, color="lightcoral", alpha=0.35, label="Optimal regime against a NRD", zorder=0)

ax1.set_xlabel("Number of non-vacuum pulse", fontsize=22)
ax1.set_ylabel("SNR", fontsize=22)
ax1.set_xlim([None, 450000])
ax1.legend(frameon=False, fontsize=22)
ax1.tick_params(axis="x", labelsize=22)
ax1.tick_params(axis="y", labelsize=22)

x_pos = 4e5 + 0.05e5
ax1.text(x_pos, snr_entangled[0, 0] - 1.75, "20%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_entangled[0, 1] - 5, "57%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_entangled[0, 2] - 5, "80%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_entangled[0, 3], "100%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_entangled[0, 3] + 30, r"$\eta_{signal}$", fontsize=28, fontweight="bold")
def top_axis_transform(x):
    x = np.array(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x != 0, param["output_power"] / x, 0)
    return result

def top_axis_inverse(x):
    x = np.array(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x != 0, param["output_power"] / x, 0)
    return result

xticks = ax1.get_xticks()
xticks_second = top_axis_transform(xticks)[::-1]

secax = ax1.secondary_xaxis('top', functions=(top_axis_transform, top_axis_inverse))
secax.set_xlabel("Average photon per non-vacuum pulse", fontsize=22)
secax.set_ticks(xticks_second)
secax.tick_params(axis="x", labelsize=22)

plt.show()