import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
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
	target_distance=3,
	receiver_diameter=0.05,
	target_albedo=0.5,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.7,
	background=250000,
	detector_dark=25,
	timing_window=0.5e-9,
)

number_sps = np.array([1, 2, 4, 8, 10])
collection_efficiency = np.array([0.35, 0.57, 0.8, 1])

# *** Single Photon Source ***

param_sps = deepcopy(param)
number_nv_pulse_sps = np.zeros((number_sps.shape[0], collection_efficiency.shape[0]))
snr_sps = np.zeros((number_sps.shape[0], collection_efficiency.shape[0]))
signal_sps = np.zeros((number_sps.shape[0], collection_efficiency.shape[0]))
noise_sps = np.zeros((number_sps.shape[0], collection_efficiency.shape[0]))
trigger_rate_sps = np.zeros((number_sps.shape[0], collection_efficiency.shape[0]))
average_photon_per_pulse_sps = np.zeros((number_sps.shape[0], collection_efficiency.shape[0]))

print("--- SINGLE PHOTON SOURCE ---")

for idx_nsps, nsps in enumerate(number_sps):
	for idx_ce, ce in enumerate(collection_efficiency):
		param_sps["sp_collection"] = ce
		extr_eff = ce * param["optics_transmitter"]
		sps = SinglePhoton(param_sps, number_sps=nsps, warning_off=True)
		number_nv_pulse_sps[idx_nsps, idx_ce] = sps.number_nv_pulse
		snr_sps[idx_nsps, idx_ce] = sps.signal_to_noise_rate()
		average_photon_per_pulse_sps[idx_nsps, idx_ce] = sps.average_photon_per_pulse
		signal_sps[idx_nsps, idx_ce] = sps.signal_rate()
		noise_sps[idx_nsps, idx_ce] = sps.noise_rate()
		trigger_rate_sps[idx_nsps, idx_ce] = sps.trigger_rate

mini = 52500
maxi = 395000

non_vacuum_number = np.linspace(mini, maxi, 100)
non_vacuum_number = np.concatenate((non_vacuum_number, number_nv_pulse_sps.flatten()))
other = np.array([75000, 100000, 150000, 200000, 250000, 300000, 350000, 375000])
non_vacuum_number = np.concatenate((non_vacuum_number, other))
non_vacuum_number = np.sort(non_vacuum_number)[::-1]

# *** Pulsed Laser ***

print("--- PULSED LASER SOURCE ---")

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

print("--- ENTANGLED PHOTON SOURCE ---")

param_eps = deepcopy(param)
snr_entangled = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
signal_eps = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
noise_eps = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
trigger_rate_eps = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))
average_photon_per_pulse_entangled = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))

last_state = np.zeros((non_vacuum_number.shape[0], collection_efficiency.shape[0]))

# Assert that the fock space dimension is large enough to accommodate the average photon per pulse.
# This allows for a more progressive increase in the fock space dimension as the average photon per pulse increases.
# This adhoc function allows that the vector |n> does not truncate information.


def adjustable_fock_space(average_photon_per_pulse):
	return round(average_photon_per_pulse * 3.5) + 10


for idx_nvp, ce in enumerate(collection_efficiency):
	param_eps["spdc_eps_collection"] = ce
	param_eps["spdc_eps_heralding"] = 0.99 * param_eps["detection_efficiency"]

	for idx_ce, nvp in enumerate(tqdm(non_vacuum_number)):
		param_eps["number_nv_pulse"] = nvp
		eps = EntangledPhotonSPDC(param_eps)
		average_photon_per_pulse_entangled[idx_ce, idx_nvp] = eps.average_photon_per_pulse
		param_eps["fock_space_dim"] = adjustable_fock_space(average_photon_per_pulse_entangled[idx_ce, idx_nvp])
		eps = EntangledPhotonSPDC(param_eps)
		#print(param_eps["fock_space_dim"], eps.prob_of_last_element_fock_space)
		snr_entangled[idx_ce, idx_nvp] = eps.signal_to_noise_rate()
		last_state[idx_ce, idx_nvp] = eps.prob_of_last_element_fock_space
		signal_eps[idx_ce, idx_nvp] = eps.signal_rate()
		noise_eps[idx_ce, idx_nvp] = eps.noise_rate()
		trigger_rate_eps[idx_ce, idx_nvp] = eps.trigger_rate
		#print(eps.prob_of_last_element_fock_space)


result = {
	"sps": {
		"0.35": {
				"signal": signal_sps[0, 0],
				"noise": noise_sps[0, 0],
				"snr": snr_sps[0, 0],
				"trigger_rate": trigger_rate_sps[0, 0],
				"average_photon_per_pulse": average_photon_per_pulse_sps[0, 0],
				"number_nv_pulse": number_nv_pulse_sps[0, 0],
			},
		"0.57": {
				"signal": signal_sps[0, 1],
				"noise": noise_sps[0, 1],
				"snr": snr_sps[0, 1],
				"trigger_rate": trigger_rate_sps[0, 1],
				"average_photon_per_pulse": average_photon_per_pulse_sps[0, 1],
				"number_nv_pulse": number_nv_pulse_sps[0, 1],
			},
		"0.8": {
				"signal": signal_sps[0, 2],
				"noise": noise_sps[0, 2],
				"snr": snr_sps[0, 2],
				"trigger_rate": trigger_rate_sps[0, 2],
				"average_photon_per_pulse": average_photon_per_pulse_sps[0, 2],
				"number_nv_pulse": number_nv_pulse_sps[0, 2],
			},
		"1": {
				"signal": signal_sps[0, 3],
				"noise": noise_sps[0, 3],
				"snr": snr_sps[0, 3],
				"trigger_rate": trigger_rate_sps[0, 3],
				"average_photon_per_pulse": average_photon_per_pulse_sps[0, 3],
				"number_nv_pulse": number_nv_pulse_sps[0, 3],
			},
		"N=2": {
			"0.35": {
				"signal": signal_sps[1, 0],
				"noise": noise_sps[1, 0],
				"snr": snr_sps[1, 0],
				"trigger_rate": trigger_rate_sps[1, 0],
				"average_photon_per_pulse": average_photon_per_pulse_sps[1, 0],
				"number_nv_pulse": number_nv_pulse_sps[1, 0],
			},
			"0.57": {
				"signal": signal_sps[1, 1],
				"noise": noise_sps[1, 1],
				"snr": snr_sps[1, 1],
				"trigger_rate": trigger_rate_sps[1, 1],
				"average_photon_per_pulse": average_photon_per_pulse_sps[1, 1],
				"number_nv_pulse": number_nv_pulse_sps[1, 1],
			},
			"0.8": {
				"signal": signal_sps[1, 2],
				"noise": noise_sps[1, 2],
				"snr": snr_sps[1, 2],
				"trigger_rate": trigger_rate_sps[1, 2],
				"average_photon_per_pulse": average_photon_per_pulse_sps[1, 2],
				"number_nv_pulse": number_nv_pulse_sps[1, 2],
			},
			"1": {
				"signal": signal_sps[1, 3],
				"noise": noise_sps[1, 3],
				"snr": snr_sps[1, 3],
				"trigger_rate": trigger_rate_sps[1, 3],
				"average_photon_per_pulse": average_photon_per_pulse_sps[1, 3],
				"number_nv_pulse": number_nv_pulse_sps[1, 3],
			},
		},
		"N=4": {
			"0.35": {
				"signal": signal_sps[2, 0],
				"noise": noise_sps[2, 0],
				"snr": snr_sps[2, 0],
				"trigger_rate": trigger_rate_sps[2, 0],
				"average_photon_per_pulse": average_photon_per_pulse_sps[2, 0],
				"number_nv_pulse": number_nv_pulse_sps[2, 0],
			},
			"0.57": {
				"signal": signal_sps[2, 1],
				"noise": noise_sps[2, 1],
				"snr": snr_sps[2, 1],
				"trigger_rate": trigger_rate_sps[2, 1],
				"average_photon_per_pulse": average_photon_per_pulse_sps[2, 1],
				"number_nv_pulse": number_nv_pulse_sps[2, 1],
			},
			"0.8": {
				"signal": signal_sps[2, 2],
				"noise": noise_sps[2, 2],
				"snr": snr_sps[2, 2],
				"trigger_rate": trigger_rate_sps[2, 2],
				"average_photon_per_pulse": average_photon_per_pulse_sps[2, 2],
				"number_nv_pulse": number_nv_pulse_sps[2, 2],
			},
			"1": {
				"signal": signal_sps[2, 3],
				"noise": noise_sps[2, 3],
				"snr": snr_sps[2, 3],
				"trigger_rate": trigger_rate_sps[2, 3],
				"average_photon_per_pulse": average_photon_per_pulse_sps[2, 3],
				"number_nv_pulse": number_nv_pulse_sps[2, 3],
			},
		},
		"N=8": {
			"0.35": {
				"signal": signal_sps[3, 0],
				"noise": noise_sps[3, 0],
				"snr": snr_sps[3, 0],
				"trigger_rate": trigger_rate_sps[3, 0],
				"average_photon_per_pulse": average_photon_per_pulse_sps[3, 0],
				"number_nv_pulse": number_nv_pulse_sps[3, 0],
			},
			"0.57": {
				"signal": signal_sps[3, 1],
				"noise": noise_sps[3, 1],
				"snr": snr_sps[3, 1],
				"trigger_rate": trigger_rate_sps[3, 1],
				"average_photon_per_pulse": average_photon_per_pulse_sps[3, 1],
				"number_nv_pulse": number_nv_pulse_sps[3, 1],
			},
			"0.8": {
				"signal": signal_sps[3, 2],
				"noise": noise_sps[3, 2],
				"snr": snr_sps[3, 2],
				"trigger_rate": trigger_rate_sps[3, 2],
				"average_photon_per_pulse": average_photon_per_pulse_sps[3, 2],
				"number_nv_pulse": number_nv_pulse_sps[3, 2],
			},
			"1": {
				"signal": signal_sps[3, 3],
				"noise": noise_sps[3, 3],
				"snr": snr_sps[3, 3],
				"trigger_rate": trigger_rate_sps[3, 3],
				"average_photon_per_pulse": average_photon_per_pulse_sps[3, 3],
				"number_nv_pulse": number_nv_pulse_sps[3, 3],
			},
		},
		"N=10": {
			"0.35": {
				"signal": signal_sps[4, 0],
				"noise": noise_sps[4, 0],
				"snr": snr_sps[4, 0],
				"trigger_rate": trigger_rate_sps[4, 0],
				"average_photon_per_pulse": average_photon_per_pulse_sps[4, 0],
				"number_nv_pulse": number_nv_pulse_sps[4, 0],
			},
			"0.57": {
				"signal": signal_sps[4, 1],
				"noise": noise_sps[4, 1],
				"snr": snr_sps[4, 1],
				"trigger_rate": trigger_rate_sps[4, 1],
				"average_photon_per_pulse": average_photon_per_pulse_sps[4, 1],
				"number_nv_pulse": number_nv_pulse_sps[4, 1],
			},
			"0.8": {
				"signal": signal_sps[4, 2],
				"noise": noise_sps[4, 2],
				"snr": snr_sps[4, 2],
				"trigger_rate": trigger_rate_sps[4, 2],
				"average_photon_per_pulse": average_photon_per_pulse_sps[4, 2],
				"number_nv_pulse": number_nv_pulse_sps[4, 2],
			},
			"1": {
				"signal": signal_sps[4, 3],
				"noise": noise_sps[4, 3],
				"snr": snr_sps[4, 3],
				"trigger_rate": trigger_rate_sps[4, 3],
				"average_photon_per_pulse": average_photon_per_pulse_sps[4, 3],
				"number_nv_pulse": number_nv_pulse_sps[4, 3],
			},
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
			"signal": signal_eps[:, 0],
			"noise": noise_eps[:, 0],
			"snr": snr_entangled[:, 0],
			"trigger_rate": trigger_rate_eps[:, 0],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 0],
			"last_state": last_state[:, 0],
		},
		"0.57": {
			"signal": signal_eps[:, 1],
			"noise": noise_eps[:, 1],
			"snr": snr_entangled[:, 1],
			"trigger_rate": trigger_rate_eps[:, 1],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 1],
			"last_state": last_state[:, 1],
		},
		"0.8": {
			"signal": signal_eps[:, 2],
			"noise": noise_eps[:, 2],
			"snr": snr_entangled[:, 2],
			"trigger_rate": trigger_rate_eps[:, 2],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 2],
			"last_state": last_state[:, 2],
		},
		"1": {
			"signal": signal_eps[:, 3],
			"noise": noise_eps[:, 3],
			"snr": snr_entangled[:, 3],
			"trigger_rate": trigger_rate_eps[:, 3],
			"average_photon_per_pulse": average_photon_per_pulse_entangled[:, 3],
			"last_state": last_state[:, 3],
		},
	},
	"param": param,
	"non_vacuum_number": non_vacuum_number,
	"collection_efficiency": collection_efficiency,
	"non_vacuum_number_sps": number_nv_pulse_sps,
}

np.save("data/Figure_3_4_5.npy", result)