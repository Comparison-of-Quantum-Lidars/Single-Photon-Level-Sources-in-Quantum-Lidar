import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
from Analysis import RangeLimitation, RocAnalysis
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

skip_computation = True

param = SetupParameters(
	fock_space_dim=35,
	output_power=400000,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	number_nv_pulse=None,
	sp_collection=0.8,
	sp_p1=0.99,
	sp_p2=5e-3,
	spdc_eps_heralding=0.99*0.7,
	spdc_eps_collection=0.8,
	atmosphere=0.5,
	target_distance=None,
	receiver_diameter=0.05,
	target_albedo=0.5,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.7,
	background=250000,
	detector_dark=25,
	timing_window=0.5e-9,
)

number_sps_array = 10
sps = SinglePhoton(param, number_sps=number_sps_array)
number_nv_pulse_sps = sps.number_nv_pulse
param["number_nv_pulse"] = number_nv_pulse_sps

range_interval = None
distance = np.linspace(1, 50, 500)
acquisition_time = np.linspace(1, 3600, 200)
target_false_positive = 0.2
target_true_positive = 0.8
colored_marker = True
precision = 25
threshold_limit_factor = 500

distance_cutoff = {
	"laser": [],
	"sps": [],
	"eps": [],
	"acquisition time": acquisition_time
}

minimal_distance = 0.0
for idx, at in enumerate(tqdm(acquisition_time)):
	if skip_computation:
		break
	distance_measurement = deepcopy(distance)[distance >= minimal_distance]
	at = np.array([at])
	results = RangeLimitation(
		params=param,
		parameter_to_match="number_nv_pulse",
		range_interval=None,
		distance=distance_measurement,
		acquisition_time=at,
		target_false_positive=target_false_positive,
		target_true_positive=target_true_positive,
		precision_roc=precision,
		threshold_limit_factor_roc=threshold_limit_factor,
		number_nv_pulse_for_match=number_nv_pulse_sps,
		number_sps_array=number_sps_array,
		early_stop=True,
		show_progress_bar=False
	).compute()

	distance_cutoff["laser"].append(results["distance_cutoff_laser"][at[0]])
	distance_cutoff["sps"].append(results["distance_cutoff_sps"][at[0]])
	distance_cutoff["eps"].append(results["distance_cutoff_eps"][at[0]])

	minimal_distance = min(distance_cutoff["laser"][-1], distance_cutoff["sps"][-1], distance_cutoff["eps"][-1]) - 1

# Uncomment .compute() to generate new data

### LOAD PREVIOUS DATA ###

results = np.load("Acquisition_time_sweep_1s_3600s.npy", allow_pickle=True).item()

#np.save("Acquisition_time_sweep_1s_3600s.npy", distance_cutoff)

if skip_computation:
	distance_cutoff = np.load("Acquisition_time_sweep_1s_3600s.npy", allow_pickle=True).item()
	acquisition_time = distance_cutoff["acquisition time"]

# *** PLOTTING ***

min_cutoff = 0.0

acquisition_time = np.array(acquisition_time)
distance_cutoff_laser = np.array(distance_cutoff["laser"])[acquisition_time >= min_cutoff]
distance_cutoff_sps = np.array(distance_cutoff["sps"])[acquisition_time >= min_cutoff]
distance_cutoff_eps = np.array(distance_cutoff["eps"])[acquisition_time >= min_cutoff]
acquisition_time = acquisition_time[acquisition_time >= min_cutoff]

fig = plt.figure(figsize=(16.2, 10))

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.plot(acquisition_time, distance_cutoff["laser"], "-.", label="Laser", color="blue", linewidth=3, zorder=2)
plt.plot(acquisition_time, distance_cutoff["sps"], "--", label="Single Photon Source", color="red", linewidth=3, zorder=3)
plt.plot(acquisition_time, distance_cutoff["eps"], "-", label="Entangled Photon Source", color="green", linewidth=3, zorder=1)
plt.scatter(0, 0, label=f"80.0% true detection\n20.0% false detection", alpha=0)

plt.xlabel("Acquisition time [s]", fontsize=22)
plt.ylabel("Distance [m]", fontsize=22)
plt.legend(fontsize=22, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=22)

plt.show()