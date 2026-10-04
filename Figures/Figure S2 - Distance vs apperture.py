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

# Figure 6
# Use Fock space dimension of 35 and number_sps_array = 10
# Figure 7
# Use Fock space dimension of 9 and number_sps_array = 1
number_sps_array = 10
sps = SinglePhoton(param, number_sps=number_sps_array)
number_nv_pulse_sps = sps.number_nv_pulse
param["number_nv_pulse"] = number_nv_pulse_sps

range_interval = None
distance = np.linspace(1, 45, 300)  #at=1s
distance = np.linspace(10, 125, 600) #at=60s
distance = np.linspace(45, 300, 1000) #at=3600s
acquisition_time = np.array([3600])
target_false_positive = 0.2
target_true_positive = 0.8
colored_marker = True
precision = 25
threshold_limit_factor = 500 #at=1s
threshold_limit_factor = 5 #at=60s
threshold_limit_factor = 0.5 #at=3600s


receiver_diameter = np.linspace(0.01, 2, 200)

distance_cutoff = {
	"laser": [],
	"sps": [],
	"eps": [],
	"receiver_diameter": receiver_diameter
}

minimal_distance = 0.0
for idx, aperture in enumerate(tqdm(receiver_diameter)):
	if skip_computation:
		break
	distance_measurement = deepcopy(distance)[distance >= minimal_distance]
	param["receiver_diameter"] = aperture
	results = RangeLimitation(
		params=param,
		parameter_to_match="number_nv_pulse",
		range_interval=None,
		distance=distance_measurement,
		acquisition_time=acquisition_time,
		target_false_positive=target_false_positive,
		target_true_positive=target_true_positive,
		precision_roc=precision,
		threshold_limit_factor_roc=threshold_limit_factor,
		number_nv_pulse_for_match=number_nv_pulse_sps,
		number_sps_array=number_sps_array,
		early_stop=True,
		show_progress_bar=False
	).compute()

	distance_cutoff["laser"].append(results["distance_cutoff_laser"][acquisition_time[0]])
	distance_cutoff["sps"].append(results["distance_cutoff_sps"][acquisition_time[0]])
	distance_cutoff["eps"].append(results["distance_cutoff_eps"][acquisition_time[0]])

	minimal_distance = min(distance_cutoff["laser"][-1], distance_cutoff["sps"][-1], distance_cutoff["eps"][-1]) - 1

# Uncomment .compute() to generate new data

### LOAD PREVIOUS DATA ###

results = np.load("Receiver_diam_sweep_0p01_2m_3600sec.npy", allow_pickle=True).item()

#np.save("Receiver_diam_sweep_0p01_2m_3600sec.npy", distance_cutoff)

if skip_computation:
	distance_cutoff = results
	receiver_diameter = results["receiver_diameter"]

# *** PLOTTING ***

min_cutoff = 0.05


receiver_diameter = np.array(receiver_diameter)
distance_cutoff_laser = np.array(distance_cutoff["laser"])[receiver_diameter >= min_cutoff]
distance_cutoff_sps = np.array(distance_cutoff["sps"])[receiver_diameter >= min_cutoff]
distance_cutoff_eps = np.array(distance_cutoff["eps"])[receiver_diameter >= min_cutoff]
receiver_diameter = receiver_diameter[receiver_diameter >= min_cutoff]

fig = plt.figure(figsize=(16.2, 10))

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.plot(receiver_diameter, distance_cutoff_laser, "-.", label="Laser", color="blue", linewidth=3, zorder=2)
plt.plot(receiver_diameter, distance_cutoff_sps, "--", label="Single Photon Source", color="red", linewidth=3, zorder=3)
plt.plot(receiver_diameter, distance_cutoff_eps, "-", label="Entangled Photon Source", color="green", linewidth=3, zorder=1)
plt.scatter(0, 0, label=f"80.0% true detection\n20.0% false detection", alpha=0)
plt.xlabel("Receiver Aperture [m]", fontsize=22)
plt.ylabel("Distance [m]", fontsize=22)
plt.legend(fontsize=22, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()