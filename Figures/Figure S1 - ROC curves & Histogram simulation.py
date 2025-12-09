import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
from tqdm import tqdm
from Analysis import RocAnalysis, HistogramAnalysis
from copy import deepcopy
import matplotlib.pyplot as plt

# *** SETUP ***

param = SetupParameters(
	fock_space_dim=35,
	output_power=400000,
	multi_photon_probability=None,
	no_vacuum_probability=None,
	number_nv_pulse=None,
	sp_collection=0.57,
	sp_p1=0.99,
	sp_p2=5e-3,
	spdc_eps_heralding=0.57*0.7,
	spdc_eps_collection=0.57,
	atmosphere=0.5,
	target_distance=5,
	receiver_diameter=0.05,
	target_albedo=0.5,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.7,
	background=250000,
	detector_dark=25,
	timing_window=0.5e-9,
)

acquisition_time = [1, 10, 100]

param_sps = deepcopy(param)
param_sps["fock_space_dim"] = 3
sps = SinglePhoton(param_sps)
nv_pulse_number = sps.compute_number_nv_pulse()

param_eps = deepcopy(param)
param_eps["number_nv_pulse"] = nv_pulse_number
eps = EntangledPhotonSPDC(param_eps)
eps_signal_rate = eps.signal_rate()
eps_noise_rate = eps.noise_rate()
eps_trigger_rate = eps.trigger_rate
threshold = 100
range_distance = 10

hist_results = {at: {} for at in acquisition_time}
roc_results = {at: {} for at in acquisition_time}

for at in tqdm(acquisition_time):
	tp, fp = RocAnalysis(
		signal_rate=eps_signal_rate,
		noise_rate=eps_noise_rate,
		trigger_rate=eps_trigger_rate,
		threshold_limit=eps_trigger_rate / threshold,
		range_interval=range_distance,
		timing_window=param["timing_window"],
		acquisition_time=at,
		precision=20
	).compute_p_d_p_fa()

	counts, _, bin_edges_distance = HistogramAnalysis(
		params=deepcopy(param_eps),
		signal_rate=eps_signal_rate,
		noise_rate=eps_noise_rate,
		acquisition_time=at,
		range_distance=range_distance,
		effective_trigger_rate=eps_trigger_rate
	).histogram_simulation()

	roc_results[at]["tp"] = tp
	roc_results[at]["fp"] = fp
	hist_results[at]["counts"] = counts
	hist_results[at]["bin_edges_distance"] = bin_edges_distance

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

# 2x2 subplots

fig, axs = plt.subplots(2, 3, figsize=(16.2, 10))

at_0 = acquisition_time[0]
axs[0, 0].plot(roc_results[at_0]["fp"], roc_results[at_0]["tp"], "-", color="black", linewidth=2.5)
str_title = f"{at_0} s"
axs[0, 0].set_title(str_title, fontsize=20, fontweight="bold")
axs[0, 0].tick_params(axis='both', which='major', labelsize=25, pad=15)

axs[1, 0].plot(hist_results[at_0]["bin_edges_distance"], hist_results[at_0]["counts"], "-", color="black", linewidth=1.5)
axs[1, 0].tick_params(axis='both', which='major', labelsize=25)

at_1 = acquisition_time[1]
axs[0, 1].plot(roc_results[at_1]["fp"], roc_results[at_1]["tp"], "-", color="black", linewidth=2.5)
str_title = f"{at_1} s"
axs[0, 1].set_title(str_title, fontsize=20, fontweight="bold")
axs[0, 1].tick_params(axis='both', which='major', labelsize=25)

axs[1, 1].plot(hist_results[at_1]["bin_edges_distance"], hist_results[at_1]["counts"], "-", color="black", linewidth=1.5)
axs[1, 1].tick_params(axis='both', which='major', labelsize=25)

at_2 = acquisition_time[2]
axs[0, 2].plot(roc_results[at_2]["fp"], roc_results[at_2]["tp"], "-", label=f"{at_2} s", color="black", linewidth=2.5)
str_title = f"{at_2} s"
axs[0, 2].set_title(str_title, fontsize=20, fontweight="bold")
axs[0, 2].tick_params(axis='both', which='major', labelsize=25)

axs[1, 2].plot(hist_results[at_2]["bin_edges_distance"], hist_results[at_2]["counts"], "-", label=f"{at_2} s", color="black", linewidth=1.5)
axs[1, 2].tick_params(axis='both', which='major', labelsize=25)

plt.subplots_adjust(hspace=0.4, wspace=0.4)

fig.text(0.525, 0.08, 'Distance [m]', ha='center', fontsize=25, fontweight="bold")
fig.text(0.06, 0.3, 'Counts', va='center', rotation='vertical', fontsize=25, fontweight="bold")
fig.text(0.525, 0.495, 'False Positive', ha='center', fontsize=25, fontweight="bold")
fig.text(0.06, 0.72, 'True Positive', va='center', rotation='vertical', fontsize=25, fontweight="bold")

plt.show()

