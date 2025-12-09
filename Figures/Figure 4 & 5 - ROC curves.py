import numpy as np
import matplotlib.pyplot as plt
from Sources import SinglePhoton
from Analysis import RocAnalysis
from copy import deepcopy

### SETUP ###

number_sps = 10

result = np.load("data/fig_3_4_5_matching_number_non_vacuum_pulses.npy", allow_pickle=True).item()

param = result["param"]

laser = result["laser"]

eps = result["eps"]
eps_0_35 = eps["0.35"]
eps_0_57 = eps["0.57"]
eps_0_8 = eps["0.8"]
eps_1 = eps["1"]

sps = result["sps"][f"N={number_sps}"] if number_sps != 1 else result["sps"]
sps_035 = sps["0.35"]
sps_057 = sps["0.57"]
sps_08 = sps["0.8"]
sps_1 = sps["1"]

number_nv_pulse = result["non_vacuum_number"]

### Compraison 35% efficiency ###

param_sps = deepcopy(param)
param_sps["sp_collection"] = 0.35
sps = SinglePhoton(param_sps, number_sps=number_sps, warning_off=True)
number_nv_pulse_sps_35 = sps.number_nv_pulse

idx_35 = np.where(number_nv_pulse == number_nv_pulse_sps_35)[0][0]

signal_laser_35 = laser["signal"][idx_35]
noise_laser_35 = laser["noise"][idx_35]
trigger_rate_laser_35 = laser["trigger_rate"][idx_35]

signal_eps_35 = eps_0_35["signal"][idx_35]
noise_eps_35 = eps_0_35["noise"][idx_35]
trigger_rate_eps_35 = eps_0_35["trigger_rate"][idx_35]

signal_sps_35 = sps_035["signal"]
noise_sps_35 = sps_035["noise"]
trigger_rate_sps_35 = sps_035["trigger_rate"]

#### Compraison 57% efficiency ###

param_sps["sp_collection"] = 0.57
sps = SinglePhoton(param_sps, number_sps=number_sps, warning_off=True)
number_nv_pulse_sps_57 = sps.number_nv_pulse
idx_57 = np.where(number_nv_pulse == number_nv_pulse_sps_57)[0][0]

signal_laser_57 = laser["signal"][idx_57]
noise_laser_57 = laser["noise"][idx_57]
trigger_rate_laser_57 = laser["trigger_rate"][idx_57]

signal_eps_57 = eps_0_57["signal"][idx_57]
noise_eps_57 = eps_0_57["noise"][idx_57]
trigger_rate_eps_57 = eps_0_57["trigger_rate"][idx_57]

signal_sps_57 = sps_057["signal"]
noise_sps_57 = sps_057["noise"]
trigger_rate_sps_57 = sps_057["trigger_rate"]

### Compraison 80% efficiency ###

param_sps["sp_collection"] = 0.8
sps = SinglePhoton(param_sps, number_sps=number_sps, warning_off=True)
number_nv_pulse_sps_80 = sps.number_nv_pulse
idx_80 = np.where(number_nv_pulse == number_nv_pulse_sps_80)[0][0]

signal_laser_80 = laser["signal"][idx_80]
noise_laser_80 = laser["noise"][idx_80]
trigger_rate_laser_80 = laser["trigger_rate"][idx_80]

signal_eps_80 = eps_0_8["signal"][idx_80]
noise_eps_80 = eps_0_8["noise"][idx_80]
trigger_rate_eps_80 = eps_0_8["trigger_rate"][idx_80]

signal_sps_80 = sps_08["signal"]
noise_sps_80 = sps_08["noise"]
trigger_rate_sps_80 = sps_08["trigger_rate"]

### Compraison 100% efficiency ###

param_sps["sp_collection"] = 1.0
sps = SinglePhoton(param_sps, number_sps=number_sps, warning_off=True)
number_nv_pulse_sps_100 = sps.number_nv_pulse
idx_100 = np.where(number_nv_pulse == number_nv_pulse_sps_100)[0][0]

signal_laser_100 = laser["signal"][idx_100]
noise_laser_100 = laser["noise"][idx_100]
trigger_rate_laser_100 = laser["trigger_rate"][idx_100]

signal_eps_100 = eps_1["signal"][idx_100]
noise_eps_100 = eps_1["noise"][idx_100]
trigger_rate_eps_100 = eps_1["trigger_rate"][idx_100]

signal_sps_100 = sps_1["signal"]
noise_sps_100 = sps_1["noise"]
trigger_rate_sps_100 = sps_1["trigger_rate"]

### ROC Curves ###

range_distance = 50

threshold = 1000

tp_laser_35, fp_laser_35 = RocAnalysis(
	signal_rate=signal_laser_35,
	noise_rate=noise_laser_35,
	trigger_rate=trigger_rate_laser_35,
	threshold_limit=trigger_rate_laser_35 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_eps_35, fp_eps_35 = RocAnalysis(
	signal_rate=signal_eps_35,
	noise_rate=noise_eps_35,
	trigger_rate=trigger_rate_eps_35,
	threshold_limit=trigger_rate_eps_35 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_35, fp_sps_35 = RocAnalysis(
	signal_rate=signal_sps_35,
	noise_rate=noise_sps_35,
	trigger_rate=trigger_rate_sps_35,
	threshold_limit=trigger_rate_sps_35 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_laser_57, fp_laser_57 = RocAnalysis(
	signal_rate=signal_laser_57,
	noise_rate=noise_laser_57,
	trigger_rate=trigger_rate_laser_57,
	threshold_limit=trigger_rate_laser_57 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_eps_57, fp_eps_57 = RocAnalysis(
	signal_rate=signal_eps_57,
	noise_rate=noise_eps_57,
	trigger_rate=trigger_rate_eps_57,
	threshold_limit=trigger_rate_eps_57 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_57, fp_sps_57 = RocAnalysis(
	signal_rate=signal_sps_57,
	noise_rate=noise_sps_57,
	trigger_rate=trigger_rate_sps_57,
	threshold_limit=trigger_rate_sps_57 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_laser_80, fp_laser_80 = RocAnalysis(
	signal_rate=signal_laser_80,
	noise_rate=noise_laser_80,
	trigger_rate=trigger_rate_laser_80,
	threshold_limit=trigger_rate_laser_80 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_eps_80, fp_eps_80 = RocAnalysis(
	signal_rate=signal_eps_80,
	noise_rate=noise_eps_80,
	trigger_rate=trigger_rate_eps_80,
	threshold_limit=trigger_rate_eps_80 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_80, fp_sps_80 = RocAnalysis(
	signal_rate=signal_sps_80,
	noise_rate=noise_sps_80,
	trigger_rate=trigger_rate_sps_80,
	threshold_limit=trigger_rate_sps_80 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_laser_100, fp_laser_100 = RocAnalysis(
	signal_rate=signal_laser_100,
	noise_rate=noise_laser_100,
	trigger_rate=trigger_rate_laser_100,
	threshold_limit=trigger_rate_laser_100 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_eps_100, fp_eps_100 = RocAnalysis(
	signal_rate=signal_eps_100,
	noise_rate=noise_eps_100,
	trigger_rate=trigger_rate_eps_100,
	threshold_limit=trigger_rate_eps_100 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_100, fp_sps_100 = RocAnalysis(
	signal_rate=signal_sps_100,
	noise_rate=noise_sps_100,
	trigger_rate=trigger_rate_sps_100,
	threshold_limit=trigger_rate_sps_100 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

### Plotting ###

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

# 2x2 subplots

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Collection efficiency: 0.2

axs[0, 0].plot(fp_sps_35, tp_sps_35, "--", label="Single Photon Source", color="red", linewidth=2.5, zorder=3, alpha=0.75)
axs[0, 0].plot(fp_laser_35, tp_laser_35, "-.", label="Pulsed Laser", color="blue", linewidth=2.5, zorder=1)
axs[0, 0].plot(fp_eps_35, tp_eps_35, "-", label="Entangled Photon Source", color="green", linewidth=2.5, zorder=2)
str_title = r"$\eta_{signal}$ = 35%, $\mathcal{N}_{nv}$= " + f"{int(number_nv_pulse_sps_35)}"
axs[0, 0].set_title(str_title, fontsize=18)
axs[0, 0].tick_params(axis='both', which='major', labelsize=25)


# Collection efficiency: 0.57

axs[0, 1].plot(fp_sps_57, tp_sps_57, "--", label="Single Photon Source", color="red", linewidth=2.5, zorder=3, alpha=0.75)
axs[0, 1].plot(fp_laser_57, tp_laser_57, "-.", label="Pulsed Laser", color="blue", linewidth=2.5, zorder=1)
axs[0, 1].plot(fp_eps_57, tp_eps_57, "-", label="Entangled Photon Source", color="green", linewidth=2.5, zorder=2)
str_title = r"$\eta_{signal}$ = 57%, $\mathcal{N}_{nv}$= " + f"{int(number_nv_pulse_sps_57)}"
axs[0, 1].set_title(str_title, fontsize=18)
axs[0, 1].tick_params(axis='both', which='major', labelsize=25)

# Collection efficiency: 0.8

axs[1, 0].plot(fp_sps_80, tp_sps_80, "--", label="Single Photon Source", color="red", linewidth=2.5, zorder=3, alpha=0.75)
axs[1, 0].plot(fp_laser_80, tp_laser_80, "-.", label="Pulsed Laser", color="blue", linewidth=2.5, zorder=1)
axs[1, 0].plot(fp_eps_80, tp_eps_80, "-", label="Entangled Photon Source", color="green", linewidth=2.5, zorder=2)
str_title = r"$\eta_{signal}$ = 80%, $\mathcal{N}_{nv}$= " + f"{int(number_nv_pulse_sps_80)}"
axs[1, 0].set_title(str_title, fontsize=20)
axs[1, 0].tick_params(axis='both', which='major', labelsize=25)

# Collection efficiency: 1

axs[1, 1].plot(fp_sps_100, tp_sps_100, "--", label="Single Photon Source", color="red", linewidth=2.5, zorder=3, alpha=0.75)
axs[1, 1].plot(fp_laser_100, tp_laser_100, "-.", label="Pulsed Laser", color="blue", linewidth=2.5, zorder=1)
axs[1, 1].plot(fp_eps_100, tp_eps_100, "-", label="Entangled Photon Source", color="green", linewidth=2.5, zorder=2)
str_title = r"$\eta_{signal}$ = 100%, $\mathcal{N}_{nv}$= " + f"{int(number_nv_pulse_sps_100)}"
axs[1, 1].set_title(str_title, fontsize=20)
axs[1, 1].tick_params(axis='both', which='major', labelsize=25)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

fig.text(0.5, 0.04, 'False Positive', ha='center', fontsize=26)
fig.text(0.075, 0.5, 'True Positive', va='center', rotation='vertical', fontsize=26)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fontsize=22, ncol=3)

plt.show()
