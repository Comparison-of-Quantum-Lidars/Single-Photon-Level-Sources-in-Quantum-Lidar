import numpy as np
import matplotlib.pyplot as plt
from Sources import SinglePhoton
from Analysis import RocAnalysis

### SETUP ###

result = np.load("5m_snr_matching_non_vacuum_pulse_2025_06_22.npy", allow_pickle=True).item()

param = result["param"]

laser = result["laser"]

eps = result["eps"]
eps_0_35 = eps["0.35"]
eps_0_57 = eps["0.57"]
eps_0_8 = eps["0.8"]
eps_1 = eps["1"]

number_nv_pulse = result["non_vacuum_number"]

### 50000 PULSE COMPARISON ###

idx_50000 = np.where(number_nv_pulse == 50000)[0][0]

signal_laser_50000 = laser["signal"][idx_50000]
noise_laser_50000 = laser["noise"][idx_50000]
trigger_rate_laser_50000 = laser["trigger_rate"][idx_50000]


signal_eps_0_35 = eps_0_35["signal"][idx_50000]
noise_eps_0_35 = eps_0_35["noise"][idx_50000]
trigger_rate_sps_0_35 = eps_0_35["trigger_rate"][idx_50000]

signal_eps_0_57 = eps_0_57["signal"][idx_50000]
noise_eps_0_57 = eps_0_57["noise"][idx_50000]
trigger_rate_eps_0_57 = eps_0_57["trigger_rate"][idx_50000]

signal_eps_0_8 = eps_0_8["signal"][idx_50000]
noise_eps_0_8 = eps_0_8["noise"][idx_50000]
trigger_rate_eps_0_8 = eps_0_8["trigger_rate"][idx_50000]

signal_eps_1 = eps_1["signal"][idx_50000]
noise_eps_1 = eps_1["noise"][idx_50000]
trigger_rate_eps_1 = eps_1["trigger_rate"][idx_50000]

### 100000 PULSE COMPARISON ###
idx_100000 = np.where(number_nv_pulse == 100000)[0][0]
signal_laser_100000 = laser["signal"][idx_100000]
noise_laser_100000 = laser["noise"][idx_100000]
trigger_rate_laser_100000 = laser["trigger_rate"][idx_100000]
signal_eps_0_35_100000 = eps_0_35["signal"][idx_100000]
noise_eps_0_35_100000 = eps_0_35["noise"][idx_100000]
trigger_rate_eps_0_35_100000 = eps_0_35["trigger_rate"][idx_100000]
signal_sps_0_57_100000 = eps_0_57["signal"][idx_100000]
noise_sps_0_57_100000 = eps_0_57["noise"][idx_100000]
trigger_rate_eps_0_57_100000 = eps_0_57["trigger_rate"][idx_100000]
signal_sps_0_8_100000 = eps_0_8["signal"][idx_100000]
noise_sps_0_8_100000 = eps_0_8["noise"][idx_100000]
trigger_rate_eps_0_8_100000 = eps_0_8["trigger_rate"][idx_100000]
signal_sps_1_100000 = eps_1["signal"][idx_100000]
noise_sps_1_100000 = eps_1["noise"][idx_100000]
trigger_rate_eps_1_100000 = eps_1["trigger_rate"][idx_100000]

### 200000 PULSE COMPARISON ###
idx_200000 = np.where(number_nv_pulse == 200000)[0][0]
signal_laser_200000 = laser["signal"][idx_200000]
noise_laser_200000 = laser["noise"][idx_200000]
trigger_rate_laser_200000 = laser["trigger_rate"][idx_200000]
signal_eps_0_35_200000 = eps_0_35["signal"][idx_200000]
noise_eps_0_35_200000 = eps_0_35["noise"][idx_200000]
trigger_rate_eps_0_35_200000 = eps_0_35["trigger_rate"][idx_200000]
signal_sps_0_57_200000 = eps_0_57["signal"][idx_200000]
noise_sps_0_57_200000 = eps_0_57["noise"][idx_200000]
trigger_rate_eps_0_57_200000 = eps_0_57["trigger_rate"][idx_200000]
signal_sps_0_8_200000 = eps_0_8["signal"][idx_200000]
noise_sps_0_8_200000 = eps_0_8["noise"][idx_200000]
trigger_rate_eps_0_8_200000 = eps_0_8["trigger_rate"][idx_200000]
signal_sps_1_200000 = eps_1["signal"][idx_200000]
noise_sps_1_200000 = eps_1["noise"][idx_200000]
trigger_rate_eps_1_200000 = eps_1["trigger_rate"][idx_200000]

### 300000 PULSE COMPARISON ###
idx_300000 = np.where(number_nv_pulse == 250000)[0][0]
signal_laser_300000 = laser["signal"][idx_300000]
noise_laser_300000 = laser["noise"][idx_300000]
trigger_rate_laser_300000 = laser["trigger_rate"][idx_300000]
signal_eps_0_35_300000 = eps_0_35["signal"][idx_300000]
noise_eps_0_35_300000 = eps_0_35["noise"][idx_300000]
trigger_rate_eps_0_35_300000 = eps_0_35["trigger_rate"][idx_300000]
signal_eps_0_57_300000 = eps_0_57["signal"][idx_300000]
noise_eps_0_57_300000 = eps_0_57["noise"][idx_300000]
trigger_rate_eps_0_57_300000 = eps_0_57["trigger_rate"][idx_300000]
signal_eps_0_8_300000 = eps_0_8["signal"][idx_300000]
noise_eps_0_8_300000 = eps_0_8["noise"][idx_300000]
trigger_rate_eps_0_8_300000 = eps_0_8["trigger_rate"][idx_300000]
signal_eps_1_300000 = eps_1["signal"][idx_300000]
noise_eps_1_300000 = eps_1["noise"][idx_300000]
trigger_rate_eps_1_300000 = eps_1["trigger_rate"][idx_300000]

print(signal_eps_1_300000)
print(noise_eps_1_300000)

print(signal_laser_300000)
print(noise_laser_300000)

### ROC CURVE 50000 PULSE COMPARISON ###

range_distance = 50
threshold = 100

tp_laser_50000, fp_laser_50000 = RocAnalysis(
	signal_rate=signal_laser_50000,
	noise_rate=noise_laser_50000,
	trigger_rate=trigger_rate_laser_50000,
	threshold_limit=trigger_rate_laser_50000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_35_50000, fp_sps_0_35_50000 = RocAnalysis(
	signal_rate=signal_eps_0_35,
	noise_rate=noise_eps_0_35,
	trigger_rate=trigger_rate_sps_0_35,
	threshold_limit=trigger_rate_sps_0_35 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_57_50000, fp_sps_0_57_50000 = RocAnalysis(
	signal_rate=signal_eps_0_57,
	noise_rate=noise_eps_0_57,
	trigger_rate=trigger_rate_eps_0_57,
	threshold_limit=trigger_rate_eps_0_57 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_8_50000, fp_sps_0_8_50000 = RocAnalysis(
	signal_rate=signal_eps_0_8,
	noise_rate=noise_eps_0_8,
	trigger_rate=trigger_rate_eps_0_8,
	threshold_limit=trigger_rate_eps_0_8 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_1_50000, fp_sps_1_50000 = RocAnalysis(
	signal_rate=signal_eps_1,
	noise_rate=noise_eps_1,
	trigger_rate=trigger_rate_eps_1,
	threshold_limit=trigger_rate_eps_1 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

### ROC CURVE 100000 PULSE COMPARISON ###

tp_laser_100000, fp_laser_100000 = RocAnalysis(
	signal_rate=signal_laser_100000,
	noise_rate=noise_laser_100000,
	trigger_rate=trigger_rate_laser_100000,
	threshold_limit=trigger_rate_laser_100000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_35_100000, fp_sps_0_35_100000 = RocAnalysis(
	signal_rate=signal_eps_0_35_100000,
	noise_rate=noise_eps_0_35_100000,
	trigger_rate=trigger_rate_eps_0_35_100000,
	threshold_limit=trigger_rate_eps_0_35_100000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_57_100000, fp_sps_0_57_100000 = RocAnalysis(
	signal_rate=signal_sps_0_57_100000,
	noise_rate=noise_sps_0_57_100000,
	trigger_rate=trigger_rate_eps_0_57_100000,
	threshold_limit=trigger_rate_eps_0_57_100000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_8_100000, fp_sps_0_8_100000 = RocAnalysis(
	signal_rate=signal_sps_0_8_100000,
	noise_rate=noise_sps_0_8_100000,
	trigger_rate=trigger_rate_eps_0_8_100000,
	threshold_limit=trigger_rate_eps_0_8_100000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_1_100000, fp_sps_1_100000 = RocAnalysis(
	signal_rate=signal_sps_1_100000,
	noise_rate=noise_sps_1_100000,
	trigger_rate=trigger_rate_eps_1_100000,
	threshold_limit=trigger_rate_eps_1_100000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

### ROC CURVE 200000 PULSE COMPARISON ###

tp_laser_200000, fp_laser_200000 = RocAnalysis(
	signal_rate=signal_laser_200000,
	noise_rate=noise_laser_200000,
	trigger_rate=trigger_rate_laser_200000,
	threshold_limit=trigger_rate_laser_200000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_35_200000, fp_sps_0_35_200000 = RocAnalysis(
	signal_rate=signal_eps_0_35_200000,
	noise_rate=noise_eps_0_35_200000,
	trigger_rate=trigger_rate_eps_0_35_200000,
	threshold_limit=trigger_rate_eps_0_35_200000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_57_200000, fp_sps_0_57_200000 = RocAnalysis(
	signal_rate=signal_sps_0_57_200000,
	noise_rate=noise_sps_0_57_200000,
	trigger_rate=trigger_rate_eps_0_57_200000,
	threshold_limit=trigger_rate_eps_0_57_200000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_8_200000, fp_sps_0_8_200000 = RocAnalysis(
	signal_rate=signal_sps_0_8_200000,
	noise_rate=noise_sps_0_8_200000,
	trigger_rate=trigger_rate_eps_0_8_200000,
	threshold_limit=trigger_rate_eps_0_8_200000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_1_200000, fp_sps_1_200000 = RocAnalysis(
	signal_rate=signal_sps_1_200000,
	noise_rate=noise_sps_1_200000,
	trigger_rate=trigger_rate_eps_1_200000,
	threshold_limit=trigger_rate_eps_1_200000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

### ROC CURVE 300000 PULSE COMPARISON ###

tp_laser_300000, fp_laser_300000 = RocAnalysis(
	signal_rate=signal_laser_300000,
	noise_rate=noise_laser_300000,
	trigger_rate=trigger_rate_laser_300000,
	threshold_limit=trigger_rate_laser_300000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_35_300000, fp_sps_0_35_300000 = RocAnalysis(
	signal_rate=signal_eps_0_35_300000,
	noise_rate=noise_eps_0_35_300000,
	trigger_rate=trigger_rate_eps_0_35_300000,
	threshold_limit=trigger_rate_eps_0_35_300000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_57_300000, fp_sps_0_57_300000 = RocAnalysis(
	signal_rate=signal_eps_0_57_300000,
	noise_rate=noise_eps_0_57_300000,
	trigger_rate=trigger_rate_eps_0_57_300000,
	threshold_limit=trigger_rate_eps_0_57_300000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_0_8_300000, fp_sps_0_8_300000 = RocAnalysis(
	signal_rate=signal_eps_0_8_300000,
	noise_rate=noise_eps_0_8_300000,
	trigger_rate=trigger_rate_eps_0_8_300000,
	threshold_limit=trigger_rate_eps_0_8_300000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

tp_sps_1_300000, fp_sps_1_300000 = RocAnalysis(
	signal_rate=signal_eps_1_300000,
	noise_rate=noise_eps_1_300000,
	trigger_rate=trigger_rate_eps_1_300000,
	threshold_limit=trigger_rate_eps_1_300000 / threshold,
	range_interval=range_distance,
	timing_window=param["timing_window"],
).compute_p_d_p_fa()

### PLOT ROC CURVES ###

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(fp_laser_300000, tp_laser_300000, "-", label="Pulsed Laser", color="blue", linewidth=2.5, zorder=2)
axs[0, 0].plot(fp_sps_0_35_300000, tp_sps_0_35_300000, "--", label="EPS 0.35", color="limegreen", linewidth=2.5, zorder=2)
axs[0, 0].plot(fp_sps_0_57_300000, tp_sps_0_57_300000, ":", label="EPS 0.57", color="seagreen", linewidth=2.5, zorder=2)
axs[0, 0].plot(fp_sps_0_8_300000, tp_sps_0_8_300000, "-", label="EPS 0.8", color="darkgreen", linewidth=2.5, zorder=2)
axs[0, 0].plot(fp_sps_1_300000, tp_sps_1_300000, "-.", label="EPS 1", color="green", linewidth=2.5, zorder=2)

str_title = r"$\mathcal{N}_{nv} = 2.5\times 10^5$"
axs[0, 0].set_title(str_title, fontsize=21)
axs[0, 0].tick_params(axis='both', which='major', labelsize=22)

axs[0, 1].plot(fp_laser_200000, tp_laser_200000, "-", color="blue", linewidth=2.5, zorder=2)
axs[0, 1].plot(fp_sps_0_35_200000, tp_sps_0_35_200000, "--", color="limegreen", linewidth=2.5, zorder=2)
axs[0, 1].plot(fp_sps_0_57_200000, tp_sps_0_57_200000, ":", color="seagreen", linewidth=2.5, zorder=2)
axs[0, 1].plot(fp_sps_0_8_200000, tp_sps_0_8_200000, "-", color="darkgreen", linewidth=2.5, zorder=2)
axs[0, 1].plot(fp_sps_1_200000, tp_sps_1_200000, "-.", color="green", linewidth=2.5, zorder=2)
str_title = r"$\mathcal{N}_{nv} = 2 \times 10^5$"
axs[0, 1].set_title(str_title, fontsize=21)
axs[0, 1].tick_params(axis='both', which='major', labelsize=22)

axs[1, 0].plot(fp_laser_100000, tp_laser_100000, "-", color="blue", linewidth=2.5, zorder=2)
axs[1, 0].plot(fp_sps_0_35_100000, tp_sps_0_35_100000, "--", color="limegreen", linewidth=2.5, zorder=2)
axs[1, 0].plot(fp_sps_0_57_100000, tp_sps_0_57_100000, ":", color="seagreen", linewidth=2.5, zorder=2)
axs[1, 0].plot(fp_sps_0_8_100000, tp_sps_0_8_100000, "-", color="darkgreen", linewidth=2.5, zorder=2)
axs[1, 0].plot(fp_sps_1_100000, tp_sps_1_100000, "-.", color="green", linewidth=2.5, zorder=2)
str_title = r"$\mathcal{N}_{nv} = 1 \times 10^5$"
axs[1, 0].set_title(str_title, fontsize=22)
axs[1, 0].tick_params(axis='both', which='major', labelsize=22)

axs[1, 1].plot(fp_laser_50000, tp_laser_50000, "-", color="blue", linewidth=2.5, zorder=2)
axs[1, 1].plot(fp_sps_0_35_50000, tp_sps_0_35_50000, "--", color="limegreen", linewidth=2.5, zorder=2)
axs[1, 1].plot(fp_sps_0_57_50000, tp_sps_0_57_50000, ":", color="seagreen", linewidth=2.5, zorder=2)
axs[1, 1].plot(fp_sps_0_8_50000, tp_sps_0_8_50000, "-", color="darkgreen", linewidth=2.5, zorder=2)
axs[1, 1].plot(fp_sps_1_50000, tp_sps_1_50000, "-.", color="green", linewidth=2.5, zorder=2)
str_title = r"$\mathcal{N}_{nv} = 0.5 \times 10^5$"
axs[1, 1].set_title(str_title, fontsize=22)
axs[1, 1].tick_params(axis='both', which='major', labelsize=22)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

fig.text(0.5, 0.04, 'False Positive', ha='center', fontsize=25)
fig.text(0.04, 0.5, 'True Positive', va='center', rotation='vertical', fontsize=25)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fontsize=20, ncol=5)

plt.show()

