import numpy as np
import matplotlib.pyplot as plt
from Sources import PulsedLaser, SetupParameters
import random
from tqdm import tqdm
from scipy.stats import binom
from Analysis import HistogramAnalysis


def random_boolean(probability: float):
	return random.random() <= probability


param = SetupParameters(
	fock_space_dim=5,
	output_power=1e6,
	laser_rate=5e6,
	sp_collection=0.2,
	sp_p1=0.9999,
	sp_p2=1e-4,
	spdc_eps_heralding=0.5,
	spdc_eps_collection=0.2,
	spdc_emission=0.1,
	target_distance=10,
	receiver_diameter=0.5,
	target_albedo=0.2,
	optics_transmitter=0.5,
	optics_receiver=0.5,
	detection_efficiency=0.9,
	background=400,
	detector_dark=200,
	timing_window=1e-9,
)

signal = PulsedLaser(param).signal_rate()
noise = PulsedLaser(param).noise_rate()
timing_window = param["timing_window"]

counts, bin_edges, bin_edges_distance = HistogramAnalysis(param, signal, noise, acquisition_time=5).histogram_simulation()

bin_edges = bin_edges / 1e-9

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

# Adjust the width to match the timing window
#plt.bar(bin_edges[:-1], counts[:, 0], width=timing_window/1e-9, align="edge", edgecolor="black")

plt.plot(bin_edges[:-1], counts[:, 0], "-")

plt.xlabel('Time of Flight (seconds)')
plt.ylabel('Coincidence Counts')
plt.title('Histogram of Photon Counts as a Function of Time of Flight')
plt.show()

plt.bar(bin_edges_distance[:-1], counts[:, 0], width=timing_window * 299792458, align="edge", edgecolor="black")
plt.xlabel('Distance (m)')
plt.ylabel('Coincidence Counts')
plt.title('Histogram of Photon Counts as a Function of Distance')
plt.show()

exit()


signal = signal - noise

print(signal, noise)

total_window = 1 / param["laser_rate"]
half_window = total_window / 2
max_range = half_window * 299792458

bins = round(2 * max_range / (299792458 * param["timing_window"]))

acquisition_time = 5

trigger_total = param["laser_rate"] * acquisition_time

counts = np.zeros((bins, 1))


noise_total = noise * acquisition_time
signal_total = signal * acquisition_time

noise_prob = noise_total / trigger_total
signal_prob = signal_total / trigger_total

max_time_of_flight = 2 * max_range / 299792458

timing_window = param["timing_window"]

tof_target = param["target_distance"] / 299792458

jitter_std_dev = 0.5e-9

total = 0


for i in tqdm(range(int(trigger_total))):
	idx_ones = np.array([-1])
	count2add_noise = np.zeros((bins, 1))
	noise2add = np.random.binomial(bins, noise_prob)
	if noise2add != 0:
		bin_with_noise = np.ones((noise2add, 1))
		other_bin = np.zeros((bins - noise2add, 1))
		count2add_noise = np.concatenate((bin_with_noise, other_bin))
		np.random.shuffle(count2add_noise)
		idx_ones = np.where(count2add_noise == 1)
		counts += count2add_noise
	add_signal = random_boolean(signal_prob)
	if add_signal:
		bin2add_signal = round(np.random.normal(tof_target, jitter_std_dev) / timing_window)
		counts[bin2add_signal] += 1
		if bin2add_signal in idx_ones:
			counts[bin2add_signal] -= 1


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

bin_edges = ((np.arange(bins + 1) * timing_window)-timing_window/2)/1e-9
bin_edges_distance = (np.arange(bins + 1) * timing_window * 299792458) - (timing_window*299792458/2)

# Adjust the width to match the timing window
plt.bar(bin_edges[:-1], counts[:, 0], width=timing_window/1e-9, align="edge", edgecolor="black")

plt.xlabel('Time of Flight (seconds)')
plt.ylabel('Coincidence Counts')
plt.title('Histogram of Photon Counts as a Function of Time of Flight')
plt.show()

plt.bar(bin_edges_distance[:-1], counts[:, 0], width=timing_window * 299792458, align="edge", edgecolor="black")
plt.xlabel('Distance (m)')
plt.ylabel('Coincidence Counts')
plt.title('Histogram of Photon Counts as a Function of Distance')
plt.show()