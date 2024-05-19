import matplotlib.pyplot as plt
from scipy.stats import binom
from typing import Optional
import numpy as np
import random
from tqdm import tqdm


class RocAnalysis:

	def __init__(self, signal_rate, noise_rate, trigger_rate, threshold_limit, range_interval, timing_window):
		self.signal_rate = signal_rate
		self.noise_rate = noise_rate
		self.trigger_rate = trigger_rate
		self.threshold_limit = threshold_limit
		self.range_interval = range_interval
		self.timing_window = timing_window

	def compute_q0_q1(self):
		q0 = self.noise_rate / self.trigger_rate
		q1 = self.signal_rate / self.trigger_rate
		return q0, q1

	def create_threshold_array(self):
		threshold = np.linspace(1, int(self.threshold_limit), int(self.threshold_limit))
		return threshold

	def compute_binom_pmf(self, threshold, q):
		p = binom.pmf(threshold, int(self.trigger_rate), q)
		return p

	def number_of_bins(self):
		n_bins = 2 * self.range_interval / (299792458 * self.timing_window)
		return n_bins

	def compute_p_d_p_fa(self):
		threshold = self.create_threshold_array()
		q0, q1 = self.compute_q0_q1()
		p_0 = self.compute_binom_pmf(threshold, q0)
		p_1 = self.compute_binom_pmf(threshold, q1)

		false_positive = 1 - (1 - np.cumsum(np.flip(p_0))) ** self.number_of_bins()
		true_positive = np.cumsum(np.flip(p_1))

		return true_positive, false_positive


class HistogramAnalysis:

	def __init__(self, params, signal_rate, noise_rate, acquisition_time, jitter_std_dev: Optional[float] = 0.5e-9):
		self.params = params
		self.signal_rate = signal_rate
		self.noise_rate = noise_rate
		self.acquisition_time = acquisition_time
		self.jitter_std_dev = jitter_std_dev
		self.bins = self.compute_bins_number()


	def compute_signal_and_noise_rate_per_bins(self):
		signal = self.signal_rate - self.noise_rate
		return signal, self.noise_rate

	def compute_window_params(self):
		total_window = 1 / self.params["laser_rate"]
		half_window = total_window / 2
		max_range = half_window * 299792458
		return max_range, total_window, half_window

	def compute_bins_number(self):
		max_range, _, __ = self.compute_window_params()
		bins = round(2 * max_range / (299792458 * self.params["timing_window"]))
		return bins

	def compute_trigger_total(self):
		trigger_total = self.params["laser_rate"] * self.acquisition_time
		return trigger_total

	def noise_and_signal_prob_per_bins(self):
		noise_total = self.noise_rate * self.acquisition_time
		signal_total = self.signal_rate * self.acquisition_time
		trigger_total = self.compute_trigger_total()

		noise_prob = noise_total / trigger_total
		signal_prob = signal_total / trigger_total

		return noise_prob, signal_prob

	def compute_max_time_of_flight(self):
		max_range, _, __ = self.compute_window_params()
		max_time_of_flight = 2 * max_range / 299792458
		return max_time_of_flight

	@staticmethod
	def random_boolean(prob):
		return random.random() < prob

	def histogram_simulation(self):
		noise_prob, signal_prob = self.noise_and_signal_prob_per_bins()
		trigger_total = self.compute_trigger_total()
		tof_target = self.params["target_distance"] / 299792458
		max_time_of_flight = self.compute_max_time_of_flight()
		timing_window = self.params["timing_window"]

		counts = np.zeros((self.bins, 1))

		for i in tqdm(range(int(trigger_total))):
			idx_ones = np.array([-1])
			count2add_noise = np.zeros((self.bins, 1))
			noise2add = np.random.binomial(self.bins, noise_prob)
			if noise2add != 0:
				bin_with_noise = np.ones((noise2add, 1))
				other_bin = np.zeros((self.bins - noise2add, 1))
				count2add_noise = np.concatenate((bin_with_noise, other_bin))
				np.random.shuffle(count2add_noise)
				idx_ones = np.where(count2add_noise == 1)
				counts += count2add_noise
			add_signal = self.random_boolean(signal_prob)
			if add_signal:
				bin2add_signal = round(np.random.normal(tof_target, self.jitter_std_dev) / timing_window)
				counts[bin2add_signal] += 1
				if bin2add_signal in idx_ones:
					counts[bin2add_signal] -= 1

		bin_edges, bin_edges_distance = self.bin_edges()

		return counts, bin_edges, bin_edges_distance

	def bin_edges(self):
		bins = self.bins
		timing_window = self.params["timing_window"]
		bin_edges = ((np.arange(bins + 1) * timing_window) - timing_window / 2)
		bin_edges_distance = (np.arange(bins + 1) * timing_window * 299792458) - (timing_window*299792458/2)
		return bin_edges, bin_edges_distance


