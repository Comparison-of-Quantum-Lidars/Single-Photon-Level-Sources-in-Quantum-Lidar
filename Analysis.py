from Sources import EntangledPhotonSPDC, SinglePhoton, PulsedLaser
from scipy.special import gammaln
from scipy.stats import binom
from typing import Optional
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class RocAnalysis:

	def __init__(
			self,
			signal_rate,
			noise_rate,
			trigger_rate,
			threshold_limit,
			range_interval,
			timing_window,
			acquisition_time=1,
			precision=50
	):
		#TODO: ADD DOCUMENTATION, INTEGRATION TIME MUST BE IN SECOND TO FIT WITH THE TRIGGERING RATE
		#TODO: ADD OPTION TO SEE MARKER WHEN THE THRESHOLD IS FOR FIX PARAMS
		self.signal_rate = signal_rate
		self.noise_rate = noise_rate
		self.trigger_rate = trigger_rate
		self.threshold_limit = threshold_limit
		self.range_interval = range_interval
		self.timing_window = timing_window
		self.acquisition_time = acquisition_time
		self.precision = precision

	def compute_q0_q1(self):
		q0 = self.noise_rate / self.trigger_rate
		q1 = self.signal_rate / self.trigger_rate
		return q0, q1

	def create_threshold_array(self):
		threshold = np.linspace(0, int(self.threshold_limit), (self.precision*int(self.threshold_limit)) + 1)
		return threshold

	def compute_binomial_experiment(self, threshold, q):

		n = int(self.trigger_rate * self.acquisition_time)
		k = threshold

		prob_mass_function_log = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) + (k * np.log(q)) + ((n - k) * np.log(1 - q))
		prob_mass_function = np.exp(prob_mass_function_log)

		distance_between_thresholds = 1/self.precision

		p = prob_mass_function * distance_between_thresholds

		return p

	def number_of_bins(self):
		n_bins = 2 * self.range_interval / (299792458 * self.timing_window)
		return n_bins

	def compute_p_d_p_fa(self):
		threshold = self.create_threshold_array()
		q0, q1 = self.compute_q0_q1()
		p_0 = self.compute_binomial_experiment(threshold, q0)
		p_1 = self.compute_binomial_experiment(threshold, q1)

		false_positive = 1 - ((1 - np.cumsum(np.flip(p_0))) ** self.number_of_bins())
		true_positive = np.cumsum(np.flip(p_1))

		return true_positive, false_positive

	def compute_discrete_binomial_experiment(self, threshold, n, q):
		p = binom.pmf(threshold, n, q)
		return p

	def markers_integer_threshold(self):
		threshold = np.linspace(0, int(self.threshold_limit), int(self.threshold_limit) + 1)
		q0, q1 = self.compute_q0_q1()
		n = int(self.trigger_rate * self.acquisition_time)
		p_0 = self.compute_discrete_binomial_experiment(threshold, n, q0)
		p_1 = self.compute_discrete_binomial_experiment(threshold, n, q1)

		false_positive = 1 - ((1 - np.cumsum(np.flip(p_0))) ** self.number_of_bins())
		true_positive = np.cumsum(np.flip(p_1))

		return true_positive, false_positive

	def debug(self):
		#TODO : REMOVE THIS FUNCTION
		threshold_dis = np.linspace(0, int(self.threshold_limit), int(self.threshold_limit) + 1)
		threshold_exp = np.linspace(0, int(self.threshold_limit), self.precision*int(self.threshold_limit) + 1)
		q0, q1 = self.compute_q0_q1()
		n = int(self.trigger_rate * self.acquisition_time)
		p_0_dis = self.compute_discrete_binomial_experiment(threshold_dis, n, q0)
		p_1_dis = self.compute_discrete_binomial_experiment(threshold_dis, n, q1)

		p_0_exp = self.compute_binomial_experiment(threshold_exp, q0)
		p_1_exp = self.compute_binomial_experiment(threshold_exp, q1)

		# plt.plot(threshold_exp, p_0_exp*self.precision, "-", label="p_0_exp", linewidth=2.5)
		# plt.plot(threshold_dis, p_0_dis, "--", label="p_0_dis", linewidth=2.5)
		# plt.show()

		plt.plot(threshold_exp, p_1_exp*self.precision, "-", label="Continuous", linewidth=2.5)
		plt.plot(threshold_dis, p_1_dis, "--", label="Discrete", linewidth=2.5)
		plt.legend(frameon=False, fontsize=22)
		plt.xlabel("k: Threshold", fontsize=22)
		plt.ylabel("Probability of having k signal photons", fontsize=22)
		plt.tick_params(labelsize=22)
		plt.show()

		false_positive_dis = 1 - ((1 - np.cumsum(np.flip(p_0_dis))) ** self.number_of_bins())
		true_positive_dis = np.cumsum(np.flip(p_1_dis))

		false_positive_exp = 1 - ((1 - np.cumsum(np.flip(p_0_exp))) ** self.number_of_bins())
		true_positive_exp = np.cumsum(np.flip(p_1_exp))

		plt.plot(false_positive_exp, true_positive_exp, "-", label="exp")
		plt.scatter(false_positive_dis, true_positive_dis, label="dis")
		plt.legend()
		plt.show()



class HistogramAnalysis:

	def __init__(self, params, signal_rate, noise_rate, acquisition_time, range_distance, effective_trigger_rate,
	             jitter_std_dev: Optional[float] = 0, **kwargs):
		self.params = params
		self.effective_trigger_rate = effective_trigger_rate
		self.signal_rate = signal_rate
		self.noise_rate = noise_rate
		self.acquisition_time = acquisition_time
		self.range_distance = range_distance
		self.jitter_std_dev = jitter_std_dev
		self.bins = self.compute_bins_number()

	def compute_signal_and_noise_rate_per_bins(self):
		signal = self.signal_rate - self.noise_rate
		return signal, self.noise_rate

	def compute_bins_number(self):
		bins = round(2 * self.range_distance / (299792458 * self.params["timing_window"]))
		return bins

	def compute_trigger_total(self):
		trigger_total = self.effective_trigger_rate * self.acquisition_time
		return trigger_total

	def noise_and_signal_prob_per_bins(self):
		signal, noise = self.compute_signal_and_noise_rate_per_bins()
		trigger_total = self.compute_trigger_total()

		noise_prob = (noise * self.acquisition_time) / trigger_total
		signal_prob = (signal * self.acquisition_time) / trigger_total

		return noise_prob, signal_prob

	def compute_max_time_of_flight(self):
		max_time_of_flight = 2 * self.range_distance / 299792458
		return max_time_of_flight

	@staticmethod
	def random_boolean(prob):
		return random.random() < prob

	def histogram_simulation(self):
		noise_prob, signal_prob = self.noise_and_signal_prob_per_bins()
		trigger_total = self.compute_trigger_total()
		tof_target = 2 * self.params["target_distance"] / 299792458
		timing_window = self.params["timing_window"]

		counts = np.zeros((self.bins, 1))

		for _ in tqdm(range(int(trigger_total))):
			idx_ones = np.array([-1])
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

		return counts[:, 0], bin_edges, bin_edges_distance

	def bin_edges(self):
		bins = self.bins
		timing_window = self.params["timing_window"]
		bin_edges = (np.arange(bins) * timing_window)
		bin_edges_distance = (bin_edges * 299792458) / 2
		return bin_edges, bin_edges_distance


class HistogramAnalysisFromAdversaryPerspective:

	def __init__(self, params_lidar, params_adversary, acquisition_time, source: str,
	             jitter_std_dev: Optional[float] = 0.5e-9):
		self.params_lidar = params_lidar
		self.params_adversary = params_adversary
		self.source = source
		if self.source not in ["pulsed", "single", "entangled"]:
			raise ValueError("Source must be either 'pulsed', 'single' or 'entangled'")
		self.acquisition_time = acquisition_time
		self.jitter_std_dev = jitter_std_dev

	def adjust_signal_noise_rate(self):
		if self.source == "pulsed":
			signal_rate, noise_rate = self.adjust_signal_noise_for_pulsed()
			return signal_rate, noise_rate
		elif self.source == "single":
			pass
		else:
			pass

	def histogram_simulation_adversary(self):
		noise_prob, _ = self.noise_and_signal_prob_per_bins()
		signal_rate, noise_rate = self.adjust_signal_noise_rate()
		ha = HistogramAnalysis(self.params_adversary, signal_rate, noise_rate, self.acquisition_time,
		                       self.jitter_std_dev)
		trigger_total = ha.compute_trigger_total()
		timing_window = self.params_adversary["timing_window"]
		bins = ha.compute_bins_number()
		counts = np.zeros((bins, 1))

		for _ in tqdm(range(int(trigger_total))):
			idx_ones = np.array([-1])
			noise2add = np.random.binomial(bins, noise_prob)
			if noise2add != 0:
				bin_with_noise = np.ones((noise2add, 1))
				other_bin = np.zeros((bins - noise2add, 1))
				count2add_noise = np.concatenate((bin_with_noise, other_bin))
				np.random.shuffle(count2add_noise)
				idx_ones = np.where(count2add_noise == 1)
				counts += count2add_noise

		signal_total = signal_rate * self.acquisition_time
		bins_total = bins * self.acquisition_time
		pass

	def noise_and_signal_prob_per_bins(self):
		signal_rate, noise_rate = self.adjust_signal_noise_rate()
		noise_total = noise_rate * self.acquisition_time
		signal_total = signal_rate * self.acquisition_time
		ha = HistogramAnalysis(self.params_adversary, signal_rate, noise_rate, self.acquisition_time,
		                       self.jitter_std_dev)
		trigger_total = ha.compute_trigger_total()

		noise_prob = noise_total / trigger_total
		signal_prob = signal_total / trigger_total

		return noise_prob, signal_prob

	def adjust_signal_noise_for_pulsed(self):
		optics_transmitter_lidar = self.params_lidar["optics_transmitter"]
		optics_receiver_adversary = self.params_adversary["optics_receiver"]
		detection_efficiency_adversary = self.params_adversary["detection_efficiency"]
		eta_detection_adversary = optics_transmitter_lidar * optics_receiver_adversary * detection_efficiency_adversary

		#Photons from the LiDAR per second detected by the adversary
		signal_rate = PulsedLaser(self.params_lidar, eta_detection_adversary=eta_detection_adversary).signal_rate()

		noise_rate = (self.params_adversary["background"] + self.params_adversary["detector_dark"]) * \
		             self.params_adversary["timing_window"]
		return signal_rate, noise_rate
