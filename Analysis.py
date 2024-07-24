from Sources import EntangledPhotonSPDC, SinglePhoton, PulsedLaser
from scipy.special import gammaln
from scipy.stats import binom
from typing import Optional
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class RocAnalysis:
	"""
	Receiver Operating Characteristic (ROC) Analysis
	This class is used to compute the ROC curve for a quantum LiDAR system.
	It returns the true positive array as a function of the false positive array.
	After creating an object RocAnalysis, the method compute_p_d_p_fa() must be called to compute the ROC curve.
	"""

	def __init__(
			self,
			signal_rate,
			noise_rate,
			trigger_rate,
			threshold_limit,
			range_interval,
			timing_window,
			acquisition_time=1,
			precision=20
	):
		"""
		:param signal_rate: Detection of photons in one second when the target is present.
		:param noise_rate: Detection of photons in one second when the target is absent.
		:param trigger_rate: Amount of triggers per second.
		:param threshold_limit: Maximum threshold value considered for the ROC Curve. It should big enough such that
		it is statistically impossible to cross it.
		:param range_interval: Maximum distance that can be resolved by the LiDAR system. The greater is the range interval,
		the more bins will be considered in the histogram which increase the probability that noise surpass the threshold.
		:param timing_window: Time interval in which the LiDAR system is able to detect photons.
		:param acquisition_time: Time in seconds that the LiDAR system is acquiring data.
		:param precision: Number of points considered in the threshold array between two integers. 1/precision is the
		threshold step.
		"""
		self.signal_rate = signal_rate
		self.noise_rate = noise_rate
		self.trigger_rate = trigger_rate
		self.threshold_limit = threshold_limit
		self.range_interval = range_interval
		self.timing_window = timing_window
		self.acquisition_time = acquisition_time
		self.precision = precision

	def compute_q0_q1(self):
		"""
		Compute the probability of measuring noise or a signal in a time window.
		:return: the probability of measuring noise and the probability of measuring a signal. [float, float]
		"""
		q0 = self.noise_rate / self.trigger_rate
		q1 = self.signal_rate / self.trigger_rate
		return q0, q1

	def create_threshold_array(self):
		"""
		Create an array of threshold values.
		:return: an array of threshold values. [np.array]
		"""
		threshold = np.linspace(0, int(self.threshold_limit), (self.precision*int(self.threshold_limit)) + 1)
		return threshold

	def compute_binomial_experiment(self, threshold, q):
		"""
		Continuous binomial experiment to compute the probability of noise or signal to go above a threshold.
		:param threshold: Array of threshold to test for the binomial experiment.
		:param q: Probability of measuring a signal photon or a noise photon (q0 or q1).
		:return: an array of probabilities of measuring a signal or noise above a threshold. [np.array]
		"""
		n = int(self.trigger_rate * self.acquisition_time)
		k = threshold

		prob_mass_function_log = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) + (k * np.log(q)) + ((n - k) * np.log(1 - q))
		prob_mass_function = np.exp(prob_mass_function_log)

		distance_between_thresholds = 1/self.precision

		p = prob_mass_function * distance_between_thresholds

		return p

	def number_of_bins(self):
		"""
		Compute the number of bins in the histogram. The number of bins is proportional to the range interval and the
		timing window.
		:return: the number of bins in the histogram. [int]
		"""
		n_bins = 2 * self.range_interval / (299792458 * self.timing_window)
		return n_bins

	def compute_p_d_p_fa(self):
		"""
		Compute the true positive and false positive arrays needed to plot the ROC curve. This is the function
		that must be called after creating the object RocAnalysis.
		:return: the true positive and false positive arrays. Both array will have the same length. [np.array, np.array]
		"""
		threshold = self.create_threshold_array()
		q0, q1 = self.compute_q0_q1()
		p_0 = self.compute_binomial_experiment(threshold, q0)
		p_1 = self.compute_binomial_experiment(threshold, q1)

		false_positive = 1 - ((1 - np.cumsum(np.flip(p_0))) ** self.number_of_bins())
		true_positive = np.cumsum(np.flip(p_1))

		return true_positive, false_positive

	@staticmethod
	def compute_discrete_binomial_experiment(threshold, n, q):
		"""
		Discrete binomial experiment to compute the probability of noise or signal to go above a threshold.
		:param threshold: Array of threshold to test for the binomial experiment.
		:param n: Number of trials in the binomial experiment: number of triggers.
		:param q: Probability of measuring a signal photon or a noise photon (q0 or q1).
		:return: an array of probabilities of measuring a signal or noise above a threshold. [np.array]
		"""
		p = binom.pmf(threshold, n, q)
		return p

	def markers_integer_threshold(self):
		"""
		Return the marker associated with the true positive and false positive for integer threshold values. This can
		be used to compare the ROC curves using the discrete or the continuous binomial experiment.
		:return: the true positive and false positive arrays for integer threshold values. [np.array, np.array]
		"""
		threshold = np.linspace(0, int(self.threshold_limit), int(self.threshold_limit) + 1)
		q0, q1 = self.compute_q0_q1()
		n = int(self.trigger_rate * self.acquisition_time)
		p_0 = self.compute_discrete_binomial_experiment(threshold, n, q0)
		p_1 = self.compute_discrete_binomial_experiment(threshold, n, q1)

		false_positive = 1 - ((1 - np.cumsum(np.flip(p_0))) ** self.number_of_bins())
		true_positive = np.cumsum(np.flip(p_1))

		return true_positive, false_positive


class HistogramAnalysis:
	"""
	Produce a simulated detection histogram for a quantum LiDAR system using a non-resolving detector.
	The simulation takes into account that if noise is detected in the bin associated with the signal, then the signal
	can't be detected. Some rare events are not taken into account in this simulation, such as:
		- Multiple noise photons in the same bin.
		- Signal is not mapped to the correct idler photon.
	"""
	def __init__(self, params, signal_rate, noise_rate, acquisition_time, range_distance, effective_trigger_rate,
	             jitter_std_dev: Optional[float] = 0, **kwargs):
		"""
		:param params: SetupParams object containing the parameters of the LiDAR system.
		:param signal_rate: Amount of signal photons detected per second.
		:param noise_rate: Amount of noise photons detected per second.
		:param acquisition_time: Time in seconds that the LiDAR system is acquiring data.
		:param range_distance: Maximum distance that can be resolved by the LiDAR system.
		:param effective_trigger_rate: Amount of triggers per second.
		:param jitter_std_dev: Standard deviation of the jitter in the detection time. To reproduce the results by the
		SNR computation, the jitter_std_dev should be set to 0.
		"""
		self.params = params
		self.effective_trigger_rate = effective_trigger_rate
		self.signal_rate = signal_rate
		self.noise_rate = noise_rate
		self.acquisition_time = acquisition_time
		self.range_distance = range_distance
		self.jitter_std_dev = jitter_std_dev
		self.bins = self.compute_bins_number()

	def compute_signal_and_noise_rate_per_bins(self):
		"""
		Compute the signal and noise rate per bins. The signal is now defined as the reflected photons from the target.
		The contribution of the background noise is now removed in order to separate the photon that are reflected from
		the target and the one that are not.
		:return: the signal (reflected photons) and noise rate per bins (photons that are not reflected from the target).
		[float, float]
		"""
		signal = self.signal_rate - self.noise_rate
		return signal, self.noise_rate

	def compute_bins_number(self):
		"""
		Compute the number of bins in the histogram. The number of bins is proportional to the range interval and the
		timing window.
		:return: the number of bins in the histogram. [int]
		"""
		bins = round(2 * self.range_distance / (299792458 * self.params["timing_window"]))
		return bins

	def compute_trigger_total(self):
		"""
		Compute the total amount of triggers that will be generated during the acquisition time.
		:return: the total amount of triggers. [float]
		"""
		trigger_total = self.effective_trigger_rate * self.acquisition_time
		return trigger_total

	def noise_and_signal_prob_per_bins(self):
		"""
		Compute the probability of measuring noise or signal in a bin. This is done by computing the overall noise/signal
		and by dividing it by the total amount of triggers.
		:return: the probability of measuring noise and the probability of measuring a signal. [float, float]
		"""
		signal, noise = self.compute_signal_and_noise_rate_per_bins()
		trigger_total = self.compute_trigger_total()

		noise_prob = (noise * self.acquisition_time) / trigger_total
		signal_prob = (signal * self.acquisition_time) / trigger_total

		return noise_prob, signal_prob

	def compute_max_time_of_flight(self):
		"""
		Compute the maximum time of flight that can be resolved by the LiDAR system based on the maximum distance.
		:return: the maximum time of flight that can be resolved by the LiDAR system. [float]
		"""
		max_time_of_flight = 2 * self.range_distance / 299792458
		return max_time_of_flight

	@staticmethod
	def random_boolean(prob):
		"""
		Return a boolean value based on a probability.
		:param prob: probability of returning True.
		:return: a boolean value. [bool]
		"""
		return random.random() < prob

	def histogram_simulation(self):
		"""
		Simulate the detection histogram that can be obtained with the given parameters.
		For each triggering rate:
			1) Generate a random number of noise photons based on the noise probability based on a binomial distribution.
			2) Distribute the noise photons randomly in the bins. Only one noise photon can be in a bin per trigger.
			3) Apply a binomial experiment to determine if a signal photon is detected in the bin associated with the
			signal.
			4) If a signal photon is detected, it is added to the bin by taking into account the jitter.
			5) If a noise photon was already in the bin, the signal photon is not added.
		:return: The histogram counts as an array, the bin edges in time of flight and the bin edges in distance.
		[np.array, np.array, np.array]
		"""
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
		"""
		Compute the bin edges in time of flight and distance.
		:return: the bin edges in time of flight and distance. [np.array, np.array].
		"""
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
