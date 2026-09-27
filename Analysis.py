from Sources import EntangledPhotonSPDC, SinglePhoton, PulsedLaser
from scipy.special import gammaln, betainc
from scipy.stats import binom
from typing import Optional
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy


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
			precision=20,
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
		threshold = np.linspace(0, int(self.threshold_limit), (self.precision * int(self.threshold_limit)) + 1)
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

		prob_mass_function_log = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) + (k * np.log(q)) + (
					(n - k) * np.log(1 - q))
		prob_mass_function = np.exp(prob_mass_function_log)

		distance_between_thresholds = 1 / self.precision
		p = prob_mass_function * distance_between_thresholds

		return p

	def number_of_bins(self):
		"""
		Compute the number of bins in the histogram. The number of bins is proportional to the range interval and the
		timing window.
		:return: the number of bins in the histogram. [int]
		"""
		n_bins = 2 * self.range_interval / (299792458 * self.timing_window)
		return round(n_bins)

	def compute_p_d_p_fa(self):
		"""
		Compute the true positive and false positive arrays needed to plot the ROC curve. This is the function
		that must be called after creating the object RocAnalysis.
		:return: the true positive and false positive arrays. Both array will have the same length. [np.array, np.array]
		"""
		threshold = self.create_threshold_array()
		q0, q1 = self.compute_q0_q1()

		p_0 = self.alternative_binomial_experiment(threshold, q0)
		p_1 = self.alternative_binomial_experiment(threshold, q1)

		false_positive = 1 - ((1 - p_0) ** self.number_of_bins())
		true_positive = p_1

		return true_positive, false_positive

	def alternative_binomial_experiment(self, threshold, q):
		n = int(self.trigger_rate * self.acquisition_time)
		k = threshold

		cumulative_sum_p = (1 - betainc(k, n+1-k, q)) / (1 - betainc(k, n+1-k, 0))  #Prob of being between 0 and k
		cumulative_sum_p[np.isnan(cumulative_sum_p)] = 0  # Handle k=0

		return 1 - cumulative_sum_p  #Prob of being above k

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


class RangeLimitation:
	"""
	Compute the SNR as a function of the distance. The maximum distance at which a certains specified true positive rate
	is obtained for a given false positive rate is computed so the user can then plot it as markers.
	"""

	def __init__(
			self,
			params,
			parameter_to_match: str,
			range_interval: Optional[float],
			distance: np.array,
			acquisition_time: np.array,
			target_false_positive: float,
			target_true_positive: float,
			precision_roc: int,
			threshold_limit_factor_roc: int,
			number_nv_pulse_for_match: Optional[float] = None,
			number_sps_array: int = 1,
			early_stop: bool = False,
			show_progress_bar: bool = True
	):
		"""
		:param params: SetupParams object containing the parameters of the LiDAR system.
		:param parameter_to_match: Specify parameters to match for the comparison. The options are: "number_nv_pulse",
		"multi_photon_probability" or "no_vacuum_probability". Another input will yield an error.
		:param range_interval: Maximum distance that can be resolved by the LiDAR system. The greater is the range interval,
		the more bins will be considered in the histogram which increase the probability that noise surpass the threshold.
		If set to None, the range interval is computed based on the current distance considered: it is therefore the best
		case scenario.
		:param distance: Array of distance at which the SNR is computed. It is recommended to use a linearly spaced array.
		:param acquisition_time: Array of acquisition time at which the SNR is computed in seconds.
		:param target_false_positive: Target false positive rate for the ROC curve.
		:param target_true_positive: Target true positive rate for the ROC curve.
		:param precision_roc: Number of points considered in the threshold array between two integers. 1/precision is the
		threshold step.
		:param threshold_limit_factor_roc: Factor to determine the threshold limit for the ROC curve. The threshold limit
		is the maximum threshold value considered for the ROC Curve. It should big enough such that it is statistically
		impossible to cross it. The better is the system, the lower should be this parameter. The threshold is computed
		using trigger_rate/threshold_limit_factor_roc.
		:param number_sps_array: Number of Single Photon sources to consider in the analysis.
		:param early_stop: If True, the computation will stop for a given acquisition time when the true positive rate
		is below the target true positive rate for all sources. This can save computation time when the distance is too large for the given acquisition time.
		:param show_progress_bar: If True, a progress bar will be shown during the computation.
		"""
		self.params = params
		self.parameter_to_match = parameter_to_match
		self.range_interval = range_interval
		self.distance = distance
		self.acquisition_time = acquisition_time
		self.target_false_positive = target_false_positive
		self.target_true_positive = target_true_positive
		self.precision_roc = precision_roc
		self.threshold_limit_factor_roc = threshold_limit_factor_roc
		self.number_sps_array = number_sps_array
		self.early_stop = early_stop
		self.show_progress_bar = show_progress_bar

		self.number_nv_pulse_for_match = number_nv_pulse_for_match if number_nv_pulse_for_match is not None else params["number_nv_pulse"]

		self.param_laser = deepcopy(self.params)
		self.param_sps = deepcopy(self.params)
		self.param_eps = deepcopy(self.params)

		try:
			self.sps = SinglePhoton(self.param_sps, number_sps=self.number_sps_array)
			if self.parameter_to_match == "number_nv_pulse" and number_nv_pulse_for_match is not None:
				nv_pulse_sps = self.sps.number_nv_pulse
				assert nv_pulse_sps == self.number_nv_pulse_for_match, "The number of NV pulse for the SPS does not match the one for the laser and the SPDC."
			self.number_nv_pulse_for_match = self.sps.number_nv_pulse
			self.keep_sps = True
		except AssertionError:
			self.keep_sps = False

		self.set_param_to_fix()

		self.laser = PulsedLaser(self.param_laser)
		self.eps = EntangledPhotonSPDC(self.param_eps)

		self.snr_laser = []
		self.snr_sps = []
		self.snr_eps = []

		self.noise_laser_all = []
		self.noise_sps_all = []
		self.noise_eps_all = []

		self.true_positive_at_target_false_value_laser = {}
		self.true_positive_at_target_false_value_sps = {}
		self.true_positive_at_target_false_value_eps = {}

		self.prepare_dict_true_positive_at_target_false_value()

		self.distance_cutoff_laser = {}
		self.distance_cutoff_sps = {}
		self.distance_cutoff_eps = {}

	@staticmethod
	def distance_at_target(distance, true_positive_at_target_false_value, target_true):
		"""
		Compute the distance at which the true positive rate is the closest to the target true positive rate.
		:param distance: Array of distance considered for the analysis.
		:param true_positive_at_target_false_value: Array of true positive rate at the target false positive rate.
		:param target_true: Target true positive rate.
		:return: the distance at which the true positive rate is the closest to the target true positive rate. [float]
		"""
		idx2keep = np.argmin(np.abs(true_positive_at_target_false_value - target_true))
		return distance[idx2keep]

	def set_param_to_fix(self):
		"""
		Set the multi-photon probability or the non-vacuum probability based on the SPS.
		"""
		assert self.parameter_to_match in ["number_nv_pulse", "multi_photon_probability", "no_vacuum_probability"], "The parameters to match is not valid. The three options are: number_nv_pulse, multi_photon_probability, no_vacuum_probability"

		if self.parameter_to_match == "multi_photon_probability":
			sps = SinglePhoton(self.param_sps, number_sps=self.number_sps_array)
			multi_photon_probability = sps.multi_photon_probability
			self.param_laser["multi_photon_probability"] = multi_photon_probability
			self.param_eps["multi_photon_probability"] = multi_photon_probability
		elif self.parameter_to_match == "no_vacuum_probability":
			sps = SinglePhoton(self.param_sps, number_sps=self.number_sps_array)
			no_vacuum_probability = sps.no_vacuum_probability
			self.param_laser["no_vacuum_probability"] = no_vacuum_probability
			self.param_eps["no_vacuum_probability"] = no_vacuum_probability
		elif self.parameter_to_match == "number_nv_pulse":
			self.param_laser["number_nv_pulse"] = self.number_nv_pulse_for_match
			self.param_eps["number_nv_pulse"] = self.number_nv_pulse_for_match
			if self.keep_sps:
				self.param_sps["number_nv_pulse"] = self.number_nv_pulse_for_match
		else:
			raise ValueError("Parameters to match not valid.")

	def prepare_dict_true_positive_at_target_false_value(self):
		"""
		Prepare the dictionary that will contain the true positive rate at the target false positive rate for each
		"""
		for at in self.acquisition_time:
			self.true_positive_at_target_false_value_laser[at] = np.zeros_like(self.distance)
			self.true_positive_at_target_false_value_sps[at] = np.zeros_like(self.distance)
			self.true_positive_at_target_false_value_eps[at] = np.zeros_like(self.distance)

	def compute(self):
		"""
		Compute the SNR as a function of the distance with the ROC Curves analysis. Here are the steps of the computation
		1) For a given distance, compute the signal rate, the noise rate and the triggering rate for each source.
		2) Compute the SNR for each source.
		3) For each acquisition time at a specific distance, compute the ROC curve for each source.
		4) Compute the true positive rate at the target false positive rate for each source. Linear interpolation is used
		to compute the true positive rate at the target false positive rate.
		5) Determine the distance at which the true positive rate is the closest to the target true positive rate. This
		is the distance cutoff.
		:return: a dictionary containing the distance, the SNR for each source, the distance cutoff for each source and
		the SNR at the distance cutoff for each source. [dict]
		"""
		early_stop_dict = {
			"laser": {acquisition_time: False for acquisition_time in self.acquisition_time},
			"sps": {acquisition_time: False for acquisition_time in self.acquisition_time},
			"eps": {acquisition_time: False for acquisition_time in self.acquisition_time}
		}

		if not self.keep_sps:
			early_stop_dict["sps"] = {acquisition_time: True for acquisition_time in self.acquisition_time}

		for idx, d in enumerate(tqdm(self.distance, disable=not self.show_progress_bar)):
			if self.early_stop and all(early_stop_dict[source][at] for source in early_stop_dict for at in early_stop_dict[source]):
				break
			self.param_laser["target_distance"] = d
			self.param_sps["target_distance"] = d
			self.param_eps["target_distance"] = d

			self.laser = PulsedLaser(self.param_laser)
			if self.keep_sps:
				self.sps = SinglePhoton(self.param_sps, number_sps=self.number_sps_array)
			self.eps = EntangledPhotonSPDC(self.param_eps)

			signal_laser = self.laser.signal_rate()
			noise_laser = self.laser.noise_rate()
			trigger_rate_laser = self.laser.trigger_rate
			snr_laser_current = (signal_laser - noise_laser) / noise_laser
			self.snr_laser.append(snr_laser_current)

			if self.keep_sps:
				signal_sps = self.sps.signal_rate()
				noise_sps = self.sps.noise_rate()
				trigger_rate_sps = self.sps.trigger_rate
				snr_sps_current = (signal_sps - noise_sps) / noise_sps
				self.snr_sps.append(snr_sps_current)

			signal_eps = self.eps.signal_rate()
			noise_eps = self.eps.noise_rate()
			trigger_rate_eps = self.eps.trigger_rate
			snr_eps_current = (signal_eps - noise_eps) / noise_eps
			self.snr_eps.append(snr_eps_current)

			for at in self.acquisition_time:

				if self.early_stop:
					if early_stop_dict["laser"][at] and early_stop_dict["sps"][at] and early_stop_dict["eps"][at]:
						continue

				true_positive_laser, false_positive_laser = self.compute_roc_curve(
					signal=signal_laser,
					noise=noise_laser,
					trigger_rate=trigger_rate_laser,
					distance=d,
					acquisition_time=at
				)

				self.true_positive_at_target_false_value_laser[at][idx] = np.interp(
					self.target_false_positive,
					np.flip(false_positive_laser),
					np.flip(true_positive_laser)
				)
				# Flip the arrays to have an increasing false/true positive array for the interpolation

				if self.keep_sps:
					true_positive_sps, false_positive_sps = self.compute_roc_curve(
						signal=signal_sps,
						noise=noise_sps,
						trigger_rate=trigger_rate_sps,
						distance=d,
						acquisition_time=at
					)

					self.true_positive_at_target_false_value_sps[at][idx] = np.interp(
						self.target_false_positive,
						np.flip(false_positive_sps),
						np.flip(true_positive_sps)
					)

				true_positive_eps, false_positive_eps = self.compute_roc_curve(
					signal=signal_eps,
					noise=noise_eps,
					trigger_rate=trigger_rate_eps,
					distance=d,
					acquisition_time=at
				)

				self.true_positive_at_target_false_value_eps[at][idx] = np.interp(
					self.target_false_positive,
					np.flip(false_positive_eps),
					np.flip(true_positive_eps)
				)

				if self.early_stop:
					if self.true_positive_at_target_false_value_laser[at][idx] < self.target_true_positive:
						early_stop_dict["laser"][at] = True
					if self.keep_sps and self.true_positive_at_target_false_value_sps[at][idx] < self.target_true_positive:
						early_stop_dict["sps"][at] = True
					if self.true_positive_at_target_false_value_eps[at][idx] < self.target_true_positive:
						early_stop_dict["eps"][at] = True

		results = self.prepare_results()

		return results

	def compute_roc_curve(self, signal, noise, trigger_rate, distance, acquisition_time):
		"""
		Compute the ROC curve for a given source.
		:param signal: signal rate for the source.
		:param noise: noise rate for the source.
		:param trigger_rate: trigger rate for the source.
		:param distance: distance at which the ROC curve is computed.
		:param acquisition_time: acquisition time for the ROC curve.
		:return: the true positive and false positive arrays. [np.array, np.array]
		"""
		true_positive, false_positive = RocAnalysis(
			signal_rate=signal,
			noise_rate=noise,
			trigger_rate=trigger_rate,
			threshold_limit=trigger_rate / self.threshold_limit_factor_roc,
			range_interval=distance if self.range_interval is None else self.range_interval,
			timing_window=self.params["timing_window"],
			acquisition_time=acquisition_time,
			precision=self.precision_roc
		).compute_p_d_p_fa()

		return true_positive, false_positive

	def prepare_results(self):
		"""
		Prepare the results in a dictionary.
		:return: a dictionary containing the distance, the SNR for each source, the distance cutoff for each source and
		the SNR at the distance cutoff for each source. [dict]
		"""
		self.snr_laser = np.array(self.snr_laser)
		self.snr_sps = np.array(self.snr_sps)
		self.snr_eps = np.array(self.snr_eps)

		self.cutoff_distance()

		results = {
			"distance": self.distance,
			"snr_laser": self.snr_laser,
			"snr_sps": self.snr_sps,
			"snr_eps": self.snr_eps,
			"distance_cutoff_laser": self.distance_cutoff_laser,
			"distance_cutoff_sps": self.distance_cutoff_sps,
			"distance_cutoff_eps": self.distance_cutoff_eps,
		}

		return results

	def cutoff_distance(self):
		"""
		Compute the distance cutoff for each source.
		"""

		for idx, at in enumerate(self.acquisition_time):
			self.distance_cutoff_laser[at] = self.distance_at_target(
				self.distance,
				self.true_positive_at_target_false_value_laser[at],
				self.target_true_positive
			)

			if self.keep_sps:
				self.distance_cutoff_sps[at] = self.distance_at_target(
					self.distance,
					self.true_positive_at_target_false_value_sps[at],
					self.target_true_positive
				)

			self.distance_cutoff_eps[at] = self.distance_at_target(
				self.distance,
				self.true_positive_at_target_false_value_eps[at],
				self.target_true_positive
			)

			key = "snr_at" + str(idx)

			self.distance_cutoff_laser[key] = self.snr_laser[np.where(self.distance == self.distance_cutoff_laser[at])]
			if self.keep_sps:
				self.distance_cutoff_sps[key] = self.snr_sps[np.where(self.distance == self.distance_cutoff_sps[at])]
			self.distance_cutoff_eps[key] = self.snr_eps[np.where(self.distance == self.distance_cutoff_eps[at])]

