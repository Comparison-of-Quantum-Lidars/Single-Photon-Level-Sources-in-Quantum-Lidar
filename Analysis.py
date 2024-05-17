import matplotlib.pyplot as plt
from scipy.stats import binom
from typing import Optional
import numpy as np


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
