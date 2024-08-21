import unittest
from Sources import SetupParameters, Source, PulsedLaser
from Analysis import RocAnalysis
from copy import deepcopy
import numpy as np


class TestRocAnalysis(unittest.TestCase):

	def setUp(self):
		self.fock_space_dim = 5
		self.output_power = 1e6
		self.multi_photon_probability = 0.1
		self.no_vacuum_probability = None
		self.sp_collection = 0.2
		self.sp_p1 = 0.99
		self.sp_p2 = 0.01
		self.spdc_eps_heralding = 0.1
		self.spdc_eps_collection = 0.2
		self.target_distance = 2
		self.atmosphere = 0.5
		self.receiver_diameter = 0.1
		self.target_albedo = 0.5
		self.optics_transmitter = 0.8
		self.optics_receiver = 0.8
		self.detection_efficiency = 0.5
		self.background = 100000
		self.detector_dark = 15
		self.timing_window = 1e-9
		self.setup_params = SetupParameters(
			fock_space_dim=self.fock_space_dim,
			output_power=self.output_power,
			multi_photon_probability=self.multi_photon_probability,
			no_vacuum_probability=self.no_vacuum_probability,
			sp_collection=self.sp_collection,
			sp_p1=self.sp_p1,
			sp_p2=self.sp_p2,
			spdc_eps_heralding=self.spdc_eps_heralding,
			spdc_eps_collection=self.spdc_eps_collection,
			target_distance=self.target_distance,
			atmosphere=self.atmosphere,
			receiver_diameter=self.receiver_diameter,
			target_albedo=self.target_albedo,
			optics_transmitter=self.optics_transmitter,
			optics_receiver=self.optics_receiver,
			detection_efficiency=self.detection_efficiency,
			background=self.background,
			detector_dark=self.detector_dark,
			timing_window=self.timing_window
		)
		self.laser = PulsedLaser(self.setup_params)
		self.signal = self.laser.signal_rate()
		self.noise = self.laser.noise_rate()
		self.trigger_rate = self.laser.trigger_rate
		self.threshold_limit = self.trigger_rate / 10
		self.range_interval = 50
		self.timing_window = self.laser.timing_window
		self.acquisition_time = 1
		self.precision = 25
		self.roc = RocAnalysis(
			signal_rate=self.signal,
			noise_rate=self.noise,
			trigger_rate=self.trigger_rate,
			threshold_limit=self.threshold_limit,
			range_interval=self.range_interval,
			timing_window=self.timing_window,
			acquisition_time=self.acquisition_time,
			precision=self.precision
		)

	def test_compute_q0_q1(self):
		q0_roc, q1_roc = self.roc.compute_q0_q1()
		q0 = self.noise / self.trigger_rate
		q1 = self.signal / self.trigger_rate

		self.assertEqual(q0, q0_roc)
		self.assertEqual(q1, q1_roc)

	def test_create_threshold_array(self):
		roc = deepcopy(self.roc)
		roc.threshold_limit = 10
		roc.precision = 2
		threshold = roc.create_threshold_array()
		expected_result = np.array(
			[0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.])

		self.assertTrue(threshold.all() == expected_result.all())

		roc.threshold_limit = 10.2
		threshold = roc.create_threshold_array()

		self.assertTrue(threshold.all() == expected_result.all())

	def test_compute_binomial_experiment(self):
		from scipy.stats import binom
		probability_of_success_on_a_trial = np.random.rand()
		number_of_trials = np.random.randint(10, 100)
		number_of_successes = int(number_of_trials * probability_of_success_on_a_trial)

		probability_scipy = binom.pmf(number_of_successes, number_of_trials, probability_of_success_on_a_trial)
		roc = deepcopy(self.roc)
		roc.trigger_rate = number_of_trials
		roc.precision = 1
		probability_roc = roc.compute_binomial_experiment(number_of_successes, probability_of_success_on_a_trial)

		self.assertAlmostEquals(probability_scipy, probability_roc, places=6)

	def test_number_of_bins(self):
		roc = deepcopy(self.roc)
		roc.range_interval = 299792458
		roc.timing_window = 1
		n_bins_roc = roc.number_of_bins()

		self.assertEqual(n_bins_roc, 2)


if __name__ == '__main__':
	unittest.main()
