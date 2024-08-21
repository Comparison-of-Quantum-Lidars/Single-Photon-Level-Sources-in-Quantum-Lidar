import unittest
import numpy as np
from Analysis import HistogramAnalysis
from Sources import SetupParameters, PulsedLaser
from copy import deepcopy


class TestHistogramAnalysis(unittest.TestCase):

	def setUp(self):
		self.fock_space_dim = 5
		self.output_power = 1e6
		self.multi_photon_probability = None
		self.no_vacuum_probability = 0.1
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
		self.trigger_rate = self.laser.trigger_rate
		self.signal = self.laser.signal_rate()
		self.noise = self.laser.noise_rate()
		self.acquisition_time = 1
		self.range_distance = 50
		self.jitter_std_dev = 0
		self.histogram = HistogramAnalysis(
			params=self.setup_params,
			effective_trigger_rate=self.trigger_rate,
			signal_rate=self.signal,
			noise_rate=self.noise,
			acquisition_time=self.acquisition_time,
			range_distance=self.range_distance,
			jitter_std_dev=self.jitter_std_dev
		)

	def test_compute_bins_number(self):
		range_distance = 299792458
		params = deepcopy(self.setup_params)
		params["timing_window"] = 1

		histo = HistogramAnalysis(
			params=params,
			effective_trigger_rate=1,
			signal_rate=1,
			noise_rate=1,
			acquisition_time=1,
			range_distance=range_distance,
			jitter_std_dev=0
		)
		n_bins = histo.compute_bins_number()

		self.assertEqual(n_bins, 2)

	def test_compute_trigger_total(self):
		trigger_total = self.trigger_rate * self.acquisition_time
		trigger_total_histo = self.histogram.compute_trigger_total()
		self.assertEqual(trigger_total, trigger_total_histo)

	def test_noise_and_signal_prob_per_bins(self):
		signal, noise = self.histogram.compute_signal_and_noise_rate_per_bins()
		trigger_total = self.histogram.compute_trigger_total()
		noise_prob = (noise * self.acquisition_time) / trigger_total
		signal_prob = (signal * self.acquisition_time) / trigger_total

		noise_histo, signal_histo = self.histogram.noise_and_signal_prob_per_bins()

		self.assertEqual(signal_prob, signal_histo)
		self.assertEqual(noise_prob, noise_histo)

	def test_compute_max_time_of_flight(self):
		max_time_of_flight = self.histogram.compute_max_time_of_flight()
		self.assertAlmostEquals(max_time_of_flight, 333.5640952e-9, places=5)


if __name__ == '__main__':
	unittest.main()
