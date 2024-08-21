import unittest
from Sources import SinglePhoton, SetupParameters
import numpy as np
from Analysis import RangeLimitation
from copy import deepcopy


class TestRangeLimitation(unittest.TestCase):

	def setUp(self):
		self.fock_space_dim = 5
		self.output_power = 1e6
		self.multi_photon_probability = None
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
		self.match_multi_photon_probability = True
		self.range_interval = 50
		self.distance = np.linspace(1, 50, 100)
		self.acquisition_time = np.array([1, 60 ,3600])
		self.target_false_positive = np.random.rand()
		self.target_true_positive = np.random.rand()
		self.precision_roc = 1
		self.threshold_limit_factor_roc = 10
		self.range_limitation = RangeLimitation(
			params=self.setup_params,
			match_multi_photon_probability=self.match_multi_photon_probability,
			range_interval=self.range_interval,
			distance=self.distance,
			acquisition_time=self.acquisition_time,
			target_false_positive=self.target_false_positive,
			target_true_positive=self.target_true_positive,
			precision_roc=self.precision_roc,
			threshold_limit_factor_roc=self.threshold_limit_factor_roc
		)

	def test_distance_at_target(self):
		distance = np.array([False, False, True, False])
		true_positive_at_target_false_value = np.array([1, 2, 3, 4])
		target_true = 3.1

		result = self.range_limitation.distance_at_target(
			distance=distance,
			true_positive_at_target_false_value=true_positive_at_target_false_value,
			target_true=target_true
		)

		self.assertTrue(result)

	def test_set_multi_or_non_vacuum_prob(self):
		sps = SinglePhoton(self.setup_params)
		multi_photon_sps = sps.multi_photon_probability
		non_vacuum_sps = sps.no_vacuum_probability

		match_multi_photon_probability = True

		rl = RangeLimitation(
			params=self.setup_params,
			match_multi_photon_probability=match_multi_photon_probability,
			range_interval=self.range_interval,
			distance=self.distance,
			acquisition_time=self.acquisition_time,
			target_false_positive=self.target_false_positive,
			target_true_positive=self.target_true_positive,
			precision_roc=self.precision_roc,
			threshold_limit_factor_roc=self.threshold_limit_factor_roc
		)

		self.assertEqual(multi_photon_sps, rl.sps.multi_photon_probability)
		self.assertEqual(multi_photon_sps, rl.laser.multi_photon_probability)
		self.assertEqual(multi_photon_sps, rl.eps.multi_photon_probability)

		match_multi_photon_probability = False

		rl = RangeLimitation(
			params=self.setup_params,
			match_multi_photon_probability=match_multi_photon_probability,
			range_interval=self.range_interval,
			distance=self.distance,
			acquisition_time=self.acquisition_time,
			target_false_positive=self.target_false_positive,
			target_true_positive=self.target_true_positive,
			precision_roc=self.precision_roc,
			threshold_limit_factor_roc=self.threshold_limit_factor_roc
		)

		self.assertEqual(non_vacuum_sps, rl.sps.no_vacuum_probability)
		self.assertEqual(non_vacuum_sps, rl.laser.no_vacuum_probability)
		self.assertEqual(non_vacuum_sps, rl.eps.no_vacuum_probability)

	def test_prepare_dict_true_positive_at_target_false_value(self):
		for at in self.acquisition_time:
			dict_laser = self.range_limitation.true_positive_at_target_false_value_laser[at]
			shape_laser = dict_laser.shape
			dict_eps = self.range_limitation.true_positive_at_target_false_value_eps[at]
			shape_eps = dict_eps.shape
			dict_sps = self.range_limitation.true_positive_at_target_false_value_sps[at]
			shape_sps = dict_sps.shape

			self.assertEqual(shape_laser, self.range_limitation.distance.shape)
			self.assertEqual(shape_eps, self.range_limitation.distance.shape)
			self.assertEqual(shape_sps, self.range_limitation.distance.shape)


if __name__ == '__main__':
	unittest.main()
