import unittest
import numpy as np
from Sources import PulsedLaser, SetupParameters, Source
from copy import deepcopy
from qutip import Qobj


class TestPulsedLaser(unittest.TestCase):

	def setUp(self):
		self.fock_space_dim = 5
		self.output_power = 1e6
		self.multi_photon_probability = 0.1
		self.no_vacuum_probability = None
		self.number_nv_pulse = None
		self.number_mp_pulse = None
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
			number_nv_pulse=self.number_nv_pulse,
			number_mp_pulse=self.number_mp_pulse,
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
		self.source = Source(self.setup_params)
		self.laser = PulsedLaser(self.setup_params)

	def test_super_for_init_params(self):
		self.assertEqual(self.laser.fock_space_dim, self.fock_space_dim)
		self.assertEqual(self.laser.output_power, self.output_power)
		self.assertEqual(self.laser.sp_collection, self.sp_collection)
		self.assertEqual(self.laser.sp_p1, self.sp_p1)
		self.assertEqual(self.laser.sp_p2, self.sp_p2)
		self.assertEqual(self.laser.spdc_eps_heralding, self.spdc_eps_heralding)
		self.assertEqual(self.laser.spdc_eps_collection, self.spdc_eps_collection)
		self.assertEqual(self.laser.target_distance, self.target_distance)
		self.assertEqual(self.laser.atmosphere, self.atmosphere)
		self.assertEqual(self.laser.receiver_diameter, self.receiver_diameter)
		self.assertEqual(self.laser.target_albedo, self.target_albedo)
		self.assertEqual(self.laser.optics_transmitter_eff, self.optics_transmitter)
		self.assertEqual(self.laser.optics_receiver_eff, self.optics_receiver)
		self.assertEqual(self.laser.detection_efficiency, self.detection_efficiency)
		self.assertEqual(self.laser.background, self.background)
		self.assertEqual(self.laser.detector_dark, self.detector_dark)
		self.assertEqual(self.laser.timing_window, self.timing_window)

		self.assertEqual(self.laser.multi_photon_probability, self.multi_photon_probability)
		laser = deepcopy(self.laser)
		laser.multi_photon_probability = None
		laser.no_vacuum_probability = 0.1
		self.assertEqual(laser.no_vacuum_probability, 0.1)

		laser = deepcopy(self.laser)
		laser.multi_photon_probability = None
		laser.number_nv_pulse = 0.9 * self.output_power
		self.assertEqual(laser.number_nv_pulse, 0.9 * self.output_power)

		background_source = self.source.background2loss()
		background_laser = self.laser.background2loss()
		self.assertEqual(background_source, background_laser)

		db2loss_source = self.source.db2loss_atmosphere()
		db2loss_laser = self.laser.db2loss_atmosphere()
		self.assertEqual(db2loss_source, db2loss_laser)

	def test_extraction_efficiency(self):
		extr_efficiency = self.optics_transmitter
		self.assertEqual(self.laser.fix_extraction_efficiency(), extr_efficiency)

	def test_cant_fix_all_probabilities(self):
		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = 0.1
		params["multi_photon_probability"] = 0.1
		params["number_nv_pulse"] = 0.1 * self.output_power

		with self.assertRaises(AssertionError):
			PulsedLaser(params)

	def test_assert_probabilities_are_realistic(self):
		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = -0.1
		params["multi_photon_probability"] = None

		with self.assertRaises(AssertionError):
			PulsedLaser(params)

		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = 1.1

		with self.assertRaises(AssertionError):
			PulsedLaser(params)

		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = None
		params["number_nv_pulse"] = 1.1 * self.output_power
		with self.assertRaises(AssertionError):
			PulsedLaser(params)

	def test_match_multi_photon_probability(self):
		params = deepcopy(self.setup_params)
		prob = np.random.rand()
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob
		laser = PulsedLaser(params)
		self.assertEqual(laser.multi_photon_probability, prob)

	def test_match_no_vacuum_probability(self):
		params = deepcopy(self.setup_params)
		prob = np.random.rand()
		params["no_vacuum_probability"] = prob
		params["multi_photon_probability"] = None
		laser = PulsedLaser(params)
		self.assertEqual(laser.no_vacuum_probability, prob)

	def test_match_number_nv_pulse(self):
		params = deepcopy(self.setup_params)
		number_nv_pulse = 0.5 * self.output_power
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = None
		params["number_nv_pulse"] = number_nv_pulse
		laser = PulsedLaser(params)
		self.assertEqual(laser.number_nv_pulse, number_nv_pulse)

	def test_reciprocal_probabilities(self):
		params = deepcopy(self.setup_params)
		prob_nvp = np.random.rand()
		params["no_vacuum_probability"] = prob_nvp
		params["multi_photon_probability"] = None
		prob_mpp_predicted = PulsedLaser(params).multi_photon_probability
		alpha_predicted_nvp = PulsedLaser(params).alpha
		trigger_rate_predicted_nvp = PulsedLaser(params).trigger_rate
		average_photon_per_pulse_nvp = PulsedLaser(params).average_photon_per_pulse
		number_nv_pulse_nvp_fixed = PulsedLaser(params).number_nv_pulse

		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob_mpp_predicted
		prob_nvp_predicted = PulsedLaser(params).no_vacuum_probability
		alpha_predicted_mpp = PulsedLaser(params).alpha
		trigger_rate_predicted_mpp = PulsedLaser(params).trigger_rate
		average_photon_per_pulse_mpp = PulsedLaser(params).average_photon_per_pulse

		self.assertAlmostEquals(prob_nvp, prob_nvp_predicted, places=5)
		self.assertAlmostEquals(alpha_predicted_nvp, alpha_predicted_mpp, places=5)
		self.assertAlmostEquals(trigger_rate_predicted_nvp, trigger_rate_predicted_mpp, places=5)
		self.assertAlmostEquals(average_photon_per_pulse_nvp, average_photon_per_pulse_mpp, places=5)

		params = deepcopy(self.setup_params)
		prob_mpp = np.random.rand()
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob_mpp
		prob_nvp_predicted = PulsedLaser(params).no_vacuum_probability
		alpha_predicted_nvp = PulsedLaser(params).alpha
		trigger_rate_predicted_nvp = PulsedLaser(params).trigger_rate
		average_photon_per_pulse_nvp = PulsedLaser(params).average_photon_per_pulse
		number_nv_pulse_mpp_fixed = PulsedLaser(params).number_nv_pulse

		params["no_vacuum_probability"] = prob_nvp_predicted
		params["multi_photon_probability"] = None
		prob_mpp_predicted = PulsedLaser(params).multi_photon_probability
		alpha_predicted_mpp = PulsedLaser(params).alpha
		trigger_rate_predicted_mpp = PulsedLaser(params).trigger_rate
		average_photon_per_pulse_mpp = PulsedLaser(params).average_photon_per_pulse
		number_nv_pulse_mpp_fixed = PulsedLaser(params).number_nv_pulse

		self.assertAlmostEquals(prob_mpp, prob_mpp_predicted, places=5)
		self.assertAlmostEquals(alpha_predicted_nvp, alpha_predicted_mpp, places=5)
		self.assertAlmostEquals(trigger_rate_predicted_nvp, trigger_rate_predicted_mpp, places=5)
		self.assertAlmostEquals(average_photon_per_pulse_nvp, average_photon_per_pulse_mpp, places=5)

		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = None
		params["number_nv_pulse"] = number_nv_pulse_nvp_fixed
		laser_nvp = PulsedLaser(params)

		self.assertAlmostEquals(laser_nvp.no_vacuum_probability, prob_nvp, places=5)

		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = None
		params["number_nv_pulse"] = number_nv_pulse_mpp_fixed
		laser_mpp = PulsedLaser(params)

		self.assertAlmostEquals(laser_mpp.multi_photon_probability, prob_mpp, places=5)

	def test_overall_detection_probability(self):
		eta_detector, eta_noise = self.laser.overall_detection_probability()
		eta_detector_source, eta_noise_source = self.source.overall_detection_probability()

		self.assertEqual(eta_detector, eta_detector_source)
		self.assertEqual(eta_noise, eta_noise_source)

	def test_detector2observable(self):
		def create_observable(efficiency, noise, fock_space_dim):
			observable = np.zeros((fock_space_dim, fock_space_dim))
			for i in range(fock_space_dim):
				observable[i, i] = (1 - (1 - efficiency) ** i) + noise * (1 - efficiency) ** i
			return Qobj(observable)

		efficiency = self.setup_params["optics_transmitter"]
		eta_detector, eta_noise = self.laser.overall_detection_probability()

		observable = create_observable(efficiency=efficiency*eta_detector, noise=eta_noise, fock_space_dim=self.fock_space_dim)
		observable_model = self.laser.detector2observable()

		self.assertTrue(observable == observable_model)

	def test_average_photon_per_pulse(self):
		params = deepcopy(self.setup_params)
		prob = np.random.rand()

		params["no_vacuum_probability"] = prob
		params["multi_photon_probability"] = None

		laser = PulsedLaser(params)

		self.assertEqual(laser.alpha**2, laser.average_photon_per_pulse)

		extr_efficiency = laser.fix_extraction_efficiency()

		average_photon_per_pulse_nvp = -np.log(-prob + 1)/extr_efficiency
		average_photon_per_pulse_model = laser.average_photon_per_pulse

		self.assertAlmostEquals(average_photon_per_pulse_nvp, average_photon_per_pulse_model, places=5)

		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob

		laser = PulsedLaser(params)

		extr_efficiency = laser.fix_extraction_efficiency()

		self.assertEqual(laser.alpha**2, laser.average_photon_per_pulse)

		from scipy.special import lambertw

		average_photon_per_pulse_mpp = (-lambertw((prob - 1)*np.exp(-1), -1) - 1)/extr_efficiency
		average_photon_per_pulse_model = laser.average_photon_per_pulse

		self.assertAlmostEquals(average_photon_per_pulse_mpp, average_photon_per_pulse_model, places=5)

	def test_triggering_rate(self):
		params = deepcopy(self.setup_params)
		prob_nvp = np.random.rand()
		params["no_vacuum_probability"] = prob_nvp
		params["multi_photon_probability"] = None
		laser = PulsedLaser(params)
		alpha = laser.alpha

		trigger_rate = laser.output_power / ((alpha**2) * laser.optics_transmitter_eff)

		self.assertEqual(laser.trigger_rate, trigger_rate)

	def test_snr(self):
		signal = self.laser.signal_rate()
		noise = self.laser.noise_rate()
		snr = (signal - noise) / noise
		snr_model = self.laser.signal_to_noise_rate()

		self.assertEqual(snr, snr_model)


if __name__ == '__main__':
	unittest.main()
