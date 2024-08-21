import unittest
import numpy as np
from Sources import EntangledPhotonSPDC, SetupParameters, Source
from copy import deepcopy
from qutip import Qobj, tensor, basis, expect, qeye


class TestEntangledPhotonSPDC(unittest.TestCase):

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
		self.source = Source(self.setup_params)
		self.eps = EntangledPhotonSPDC(self.setup_params)

	def test_super_for_init_params(self):
		self.assertEqual(self.eps.fock_space_dim, self.fock_space_dim)
		self.assertEqual(self.eps.output_power, self.output_power)
		self.assertEqual(self.eps.sp_collection, self.sp_collection)
		self.assertEqual(self.eps.sp_p1, self.sp_p1)
		self.assertEqual(self.eps.sp_p2, self.sp_p2)
		self.assertEqual(self.eps.spdc_eps_heralding, self.spdc_eps_heralding)
		self.assertEqual(self.eps.spdc_eps_collection, self.spdc_eps_collection)
		self.assertEqual(self.eps.target_distance, self.target_distance)
		self.assertEqual(self.eps.atmosphere, self.atmosphere)
		self.assertEqual(self.eps.receiver_diameter, self.receiver_diameter)
		self.assertEqual(self.eps.target_albedo, self.target_albedo)
		self.assertEqual(self.eps.optics_transmitter_eff, self.optics_transmitter)
		self.assertEqual(self.eps.optics_receiver_eff, self.optics_receiver)
		self.assertEqual(self.eps.detection_efficiency, self.detection_efficiency)
		self.assertEqual(self.eps.background, self.background)
		self.assertEqual(self.eps.detector_dark, self.detector_dark)
		self.assertEqual(self.eps.timing_window, self.timing_window)

		self.assertEqual(self.eps.multi_photon_probability, self.multi_photon_probability)
		eps = deepcopy(self.eps)
		eps.multi_photon_probability = None
		eps.no_vacuum_probability = 0.1
		self.assertEqual(eps.no_vacuum_probability, 0.1)

		background_source = self.source.background2loss()
		background_eps = self.eps.background2loss()
		self.assertEqual(background_eps, background_source)

		db2loss_source = self.source.db2loss_atmosphere()
		db2loss_eps = self.eps.db2loss_atmosphere()
		self.assertEqual(db2loss_eps, db2loss_source)

	def test_extraction_efficiency(self):
		extr_efficiency = self.optics_transmitter * self.spdc_eps_collection
		self.assertEqual(self.eps.fix_extraction_efficiency(), extr_efficiency)

	def test_cant_fix_both_probabilities(self):
		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = 0.1
		params["multi_photon_probability"] = 0.1

		with self.assertRaises(AssertionError):
			EntangledPhotonSPDC(params)

	def test_assert_probabilities_are_realistic(self):
		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = -0.1
		params["multi_photon_probability"] = None

		with self.assertRaises(AssertionError):
			EntangledPhotonSPDC(params)

		params = deepcopy(self.setup_params)
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = 1.1

		with self.assertRaises(AssertionError):
			EntangledPhotonSPDC(params)

	def test_match_multi_photon_probability(self):
		params = deepcopy(self.setup_params)
		prob = np.random.rand()
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob
		eps = EntangledPhotonSPDC(params)
		self.assertEqual(eps.multi_photon_probability, prob)

	def test_match_no_vacuum_probability(self):
		params = deepcopy(self.setup_params)
		prob = np.random.rand()
		params["no_vacuum_probability"] = prob
		params["multi_photon_probability"] = None
		eps = EntangledPhotonSPDC(params)
		self.assertEqual(eps.no_vacuum_probability, prob)

	def test_reciprocal_probabilities(self):
		params = deepcopy(self.setup_params)
		prob_nvp = np.random.rand()
		params["no_vacuum_probability"] = prob_nvp
		params["multi_photon_probability"] = None
		prob_mpp_predicted = EntangledPhotonSPDC(params).multi_photon_probability
		average_photon_per_pulse_predicted_nvp = EntangledPhotonSPDC(params).average_photon_per_pulse
		trigger_rate_predicted_nvp = EntangledPhotonSPDC(params).trigger_rate

		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob_mpp_predicted
		prob_nvp_predicted = EntangledPhotonSPDC(params).no_vacuum_probability
		average_photon_per_pulse_predicted_mpp = EntangledPhotonSPDC(params).average_photon_per_pulse
		trigger_rate_predicted_mpp = EntangledPhotonSPDC(params).trigger_rate

		self.assertAlmostEquals(prob_nvp, prob_nvp_predicted, places=5)
		self.assertAlmostEquals(average_photon_per_pulse_predicted_nvp, average_photon_per_pulse_predicted_mpp, places=5)
		self.assertAlmostEquals(trigger_rate_predicted_nvp, trigger_rate_predicted_mpp, places=5)

		params = deepcopy(self.setup_params)
		prob_mpp = np.random.rand()
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob_mpp
		prob_nvp_predicted = EntangledPhotonSPDC(params).no_vacuum_probability
		average_photon_per_pulse_predicted_mpp = EntangledPhotonSPDC(params).average_photon_per_pulse
		trigger_rate_predicted_mpp = EntangledPhotonSPDC(params).trigger_rate

		params["no_vacuum_probability"] = prob_nvp_predicted
		params["multi_photon_probability"] = None
		prob_mpp_predicted = EntangledPhotonSPDC(params).multi_photon_probability
		average_photon_per_pulse_predicted_nvp = EntangledPhotonSPDC(params).average_photon_per_pulse
		trigger_rate_predicted_nvp = EntangledPhotonSPDC(params).trigger_rate

		self.assertAlmostEquals(prob_mpp, prob_mpp_predicted, places=5)
		self.assertAlmostEquals(average_photon_per_pulse_predicted_nvp, average_photon_per_pulse_predicted_mpp, places=5)
		self.assertAlmostEquals(trigger_rate_predicted_nvp, trigger_rate_predicted_mpp, places=5)

	def test_overall_detection_probability(self):
		eta_detector, eta_noise = self.eps.overall_detection_probability()
		eta_detector_source, eta_noise_source = self.source.overall_detection_probability()

		self.assertEqual(eta_detector, eta_detector_source)
		self.assertEqual(eta_noise, eta_noise_source)

	def test_detector2observable_signal(self):
		def create_observable(efficiency, noise, fock_space_dim):
			observable = np.zeros((fock_space_dim, fock_space_dim))
			for i in range(fock_space_dim):
				observable[i, i] = (1 - (1 - efficiency) ** i) + noise * (1 - efficiency) ** i
			return Qobj(observable)

		efficiency = self.eps.fix_extraction_efficiency()
		eta_detector, eta_noise = self.eps.overall_detection_probability()

		observable = create_observable(efficiency*eta_detector, eta_noise, self.fock_space_dim)
		observable_model = self.eps.detector2observable_signal()

		self.assertEqual(observable, observable_model)

	def test_detector2observable_idler(self):
		def create_observable(efficiency, noise, fock_space_dim):
			observable = np.zeros((fock_space_dim, fock_space_dim))
			for i in range(fock_space_dim):
				observable[i, i] = (1 - (1 - efficiency) ** i) + noise * (1 - efficiency) ** i
			return Qobj(observable)

		efficiency = self.spdc_eps_heralding
		eta_noise = self.eps.detector_dark * self.eps.timing_window

		observable = create_observable(efficiency, eta_noise, self.fock_space_dim)
		observable_model = self.eps.detector2observable_idler()

		self.assertEqual(observable, observable_model)

	def test_average_photon_per_pulse(self):
		params = deepcopy(self.setup_params)
		prob = np.random.rand()

		params["no_vacuum_probability"] = prob
		params["multi_photon_probability"] = None

		eps = EntangledPhotonSPDC(params)

		extr_efficiency = eps.fix_extraction_efficiency()

		average_photon_per_pulse_nvp = prob / (extr_efficiency * (1 - prob))
		self.assertEqual(eps.average_photon_per_pulse, average_photon_per_pulse_nvp)

		prob = np.random.rand()
		params["no_vacuum_probability"] = None
		params["multi_photon_probability"] = prob

		eps = EntangledPhotonSPDC(params)

		extr_efficiency = eps.fix_extraction_efficiency()

		average_photon_per_pulse_mpp = np.sqrt(prob) / (extr_efficiency * (1 - np.sqrt(prob)))
		self.assertEqual(eps.average_photon_per_pulse, average_photon_per_pulse_mpp)

	def test_eps_rate(self):
		params = deepcopy(self.setup_params)
		prob_nvp = np.random.rand()
		params["no_vacuum_probability"] = prob_nvp
		params["multi_photon_probability"] = None
		eps = EntangledPhotonSPDC(params)

		eps_rate = eps.output_power / (eps.fix_extraction_efficiency() * eps.average_photon_per_pulse)

		self.assertEqual(eps.compute_eps_rate(), eps_rate)

	def test_trigger_rate(self):
		params = deepcopy(self.setup_params)
		prob_nvp = np.random.rand()
		params["fock_space_dim"] = 20
		params["no_vacuum_probability"] = 0.1
		params["multi_photon_probability"] = None
		eps = EntangledPhotonSPDC(params)

		idler_detector = eps.detector2observable_idler()
		operator_idler_only = tensor(idler_detector, qeye(eps.fock_space_dim))
		joint_vacuum_state = tensor(basis(eps.fock_space_dim, 0), basis(eps.fock_space_dim, 0))
		eps_state = eps.compute_spdc_eps_state()
		eps_rate = eps.compute_eps_rate()

		trigger_rate = eps_rate * (expect(operator_idler_only, eps_state) + expect(operator_idler_only, joint_vacuum_state) * ((1/(eps.timing_window*eps_rate))-1))
		trigger_rate_model = eps.trigger_rate

		self.assertAlmostEquals(trigger_rate, trigger_rate_model)

	def test_snr(self):
		signal = self.eps.signal_rate()
		noise = self.eps.noise_rate()
		snr = (signal - noise) / noise

		self.assertEqual(self.eps.signal_to_noise_rate(), snr)


if __name__ == '__main__':
	unittest.main()
