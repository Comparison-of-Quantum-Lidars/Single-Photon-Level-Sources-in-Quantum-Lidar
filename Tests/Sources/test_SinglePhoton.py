import unittest
import numpy as np
from Sources import SinglePhoton, SetupParameters, Source
from copy import deepcopy
from qutip import Qobj


class TestSinglePhoton(unittest.TestCase):

	def setUp(self):
		self.fock_space_dim = 3
		self.output_power = 1e6
		self.multi_photon_probability = None
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
		self.sps = SinglePhoton(self.setup_params)

	def test_super_for_init_params(self):
		self.assertEqual(self.sps.fock_space_dim, self.fock_space_dim)
		self.assertEqual(self.sps.output_power, self.output_power)
		self.assertEqual(self.sps.sp_collection, self.sp_collection)
		self.assertEqual(self.sps.sp_p1, self.sp_p1)
		self.assertEqual(self.sps.sp_p2, self.sp_p2)
		self.assertEqual(self.sps.spdc_eps_heralding, self.spdc_eps_heralding)
		self.assertEqual(self.sps.spdc_eps_collection, self.spdc_eps_collection)
		self.assertEqual(self.sps.target_distance, self.target_distance)
		self.assertEqual(self.sps.atmosphere, self.atmosphere)
		self.assertEqual(self.sps.receiver_diameter, self.receiver_diameter)
		self.assertEqual(self.sps.target_albedo, self.target_albedo)
		self.assertEqual(self.sps.optics_transmitter_eff, self.optics_transmitter)
		self.assertEqual(self.sps.optics_receiver_eff, self.optics_receiver)
		self.assertEqual(self.sps.detection_efficiency, self.detection_efficiency)
		self.assertEqual(self.sps.background, self.background)
		self.assertEqual(self.sps.detector_dark, self.detector_dark)
		self.assertEqual(self.sps.timing_window, self.timing_window)

		background_source = self.source.background2loss()
		background_sps = self.sps.background2loss()
		self.assertEqual(background_sps, background_source)

		db2loss_source = self.source.db2loss_atmosphere()
		db2loss_sps = self.sps.db2loss_atmosphere()
		self.assertEqual(db2loss_sps, db2loss_source)

	def test_overall_detection_probability(self):
		eta_detector_source, eta_noise_source = self.source.overall_detection_probability()
		eta_detector_sps, eta_noise_sps = self.sps.overall_detection_probability()

		self.assertEqual(eta_detector_source, eta_detector_sps)
		self.assertEqual(eta_noise_source, eta_noise_sps)

		params = deepcopy(self.setup_params)
		params["background"] = self.setup_params["background"] * np.random.rand()
		params["atmosphere"] = self.setup_params["atmosphere"] * np.random.rand()
		params["target_distance"] = self.setup_params["target_distance"] * np.random.rand()

		source_new = Source(params)
		sps_new = SinglePhoton(params)

		eta_detector_source_new, eta_noise_source_new = source_new.overall_detection_probability()
		eta_detector_sps_new, eta_noise_sps_new = sps_new.overall_detection_probability()

		self.assertEqual(eta_detector_source_new, eta_detector_sps_new)
		self.assertEqual(eta_noise_source_new, eta_noise_sps_new)


	def test_cant_fix_both_probabilities(self):
		params = deepcopy(self.setup_params)
		params["multi_photon_probability"] = 0.2
		params["no_vacuum_probability"] = 0.25
		params["number_nv_pulse"] = 100

		with self.assertRaises(AssertionError):
			SinglePhoton(params)

	def test_extraction_efficiency(self):
		extr_eff = self.optics_transmitter * self.sp_collection
		self.assertEqual(extr_eff, self.sps.fix_extraction_efficiency())

	def test_cant_fix_specific_probabilities_for_sps(self):
		params = deepcopy(self.setup_params)
		prob = np.random.rand()
		params["multi_photon_probability"] = prob
		params["no_vacuum_probability"] = None
		params["number_nv_pulse"] = None

		with self.assertRaises(AssertionError):
			SinglePhoton(params)

		params["multi_photon_probability"] = None
		params["number_nv_pulse"] = None
		params["no_vacuum_probability"] = prob

		with self.assertRaises(AssertionError):
			SinglePhoton(params)

		params["multi_photon_probability"] = None
		params["no_vacuum_probability"] = None
		params["number_nv_pulse"] = prob * self.output_power

		with self.assertRaises(AssertionError):
			SinglePhoton(params)

		params["multi_photon_probability"] = None
		params["no_vacuum_probability"] = None
		params["number_nv_pulse"] = None
		sps = SinglePhoton(params)

		extr_eff = self.optics_transmitter * self.sp_collection
		aimed_multi_photon_prob = self.sp_p2 * extr_eff ** 2
		aimed_non_vacuum_prob = self.sp_p1 * extr_eff + self.sp_p2 * extr_eff * (2 - extr_eff)

		self.assertEqual(aimed_multi_photon_prob, sps.multi_photon_probability)
		self.assertEqual(aimed_non_vacuum_prob, sps.no_vacuum_probability)

		params["multi_photon_probability"] = aimed_multi_photon_prob
		params["no_vacuum_probability"] = aimed_non_vacuum_prob

		try:
			SinglePhoton(params)
			self.assertTrue(True)
		except AssertionError:
			self.fail("Unexpected AssertionError")

	def test_average_photon_per_pulse(self):
		params = deepcopy(self.setup_params)
		sps = SinglePhoton(params)
		average_photon_per_pulse = self.sp_p1 + (2 * self.sp_p2)

		self.assertEqual(average_photon_per_pulse, sps.average_photon_per_pulse)

	def test_triggering_rate(self):
		average_photon_prob = self.sps.average_photon_per_pulse
		extr_eff = self.sps.fix_extraction_efficiency()
		trigger_rate = self.output_power / (average_photon_prob * extr_eff)

		self.assertEqual(trigger_rate, self.sps.trigger_rate)

	def test_detector2observable(self):
		def create_observable(efficiency, noise, fock_space_dim):
			observable = np.zeros((fock_space_dim, fock_space_dim))
			for i in range(fock_space_dim):
				observable[i, i] = (1 - (1 - efficiency) ** i) + noise * (1 - efficiency) ** i
			return Qobj(observable)

		efficiency = self.sps.fix_extraction_efficiency()
		eta_detector, eta_noise = self.sps.overall_detection_probability()

		observable = create_observable(efficiency=efficiency * eta_detector, noise=eta_noise, fock_space_dim=self.fock_space_dim)
		observable_model = self.sps.detector2observable()

		self.assertTrue(observable == observable_model)

	def test_snr(self):
		signal = self.sps.signal_rate()
		noise = self.sps.noise_rate()
		snr = (signal - noise) / noise

		self.assertEqual(snr, self.sps.signal_to_noise_rate())


if __name__ == '__main__':
	unittest.main()
