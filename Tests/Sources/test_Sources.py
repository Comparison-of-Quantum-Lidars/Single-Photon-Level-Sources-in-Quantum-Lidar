import unittest
import numpy as np
from copy import deepcopy
from qutip import Qobj

from Sources import Source, SetupParameters


class TestSource(unittest.TestCase):

	def setUp(self):
		self.fock_space_dim = 5
		self.output_power = 1e6
		self.multi_photon_probability = 0.1
		self.no_vacuum_probability = 0.2
		self.number_nv_pulse = 0.5 * self.output_power
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

	def test_init_regular_params(self):
		self.assertEqual(self.source.params, self.setup_params)
		self.assertEqual(self.source.fock_space_dim, self.fock_space_dim)
		self.assertEqual(self.source.output_power, self.output_power)
		self.assertEqual(self.source.multi_photon_probability, self.multi_photon_probability)
		self.assertEqual(self.source.no_vacuum_probability, self.no_vacuum_probability)
		self.assertEqual(self.source.number_nv_pulse, self.number_nv_pulse)
		self.assertEqual(self.source.sp_collection, self.sp_collection)
		self.assertEqual(self.source.sp_p1, self.sp_p1)
		self.assertEqual(self.source.sp_p2, self.sp_p2)
		self.assertEqual(self.source.spdc_eps_heralding, self.spdc_eps_heralding)
		self.assertEqual(self.source.spdc_eps_collection, self.spdc_eps_collection)
		self.assertEqual(self.source.target_distance, self.target_distance)
		self.assertEqual(self.source.atmosphere, self.atmosphere)
		self.assertEqual(self.source.receiver_diameter, self.receiver_diameter)
		self.assertEqual(self.source.target_albedo, self.target_albedo)
		self.assertEqual(self.source.optics_transmitter_eff, self.optics_transmitter)
		self.assertEqual(self.source.optics_receiver_eff, self.optics_receiver)
		self.assertEqual(self.source.detection_efficiency, self.detection_efficiency)
		self.assertEqual(self.source.background, self.background)
		self.assertEqual(self.source.detector_dark, self.detector_dark)
		self.assertEqual(self.source.timing_window, self.timing_window)

	def test_distance_init(self):
		params = deepcopy(self.setup_params)
		params["target_distance"] = None
		source = Source(params)
		self.assertEqual(source.target_distance, 1)

	def test_background2loss(self):
		receiver_diameter_in_cm = self.receiver_diameter
		receiver_radius_in_cm = receiver_diameter_in_cm / 2
		receiver_area_in_cm_squared = np.pi * receiver_radius_in_cm ** 2
		photons_per_second_received_background = self.background * receiver_area_in_cm_squared
		loss_model = self.source.background2loss()
		self.assertAlmostEquals(loss_model, photons_per_second_received_background, places=5)

	def test_db2loss_atmosphere(self):
		attenuation_db_per_m = self.atmosphere / 1000
		db_for_distance_of_1000m = attenuation_db_per_m * 1000
		db_to_loss = 10 ** (-db_for_distance_of_1000m / 10)

		params = deepcopy(self.setup_params)
		params["target_distance"] = 1000
		source = Source(params)
		loss_model = source.db2loss_atmosphere()

		self.assertAlmostEquals(loss_model, db_to_loss, places=5)

	def test_overall_detection_probability_regular(self):
		atmosphere_loss = self.source.db2loss_atmosphere()
		solid_angle = (np.pi * (self.receiver_diameter ** 2) / 4) / (2 * np.pi * (self.target_distance ** 2))
		eta_detector = solid_angle * self.optics_receiver * self.target_albedo * self.detection_efficiency * (
				atmosphere_loss ** 2)

		background_loss = self.source.background2loss()
		eta_noise = (background_loss + self.detector_dark) * self.timing_window

		eta_detector_model, eta_noise_model = self.source.overall_detection_probability()

		self.assertAlmostEquals(eta_detector, eta_detector_model, places=5)
		self.assertAlmostEquals(eta_noise, eta_noise_model, places=5)

	def test_overall_detection_probability_modify_params(self):
		source = deepcopy(self.source)
		source.background = 1000

		self.assertEqual(source.background, 1000)

		background_loss_model = source.background2loss()
		background_loss = 1000 * np.pi * ((source.receiver_diameter/ 2) ** 2)

		self.assertAlmostEquals(background_loss, background_loss_model, places=5)

		eta_noise = (background_loss + source.detector_dark) * source.timing_window

		_, eta_noise_model = source.overall_detection_probability()

		self.assertAlmostEquals(eta_noise, eta_noise_model, places=5)

	def test_overall_detection_probability_extreme_case(self):
		params = deepcopy(self.setup_params)
		params["optics_receiver"] = 0
		source = Source(params)

		eta_detector, _ = source.overall_detection_probability()

		print(eta_detector)

		self.assertEqual(eta_detector, 0)

	def test_bucket_detector(self):
		efficiency = 1
		noise = 0

		def create_observable(efficiency, noise, fock_space_dim):
			observable = np.zeros((fock_space_dim, fock_space_dim))
			for i in range(fock_space_dim):
				observable[i, i] = (1 - (1 - efficiency) ** i) + noise * (1 - efficiency) ** i
			return Qobj(observable)

		observable = create_observable(efficiency, noise, self.fock_space_dim)
		observable_model = self.source.bucket_detector(n=self.fock_space_dim, efficiency=efficiency, noise=noise)

		self.assertTrue(observable == observable_model)

		efficiency = 0
		noise = 1

		observable = create_observable(efficiency, noise, self.fock_space_dim)
		observable_model = self.source.bucket_detector(n=self.fock_space_dim, efficiency=efficiency, noise=noise)

		self.assertTrue(observable == observable_model)

		efficiency = np.random.rand()
		noise = np.random.rand()

		observable = create_observable(efficiency, noise, self.fock_space_dim)
		observable_model = self.source.bucket_detector(n=self.fock_space_dim, efficiency=efficiency, noise=noise)

		self.assertTrue(observable == observable_model)


if __name__ == '__main__':
	unittest.main()
