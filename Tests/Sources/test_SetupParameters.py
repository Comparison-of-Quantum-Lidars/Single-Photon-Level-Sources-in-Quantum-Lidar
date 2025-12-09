import unittest

from Sources import SetupParameters


class TestSetupParameters(unittest.TestCase):

	def setUp(self):
		self.fock_space_dim = 5
		self.output_power = 1e6
		self.multi_photon_probability = 0.1
		self.no_vacuum_probability = 0.2
		self.number_nv_pulse = self.output_power * 0.9
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

	def test_fock_space_dim(self):
		self.assertEqual(self.setup_params["fock_space_dim"], self.fock_space_dim)

	def test_output_power(self):
		self.assertEqual(self.setup_params["output_power"], self.output_power)

	def test_multi_photon_probability(self):
		self.assertEqual(self.setup_params["multi_photon_probability"], self.multi_photon_probability)

	def test_no_vacuum_probability(self):
		self.assertEqual(self.setup_params["no_vacuum_probability"], self.no_vacuum_probability)

	def test_number_nv_pulse(self):
		self.assertEqual(self.setup_params["number_nv_pulse"], self.number_nv_pulse)

	def test_sp_collection(self):
		self.assertEqual(self.setup_params["sp_collection"], self.sp_collection)

	def test_sp_p1(self):
		self.assertEqual(self.setup_params["sp_p1"], self.sp_p1)

	def test_sp_p2(self):
		self.assertEqual(self.setup_params["sp_p2"], self.sp_p2)

	def test_spdc_eps_heralding(self):
		self.assertEqual(self.setup_params["spdc_eps_heralding"], self.spdc_eps_heralding)

	def test_spdc_eps_collection(self):
		self.assertEqual(self.setup_params["spdc_eps_collection"], self.spdc_eps_collection)

	def test_target_distance(self):
		self.assertEqual(self.setup_params["target_distance"], self.target_distance)

	def test_atmosphere(self):
		self.assertEqual(self.setup_params["atmosphere"], self.atmosphere)

	def test_receiver_diameter(self):
		self.assertEqual(self.setup_params["receiver_diameter"], self.receiver_diameter)

	def test_target_albedo(self):
		self.assertEqual(self.setup_params["target_albedo"], self.target_albedo)

	def test_optics_transmitter(self):
		self.assertEqual(self.setup_params["optics_transmitter"], self.optics_transmitter)

	def test_optics_receiver(self):
		self.assertEqual(self.setup_params["optics_receiver"], self.optics_receiver)

	def test_detection_efficiency(self):
		self.assertEqual(self.setup_params["detection_efficiency"], self.detection_efficiency)

	def test_background(self):
		self.assertEqual(self.setup_params["background"], self.background)

	def test_detector_dark(self):
		self.assertEqual(self.setup_params["detector_dark"], self.detector_dark)

	def test_timing_window(self):
		self.assertEqual(self.setup_params["timing_window"], self.timing_window)

	def test_modify_params_after_init(self):
		self.setup_params["fock_space_dim"] = 10
		self.assertEqual(self.setup_params["fock_space_dim"], 10)


if __name__ == '__main__':
	unittest.main()
