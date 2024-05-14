import numpy as np
from Operators import State, Operators


class Source:
	"""
	This class is the parent class for all the sources that can be used in the simulation. It defines the general
	parameters that are common to most sources. The overall detection is also computed here.
	"""

	def __init__(self, params):
		self.params = params
		self.check_all_params()
		self.fock_space_dim = params["fock_space_dim"]
		self.output_power = params["output_power"]
		self.laser_rate = params["laser_rate"]
		self.sp_collection = params["sp_collection"]
		self.sp_p1 = params["sp_p1"]
		self.sp_p2 = params["sp_p2"]
		self.spdc_eps_heralding = params["spdc_eps_heralding"]
		self.spdc_eps_collection = params["spdc_eps_collection"]
		self.spdc_emission = params["spdc_emission"]
		self.target_distance = params["target_distance"]
		self.receiver_diameter = params["receiver_diameter"]
		self.target_albedo = params["target_albedo"]
		self.optics_transmitter_eff = params["optics_transmitter"]
		self.optics_receiver_eff = params["optics_receiver"]
		self.detection_efficiency = params["detection_efficiency"]
		self.background = params["background"]
		self.detector_dark = params["detector_dark"]
		self.timing_window = params["timing_window"]

	def check_all_params(self):
		pass

	def overall_detection_probability(self):
		"""
		Compute the overall detection probability of the detector and the noise detection probability
		:return: Overall detection probability and noise detection probability
		"""
		# Overall detection of the detector

		solid_angle_approx = (np.pi * self.receiver_diameter ** 2 / 4) / (2 * np.pi * self.target_distance ** 2)
		eta_detector = solid_angle_approx * self.optics_receiver_eff * self.target_albedo * self.optics_transmitter_eff * self.detection_efficiency

		# Overall noise detection
		eta_noise = (self.background + self.detector_dark) * self.timing_window

		return eta_detector, eta_noise


class PulsedLaser(Source):

	def __init__(self, params):
		"""
		:param params: SetupParameters object containing the parameters of the simulation
		"""
		super().__init__(params)

	def compute_alpha(self):
		"""
		Compute the alpha parameter for the displacement operator
		:return: Alpha parameter
		"""
		return np.sqrt(self.output_power / self.laser_rate)

	def compute_laser_state(self):
		"""
		Compute the laser state using the displacement operator on the vacuum state
		:return: Laser state
		"""
		alpha = self.compute_alpha()
		vacuum = State.vacuum(self.fock_space_dim)
		displacement = Operators.displacement(self.fock_space_dim, alpha)
		laser_state = displacement * vacuum
		return laser_state

	def effective_trigger_rate(self):
		"""
		Compute the effective trigger rate of the laser source. For the laser source, the effective trigger rate is
		always equal to the laser rate.
		:return: Effective trigger rate
		"""
		return self.laser_rate

	def detector2observable(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector
		"""
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = Operators.bucket_detector(self.fock_space_dim, eta_detector, eta_noise)
		return apd_detector

	def signal_rate(self):
		"""
		Compute the signal rate of the laser source
		:return: Signal rate
		"""
		apd_detector = self.detector2observable()
		laser_state = self.compute_laser_state()
		signal = self.effective_trigger_rate() * apd_detector.expect(laser_state)
		return signal

	def noise_rate(self):
		"""
		Compute the noise rate of the laser source
		:return: Noise rate
		"""
		apd_detector = self.detector2observable()
		vacuum = State.vacuum(self.fock_space_dim)
		noise = self.effective_trigger_rate() * apd_detector.expect(vacuum)
		return noise

	def signal_to_noise_rate(self):
		"""
		Compute the signal-to-noise ratio of the laser source
		:return: Signal-to-noise ratio
		"""
		signal = self.signal_rate()
		vacuum = State.vacuum(self.fock_space_dim)
		apd_detector = self.detector2observable()
		noise = self.effective_trigger_rate() * apd_detector.expect(vacuum)
		signal_to_noise = signal / noise
		return signal_to_noise


class SinglePhoton(Source):
	def __init__(self, params):
		"""
		:param params: SetupParameters object containing the parameters of the simulation
		"""
		super().__init__(params)

	def single_photon_state(self):
		"""
		Compute the single photon state based on the probabilities of the single photon source
		:return: Single photon state
		"""
		vacuum = np.sqrt(1 - (self.sp_p1 + self.sp_p2)) * State.vacuum(self.fock_space_dim)
		one_photon = np.sqrt(self.sp_p1) * State.one_photon(self.fock_space_dim)
		two_photon = np.sqrt(self.sp_p2) * State.two_photon(self.fock_space_dim)
		return vacuum + one_photon + two_photon

	def effective_trigger_rate(self):
		"""
		Compute the effective trigger rate of the single photon source considering the efficiency of the source
		:return: Effective trigger rate
		"""
		#TODO : CONFIRM THIS EQUATION -> NOT SURE
		return self.output_power / (self.sp_p1 + 2 * self.sp_p2) / self.sp_collection

	def detector2observable(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector
		"""
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = Operators.bucket_detector(self.fock_space_dim, self.sp_collection * eta_detector, eta_noise)
		return apd_detector

	def signal_rate(self):
		"""
		Compute the signal rate of the single photon source
		:return: The signal rate for the single photon source
		"""
		apd_detector = self.detector2observable()
		single_photon_state = self.single_photon_state()
		signal_rate = self.effective_trigger_rate() * apd_detector.expect(single_photon_state)
		return signal_rate

	def noise_rate(self):
		"""
		Compute the noise rate of the single photon source
		:return: The noise rate for the single photon source
		"""
		apd_detector = self.detector2observable()
		vacuum = State.vacuum(self.fock_space_dim)
		noise = self.effective_trigger_rate() * apd_detector.expect(vacuum)
		return noise

	def signal_to_noise_rate(self):
		"""
		Compute the signal-to-noise ratio of the single photon source
		:return: The signal-to-noise ratio for the single photon source
		"""
		signal = self.signal_rate()
		apd_detector = self.detector2observable()
		vacuum = State.vacuum(self.fock_space_dim)
		noise = self.effective_trigger_rate() * apd_detector.expect(vacuum)
		signal_to_noise = signal / noise
		return signal_to_noise


class EntangledPhoton(Source):

	def __init__(self, params):
		super().__init__(params)

	def compute_spdc_eps_state(self):
		vacuum = State.vacuum(self.fock_space_dim)
		squeezed_operator = Operators.squeezed(self.fock_space_dim, self.spdc_emission)
		return squeezed_operator * (vacuum @ vacuum)

	def detector2observable(self):
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = Operators.bucket_detector(self.fock_space_dim, self.spdc_eps_collection * eta_detector, eta_noise)
		return apd_detector

	def local_detector2observable(self):
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector_local = Operators.bucket_detector(self.fock_space_dim, self.spdc_eps_heralding, self.detector_dark * self.timing_window)
		return apd_detector_local

	def compute_eps_rate(self):
		# TODO : Fix bug of size mismatch
		ida_vacc = Operators.identity(self.fock_space_dim) @ State.vacuum(self.fock_space_dim)
		eps_state = self.compute_spdc_eps_state()
		#expected_value = ida_vacc.expect(eps_state)
		return None

class SetupParameters:

	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	@staticmethod
	def help():
		print("Here are the parameters you can set:")
		print("--- General ---")
		print("fock_space_dim: Dimension of the fock space, must be an integer")
		print("--- Sources ---")
		print("output_power: optical output power of the source in photon per second [s^-1]")
		print("laser_rate: Pulsing rate of the laser source [Hz]")
		print(
			"sp_collection: Efficiency of the single photon source collection, does not include detector efficiency [-]")
		print("sp_p1: Probability that the single photon source emits 1 photons [-]")
		print("sp_p2: Probability that the single photon source emits 2 photons [-]")
		print(
			"spdc_eps_heralding: Efficiency of the EPS local measurement, includes collection and detector efficiency [-]")
		print("spdc_eps_collection: Efficiency of the EPS collection for the sensing photon, does not include detector "
		      "efficiency [-]")
		print("spdc_emission: SPDC average pair photon number [-]")
		print("--- Detection ---")
		print("target_distance: Distance to target, loss is assumed as a 2*pi sphere scattering from target, "
		      "no other channel losses included")
		print("receiver_diameter: Diameter of the receiver telescope, assume no obstruction [m]")
		print("target_albedo: Albedo of the target [-]")
		print("optics_transmitter: Optical efficiency of the transmitter [-]")
		print("optics_receiver: Optical efficiency of the receiver [-]")
		print("detection_efficiency: Detection efficiency of single photon detector [-]")
		print("background: Background noise detection rate [Hz]")
		print("detector_dark: Dark count rate of the detector [Hz]")
		print("timing_window: Time window for detection -> Must be larger than all timing jitters [s]")

	def __repr__(self):
		return str(self.__dict__)

	def __str__(self):
		return str(self.__dict__)

	def __getitem__(self, item):
		return self.__dict__[item]

	def __setitem__(self, key, value):
		self.__dict__[key] = value

	def __delitem__(self, key):
		del self.__dict__[key]
