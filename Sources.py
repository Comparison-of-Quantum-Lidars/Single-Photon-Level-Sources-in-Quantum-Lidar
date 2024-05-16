from qutip import Qobj, basis, displace, qeye, expect, destroy, create, tensor
import numpy as np
from typing import Optional

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

	@staticmethod
	def bucket_detector(n: int, efficiency: float, noise: float):
		"""
		Build the bucket detector operator for APD detection
		:param n: Dimension of the operator in the fock space
		:param efficiency: Detection efficiency
		:param noise: Noise in the detector
		:return: Bucket detector operator
		"""
		observable = np.zeros((n, n))
		for i in range(n):
			observable[i, i] = 1 - ((1 - efficiency) ** i)
		observable = Qobj(observable) + qeye(n) * noise
		return observable


class PulsedLaser(Source):

	def __init__(self, params, **kwargs):
		super().__init__(params)
		self.apd_detector = self.detector2observable()
		self.vacuum = basis(self.fock_space_dim, 0)

		self.kwargs = kwargs
		self.alpha = kwargs.get("alpha", None)
	def compute_alpha(self):
		"""
		Compute the alpha parameter of the laser
		:return: Alpha parameter
		"""
		if self.alpha is None:
			return np.sqrt(self.output_power / self.laser_rate)
		else:
			return self.alpha

	def compute_laser_state(self):
		"""
		Compute the state of the laser
		:return: State of the laser
		"""
		alpha = self.compute_alpha()
		displacement = displace(self.fock_space_dim, alpha)
		laser_state = displacement * self.vacuum
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
		apd_detector = self.bucket_detector(self.fock_space_dim, eta_detector, eta_noise)
		return apd_detector

	def signal_rate(self):
		laser_state = self.compute_laser_state()
		signal_rate = self.effective_trigger_rate() * expect(self.apd_detector, laser_state)
		return signal_rate

	def noise_rate(self):
		noise_rate = self.effective_trigger_rate() * expect(self.apd_detector, self.vacuum)
		return noise_rate

	def signal_to_noise_rate(self):
		signal = self.signal_rate()
		noise = self.noise_rate()
		return signal/noise


class SinglePhoton(Source):

	def __init__(self, params, **kwargs):
		super().__init__(params)
		self.kwargs = kwargs

		self.apd_detector = self.detector2observable()
		self.vacuum = basis(self.fock_space_dim, 0)

	def single_photon_state(self):
		"""
		Compute the state of the single photon source
		:return: State of the single photon source
		"""
		vacuum = np.sqrt(1 - (self.sp_p1 + self.sp_p2)) * self.vacuum
		single_photon = np.sqrt(self.sp_p1) * basis(self.fock_space_dim, 1)
		two_photon = np.sqrt(self.sp_p2) * basis(self.fock_space_dim, 2)
		return vacuum + single_photon + two_photon

	def effective_trigger_rate(self):
		return self.output_power / (self.sp_p1 + 2 * self.sp_p2) / self.sp_collection

	def detector2observable(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector
		"""
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = self.bucket_detector(self.fock_space_dim, self.sp_collection * eta_detector, eta_noise)
		return apd_detector

	def signal_rate(self):
		single_photon_state = self.single_photon_state()
		signal_rate = self.effective_trigger_rate() * expect(self.apd_detector, single_photon_state)
		return signal_rate

	def noise_rate(self):
		noise = self.effective_trigger_rate() * expect(self.apd_detector, self.vacuum)
		return noise

	def signal_to_noise_rate(self):
		signal_rate = self.signal_rate()
		noise = self.noise_rate()
		return signal_rate/noise


class EntangledPhotonSPDC(Source):

	def __init__(self, params, **kwargs):
		super().__init__(params)
		self.kwargs = kwargs

		self.apd_detector_signal = self.detector2observable_signal()
		self.apd_detector_idler = self.detector2observable_idler()
		self.vacuum = basis(self.fock_space_dim, 0)

	def squeezed_operator(self):
		a = destroy(self.fock_space_dim)
		a_dagger = create(self.fock_space_dim)
		spdc_epsilon = np.arcsinh(np.sqrt(self.spdc_emission))
		argument = -1j*(tensor(a, a) + tensor(a_dagger, a_dagger))* spdc_epsilon
		squeezed_operator = argument.expm()
		return squeezed_operator
	def compute_spdc_eps_state(self):
		squeezed_operator = self.squeezed_operator()
		return squeezed_operator * tensor(self.vacuum, self.vacuum)

	def detector2observable_signal(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector
		"""
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = self.bucket_detector(self.fock_space_dim, self.spdc_eps_collection * eta_detector, eta_noise)
		return apd_detector

	def detector2observable_idler(self):
		apd_detector_local = self.bucket_detector(self.fock_space_dim, self.spdc_eps_heralding, self.detector_dark * self.timing_window)
		return apd_detector_local

	def compute_eps_rate(self):
		operator_detection_vacuum_pair = tensor(qeye(self.fock_space_dim), self.vacuum * self.vacuum.dag())
		prob_pair_vacuum = expect(operator_detection_vacuum_pair, self.compute_spdc_eps_state())
		eps_rate = self.output_power/((1-prob_pair_vacuum)*self.spdc_eps_collection)
		return eps_rate

	def effective_trigger_rate(self):
		operator_detection_idler = tensor(self.apd_detector_idler, qeye(self.fock_space_dim))
		return self.compute_eps_rate() * expect(operator_detection_idler, self.compute_spdc_eps_state())

	def signal_rate(self):
		operator_joint_detection = tensor(self.apd_detector_idler, self.apd_detector_signal)
		return self.compute_eps_rate() * expect(operator_joint_detection, self.compute_spdc_eps_state())

	def noise_rate(self):
		operator_detector_signal_only = tensor(qeye(self.fock_space_dim), self.apd_detector_signal)
		joint_vacuum = tensor(self.vacuum, self.vacuum)
		return self.effective_trigger_rate() * expect(operator_detector_signal_only, joint_vacuum)

	def signal_to_noise_rate(self):
		signal = self.signal_rate()
		noise = self.noise_rate()
		return signal/noise



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


