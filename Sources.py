from qutip import Qobj, basis, displace, qeye, expect, destroy, create, tensor
import numpy as np
from scipy.special import lambertw


class Source:
	"""
	This class is the parent class for all the sources that can be used in the simulation. It defines the general
	parameters that are common to most sources. The overall detection is also computed here.
	Please use SetupParameters.help() to understand the parameters that can be set.
	"""

	def __init__(self, params):
		self.params = params
		self.check_all_params()
		self.fock_space_dim = params["fock_space_dim"]
		self.output_power = params["output_power"]
		self.trigger_rate = params["trigger_rate"]
		self.multi_photon_probability = params["multi_photon_probability"]
		self.no_vacuum_probability = params["no_vacuum_probability"]
		self.sp_collection = params["sp_collection"]
		self.sp_p1 = params["sp_p1"]
		self.sp_p2 = params["sp_p2"]
		self.spdc_eps_heralding = params["spdc_eps_heralding"]
		self.spdc_eps_collection = params["spdc_eps_collection"]
		self.atmosphere = params["atmosphere"]
		self.adversary_eff = params.__dict__.get("adversary_eff", None)
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

		solid_angle_approx = (np.pi * (self.receiver_diameter ** 2) / 4) / (2 * np.pi * (self.target_distance ** 2))
		eta_detector = solid_angle_approx * self.optics_receiver_eff * self.target_albedo * self.optics_transmitter_eff * self.detection_efficiency * (
				self.atmosphere ** 2)

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
	"""
	This class models an attenuated pulsed laser source. The user can specify the output power and the trigger rate.
	For this source, the power is assumed to be always fixed. From there, one of the three following parameters can be
	fixed:
		1) The trigger rate
		2) The multi-photon probability
		3) The no-vacuum probability

	Setting one of them will automatically compute the two others.
	"""
	def __init__(self, params, **kwargs):
		"""
		:param params: Parameters of the source using the object SetupParameters
		:param kwargs: N/A
		"""
		super().__init__(params)
		self.kwargs = kwargs
		self.extr_efficiency = self.fix_extraction_efficiency()
		self.fix_parameters()
		self.vacuum = basis(self.fock_space_dim, 0)
		self.apd_detector = self.detector2observable()

	def fix_extraction_efficiency(self):
		"""
		Fix the extraction efficiency of the source. For an attenuated pulse laser, the collection efficiency is 100%.
		A contribution is added if the efficiency of the adversary is specified. In most cases, we prefer not to specify
		the adversary efficiency because we do not have any prior information about it.
		:return: Extraction efficiency [float]
		"""
		if self.adversary_eff is None:
			return 1
		return self.adversary_eff

	def fix_parameters(self):
		"""
		Four parameters can be played with: Output Power, Trigger Rate, Multi-photon probability and No-Vacuum probability
		Only two of those parameters can be fixed. Since P_{out} is always fixed, it is only possible to fix the
		trigger rate, the multi-photon probability or the no-vacuum probability.
		"""
		assert self.trigger_rate is not None or self.multi_photon_probability is not None or self.no_vacuum_probability is not None, "The trigger rate, multi-photon probability or no-vacuum probability must be fixed"

		if self.trigger_rate is not None:
			assert self.multi_photon_probability is None and self.no_vacuum_probability is None, "multi-photon probability and no-vacuum probability can't be fixed if trigger rate is fixed"

			self.multi_photon_probability = self.compute_multi_photon_probability()
			self.no_vacuum_probability = self.compute_no_vacuum_probability()

		elif self.multi_photon_probability is not None:
			assert self.trigger_rate is None and self.no_vacuum_probability is None, "trigger rate and no-vacuum probability can't be fixed if multi-photon probability is fixed"
			assert 0 < self.multi_photon_probability < 1, "The multi-photon probability must be between 0 and 1"

			alpha = np.sqrt(
				(-lambertw(z=((self.multi_photon_probability - 1) / np.exp(1)), k=-1) - 1) / (self.extr_efficiency)
			)
			assert alpha.imag == 0, "The multi-photon probability is not valid: alpha is complex"
			alpha = alpha.real
			self.trigger_rate = self.output_power / (alpha ** 2)
			self.no_vacuum_probability = self.compute_no_vacuum_probability()

		elif self.no_vacuum_probability is not None:
			assert self.trigger_rate is None and self.multi_photon_probability is None, "trigger rate and multi-photon probability can't be fixed if no-vacuum probability is fixed"
			assert 0 < self.no_vacuum_probability < 1, "The no-vacuum probability must be between 0 and 1"

			log_argument = -self.no_vacuum_probability + 1
			alpha = np.sqrt(-np.log(log_argument) / self.extr_efficiency)

			self.trigger_rate = self.output_power / (alpha ** 2)
			self.multi_photon_probability = self.compute_multi_photon_probability()
		else:
			raise ValueError("The trigger rate, multi-photon probability or no-vacuum probability must be fixed")

	def compute_multi_photon_probability(self):
		"""
		Compute the multi-photon probability of the laser when either the triggering rate or the non-vacuum probability
		is fixed.
		:return: Multi-photon probability [float]
		"""
		alpha = self.compute_alpha()
		return 1 - (np.exp(-self.extr_efficiency * (alpha ** 2)) * (1 + self.extr_efficiency * (alpha ** 2)))

	def compute_no_vacuum_probability(self):
		"""
		Compute the no-vacuum probability of the laser when either the triggering rate or the multi-photon probability
		:return: Non-vacuum probability [float]
		"""
		alpha = self.compute_alpha()
		return 1 - np.exp(-self.extr_efficiency * (alpha ** 2))

	def compute_effective_trigger_rate(self):
		"""
		Compute the effective trigger rate of the laser source. For the laser source, the effective trigger rate is
		always equal to the laser rate.
		:return: Effective trigger rate [float]
		"""
		return self.trigger_rate

	def compute_alpha(self):
		"""
		Compute the alpha parameter of the laser
		:return: Alpha parameter [float]
		"""
		return np.sqrt(self.output_power / self.trigger_rate)

	def compute_laser_state(self):
		"""
		Compute the state of the laser
		:return: State of the laser source [Qobj]
		"""
		alpha = self.compute_alpha()
		displacement = displace(self.fock_space_dim, alpha)
		laser_state = displacement * self.vacuum
		return laser_state

	def detector2observable(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector [Qobj]
		"""
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = self.bucket_detector(self.fock_space_dim, eta_detector, eta_noise)
		return apd_detector

	def signal_rate(self):
		"""
		Compute the signal rate which is the signal photons + noise photons. It is therefore the probability of
		having a detection event multiplied by the effective triggering rate.
		:return: The signal rate [float]
		"""
		laser_state = self.compute_laser_state()
		signal_rate = self.trigger_rate * expect(self.apd_detector, laser_state)
		return signal_rate

	def noise_rate(self):
		"""
		Compute the noise rate which is the noise photons that are detected. It is possible to determine the noise
		photons rate by using the system when the target is not present.
		:return: The noise rate [float]
		"""
		noise_rate = self.compute_effective_trigger_rate() * expect(self.apd_detector, self.vacuum)
		return noise_rate

	def signal_to_noise_rate(self):
		"""
		Compute the signal-to-noise rate (SNR) of the laser source.
		:return: The SNR [float]
		"""
		signal = self.signal_rate()
		noise = self.noise_rate()
		return (signal - noise) / noise

	@property
	def average_photon_per_pulse(self):
		"""
		:return: Average number of photons per pulse [float]
		"""
		return self.compute_alpha() ** 2


class SinglePhoton(Source):
	"""
	This class models a single photon source. The optical power is always fixed by the user. Therefore, the triggering
	rate, the multi-photon probability and the no-vacuum probability cannot be fixed since the purity of the source is
	also always fixed. The average photon per pulse cannot be modified for this source which remove our ability to
	fix the trigger rate, the multi-photon probability or the no-vacuum probability. If one of them is given, it must
	be the correct value or an error will be raised.
	"""
	def __init__(self, params, **kwargs):
		"""
		:param params: Parameters of the source using the object SetupParameters
		:param kwargs: N/A
		"""
		super().__init__(params)
		self.extr_efficiency = self.fix_extraction_efficiency()
		self.kwargs = kwargs
		self.fix_parameters()
		self.apd_detector = self.detector2observable()
		self.vacuum = basis(self.fock_space_dim, 0)

	def fix_extraction_efficiency(self):
		"""
		Fix the extraction efficiency of the source. This extraction efficiency is given by the collection efficiency
		of the system. The contribution of an adversary "stealing" some photons can be added. In most cases, we prefer
		not to specify the adversary efficiency because we do not have any prior information about it.
		:return: Extraction efficiency [float]
		"""
		if self.adversary_eff is None:
			return self.sp_collection
		return self.sp_collection * self.adversary_eff

	def fix_parameters(self):
		"""
		Four parameters can be played with: Output Power, Trigger Rate, Multi-photon probability and No-Vacuum probability
		For the Single Photon Source, the user can specified the Trigger Rate, the Multi-Photon Probability and the
		No Vaccum to None. The Single Photon Source will then compute the missing parameter.
		"""

		if self.trigger_rate is not None:
			aimed_trigger_rate = self.compute_effective_trigger_rate()
			assert self.trigger_rate == aimed_trigger_rate, f"The trigger rate is not valid it should be {aimed_trigger_rate}Hz but is {self.trigger_rate}"
		if self.multi_photon_probability is not None:
			aimed_multi_photon_probability = self.compute_multi_photon_probability()
			assert self.multi_photon_probability == aimed_multi_photon_probability, f"The multi-photon probability is not valid it should be {aimed_multi_photon_probability} but is {self.multi_photon_probability}"
		if self.no_vacuum_probability is not None:
			aimed_no_vacuum_probability = self.compute_no_vacuum_probability()
			assert self.no_vacuum_probability == aimed_no_vacuum_probability, f"The no-vacuum probability is not valid it should be {aimed_no_vacuum_probability} but is {self.no_vacuum_probability}"

		self.trigger_rate = self.compute_effective_trigger_rate()
		self.multi_photon_probability = self.compute_multi_photon_probability()
		self.no_vacuum_probability = self.compute_no_vacuum_probability()

	def compute_multi_photon_probability(self):
		"""
		Compute the multi-photon probability of the single photon source
		:return: Multi-photon probability [float]
		"""
		return self.sp_p2 * (self.extr_efficiency ** 2)

	def compute_no_vacuum_probability(self):
		"""
		Compute the no-vacuum probability of the single photon source
		:return: No-vacuum probability [float]
		"""
		return self.sp_p1 * self.extr_efficiency + self.sp_p2 * self.extr_efficiency * (2 - self.extr_efficiency)

	def compute_effective_trigger_rate(self):
		"""
		Compute the effective trigger rate of the single photon source
		:return: Effective trigger rate [float]
		"""
		return self.output_power / ((self.sp_p1 + 2 * self.sp_p2) * self.sp_collection)

	def single_photon_state(self):
		"""
		Compute the state of the single photon source
		:return: State of the single photon source [Qobj]
		"""
		vacuum = np.sqrt(1 - (self.sp_p1 + self.sp_p2)) * self.vacuum
		single_photon = np.sqrt(self.sp_p1) * basis(self.fock_space_dim, 1)
		two_photon = np.sqrt(self.sp_p2) * basis(self.fock_space_dim, 2)
		return vacuum + single_photon + two_photon

	def detector2observable(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector [Qobj]
		"""
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = self.bucket_detector(self.fock_space_dim, self.sp_collection * eta_detector, eta_noise)
		return apd_detector

	def signal_rate(self):
		"""
		Compute the signal rate of the single photon source
		:return: Signal rate [float]
		"""
		single_photon_state = self.single_photon_state()
		signal_rate = self.trigger_rate * expect(self.apd_detector, single_photon_state)
		return signal_rate

	def noise_rate(self):
		"""
		Compute the noise rate of the single photon source
		:return: Noise rate [float]
		"""
		noise = self.trigger_rate * expect(self.apd_detector, self.vacuum)
		return noise

	def signal_to_noise_rate(self):
		"""
		Compute the signal-to-noise rate (SNR) of the single photon source. The signal rate includes the noise photons.
		The SNR can therefore be understood as : (signal photons + noise photons) / noise photons
		:return: The SNR [float]
		"""
		signal_rate = self.signal_rate()
		noise = self.noise_rate()
		return (signal_rate - noise) / noise

	@property
	def average_photon_per_pulse(self):
		"""
		:return: Average number of photons per pulse [float]
		"""
		return self.sp_p1 + 2 * self.sp_p2


class EntangledPhotonSPDC(Source):
	"""
	This class models an entangled photon source using SPDC. The output power is always fixed by the user. From there,
	the user can only fix one of the three following parameters:
		1) The trigger rate
		2) The multi-photon probability
		3) The no-vacuum probability

	Setting one of them will automatically compute the two others. This is possible because, for the SPDC source, the
	average photon per pulse can easily be modified by playing with the power of the pump laser.
	"""
	def __init__(self, params, **kwargs):
		"""
		:param params: Parameters of the source using the object SetupParameters
		:param kwargs: N/A
		"""
		super().__init__(params)
		self.kwargs = kwargs
		self.epsilon = None
		self.extr_efficiency = self.fix_extraction_efficiency()
		self.apd_detector_signal = self.detector2observable_signal()
		self.apd_detector_idler = self.detector2observable_idler()
		self.vacuum = basis(self.fock_space_dim, 0)
		self.fix_parameters()

	def fix_extraction_efficiency(self):
		"""
		Fix the extraction efficiency of the source. This extraction efficiency is given by the collection efficiency
		:return: Extraction efficiency [float]
		"""
		if self.adversary_eff is None:
			return self.spdc_eps_collection
		return self.spdc_eps_collection * self.adversary_eff

	def fix_parameters(self):
		"""
		Fix the parameters of the SPDC requested by the user. The other parameters are then computed.
		"""
		assert self.trigger_rate is not None or self.multi_photon_probability is not None or self.no_vacuum_probability is not None, "The trigger rate, multi-photon probability or no-vacuum probability must be fixed"

		if self.trigger_rate is not None:
			assert self.multi_photon_probability is None and self.no_vacuum_probability is None, "multi-photon probability and no-vacuum probability can't be fixed if trigger rate is fixed"

			self.epsilon = (self.output_power / (self.trigger_rate * self.spdc_eps_collection)) - (
						1 / self.spdc_eps_heralding)

			self.multi_photon_probability = self.compute_multi_photon_probability()
			self.no_vacuum_probability = self.compute_no_vacuum_probability()
		elif self.multi_photon_probability is not None:
			assert self.trigger_rate is None and self.no_vacuum_probability is None, "trigger rate and no-vacuum probability can't be fixed if multi-photon probability is fixed"
			assert 0 < self.multi_photon_probability < 1, "The multi-photon probability must be between 0 and 1"

			self.epsilon = np.sqrt(self.multi_photon_probability) / (
					(1 - np.sqrt(self.multi_photon_probability)) * self.extr_efficiency)

			self.trigger_rate = self.compute_effective_trigger_rate()
			self.no_vacuum_probability = self.compute_no_vacuum_probability()

		elif self.no_vacuum_probability is not None:
			assert self.trigger_rate is None and self.multi_photon_probability is None, "trigger rate and multi-photon probability can't be fixed if no-vacuum probability is fixed"
			assert 0 < self.no_vacuum_probability < 1, "The no-vacuum probability must be between 0 and 1"

			self.epsilon = self.no_vacuum_probability / (self.extr_efficiency * (1 - self.no_vacuum_probability))

			self.trigger_rate = self.compute_effective_trigger_rate()
			self.multi_photon_probability = self.compute_multi_photon_probability()
		else:
			raise ValueError("The trigger rate, multi-photon probability or no-vacuum probability must be fixed")

	def compute_multi_photon_probability(self):
		"""
		Compute the multi-photon probability of the SPDC source
		:return: Multi-photon probability [float]
		"""
		return ((self.extr_efficiency ** 2) * (self.epsilon ** 2)) / ((self.extr_efficiency * self.epsilon + 1) ** 2)

	def compute_no_vacuum_probability(self):
		"""
		Compute the no-vacuum probability of the SPDC source
		:return: No-vacuum probability [float]
		"""
		return self.extr_efficiency * self.epsilon / (self.extr_efficiency * self.epsilon + 1)

	def compute_effective_trigger_rate(self):
		"""
		Compute the effective trigger rate of the SPDC source
		:return: Effective trigger rate [float]
		"""
		return (self.output_power * self.spdc_eps_heralding) / (
				self.spdc_eps_collection * (1 + self.epsilon * self.spdc_eps_heralding))

	def compute_eps_rate(self):
		"""
		Compute the rate of the SPDC source. The rate is the rate at which a pair of photons is emitted.
		:return: Rate of the SPDC source [float]
		"""
		return (self.trigger_rate * (1 + (self.epsilon * self.spdc_eps_heralding))) / (
				self.spdc_eps_heralding * self.epsilon)

	def squeezed_operator(self):
		"""
		Compute the squeezed operator of the SPDC source for a pair of squeezed state. The strength of the squeezing
		opterator is: r = arcsinh(sqrt(epsilon)) while theta = 0. Epsilon is the average number of photons per pulse.
		The value of theta is not that much important since it is a global phase.
		:return: Squeezed operator [Qobj]
		"""
		a = destroy(self.fock_space_dim)
		a_dagger = create(self.fock_space_dim)
		spdc_epsilon = np.arcsinh(np.sqrt(self.epsilon))
		argument = -1j * (tensor(a, a) + tensor(a_dagger, a_dagger)) * spdc_epsilon
		squeezed_operator = argument.expm()
		return squeezed_operator

	def compute_spdc_eps_state(self):
		"""
		Compute the state of the SPDC source using the squeezed operator
		:return: State of the SPDC source [Qobj]
		"""
		squeezed_operator = self.squeezed_operator()
		return squeezed_operator * tensor(self.vacuum, self.vacuum)

	def detector2observable_signal(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector [Qobj]
		"""
		eta_detector, eta_noise = self.overall_detection_probability()
		apd_detector = self.bucket_detector(self.fock_space_dim, self.spdc_eps_collection * eta_detector, eta_noise)
		return apd_detector

	def detector2observable_idler(self):
		"""
		Model the detector as an observable based on its detection efficiency and the noise detection probability
		:return: Observable representing the detector [Qobj]
		"""
		apd_detector_local = self.bucket_detector(self.fock_space_dim, self.spdc_eps_heralding,
		                                          self.detector_dark * self.timing_window)
		return apd_detector_local

	def signal_rate(self):
		"""
		Compute the signal rate of the SPDC source. The signal rate is the rate at which a signal photon is detected.
		:return: Signal rate [float]
		"""
		operator_joint_detection = tensor(self.apd_detector_idler, self.apd_detector_signal)
		return self.compute_eps_rate() * expect(operator_joint_detection, self.compute_spdc_eps_state())
	def noise_rate(self):
		"""
		Compute the noise rate of the SPDC source. The noise rate is the rate at which noise photons are detected.
		:return: Noise rate [float]
		"""
		operator_detector_signal_only = tensor(qeye(self.fock_space_dim), self.apd_detector_signal)
		joint_vacuum = tensor(self.vacuum, self.vacuum)
		return self.trigger_rate * expect(operator_detector_signal_only, joint_vacuum)

	def signal_to_noise_rate(self):
		"""
		Compute the signal-to-noise rate (SNR) of the SPDC source. The signal rate includes the noise photons.
		:return: The SNR [float]
		"""
		signal = self.signal_rate()
		noise = self.noise_rate()
		return (signal - noise) / noise

	@property
	def average_photon_per_pulse(self):
		"""
		:return: Average number of photons per pulse [float]
		"""
		return self.epsilon

	@property
	def prob_of_last_element_fock_space(self):
		"""
		A Fock space of n will imply that a vector of n elements is created. This function returns the norm squared
		of the n element of the vector. If this norm squared is too high, it means that the Fock space is not large
		enough. This function is used to check if the Fock space is large enough. Only interesting for debugging.
		:return: Probability of the last element of the Fock space [float]
		"""
		last_element_prob = np.abs(self.compute_spdc_eps_state()[-1])**2
		return last_element_prob[0]
	@property
	def g2(self):
		#TODO : NOT WORKING
		operator_joint_detection = tensor(self.apd_detector_idler, self.apd_detector_signal)
		joint_measurement_prob = expect(operator_joint_detection, self.compute_spdc_eps_state())

		prob_idler = (1 - (1 / ((self.epsilon + 1) - (self.epsilon * (1 - self.spdc_eps_heralding)))))

		efficiency_signal = self.spdc_eps_collection * self.overall_detection_probability()[0]
		prob_signal = (1 - (1 / ((efficiency_signal + 1) - (efficiency_signal * (1 - efficiency_signal)))))

		#operator_detector_signal_only = tensor(qeye(self.fock_space_dim), self.apd_detector_signal)
		#prob_signal = expect(operator_detector_signal_only, self.compute_spdc_eps_state())

		g2 = joint_measurement_prob / (prob_idler * prob_signal)

		return g2


class SetupParameters:
	"""
	This class is used to store the parameters of the simulation. It is used to store the parameters of the sources,
	the detection and the environment. The user can set the parameters using the __init__ method or by setting the
	parameters directly. A help function is available to understand the parameters that can be set.
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	@staticmethod
	def help():
		print("Here are the parameters you can set:")
		print("--- General ---")
		print("fock_space_dim: Dimension of the fock space, must be an integer")
		print("--- Sources ---")
		print("output_power: optical output power of the source in photon per second [s^-1]")
		print("trigger_rate: Pulsing rate of the laser source [Hz]")
		print("multi_photon_probability: Probability of the laser source emitting more than 1 photon [-]")
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
		print("atmosphere: Efficiency of propagation in the atmosphere")
		print("adversary_eff: Overall efficiency of the adversary at detecting (includes the atmospheric efficiency)")
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

	def __copy__(self):
		return SetupParameters(**self.__dict__)

	def __deepcopy__(self, memodict={}):
		return SetupParameters(**self.__dict__)
