import numpy as np
from scipy.constants import h, c
from scipy.special import lambertw


class MeasureDetectorEfficiency:

	def __init__(
			self,
			power_after_attenuator: float,
			wavelength: float,
			real_count_rate: float,
	):
		self.power_after_attenuator = power_after_attenuator
		self.wavelength = wavelength
		self.real_count_rate = real_count_rate

	def calculate_detector_efficiency(self):
		detector_efficiency = self.real_count_rate / self.compute_expected_counts()
		self.display_results(detector_efficiency)
		return detector_efficiency

	def compute_expected_counts(self):
		return (self.power_after_attenuator * self.wavelength) / (h * c)

	@staticmethod
	def display_results(detector_efficiency):
		print(f"Detector efficiency: {detector_efficiency}")


class MeasureOverallLoss:

	def __init__(
			self,
			detector_efficiency: float,
			triggering_rate: float,
			signal_with_laser_no_target: float,
			noise_level_signal_with_laser_no_target: float,
			signal_with_laser_target: float,
			noise_level_signal_with_laser_target: float,
	):
		self.detector_efficiency = detector_efficiency
		self.triggering_rate = triggering_rate
		self.signal_with_laser_no_target = signal_with_laser_no_target
		self.noise_level_signal_with_laser_no_target = noise_level_signal_with_laser_no_target
		self.signal_with_laser_target = signal_with_laser_target
		self.noise_level_signal_with_laser_target = noise_level_signal_with_laser_target

	def calculate_overall_loss(self):
		eta_loss = self.calculate_loss()
		self.display_results(eta_loss)
		return eta_loss

	def calculate_average_photons_per_pulse(self):
		signal_laser = self.signal_with_laser_no_target - self.noise_level_signal_with_laser_no_target
		arg_ln = -(signal_laser / self.triggering_rate) + 1
		return (-1 / self.detector_efficiency) * np.log(arg_ln)

	def calculate_loss(self, mu=None):
		signal_laser_target = self.signal_with_laser_target - self.noise_level_signal_with_laser_target
		arg_ln = -(signal_laser_target / self.triggering_rate) + 1
		average_photons_per_pulse = self.calculate_average_photons_per_pulse() if mu is None else mu
		eta_loss = (-1 / (average_photons_per_pulse * self.detector_efficiency)) * np.log(arg_ln)
		return eta_loss

	@staticmethod
	def display_results(eta_loss):
		print(f"The Overall Loss is {eta_loss}")


class SPSMeasurement:

	def __init__(
			self,
			signal_no_attenuation: float,
			noise_level_no_attenuation: float,
			signal_with_attenuation: float,
			attenuation: float,
			noise_level_with_attenuation: float,
			detector_efficiency: float,
			triggering_rate: float,
	):
		self.signal_no_attenuation = signal_no_attenuation
		self.noise_level_no_attenuation = noise_level_no_attenuation
		self.signal_with_attenuation = signal_with_attenuation
		self.attenuation = attenuation
		self.noise_level_with_attenuation = noise_level_with_attenuation
		self.detector_efficiency = detector_efficiency
		self.triggering_rate = triggering_rate

	def parameters_estimation(self):
		multi_photon_prob = self.multi_photon_probability()
		non_vacuum_prob = self.non_vacuum_probability()
		p_out = self.optical_power()
		self.display_results(multi_photon_prob, non_vacuum_prob, p_out)
		return multi_photon_prob, non_vacuum_prob, p_out

	def multi_photon_probability(self):
		signal_sps_no_attenuation = self.signal_no_attenuation - self.noise_level_no_attenuation
		signal_sps_with_attenuation = self.signal_with_attenuation - self.noise_level_with_attenuation
		multi_photon_prob = (signal_sps_no_attenuation * self.attenuation - signal_sps_with_attenuation) / (
				self.triggering_rate * (self.detector_efficiency ** 2) * self.attenuation * (self.attenuation - 1))
		return multi_photon_prob

	def non_vacuum_probability(self):
		signal_sps_no_attenuation = self.signal_no_attenuation - self.noise_level_no_attenuation
		multi_photon_prob = self.multi_photon_probability()
		non_vacuum_prob = ((signal_sps_no_attenuation / (
				self.triggering_rate * self.detector_efficiency)) + multi_photon_prob * (
				                   self.detector_efficiency - 1))
		return non_vacuum_prob

	def optical_power(self):
		signal_sps_no_attenuation = self.signal_no_attenuation - self.noise_level_no_attenuation
		multi_photon_prob = self.multi_photon_probability()
		p_out = self.triggering_rate * (signal_sps_no_attenuation / (
				self.detector_efficiency * self.triggering_rate) + multi_photon_prob * self.detector_efficiency)
		return p_out

	@staticmethod
	def display_results(multi_photon_prob, non_vacuum_prob, p_out):
		print(f"SPS: Multi-Photon Probability = {multi_photon_prob + 1e-12}")
		print(f"SPS: Non-Vacuum Probability = {non_vacuum_prob}")
		print(f"SPS: Output Power = {p_out}")


class EPSMeasurement:

	def __init__(
			self,
			signal_signal_photon: float,
			noise_level_signal_photon: float,
			signal_detector_efficiency: float,
			signal_idler_photon: float,
			noise_level_idler_photon: float,
			idler_detector_efficiency: float,
			signal_coincidence: float,
			aimed_p_out: float,
			aimed_multi_photon_probability: float = None,
			aimed_non_vacuum_probability: float = None,
	):
		self.signal_signal_photon = signal_signal_photon
		self.noise_level_signal_photon = noise_level_signal_photon
		self.signal_detector_efficiency = signal_detector_efficiency
		self.signal_idler_photon = signal_idler_photon
		self.noise_level_idler_photon = noise_level_idler_photon
		self.idler_detector_efficiency = idler_detector_efficiency
		self.signal_coincidence = signal_coincidence
		self.aimed_multi_photon_probability = aimed_multi_photon_probability
		self.aimed_non_vacuum_probability = aimed_non_vacuum_probability
		self.aimed_p_out = aimed_p_out
		self.check_params()

	def check_params(self):
		assert self.aimed_multi_photon_probability is None or self.aimed_non_vacuum_probability is None, "Only one parameter can be estimated at a time"
		assert not (
				self.aimed_multi_photon_probability is None and self.aimed_non_vacuum_probability is None), "At least one parameter must be estimated"
		assert self.aimed_p_out is not None, "The output power must be estimated"

	def parameters_estimation(self):
		eps_rate_current = self.current_eps_rate()
		eps_rate_aimed = self.aimed_eps_rate()
		self.display_results(eps_rate_current, eps_rate_aimed)
		return eps_rate_current, eps_rate_aimed

	def signal_collection_efficiency(self):
		idler_signal_photon = self.signal_idler_photon - self.noise_level_idler_photon
		return self.signal_coincidence / (self.signal_detector_efficiency * idler_signal_photon)

	def idler_collection_efficiency(self):
		signal_signal_photon = self.signal_signal_photon - self.noise_level_signal_photon
		return self.signal_coincidence / (self.idler_detector_efficiency * signal_signal_photon)

	def current_eps_rate(self):
		signal_signal_photon = self.signal_signal_photon - self.noise_level_signal_photon
		idler_signal_photon = self.signal_idler_photon - self.noise_level_idler_photon
		return (signal_signal_photon * idler_signal_photon) / self.signal_coincidence

	def average_photons_per_pulse_if_non_vacuum_fix(self):
		collection_efficiency_signal = self.signal_collection_efficiency()
		non_vacuum_prob = self.aimed_non_vacuum_probability
		average_photon_per_pulse = non_vacuum_prob / ((1 - non_vacuum_prob) * collection_efficiency_signal)
		return average_photon_per_pulse

	def average_photons_per_pulse_if_multi_photon_fix(self):
		collection_efficiency_signal = self.signal_collection_efficiency()
		multi_photon_prob = self.aimed_multi_photon_probability
		sqrt_multi_photon_prob = np.sqrt(multi_photon_prob)
		average_photon_per_pulse = sqrt_multi_photon_prob / (
				(1 - sqrt_multi_photon_prob) * collection_efficiency_signal)
		return average_photon_per_pulse

	def aimed_eps_rate(self):
		if self.aimed_multi_photon_probability is not None:
			average_photon_per_pulse = self.average_photons_per_pulse_if_multi_photon_fix()
		elif self.aimed_non_vacuum_probability is not None:
			average_photon_per_pulse = self.average_photons_per_pulse_if_non_vacuum_fix()
		else:
			raise ValueError("No parameter to estimate")

		collection_efficiency_signal = self.signal_collection_efficiency()

		return self.aimed_p_out / (average_photon_per_pulse * collection_efficiency_signal)

	@staticmethod
	def display_results(eps_rate_current, eps_rate_aimed):
		print(f"EPS: Current EPS Rate = {eps_rate_current}")
		print(f"EPS: Aimed EPS Rate = {eps_rate_aimed}")
		print(f"EPS: Difference : [{(eps_rate_aimed - eps_rate_current)}]")


class LaserMeasurement:

	def __init__(
			self,
			signal: float,
			noise_level: float,
			detector_efficiency: float,
			aimed_p_out: float,
			aimed_multi_photon_probability: float = None,
			aimed_non_vacuum_probability: float = None,
	):
		self.signal = signal
		self.noise_level = noise_level
		self.detector_efficiency = detector_efficiency
		self.aimed_p_out = aimed_p_out
		self.aimed_multi_photon_probability = aimed_multi_photon_probability
		self.aimed_non_vacuum_probability = aimed_non_vacuum_probability
		self.check_params()

	def check_params(self):
		assert self.aimed_multi_photon_probability is None or self.aimed_non_vacuum_probability is None, "Only one parameter can be estimated at a time"
		assert not (
				self.aimed_multi_photon_probability is None and self.aimed_non_vacuum_probability is None), "At least one parameter must be estimated"
		assert self.aimed_p_out is not None, "The output power must be estimated"

	def parameters_estimation(self):
		optimal_trigger_rate = self.optimal_triggering_rate()
		current_p_out = self.current_optical_power()
		aimed_signal_rate = self.aimed_signal_rate()
		# current_average_photon_per_pulse = self.current_average_photon_per_pulse()
		# if self.aimed_multi_photon_probability is not None:
		# 	average_photon_per_pulse = self.average_photons_per_pulse_if_multi_photon_fix()
		# elif self.aimed_non_vacuum_probability is not None:
		# 	average_photon_per_pulse = self.average_photons_per_pulse_if_non_vacuum_fix()
		# else:
		# 	raise ValueError("No parameter to estimate")

		self.display_results(optimal_trigger_rate, current_p_out, aimed_signal_rate)

	def average_photons_per_pulse_if_non_vacuum_fix(self):
		return np.log(-self.aimed_non_vacuum_probability + 1)

	def average_photons_per_pulse_if_multi_photon_fix(self):
		average_photon_per_pulse = -lambertw(z=((self.aimed_multi_photon_probability - 1) / np.exp(1)), k=-1) - 1
		assert average_photon_per_pulse.imag == 0, "The average photon per pulse is not real"
		return average_photon_per_pulse.real

	def optimal_triggering_rate(self):
		if self.aimed_multi_photon_probability is not None:
			average_photon_per_pulse = self.average_photons_per_pulse_if_multi_photon_fix()
		elif self.aimed_non_vacuum_probability is not None:
			average_photon_per_pulse = self.average_photons_per_pulse_if_non_vacuum_fix()
		else:
			raise ValueError("No parameter to estimate")

		return self.aimed_p_out / average_photon_per_pulse

	def current_average_photon_per_pulse(self):
		triggering_rate = self.optimal_triggering_rate()
		signal = self.signal - self.noise_level

		ln_arg = -(signal / triggering_rate) + 1

		return (-1 / self.detector_efficiency) * np.log(ln_arg)

	def current_optical_power(self):
		triggering_rate = self.optimal_triggering_rate()
		current_average_photon_per_pulse = self.current_average_photon_per_pulse()
		return triggering_rate * current_average_photon_per_pulse

	def aimed_signal_rate(self):
		if self.aimed_multi_photon_probability is not None:
			average_photon_per_pulse = self.average_photons_per_pulse_if_multi_photon_fix()
		elif self.aimed_non_vacuum_probability is not None:
			average_photon_per_pulse = self.average_photons_per_pulse_if_non_vacuum_fix()
		else:
			raise ValueError("No parameter to estimate")
		trigger_rate = self.optimal_triggering_rate()

		return trigger_rate * (1 - np.exp(-average_photon_per_pulse * self.detector_efficiency))

	def display_results(self, optimal_trigger_rate, current_p_out, signal_rate):
		print(f"MAKE SURE THE TRIGGERING IS SET TO THE CORRECT VALUE BEFORE CONTINUING WITH THE EXPERIMENT!")
		print(f"Laser: Optimal Triggering Rate = {optimal_trigger_rate}")
		print(f"Laser: Current Output Power = {current_p_out}")
		print(f"Laser: Aimed Output Power = {self.aimed_p_out}")
		print(f"Difference : [{(self.aimed_p_out - current_p_out)}]")
		print(f"Laser: Current signal rate = {self.signal - self.noise_level}")
		print(f"Laser: Aimed Signal Rate = {signal_rate}")
		print(f"Difference : [{(signal_rate - (self.signal - self.noise_level))}]")


if __name__ == '__main__':
	# - * - Experiment 1 : Detector Efficiency - * -

	# MeasureDetectorEfficiency(
	# 	power_after_attenuator=2.5e-9,
	# 	wavelength=1550*10**-9,
	# 	real_count_rate=10e9
	# ).calculate_detector_efficiency()

	# - * - Experiment 2 : Overall loss - * -

	# MeasureOverallLoss(
	# 	detector_efficiency=0.5,
	# 	triggering_rate=1e6,
	# 	signal_with_laser_no_target=100000,
	# 	noise_level_signal_with_laser_no_target=50000,
	# 	signal_with_laser_target=10,
	# 	noise_level_signal_with_laser_target=5,
	# ).calculate_overall_loss()

	# - * - Experiment 3 : SPS Measurements - * -

	# SPSMeasurement(
	# 	signal_no_attenuation=100000,
	# 	noise_level_no_attenuation=50000,
	# 	signal_with_attenuation=10852,
	# 	attenuation=0.1,
	# 	noise_level_with_attenuation=5095,
	# 	detector_efficiency=0.5,
	# 	triggering_rate=0.5e6,
	# ).parameters_estimation()

	# - * - Experiment 4 : EPS Measurements - * -

	# EPSMeasurement(
	# 	signal_signal_photon=187800,
	# 	noise_level_signal_photon=5000,
	# 	signal_detector_efficiency=0.5,
	# 	signal_idler_photon=187800,
	# 	noise_level_idler_photon=5000,
	# 	idler_detector_efficiency=0.5,
	# 	signal_coincidence=100000,
	# 	aimed_multi_photon_probability=0.06728888888988889,
	# 	#aimed_non_vacuum_probability=0.16635555555555556,
	# 	aimed_p_out=116822.22222222223,
	# ).parameters_estimation()

	# - * - Experiment 5 : Laser Measurements - * -

	# LaserMeasurement(
	# 	signal=53172,
	# 	noise_level=500,
	# 	detector_efficiency=0.5,
	# 	aimed_multi_photon_probability=0.06728888888988889,
	# 	aimed_p_out=116822.22222222223,
	# ).parameters_estimation()

	print("done")
