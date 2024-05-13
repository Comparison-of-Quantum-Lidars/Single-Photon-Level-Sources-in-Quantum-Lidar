import numpy as np


class EntangledPhotonSource:

	def __init__(self, params):
		self.params = params

	def check_all_params(self):
		pass


class SinglePhotonSource:

	def __init__(self):
		pass


class PulsedLaserSource:

	def __init__(self):
		pass


class SetupParameters:

	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	@staticmethod
	def help():
		print("Here are the parameters you can set:")
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
