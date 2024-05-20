import numpy as np
import matplotlib.pyplot as plt
from Sources import *
from copy import copy, deepcopy
from Analysis import *


param = SetupParameters(
	fock_space_dim=10,
	output_power=2e7,
	laser_rate=1e8,
	sp_collection=0.2,
	sp_p1=0.9999,
	sp_p2=1e-4,
	spdc_eps_heralding=0.18,
	spdc_eps_collection=0.2,
	spdc_emission=0.1,
	target_distance=40,
	receiver_diameter=0.5,
	target_albedo=0.2,
	optics_transmitter=0.5,
	optics_receiver=0.5,
	detection_efficiency=0.9,
	background=100,
	detector_dark=50,
	timing_window=0.5e-9,
)

range_interval = 50

### PULSED LASER ###
signal_laser = PulsedLaser(param).signal_rate()
noise_laser = PulsedLaser(param).noise_rate()
trigger_rate_laser = PulsedLaser(param).effective_trigger_rate()

true_positive_laser, false_positive_laser = RocAnalysis(
	signal_rate=signal_laser,
	noise_rate=noise_laser,
	trigger_rate=trigger_rate_laser,
	threshold_limit=trigger_rate_laser/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

### SINGLE PHOTON ###
signal_sp = SinglePhoton(param).signal_rate()
noise_sp = SinglePhoton(param).noise_rate()
trigger_rate_sp = SinglePhoton(param).effective_trigger_rate()

true_positive_sps, false_positive_sps = RocAnalysis(
	signal_rate=signal_sp,
	noise_rate=noise_sp,
	trigger_rate=trigger_rate_sp,
	threshold_limit=trigger_rate_sp/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

### Entangled Source ###
signal_spdc = EntangledPhotonSPDC(param).signal_rate()
noise_spdc = EntangledPhotonSPDC(param).noise_rate()
trigger_rate_spdc = EntangledPhotonSPDC(param).effective_trigger_rate()

true_positive_spdc, false_positive_spdc = RocAnalysis(
	signal_rate=signal_spdc,
	noise_rate=noise_spdc,
	trigger_rate=trigger_rate_spdc,
	threshold_limit=trigger_rate_spdc/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig = plt.figure(figsize=(16.1, 10))


plt.plot(false_positive_laser, true_positive_laser, "-", label="Pulsed laser", linewidth=2.5, color="blue")
plt.plot(false_positive_sps, true_positive_sps, "--", label="Single photon", linewidth=2.5, color="red")
plt.plot(false_positive_spdc, true_positive_spdc, "-.", label="Entangled photons (SPDC)", linewidth=2.5, color="green")
plt.xlabel("False positive", fontsize=22)
plt.ylabel("True positive", fontsize=22)
#plt.title("ROC curve with matched output power", fontsize=22, fontweight="bold")
plt.legend(fontsize=22, frameon=False)
plt.tick_params(labelsize=22)
plt.show()

### ROC curve with matched matched trigger rate (as seen by an adversary) ###

aimed_effective_trigger_rate = 1e7

param_laser = deepcopy(param)
param_laser["output_power"] = 1.0536e7
param_laser["laser_rate"] = 1e8

true_positive_laser, false_positive_laser = RocAnalysis(
	signal_rate=PulsedLaser(param_laser).signal_rate(),
	noise_rate=PulsedLaser(param_laser).noise_rate(),
	trigger_rate=PulsedLaser(param_laser).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_laser = PulsedLaser(param_laser).signal_to_noise_rate()
avg_power_per_pulse_laser = 1.0536e7/aimed_effective_trigger_rate

print(f"Detectable triggering rate laser : {1e8*(1-np.exp(-(1.0536e7/1e8)))}")

param_sp = deepcopy(param)
param_sp["output_power"] = 1.0001e7

true_positive_sp, false_positive_sp = RocAnalysis(
	signal_rate=SinglePhoton(param_sp).signal_rate(),
	noise_rate=SinglePhoton(param_sp).noise_rate(),
	trigger_rate=SinglePhoton(param_sp).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_sps = SinglePhoton(param_sp).signal_to_noise_rate()
avg_power_per_pulse_sps = 1.0001e7/aimed_effective_trigger_rate

print(f"Detectable triggering rate single photon : {1.0001e7/1.0001}")

param_spdc = deepcopy(param)
param_spdc["output_power"] = 1.08e7

true_positive_spdc, false_positive_spdc = RocAnalysis(
	signal_rate=EntangledPhotonSPDC(param_spdc).signal_rate(),
	noise_rate=EntangledPhotonSPDC(param_spdc).noise_rate(),
	trigger_rate=EntangledPhotonSPDC(param_spdc).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_spdc = EntangledPhotonSPDC(param_spdc).signal_to_noise_rate()
avg_power_per_pulse_spdc = 1.08e7/aimed_effective_trigger_rate

print(f"Detectable triggering rate SPDC : {1.08e7/1.08}")

plt.plot(false_positive_laser, true_positive_laser, "-", label=f"Pulsed laser:\n-SNR: {snr_laser:.2f}\n-Photon/pulse: {avg_power_per_pulse_laser:.2f}", linewidth=2.5, color="blue")
plt.plot(false_positive_sp, true_positive_sp, "--", label=f"Single photon:\n-SNR: {snr_sps:.2f}\n-Photon/pulse: {avg_power_per_pulse_sps:.2f}", linewidth=2.5, color="red")
plt.plot(false_positive_spdc, true_positive_spdc, "-.", label=f"SPDC:\n-SNR: {snr_spdc:.2f}\n-Photon/pulse: {avg_power_per_pulse_spdc:.2f}", linewidth=2.5, color="green")
plt.xlabel("False positive", fontsize=22)
plt.ylabel("True positive", fontsize=22)
plt.tick_params(labelsize=22)
plt.legend(fontsize=18, frameon=False, loc="upper left")
plt.show()

print(f"SNR laser : {snr_laser}", f"Average power per pulse : {avg_power_per_pulse_laser}")
print(f"SNR single photon : {snr_sps}", f"Average power per pulse : {avg_power_per_pulse_sps}")
print(f"SNR SPDC : {snr_spdc}", f"Average power per pulse : {avg_power_per_pulse_spdc}")


### Aimed triggering rate = 2.5e7 ###

aimed_effective_trigger_rate = 2.5e7

param_laser = deepcopy(param)
param_laser["output_power"] = 2.8768e7
param_laser["laser_rate"] = 1e8

true_positive_laser, false_positive_laser = RocAnalysis(
	signal_rate=PulsedLaser(param_laser).signal_rate(),
	noise_rate=PulsedLaser(param_laser).noise_rate(),
	trigger_rate=PulsedLaser(param_laser).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_laser = PulsedLaser(param_laser).signal_to_noise_rate()
avg_power_per_pulse_laser = 2.8768e7/aimed_effective_trigger_rate

param_sp = deepcopy(param)
param_sp["output_power"] = 2.5e7

true_positive_sp, false_positive_sp = RocAnalysis(
	signal_rate=SinglePhoton(param_sp).signal_rate(),
	noise_rate=SinglePhoton(param_sp).noise_rate(),
	trigger_rate=SinglePhoton(param_sp).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_sps = SinglePhoton(param_sp).signal_to_noise_rate()
avg_power_per_pulse_sps = 2.5e7/aimed_effective_trigger_rate

param_spdc = deepcopy(param)
param_spdc["output_power"] = 2.7e7

true_positive_spdc, false_positive_spdc = RocAnalysis(
	signal_rate=EntangledPhotonSPDC(param_spdc).signal_rate(),
	noise_rate=EntangledPhotonSPDC(param_spdc).noise_rate(),
	trigger_rate=EntangledPhotonSPDC(param_spdc).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_spdc = EntangledPhotonSPDC(param_spdc).signal_to_noise_rate()
avg_power_per_pulse_spdc = 2.7e7/aimed_effective_trigger_rate

plt.plot(false_positive_laser, true_positive_laser, "-", label=f"Pulsed laser:\n-SNR: {snr_laser:.2f}\n-Photon/pulse: {avg_power_per_pulse_laser:.2f}", linewidth=2.5, color="blue")
plt.plot(false_positive_sp, true_positive_sp, "--", label=f"Single photon:\n-SNR: {snr_sps:.2f}\n-Photon/pulse: {avg_power_per_pulse_sps:.2f}", linewidth=2.5, color="red")
plt.plot(false_positive_spdc, true_positive_spdc, "-.", label=f"SPDC:\n-SNR: {snr_spdc:.2f}\n-Photon/pulse: {avg_power_per_pulse_spdc:.2f}", linewidth=2.5, color="green")
plt.xlabel("False positive", fontsize=22)
plt.ylabel("True positive", fontsize=22)
plt.tick_params(labelsize=22)
plt.legend(fontsize=22, frameon=False)
plt.show()

print(f"SNR laser : {snr_laser}", f"Average power per pulse : {avg_power_per_pulse_laser}")
print(f"SNR single photon : {snr_sps}", f"Average power per pulse : {avg_power_per_pulse_sps}")
print(f"SNR SPDC : {snr_spdc}", f"Average power per pulse : {avg_power_per_pulse_spdc}")


### Aimed triggering rate = 5e7 ###

aimed_effective_trigger_rate = 5e7

param_laser = deepcopy(param)
param_laser["output_power"] = 6.9315e7
param_laser["laser_rate"] = 1e8

true_positive_laser, false_positive_laser = RocAnalysis(
	signal_rate=PulsedLaser(param_laser).signal_rate(),
	noise_rate=PulsedLaser(param_laser).noise_rate(),
	trigger_rate=PulsedLaser(param_laser).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_laser = PulsedLaser(param_laser).signal_to_noise_rate()
avg_power_per_pulse_laser = 6.9315e7/aimed_effective_trigger_rate

param_sp = deepcopy(param)
param_sp["output_power"] = 5.0005e7

true_positive_sp, false_positive_sp = RocAnalysis(
	signal_rate=SinglePhoton(param_sp).signal_rate(),
	noise_rate=SinglePhoton(param_sp).noise_rate(),
	trigger_rate=SinglePhoton(param_sp).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_sps = SinglePhoton(param_sp).signal_to_noise_rate()
avg_power_per_pulse_sps = 5.0005e7/aimed_effective_trigger_rate

param_spdc = deepcopy(param)
param_spdc["output_power"] = 5.4e7

true_positive_spdc, false_positive_spdc = RocAnalysis(
	signal_rate=EntangledPhotonSPDC(param_spdc).signal_rate(),
	noise_rate=EntangledPhotonSPDC(param_spdc).noise_rate(),
	trigger_rate=EntangledPhotonSPDC(param_spdc).effective_trigger_rate(),
	threshold_limit=aimed_effective_trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

snr_spdc = EntangledPhotonSPDC(param_spdc).signal_to_noise_rate()
avg_power_per_pulse_spdc = 5.4e7/aimed_effective_trigger_rate

plt.plot(false_positive_laser, true_positive_laser, "-", label=f"Pulsed laser:\n-SNR: {snr_laser:.2f}\n-Photon/pulse: {avg_power_per_pulse_laser:.2f}", linewidth=2.5, color="blue")
plt.plot(false_positive_sp, true_positive_sp, "--", label=f"Single photon:\n-SNR: {snr_sps:.2f}\n-Photon/pulse: {avg_power_per_pulse_sps:.2f}", linewidth=2.5, color="red")
plt.plot(false_positive_spdc, true_positive_spdc, "-.", label=f"SPDC:\n-SNR: {snr_spdc:.2f}\n-Photon/pulse: {avg_power_per_pulse_spdc:.2f}", linewidth=2.5, color="green")
plt.xlabel("False positive", fontsize=22)
plt.ylabel("True positive", fontsize=22)
plt.tick_params(labelsize=22)
plt.legend(fontsize=22, frameon=False)
plt.show()

print(f"SNR laser : {snr_laser}", f"Average power per pulse : {avg_power_per_pulse_laser}")
print(f"SNR single photon : {snr_sps}", f"Average power per pulse : {avg_power_per_pulse_sps}")
print(f"SNR SPDC : {snr_spdc}", f"Average power per pulse : {avg_power_per_pulse_spdc}")


### Match Multi-photon probability with source & entangled###

### Pulsed Laser with sps ###

alpha_range = np.linspace(0, 2, 10000)
sps_p2 = param["sp_p2"]
idx = np.argmin(np.abs(alpha_range - sps_p2))
alpha=alpha_range[idx]

signal = PulsedLaser(param, alpha=alpha).signal_rate()
noise = PulsedLaser(param, alpha=alpha).noise_rate()
trigger_rate = PulsedLaser(param, alpha=alpha).effective_trigger_rate()

true_positive_laser_sps, false_positive_laser_sps = RocAnalysis(
	signal_rate=signal,
	noise_rate=noise,
	trigger_rate=trigger_rate,
	threshold_limit=trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

### Pulsed Laser with entangled ###

spdc_p2 = 0.0832
idx = np.argmin(np.abs(alpha_range - spdc_p2))
alpha=alpha_range[idx]

signal = PulsedLaser(param, alpha=alpha).signal_rate()
noise = PulsedLaser(param, alpha=alpha).noise_rate()
trigger_rate = PulsedLaser(param, alpha=alpha).effective_trigger_rate()

true_positive_laser_spdc, false_positive_laser_spdc = RocAnalysis(
	signal_rate=signal,
	noise_rate=noise,
	trigger_rate=trigger_rate,
	threshold_limit=trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()


### Single photon ###

signal = SinglePhoton(param).signal_rate()
noise = SinglePhoton(param).noise_rate()
trigger_rate = SinglePhoton(param).effective_trigger_rate()

true_positive_sps, false_positive_sps = RocAnalysis(
	signal_rate=signal,
	noise_rate=noise,
	trigger_rate=trigger_rate,
	threshold_limit=trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()

### Entangled ###

signal = EntangledPhotonSPDC(param).signal_rate()
noise = EntangledPhotonSPDC(param).noise_rate()
trigger_rate = EntangledPhotonSPDC(param).effective_trigger_rate()

true_positive_spdc, false_positive_spdc = RocAnalysis(
	signal_rate=signal,
	noise_rate=noise,
	trigger_rate=trigger_rate,
	threshold_limit=trigger_rate/1000,
	range_interval=range_interval,
	timing_window=param["timing_window"]
).compute_p_d_p_fa()


fig, ax = plt.subplots(1, 2, figsize=(16.1, 10))

ax[0].plot(false_positive_laser_sps, true_positive_laser_sps, "-", label="Pulsed laser", linewidth=2.5, color="blue")
ax[0].plot(false_positive_sps, true_positive_sps, "--", label="SPS", linewidth=2.5, color="red")
ax[0].legend(fontsize=22, bbox_to_anchor=(0.5, 1.15), loc='upper center', frameon=False)
ax[0].set_xlabel("False positive", fontsize=22)
ax[0].set_ylabel("True positive", fontsize=22)
ax[0].tick_params(labelsize=22)

ax[1].plot(false_positive_laser_spdc, true_positive_laser_spdc, "-", label="Pulsed laser", linewidth=2.5, color="blue")
ax[1].plot(false_positive_spdc, true_positive_spdc, "--", label="SPDC", linewidth=2.5, color="red")
ax[1].legend(fontsize=22, bbox_to_anchor=(0.5, 1.15), loc='upper center', frameon=False)
ax[1].set_xlabel("False positive", fontsize=22)
ax[1].set_ylabel("True positive", fontsize=22)
ax[1].tick_params(labelsize=22)


plt.show()