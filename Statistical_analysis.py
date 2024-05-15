import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from Operators import *
from Sources import *

param = SetupParameters(
	fock_space_dim=10,
	output_power=2e7,
	laser_rate=1e8,
	sp_collection=0.2,
	sp_p1=0.9999,
	sp_p2=1e-4,
	spdc_eps_heralding=0.05,
	spdc_eps_collection=0.1,
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

# Reproducing graph for the ROC curve

### PULSED LASER ###
signal_laser = PulsedLaser(param).signal_rate()
noise_laser = PulsedLaser(param).noise_rate()
trigger_rate_laser = PulsedLaser(param).effective_trigger_rate()

#q0 = noise_laser/trigger_rate_laser
#q1 = signal_laser/trigger_rate_laser
q0 = PulsedLaser(param).false_positive_probability()
q1 = PulsedLaser(param).true_positive_probability()


M = trigger_rate_laser
x = np.linspace(1, M/1000, int(M/1000))


p0 = binom.pmf(x, M, q0)
p1 = binom.pmf(x, M, q1)

range_interval = 80
n_bins_over_range_interval = range_interval/299792458/param["timing_window"]

sum_p0 = 1-(1-np.cumsum(np.flip(p0)))**n_bins_over_range_interval
sum_p1 = np.cumsum(np.flip(p1))

p_d_pulsed = sum_p1
p_fa_pulsed = sum_p0

### SINGLE PHOTON ###
signal_sp = SinglePhoton(param).signal_rate()
noise_sp = SinglePhoton(param).noise_rate()
trigger_rate_sp = SinglePhoton(param).effective_trigger_rate()

#q0 = noise_sp/trigger_rate_sp
#q1 = signal_sp/trigger_rate_sp
q0 = SinglePhoton(param).false_positive_probability()
q1 = SinglePhoton(param).true_positive_probability()



L = int(trigger_rate_sp)

i = np.linspace(1, L/1000, int(L/1000))

p0 = binom.pmf(x, M, q0)
p1 = binom.pmf(x, M, q1)

n_bins_over_range_interval = 50/299792458/param["timing_window"]

sum_p0 = 1-(1-np.cumsum(np.flip(p0)))**n_bins_over_range_interval
sum_p1 = np.cumsum(np.flip(p1))

p_d_sp = sum_p1
p_fa_sp = sum_p0

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig = plt.figure(figsize=(16, 8))


plt.plot(p_fa_pulsed, p_d_pulsed, "-", label="Pulsed laser", linewidth=2.5, color="blue")
plt.plot(p_fa_sp, p_d_sp, "--", label="Single photon", linewidth=2.5, color="red")
plt.xlabel("False positive", fontsize=22)
plt.ylabel("True positive", fontsize=22)
plt.title("ROC curve with matched output power", fontsize=22, fontweight="bold")
plt.legend(fontsize=22, frameon=False)
plt.tick_params(labelsize=22)
plt.show()