import numpy as np
from Sources import SinglePhoton, PulsedLaser, EntangledPhotonSPDC, SetupParameters
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

param = SetupParameters(
	fock_space_dim=6,
	output_power=2e6,
	trigger_rate=None,
	multi_photon_probability=0.5,
	no_vacuum_probability=None,
	sp_collection=None,
	sp_p1=0.99,
	sp_p2=1e-3,
	spdc_eps_heralding=0.18,
	spdc_eps_collection=0.2,
	channel_efficiency=1,
	target_distance=1,
	receiver_diameter=0.05,
	target_albedo=0.2,
	optics_transmitter=0.8,
	optics_receiver=0.5,
	detection_efficiency=0.5,
	background=400,
	detector_dark=200,
	timing_window=0.5e-9,
)


eps = EntangledPhotonSPDC(param)

mpp = np.linspace(0.001, 0.99, 600)

g2 = []

for multi in tqdm(mpp):
	param.multi_photon_probability = multi
	eps = EntangledPhotonSPDC(param)
	g2.append(eps.g2)

plt.plot(mpp, g2)
plt.show()
