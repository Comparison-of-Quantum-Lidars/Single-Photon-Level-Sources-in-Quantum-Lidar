import numpy as np
import matplotlib.pyplot as plt
from Sources import SinglePhoton
from copy import deepcopy

result = np.load("data/Figure_3_4_5.npy", allow_pickle=True).item()

non_vacuum_number = result["non_vacuum_number"]
snr_entangled_35 = result["eps"]['0.35']['snr']
snr_entangled_57 = result["eps"]['0.57']['snr']
snr_entangled_80 = result["eps"]['0.8']['snr']
snr_entangled_100 = result["eps"]['1']['snr']

param = result["param"]

param_sps = deepcopy(param)
param_sps["fock_space_dim"] = 3
param_sps["sp_collection"] = 0.35
number_nv_pulse_sps_35 = SinglePhoton(param_sps).number_nv_pulse
snr_sps_35 = result["sps"]["0.35"]["snr"]
param_sps["sp_collection"] = 0.57
number_nv_pulse_sps_57 = SinglePhoton(param_sps).number_nv_pulse
snr_sps_57 = result["sps"]["0.57"]["snr"]
param_sps["sp_collection"] = 0.8
number_nv_pulse_sps_80 = SinglePhoton(param_sps).number_nv_pulse
snr_sps_80 = result["sps"]["0.8"]["snr"]
param_sps["sp_collection"] = 1.0
number_nv_pulse_sps_100 = SinglePhoton(param_sps).number_nv_pulse
snr_sps_100 = result["sps"]["1"]["snr"]

number_nv_pulse_sps_35_array10 = result["sps"]["N=10"]["0.35"]["number_nv_pulse"]
snr_sps_35_array10 = result["sps"]["N=10"]["0.35"]["snr"]
number_nv_pulse_sps_57_array10 = result["sps"]["N=10"]["0.57"]["number_nv_pulse"]
snr_sps_57_array10 = result["sps"]["N=10"]["0.57"]["snr"]
number_nv_pulse_sps_80_array10 = result["sps"]["N=10"]["0.8"]["number_nv_pulse"]
snr_sps_80_array10 = result["sps"]["N=10"]["0.8"]["snr"]
number_nv_pulse_sps_100_array10 = result["sps"]["N=10"]["1"]["number_nv_pulse"]
snr_sps_100_array10 = result["sps"]["N=10"]["1"]["snr"]

number_nv_pulse_sps_array_10 = np.array([number_nv_pulse_sps_35_array10, number_nv_pulse_sps_57_array10,
                                         number_nv_pulse_sps_80_array10, number_nv_pulse_sps_100_array10])
snr_sps_array_10 = np.array([snr_sps_35_array10, snr_sps_57_array10, snr_sps_80_array10, snr_sps_100_array10])

number_nv_pulse_sps = np.array([number_nv_pulse_sps_35, number_nv_pulse_sps_57, number_nv_pulse_sps_80, number_nv_pulse_sps_100])
snr_sps = np.array([snr_sps_35, snr_sps_57, snr_sps_80, snr_sps_100])

snr_laser = result["laser"]["snr"]

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig, ax1 = plt.subplots(figsize=(16.2, 10))

ax1.semilogy(non_vacuum_number / 1e3, snr_entangled_35, "-", label="Correlated Photon Pair Source", color="green",
             linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number / 1e3, snr_entangled_57, "-", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number / 1e3, snr_entangled_80, "-", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number / 1e3, snr_entangled_100, "-", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number / 1e3, snr_laser, "-.", label="Pulsed Laser", color="blue", linewidth=3, zorder=2)
ax1.scatter(number_nv_pulse_sps / 1e3, snr_sps, label="Single-Photon Source", color="red", s=100, zorder=3)
ax1.scatter(number_nv_pulse_sps_array_10 / 1e3, snr_sps_array_10, label="Array of 10 Single-Photon Sources", color="red", s=75, marker="s", zorder=3, alpha=0.7)

ax1.axvspan(50000 / 1e3, 100000 / 1e3, color="dodgerblue", alpha=0.35, hatch='//', label="Preferred regime against a non-PNR det.", zorder=0)
ax1.axvspan(350000 / 1e3, 400000 / 1e3, color="lightcoral", alpha=0.35, hatch='..', label="Preferred regime against a PNR det.", zorder=0)

ax1.set_xlabel(r"Number of non-vacuum pulses $(\times 10^3)$", fontsize=25)
ax1.set_ylabel("SNR", fontsize=25)
ax1.set_xlim([None, 450000 / 1e3])
ax1.legend(frameon=False, fontsize=20, loc="lower center")
ax1.tick_params(axis="x", labelsize=25)
ax1.tick_params(axis="y", labelsize=25)
xticks = ax1.get_xticks()
xticks = [x for x in xticks if x <= 400000 / 1e3]
ax1.set_xticks(xticks)

x_pos = 4e5 + 0.05e5
y_factor = 1
text_fontsize = 22
ax1.text(x_pos / 1e3, snr_sps_35 - 1/y_factor, "35%", fontsize=text_fontsize, fontweight="bold")
ax1.text(x_pos / 1e3, snr_sps_57 - 2/y_factor, "57%", fontsize=text_fontsize, fontweight="bold")
ax1.text(x_pos / 1e3, snr_sps_80 - 2/y_factor, "80%", fontsize=text_fontsize, fontweight="bold")
ax1.text(x_pos / 1e3, snr_sps_100, "100%", fontsize=text_fontsize, fontweight="bold")
ax1.text(x_pos / 1e3, snr_sps_100 + 20/y_factor, r"$\eta_{signal}$", fontsize=28, fontweight="bold")

offset_x_axis_100p = 40000  # Normally 40000
ax1.text((number_nv_pulse_sps_array_10[-1] - offset_x_axis_100p) / 1e3, snr_sps_array_10[-1] - 20/y_factor, "100%", fontsize=text_fontsize, fontweight="bold")
ax1.text((number_nv_pulse_sps_array_10[-2] - 20000) / 1e3, snr_sps_array_10[-2] - 65/y_factor, r"80%", fontsize=text_fontsize, fontweight="bold")
ax1.text((number_nv_pulse_sps_array_10[-3] - 10000) / 1e3, snr_sps_array_10[-3] - 55/y_factor, r"57%", fontsize=text_fontsize, fontweight="bold")
ax1.text((number_nv_pulse_sps_array_10[-4] - 7500) / 1e3, snr_sps_array_10[-4] - 30/y_factor, r"35%", fontsize=text_fontsize, fontweight="bold")



def top_axis_transform(x):
    x = np.array(x) * 1e3
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x != 0, param["output_power"] / x, 0)
    return result

def top_axis_inverse(x):
    x = np.array(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x != 0, param["output_power"] / x, 0)
    return result

xticks = ax1.get_xticks()
xticks_second = np.round(top_axis_transform(xticks)[::-1], 1)
xticks_second = xticks_second[:-1]


secax = ax1.secondary_xaxis('top', functions=(top_axis_transform, top_axis_inverse))
secax.set_xlabel("Average photon number per non-vacuum pulse", fontsize=25)
secax.set_ticks(xticks_second)
secax.tick_params(axis="x", labelsize=25)

#plt.savefig("SNR_vs_number_non_vacuum_pulse.png", dpi=600, bbox_inches="tight")
plt.show()