import numpy as np
import matplotlib.pyplot as plt
from Sources import SinglePhoton
from copy import deepcopy

result = np.load("5m_snr_matching_non_vacuum_pulse_2025_06_22.npy", allow_pickle=True).item()

non_vacuum_number = result["non_vacuum_number"]
snr_entangled_35 = result["eps"]['0.35']['snr']
snr_entangled_57 = result["eps"]['0.57']['snr']
snr_entangled_80 = result["eps"]['0.8']['snr']
snr_entangled_100 = result["eps"]['1']['snr']

param = result["param"]

param_sps = deepcopy(param)
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

number_nv_pulse_sps = np.array([number_nv_pulse_sps_35, number_nv_pulse_sps_57, number_nv_pulse_sps_80, number_nv_pulse_sps_100])
snr_sps = np.array([snr_sps_35, snr_sps_57, snr_sps_80, snr_sps_100])

snr_laser = result["laser"]["snr"]


plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

fig, ax1 = plt.subplots(figsize=(16.2, 10))

ax1.semilogy(non_vacuum_number, snr_entangled_35, "-.", label="Entangled Photon Source", color="green",
             linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_entangled_57, "-.", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_entangled_80, "-.", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_entangled_100, "-.", color="green", linewidth=3, zorder=1)
ax1.semilogy(non_vacuum_number, snr_laser, "-", label="Pulsed Laser", color="blue", linewidth=3, zorder=1)
ax1.scatter(number_nv_pulse_sps, snr_sps, label="Single Photon Source", color="red", s=100, zorder=2)

ax1.axvspan(50000, 100000, color="lightblue", alpha=0.35, label="Optimal regime against a NNRD", zorder=0)
ax1.axvspan(350000, 400000, color="lightcoral", alpha=0.35, label="Optimal regime against a NRD", zorder=0)

ax1.set_xlabel("Number of non-vacuum pulse", fontsize=22)
ax1.set_ylabel("SNR", fontsize=22)
ax1.set_xlim([None, 450000])
ax1.legend(frameon=False, fontsize=22)
ax1.tick_params(axis="x", labelsize=22)
ax1.tick_params(axis="y", labelsize=22)
xticks = ax1.get_xticks()
xticks = [x for x in xticks if x <= 400000]
ax1.set_xticks(xticks)


x_pos = 4e5 + 0.05e5
ax1.text(x_pos, snr_sps_35 - 1, "35%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_sps_57 - 2, "57%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_sps_80 - 2, "80%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_sps_100, "100%", fontsize=22, fontweight="bold")
ax1.text(x_pos, snr_sps_100 + 20, r"$\eta_{signal}$", fontsize=28, fontweight="bold")
def top_axis_transform(x):
    x = np.array(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x != 0, param["output_power"] / x, 0)
    return result

def top_axis_inverse(x):
    x = np.array(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x != 0, param["output_power"] / x, 0)
    return result

xticks = ax1.get_xticks()
xticks_second = top_axis_transform(xticks)[::-1]
xticks_second = xticks_second[:-1]


secax = ax1.secondary_xaxis('top', functions=(top_axis_transform, top_axis_inverse))
secax.set_xlabel("Average photon per non-vacuum pulse", fontsize=22)
secax.set_ticks(xticks_second)
secax.tick_params(axis="x", labelsize=22)

plt.show()