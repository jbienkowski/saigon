import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .entities import SteadDataObject
from scipy.signal import spectrogram, stft, istft


class SteadPlotter:
    def __init__(self):
        pass

    def plot_all(self, do: SteadDataObject):
        timespan = do.get_timespan()
        comp_e = do.get_component("e")
        comp_n = do.get_component("n")
        comp_z = do.get_component("z")

        d0 = pd.DataFrame(data=comp_e, index=timespan)
        d1 = pd.DataFrame(data=comp_n, index=timespan)
        d2 = pd.DataFrame(data=comp_z, index=timespan)

        _, axes = plt.subplots(3, 2, figsize=(14, 6))
        plt.subplots_adjust(hspace=0.80)

        sns.lineplot(data=d0, ax=axes[0, 0], linewidth=1, legend=None)
        sns.lineplot(data=d1, ax=axes[1, 0], linewidth=1, legend=None)
        sns.lineplot(data=d2, ax=axes[2, 0], linewidth=1, legend=None)

        axes[0, 0].set_title("Horizontal (E) component")
        axes[0, 0].set(xlabel="Time (UTC)", ylabel="Amplitude counts")
        axes[0, 0].locator_params(nbins=6, axis="y")

        axes[1, 0].set_title("Horizontal (N) component")
        axes[1, 0].set(xlabel="Time (UTC)", ylabel="Amplitude counts")
        axes[1, 0].locator_params(nbins=6, axis="y")

        axes[2, 0].set_title("Vertical (Z) component")
        axes[2, 0].set(xlabel="Time (UTC)", ylabel="Amplitude counts")
        axes[2, 0].locator_params(nbins=6, axis="y")

        s0, f0, t0, _ = plt.specgram(x=comp_e, Fs=100)
        s1, f1, t1, _ = plt.specgram(x=comp_n, Fs=100)
        s2, f2, t2, _ = plt.specgram(x=comp_z, Fs=100)

        axes[0, 1].clear()
        axes[0, 1].set_title("Horizontal (E) component")
        axes[0, 1].pcolormesh(t0, f0, s0, shading="gouraud")
        axes[0, 1].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        axes[1, 1].clear()
        axes[1, 1].set_title("Horizontal (N) component")
        axes[1, 1].pcolormesh(t1, f1, s1, shading="gouraud")
        axes[1, 1].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        axes[2, 1].clear()
        axes[2, 1].set_title("Vertical (Z) component")
        axes[2, 1].pcolormesh(t2, f2, s2, shading="gouraud")
        axes[2, 1].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        plt.subplots_adjust(top=0.88)

        type = "Earthquake" if do.trace_category == "earthquake_local" else "Noise"
        plt.suptitle(f"{type}: {do.trace_name} at {do.get_ts_short()}", fontsize=14)

        plt.savefig(f"out/{do.trace_name}.png")

    def plot_time_series(self, do: SteadDataObject):
        timespan = do.get_timespan()
        d0 = pd.DataFrame(data=do.data[0], index=timespan)
        d1 = pd.DataFrame(data=do.data[1], index=timespan)
        d2 = pd.DataFrame(data=do.data[2], index=timespan)

        _, axes = plt.subplots(3, 1, figsize=(7, 6))
        plt.subplots_adjust(hspace=0.80)

        sns.lineplot(data=d0, ax=axes[0], linewidth=1, legend=None)
        sns.lineplot(data=d1, ax=axes[1], linewidth=1, legend=None)
        sns.lineplot(data=d2, ax=axes[2], linewidth=1, legend=None)

        axes[0].set_title("Vertical (Z) component")
        axes[0].set(xlabel="Time (UTC)", ylabel="Velocity [$km \cdot s^{-1}$]")
        axes[0].locator_params(nbins=6, axis="y")

        axes[1].set_title("Horizontal (N) component")
        axes[1].set(xlabel="Time (UTC)", ylabel="Velocity [$km \cdot s^{-1}$]")
        axes[1].locator_params(nbins=6, axis="y")

        axes[2].set_title("Horizontal (E) component")
        axes[2].set(xlabel="Time (UTC)", ylabel="Velocity [$km \cdot s^{-1}$]")
        axes[2].locator_params(nbins=6, axis="y")

        plt.subplots_adjust(top=0.88)

        type = "Earthquake" if do.type == "EQ" else "Noise"
        plt.suptitle(f"{type}: {do.net}.{do.sta} at {do.get_ts_short()}", fontsize=14)

        plt.savefig(f"out/{do.type}-{do.id}-series.png")

    def plot_spectrogram(self, do: SteadDataObject):
        s0, f0, t0, _ = plt.specgram(x=do.data[0], Fs=20)
        s1, f1, t1, _ = plt.specgram(x=do.data[1], Fs=20)
        s2, f2, t2, _ = plt.specgram(x=do.data[2], Fs=20)

        _, axes = plt.subplots(3, 1, figsize=(7, 6))
        plt.subplots_adjust(hspace=0.80)

        axes[0].set_title("Vertical (Z) component")
        axes[0].pcolormesh(t0, f0, s0, shading="gouraud")
        axes[0].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        axes[2].set_title("Horizontal (N) component")
        axes[1].pcolormesh(t1, f1, s1, shading="gouraud")
        axes[1].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        axes[2].set_title("Horizontal (E) component")
        axes[2].pcolormesh(t2, f2, s2, shading="gouraud")
        axes[2].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        type = "Earthquake" if do.type == "EQ" else "Noise"
        plt.suptitle(f"{type}: {do.net}.{do.sta} at {do.get_ts_short()}", fontsize=14)

        plt.savefig(f"out/{do.type}-{do.id}-spectrogram.png")

    def plot_single_stream(self, do, label, fs=100, nperseg=155, file_path=None):
        d0 = pd.DataFrame(data=do)

        fig = plt.figure(figsize=(16, 16), dpi=80)
        ax1 = plt.subplot2grid((4, 1), (0, 0))
        ax2 = plt.subplot2grid((4, 1), (1, 0))
        ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)

        plt.subplots_adjust(hspace=0.5)

        sns.lineplot(data=do, ax=ax1, linewidth=1, legend=None)

        ax1.set_title("Waveform")
        ax1.set(xlabel="Samples", ylabel="Amplitude counts")
        ax1.locator_params(nbins=6, axis="y")

        f, t, Sxx = spectrogram(x=do, fs=fs)

        ax2.clear()
        ax2.set_title("Spectrogram")
        ax2.pcolormesh(t, f, Sxx, shading="gouraud")
        ax2.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        f_sftt, t_sftt, Zxx = stft(do, window="hanning", fs=fs, nperseg=nperseg)

        ax3.clear()
        ax3.set_title("STFT")
        ax3.pcolormesh(t_sftt, f_sftt, np.abs(Zxx), shading="auto")

        plt.suptitle(label, fontsize=14)

        if file_path != None:
            plt.savefig(file_path)