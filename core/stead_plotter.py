import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .entities import SteadDataObject
from scipy.signal import spectrogram, stft, istft


class SteadPlotter:
    def __init__(self):
        pass

    def plot_all(self, do: SteadDataObject, fs=100, nperseg=155):
        timespan = do.get_timespan()
        comp_e = do.get_component("e")
        comp_n = do.get_component("n")
        comp_z = do.get_component("z")

        d0 = pd.DataFrame(data=comp_e, index=timespan)
        d1 = pd.DataFrame(data=comp_n, index=timespan)
        d2 = pd.DataFrame(data=comp_z, index=timespan)

        fig = plt.figure(figsize=(16, 10), dpi=80)
        ax1 = plt.subplot2grid((5, 6), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((5, 6), (1, 0), colspan=3)
        ax3 = plt.subplot2grid((5, 6), (2, 0), colspan=3)
        ax4 = plt.subplot2grid((5, 6), (0, 3), colspan=3)
        ax5 = plt.subplot2grid((5, 6), (1, 3), colspan=3)
        ax6 = plt.subplot2grid((5, 6), (2, 3), colspan=3)
        ax7 = plt.subplot2grid((5, 6), (3, 0), colspan=2, rowspan=2)
        ax8 = plt.subplot2grid((5, 6), (3, 2), colspan=2, rowspan=2)
        ax9 = plt.subplot2grid((5, 6), (3, 4), colspan=2, rowspan=2)

        plt.subplots_adjust(hspace=1, wspace=1)

        sns.lineplot(data=d0, ax=ax1, linewidth=1, legend=None)
        sns.lineplot(data=d1, ax=ax2, linewidth=1, legend=None)
        sns.lineplot(data=d2, ax=ax3, linewidth=1, legend=None)

        ax1.set_title("East component waveform")
        ax1.set(xlabel="Samples", ylabel="Amplitude counts")
        ax1.locator_params(nbins=6, axis="y")

        ax2.set_title("North component waveform")
        ax2.set(xlabel="Samples", ylabel="Amplitude counts")
        ax2.locator_params(nbins=6, axis="y")

        ax3.set_title("Vertical component waveform")
        ax3.set(xlabel="Samples", ylabel="Amplitude counts")
        ax3.locator_params(nbins=6, axis="y")

        f_0, t_0, Sxx_0 = spectrogram(x=comp_e, fs=fs)
        f_1, t_1, Sxx_1 = spectrogram(x=comp_n, fs=fs)
        f_2, t_2, Sxx_2 = spectrogram(x=comp_z, fs=fs)

        ax4.clear()
        ax4.set_title("East component spectrogram")
        _ax4 = ax4.pcolormesh(t_0, f_0, Sxx_0, shading="gouraud")
        ax4.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")
        fig.colorbar(_ax4, ax=ax4)

        ax5.clear()
        ax5.set_title("North component spectrogram")
        _ax5 = ax5.pcolormesh(t_1, f_1, Sxx_1, shading="gouraud")
        ax5.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")
        fig.colorbar(_ax5, ax=ax5)

        ax6.clear()
        ax6.set_title("Vertical component spectrogram")
        _ax6 = ax6.pcolormesh(t_2, f_2, Sxx_2, shading="gouraud")
        ax6.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")
        fig.colorbar(_ax6, ax=ax6)

        f_sftt_0, t_sftt_0, Zxx_0 = stft(
            comp_e, window="hanning", fs=fs, nperseg=nperseg
        )
        f_sftt_1, t_sftt_1, Zxx_1 = stft(
            comp_n, window="hanning", fs=fs, nperseg=nperseg
        )
        f_sftt_2, t_sftt_2, Zxx_2 = stft(
            comp_z, window="hanning", fs=fs, nperseg=nperseg
        )

        ax7.clear()
        ax7.set_title("East component STFT")
        _ax7 = ax7.pcolormesh(t_sftt_0, f_sftt_0, np.abs(Zxx_0), shading="auto")
        fig.colorbar(_ax7, ax=ax7)

        ax8.clear()
        ax8.set_title("North component STFT")
        _ax8 = ax8.pcolormesh(t_sftt_1, f_sftt_1, np.abs(Zxx_1), shading="auto")
        fig.colorbar(_ax8, ax=ax8)

        ax9.clear()
        ax9.set_title("Vertical component STFT")
        _ax9 = ax9.pcolormesh(t_sftt_2, f_sftt_2, np.abs(Zxx_2), shading="auto")
        fig.colorbar(_ax9, ax=ax9)

        type = "Earthquake" if do.trace_category == "earthquake_local" else "Noise"
        plt.suptitle(f"{type}: {do.trace_name} at {do.get_ts_short()}", fontsize=14)

        plt.savefig(f"out/{do.trace_name}.png")
        plt.close(fig)

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