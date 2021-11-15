import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import spectrogram, stft
import numpy as np


class GANPlotter:
    def plot_all(self, do, label, fs=100, nperseg=155, file_path=None):
        d0 = pd.DataFrame(data=do[0])
        d1 = pd.DataFrame(data=do[1])
        d2 = pd.DataFrame(data=do[2])

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

        ax1.set_title("Vertical component waveform")
        ax1.set(xlabel="Samples", ylabel="Amplitude counts")
        ax1.locator_params(nbins=6, axis="y")

        ax2.set_title("North component waveform")
        ax2.set(xlabel="Samples", ylabel="Amplitude counts")
        ax2.locator_params(nbins=6, axis="y")

        ax3.set_title("East component waveform")
        ax3.set(xlabel="Samples", ylabel="Amplitude counts")
        ax3.locator_params(nbins=6, axis="y")

        f_0, t_0, Sxx_0 = spectrogram(x=do[0], fs=fs)
        f_1, t_1, Sxx_1 = spectrogram(x=do[1], fs=fs)
        f_2, t_2, Sxx_2 = spectrogram(x=do[2], fs=fs)

        ax4.clear()
        ax4.set_title("Vertical component spectrogram")
        ax4.pcolormesh(t_0, f_0, Sxx_0, shading="gouraud")
        ax4.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        ax5.clear()
        ax5.set_title("North component spectrogram")
        ax5.pcolormesh(t_1, f_1, Sxx_1, shading="gouraud")
        ax5.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        ax6.clear()
        ax6.set_title("East component spectrogram")
        ax6.pcolormesh(t_2, f_2, Sxx_2, shading="gouraud")
        ax6.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        f_sftt_0, t_sftt_0, Zxx_0 = stft(
            do[0], window="hanning", fs=fs, nperseg=nperseg
        )
        f_sftt_1, t_sftt_1, Zxx_1 = stft(
            do[1], window="hanning", fs=fs, nperseg=nperseg
        )
        f_sftt_2, t_sftt_2, Zxx_2 = stft(
            do[2], window="hanning", fs=fs, nperseg=nperseg
        )

        ticks = None

        if fs == 100:
            ticks = np.arange(78)
        else:
            ticks = np.arange(64)

        ax7.clear()
        ax7.set_title("Vertical component STFT")
        ax7.pcolormesh(ticks, ticks, np.abs(Zxx_0), shading="auto")

        ax8.clear()
        ax8.set_title("North component STFT")
        ax8.pcolormesh(ticks, ticks, np.abs(Zxx_1), shading="auto")

        ax9.clear()
        ax9.set_title("East component STFT")
        ax9.pcolormesh(ticks, ticks, np.abs(Zxx_2), shading="auto")

        plt.suptitle(label, fontsize=14)

        if file_path != None:
            plt.savefig(file_path)
            plt.close(fig)

    def plot_single_stream(self, do, label, fs=100, nperseg=155, file_path=None):
        d0 = pd.DataFrame(data=do)

        fig = plt.figure(figsize=(16, 16), dpi=80)
        ax1 = plt.subplot2grid((6, 1), (0, 0))
        ax2 = plt.subplot2grid((6, 1), (1, 0))
        ax3 = plt.subplot2grid((6, 1), (2, 0), rowspan=4)

        plt.subplots_adjust(hspace=0.5)

        sns.lineplot(data=do, ax=ax1, linewidth=1, legend=None)

        ax1.set_title("Waveform")
        ax1.set(xlabel="Samples", ylabel="Amplitude counts")
        ax1.locator_params(nbins=6, axis="y")

        f, t, Sxx = spectrogram(x=do, fs=fs)

        ax2.clear()
        ax2.set_title("Spectrogram")
        _ax2 = ax2.pcolormesh(t, f, Sxx, shading="gouraud")
        ax2.set(xlabel="Time [sec]", ylabel="Frequency [Hz]")
        fig.colorbar(_ax2, ax=ax2)

        f_sftt, t_sftt, Zxx = stft(do, window="hanning", fs=fs, nperseg=nperseg)

        ticks = None

        if fs == 100:
            ticks = np.arange(78)
        else:
            ticks = np.arange(64)

        ax3.clear()
        ax3.set_title("STFT")
        _ax3 = ax3.pcolormesh(ticks, ticks, np.abs(Zxx), shading="auto")
        fig.colorbar(_ax3, ax=ax3)

        plt.suptitle(label, fontsize=14)

        if file_path != None:
            plt.savefig(file_path)
            plt.close(fig)

    def plot_stft(self, stream, fs=100, nperseg=155):
        f, t, Zxx = stft(stream, window="hanning", fs=fs, nperseg=nperseg)
        # plt.specgram(x_1[0][0], cmap='plasma', Fs=100)
        plt.pcolormesh(t, f, np.abs(Zxx), shading="auto")
