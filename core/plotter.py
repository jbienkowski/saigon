import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from .models import DataObject
from scipy import signal


class Plotter:
    def __init__(self):
        pass

    def plot_time_series(self, do: DataObject):
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

    def plot_spectrogram(self, do: DataObject):        
        f0, t0, Sxx0 = signal.spectrogram(do.data[0])
        f1, t1, Sxx1 = signal.spectrogram(do.data[1])
        f2, t2, Sxx2 = signal.spectrogram(do.data[2])

        _, axes = plt.subplots(3, 1, figsize=(7, 6))
        plt.subplots_adjust(hspace=0.80)

        axes[0].set_title("Vertical (Z) component")
        axes[0].pcolormesh(t0, f0, Sxx0, shading='gouraud')
        axes[0].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        axes[2].set_title("Horizontal (N) component")
        axes[1].pcolormesh(t1, f1, Sxx1, shading='gouraud')
        axes[1].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        axes[2].set_title("Horizontal (E) component")
        axes[2].pcolormesh(t2, f2, Sxx2, shading='gouraud')
        axes[2].set(xlabel="Time [sec]", ylabel="Frequency [Hz]")

        type = "Earthquake" if do.type == "EQ" else "Noise"
        plt.suptitle(f"{type}: {do.net}.{do.sta} at {do.get_ts_short()}", fontsize=14)

        plt.savefig(f"out/{do.type}-{do.id}-spectrogram.png")
