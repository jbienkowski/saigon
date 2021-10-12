import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Hd5Reader:
    def __init__(self, path):
        self._path = path

    def plot_an(self, id):
        with h5py.File(self._path, "r") as f:
            obj = f.get(id)
            for at in obj.attrs:
                print(f"{at}: {obj.attrs[at]}")

            data = np.array(obj)

            d0 = pd.DataFrame(data=data[0])
            d1 = pd.DataFrame(data=data[1])
            d2 = pd.DataFrame(data=data[2])

            _, axes = plt.subplots(3, 1)

            sns.lineplot(data = d0, ax = axes[0])
            sns.lineplot(data = d1, ax = axes[1])
            sns.lineplot(data = d2, ax = axes[2])

            axes[0].set_title("Z")
            axes[1].set_title("N")
            axes[2].set_title("E")
            plt.suptitle(id)
            return
