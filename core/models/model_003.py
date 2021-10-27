import numpy as np
from .base.model_base import ModelBase
from keras.models import Model
from keras import layers
from core.stead_reader import SteadReader
from scipy.signal import stft

class Model003:
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    def __init__(self, cfg):
        self._cfg = cfg
        self._prepare_data()
        self._define_model()

    def _prepare_data(self):
        sr = SteadReader(self._cfg)
        (x_1, y_1, x_2, y_2) = sr.get_processed_data(10000, 20000, 18000)
        self.x_train, self.y_train = self._build_spectrograms(x_1, y_1)
        self.x_test, self.y_test = self._build_spectrograms(x_2, y_2)

    def _build_spectrograms(self, x, y):
        x_train = []
        y_train = []
        
        for idx, triplet in enumerate(x):
            for stream in triplet:
                _, _, zxx = stft(stream, window='hanning', nperseg=155)
                x_train.append(np.abs(zxx))
                y_train.append(y[idx])

        return (x_train, y_train)


    def _define_model(self):
        pass
