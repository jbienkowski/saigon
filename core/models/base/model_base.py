import os
from time import strftime
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard


class ModelBase:
    def __init__(self, cfg: dict, model_name: str):
        self._cfg = cfg
        self._model_name = model_name
        self.samples_per_batch = 1000
        self.epochs = 100
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.model = None

    @property
    def total_inputs(self):
        components = self._cfg["component_count"]
        samples = self._cfg["component_samples_count"]
        return components * samples

    def summarize_model(self):
        return self.model.summary()

    def fit_model(self):
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.samples_per_batch,
            epochs=self.epochs,
            callbacks=[self.get_tensorboard(self._model_name)],
            verbose=0,
            validation_data=(self.x_val, self.y_val),
        )

    def get_confusion_matrix(self):
        pass

    def get_recall(self):
        pass

    def get_precision(self):
        pass

    def get_avg_presicion(self):
        pass

    def get_f1_score(self):
        pass

    def get_tensorboard(self):
        folder_name = f'{self._model_name} at {strftime("%H %M")}'
        dir_paths = os.path.join(self._cfg["tf_log_path"], folder_name)

        try:
            os.makedirs(dir_paths)
        except OSError as err:
            print(err.strerror)
        else:
            print("Successfully created directory")

        return TensorBoard(log_dir=dir_paths)
