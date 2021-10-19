import os
from time import strftime
from typing import NoReturn
from keras.callbacks import TensorBoard


class ModelBase:
    NR_CLASSES = 2
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    SAMPLES_PER_BATCH = 1000
    MODEL = None

    def __init__(
        self,
        cfg: dict,
        model_name: str,
        x_train: list,
        y_train: list,
        x_test: list,
        y_test: list,
    ):
        self._cfg = cfg
        self.model_name = model_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

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
            callbacks=[self.get_tensorboard()],
            verbose=0,
            validation_data=(self.x_test, self.y_test),
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
        folder_name = f'{self.model_name} at {strftime("%H %M")}'
        dir_paths = os.path.join(self._cfg["tf_log_path"], folder_name)

        try:
            os.makedirs(dir_paths)
        except OSError as err:
            print(err.strerror)
        else:
            print("Successfully created directory")

        return TensorBoard(log_dir=dir_paths)
