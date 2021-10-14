import os
from time import strftime
from keras.callbacks import TensorBoard


class TfHelper:
    def __init__(self, config):
        self._cfg = config

    def get_tensorboard(self, model_name):
        folder_name = f'{model_name} at {strftime("%H %M")}'
        dir_paths = os.path.join(self._cfg["tf_log_path"], folder_name)

        try:
            os.makedirs(dir_paths)
        except OSError as err:
            print(err.strerror)
        else:
            print("Successfully created directory")

        return TensorBoard(log_dir=dir_paths)
