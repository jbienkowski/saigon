import os
from time import strftime
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard


class ModelBase:
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
        self._model_name = model_name
        self.samples_per_batch = 1000
        self.epochs = 100
        self.learning_rate = 1e-3
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None

    @property
    def total_inputs(self):
        components = self._cfg["component_count"]
        samples = self._cfg["component_samples_count"]
        return components * samples

    def summarize_model(self):
        return self.model.summary()

    def setup_layer(self, input, weight_dim, bias_dim, name):
        with tf.name_scope(name):
            initial_w = tf.random.truncated_normal(shape=weight_dim, stddev=0.1, seed=42)
            w = tf.Variable(initial_value=initial_w, name='W')

            initial_b = tf.constant(value=0.0, shape=bias_dim)
            b = tf.Variable(initial_value=initial_b, name='B')

            layer_in = tf.matmul(input, w) + b
            
            if name=='out':
                layer_out = tf.nn.softmax(layer_in)
            else:
                layer_out = tf.nn.relu(layer_in)
            
            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
            
            return layer_out

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
        folder_name = f'{self._model_name} at {strftime("%H %M")}'
        dir_paths = os.path.join(self._cfg["tf_log_path"], folder_name)

        try:
            os.makedirs(dir_paths)
        except OSError as err:
            print(err.strerror)
        else:
            print("Successfully created directory")

        return TensorBoard(log_dir=dir_paths)
