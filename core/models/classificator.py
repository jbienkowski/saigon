import os
import h5py
from keras import callbacks
import numpy as np

from time import strftime
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import layers


# Define the constants
MODEL_NAME = "disc_solo"
LOG_DIR = 'out/tensorboard_logs/'
FS = 100
NPERSEG = 155
SAMPLES = 6000
STFT_SIZE = 78


def get_tensorboard(model_name):

    folder_name = f'{model_name} at {strftime("%H %M")}'
    dir_paths = os.path.join(LOG_DIR, folder_name)

    try:
        os.makedirs(dir_paths)
    except OSError as err:
        print(err.strerror)
    else:
        print('Directory created')

    return TensorBoard(log_dir=dir_paths, histogram_freq=1)


def make_discriminator_model():
    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), padding="same", input_shape=[STFT_SIZE, STFT_SIZE, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def load_samples(n_samples):
    with h5py.File("data/stead_test_100_hz.hdf5", "r") as f:
        keys = f["keys"][:n_samples]
        labels = f["labels"][:n_samples]
        data = f["data"][:n_samples]
        return data, keys, labels

def generate_validation_samples(n_samples):
    dataset, keys, labels = load_samples(n_samples)
    return dataset, keys, np.expand_dims(labels, axis=1)


X_VALID, KEYS_VALID, Y_VALID = generate_validation_samples(15000)
d_model = make_discriminator_model()

d_model.fit(
    x=X_VALID[:10000],
    y=Y_VALID[:10000],
    epochs=10,
    validation_data=(X_VALID[10000:15000], Y_VALID[10000:15000]),
    callbacks=[get_tensorboard(MODEL_NAME)]
)

d_model.save("out/disc_solo")