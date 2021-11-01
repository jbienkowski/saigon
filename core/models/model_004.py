import os
import h5py
import tensorflow as tf
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns

from IPython import display
from tensorflow.keras import layers
from time import strftime
from scipy.signal import spectrogram, stft, istft

MODEL_NAME = "GanPlayground"
BUFFER_SIZE = 60000
BATCH_SIZE = 256


def plot_all(do, label, fs=100, nperseg=150):
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

    f_sftt_0, t_sftt_0, Zxx_0 = stft(do[0], window="hanning", fs=fs, nperseg=nperseg)
    f_sftt_1, t_sftt_1, Zxx_1 = stft(do[1], window="hanning", fs=fs, nperseg=nperseg)
    f_sftt_2, t_sftt_2, Zxx_2 = stft(do[2], window="hanning", fs=fs, nperseg=nperseg)

    ax7.clear()
    ax7.set_title("Vertical component STFT")
    ax7.pcolormesh(t_sftt_0, f_sftt_0, np.abs(Zxx_0), shading="auto")

    ax8.clear()
    ax8.set_title("North component STFT")
    ax8.pcolormesh(t_sftt_1, f_sftt_1, np.abs(Zxx_1), shading="auto")

    ax9.clear()
    ax9.set_title("East component STFT")
    ax9.pcolormesh(t_sftt_2, f_sftt_2, np.abs(Zxx_2), shading="auto")

    plt.suptitle(label, fontsize=14)


def plot_single_stream(do, label, fs=100, nperseg=150):
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


def plot_stft(stream, fs=100, nperseg=155):
    f, t, Zxx = stft(stream, window="hanning", fs=fs, nperseg=nperseg)
    # plt.specgram(x_1[0][0], cmap='plasma', Fs=100)
    plt.pcolormesh(t, f, np.abs(Zxx), shading="auto")


def get_data(file_path, idx_start, idx_end, idx_slice):
    x_train = None
    #     y_train = None
    evi_train = None
    x_test = None
    #     y_test = None
    evi_test = None
    with h5py.File(file_path, "r") as f:
        x_train = f["data"][idx_start:idx_slice]
        #         y_train = f["labels"][idx_start:idx_slice]
        evi_train = f["keys"][idx_start:idx_slice]
        x_test = f["data"][idx_slice:idx_end]
        #         y_test = f["labels"][idx_slice:idx_end]
        evi_test = f["keys"][idx_slice:idx_end]
        #         return (x_train, y_train, evi_train, x_test, y_test, evi_test)
        return (x_train, evi_train, x_test, evi_test)


def build_stfts(x):
    x_train = []

    for idx, triplet in enumerate(x):
        for stream in triplet:
            _, _, zxx = stft(stream, window="hanning", nperseg=155)
            x_train.append(np.abs(zxx))

    return np.array(x_train)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(3 * 3 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((3, 3, 1024)))

    model.add(
        layers.Conv2DTranspose(
            512, (10, 10), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            256, (10, 10), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            128, (10, 10), strides=(13, 13), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            1,
            (10, 10),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="tanh",
        )
    )

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[78, 78, 1]
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.plot(predictions[i, :, :, 0][:6000])
        plt.axis("off")

    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


folder_name = f"{MODEL_NAME} at {strftime('%H:%M')}"
log_dir = os.path.join("log/", folder_name)

try:
    os.makedirs(log_dir)
except OSError as exception:
    print(exception.strerror)
else:
    print("Successfully created dirs!")

# (x_1, y_1, evi_1, x_2, y_2, evi_2) = get_data("../data/STEAD-processed-gan.hdf5", 10000, 20000, 18000)
(x_1, evi_1, x_2, evi_2) = get_data("data/STEAD-processed-gan.hdf5", 1000, 2000, 1800)

x_train = build_stfts(x_1)
x_test = build_stfts(x_2)

x_train = x_train.reshape(x_train.shape[0], 78, 78, 1).astype("float32")
x_test = x_test.reshape(x_test.shape[0], 78, 78, 1).astype("float32")

train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

generator = make_generator_model()
discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)
