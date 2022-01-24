import logging
import h5py

import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras import layers

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from scipy.signal import spectrogram, stft, istft


# Define the constants
FS = 66
NPERSEG = 127
SAMPLES = 4000
STFT_SIZE = 64


def plot_all(do, label, file_path=None):
    d0 = pd.DataFrame(data=do[0][:SAMPLES])
    d1 = pd.DataFrame(data=do[1][:SAMPLES])
    d2 = pd.DataFrame(data=do[2][:SAMPLES])

    plt.rc('font', size=11)
    plt.rc('axes', titlesize=16)

    fig = plt.figure(figsize=(16, 10), dpi=227)
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
    ax1.set(xlabel="Samples", ylabel="Amp. counts")
    ax1.locator_params(nbins=6, axis="y")

    ax2.set_title("North component waveform")
    ax2.set(xlabel="Samples", ylabel="Amp. counts")
    ax2.locator_params(nbins=6, axis="y")

    ax3.set_title("East component waveform")
    ax3.set(xlabel="Samples", ylabel="Amp. counts")
    ax3.locator_params(nbins=6, axis="y")

    f_0, t_0, Sxx_0 = spectrogram(x=do[0], fs=FS)
    f_1, t_1, Sxx_1 = spectrogram(x=do[1], fs=FS)
    f_2, t_2, Sxx_2 = spectrogram(x=do[2], fs=FS)

    ax4.clear()
    ax4.set_title("Vertical component spectrogram")
    _ax4 = ax4.pcolormesh(t_0, f_0, Sxx_0, shading="gouraud")
    ax4.set(xlabel="Time [sec]", ylabel="Freq. [Hz]")
    fig.colorbar(_ax4, ax=ax4)

    ax5.clear()
    ax5.set_title("North component spectrogram")
    _ax5 = ax5.pcolormesh(t_1, f_1, Sxx_1, shading="gouraud")
    ax5.set(xlabel="Time [sec]", ylabel="Freq. [Hz]")
    fig.colorbar(_ax5, ax=ax5)

    ax6.clear()
    ax6.set_title("East component spectrogram")
    _ax6 = ax6.pcolormesh(t_2, f_2, Sxx_2, shading="gouraud")
    ax6.set(xlabel="Time [sec]", ylabel="Freq. [Hz]")
    fig.colorbar(_ax6, ax=ax6)

    f_sftt_0, t_sftt_0, Zxx_0 = stft(do[0], window="hanning", fs=FS, nperseg=NPERSEG)
    f_sftt_1, t_sftt_1, Zxx_1 = stft(do[1], window="hanning", fs=FS, nperseg=NPERSEG)
    f_sftt_2, t_sftt_2, Zxx_2 = stft(do[2], window="hanning", fs=FS, nperseg=NPERSEG)

    ticks = np.arange(STFT_SIZE)

    ax7.clear()
    ax7.set_title("Vertical component STFT")
    _ax7 = ax7.pcolormesh(ticks, ticks, np.abs(Zxx_0), shading="auto")
    fig.colorbar(_ax7, ax=ax7)

    ax8.clear()
    ax8.set_title("North component STFT")
    _ax8 = ax8.pcolormesh(ticks, ticks, np.abs(Zxx_1), shading="auto")
    fig.colorbar(_ax8, ax=ax8)

    ax9.clear()
    ax9.set_title("East component STFT")
    _ax9 = ax9.pcolormesh(ticks, ticks, np.abs(Zxx_2), shading="auto")
    fig.colorbar(_ax9, ax=ax9)

    plt.suptitle(label, fontsize=14)

    if file_path != None:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)


def make_generator_model(latent_dim):
    model = Sequential()
    model.add(layers.Dense(2 * 2 * 256, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((2, 2, 256)))

    model.add(
        layers.Conv2DTranspose(
            32, (20, 20), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(
        layers.Conv2DTranspose(
            32, (4, 4), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(
        layers.Conv2DTranspose(
            32, (4, 4), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(
        layers.Conv2DTranspose(
            32, (4, 4), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(
        layers.Conv2DTranspose(
            32, (4, 4), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(
        layers.Conv2DTranspose(
            3,
            (3, 3),
            padding="same",
            use_bias=False,
            activation="linear",
        )
    )

    return model


def make_discriminator_model():
    model = Sequential()

    model.add(layers.Conv2D(32, (1, 1), padding="same", input_shape=[STFT_SIZE, STFT_SIZE, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model


def load_real_samples(arr_len=10000):
    with h5py.File("data/stead_learn_stft_64.hdf5", "r") as f:
        keys = f["keys"][:arr_len]
        labels = f["labels"][:arr_len]
        data = f["data"][:arr_len]
        return data, keys, labels

def load_validation_samples(n_samples):
    with h5py.File("data/stead_test_stft_64.hdf5", "r") as f:
        keys = f["keys"][:n_samples]
        labels = f["labels"][:n_samples]
        data = f["data"][:n_samples]
        return data, keys, labels

def generate_validation_samples(n_samples):
    dataset, keys, labels = load_validation_samples(n_samples)
    return dataset, keys, np.expand_dims(labels, axis=1)


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # Generate latent space points; we multiply by 2 due to complex number data type
    x_input = randn(2 * latent_dim * n_samples).view(np.complex128)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


# create and save a plot of generated images
def save_plot(examples, epoch):
    """Plot the examples."""
    for idx, _ in enumerate(examples[:5]):
        plot_all(
            istft(examples[idx].transpose(2, 0, 1), fs=FS, nperseg=NPERSEG)[1],
            f"GAN Event (epoch {epoch+1})",
            file_path=f"out/epoch_{epoch+1}_image_{idx}.png",
        )


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # Prepare validation samples
    _, acc_valid = d_model.evaluate(X_VALID, Y_VALID, verbose=0)
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    logging.warning(f">Accuracy real: {acc_real * 100}, fake: {acc_fake * 100}")
    logging.warning(f"Discriminator validation score: {acc_valid * 100}")

    save_plot(x_fake, epoch)

    if (acc_valid > 0.9) or (epoch + 1) % 10 == 0:
        # save the generator model tile file
        g_model.save(f"out/gen-{epoch+1}")
        d_model.save(f"out/desc-{epoch+1}")


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            logging.info(
                f"Epoch {i + 1}, batch {j + 1}/{bat_per_epo}, {d_loss1=}, {d_loss2=}, {g_loss=}"
            )
        # evaluate the model performance, sometimes
        if (i + 1) % 2 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


X_VALID, KEYS_VALID, Y_VALID = generate_validation_samples(1000)
# size of the latent space
latent_dim = 512
# create the discriminator
d_model = make_discriminator_model()
# create the generator
g_model = make_generator_model(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset, _, _ = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
