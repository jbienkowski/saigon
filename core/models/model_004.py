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


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.gp = GANPlotter()

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated = self.model.generator(random_latent_vectors)
        for i in range(generated.shape[0]):
            inversed_z = istft(
                generated[i, :, :, 0][:6000], window="hanning", fs=100, nperseg=155
            )
            inversed_n = istft(
                generated[i, :, :, 1][:6000], window="hanning", fs=100, nperseg=155
            )
            inversed_e = istft(
                generated[i, :, :, 2][:6000], window="hanning", fs=100, nperseg=155
            )
            self.gp.plot_all(
                [inversed_z[1][:6000], inversed_n[1][:6000], inversed_e[1][:6000]],
                f"GAN Event (epoch {epoch})",
                file_path=f"image_at_epoch_{epoch}.png",
            )


class GANPlotter:
    def plot_all(self, do, label, fs=100, nperseg=150, file_path=None):
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

        f_sftt_0, t_sftt_0, Zxx_0 = stft(
            do[0], window="hanning", fs=fs, nperseg=nperseg
        )
        f_sftt_1, t_sftt_1, Zxx_1 = stft(
            do[1], window="hanning", fs=fs, nperseg=nperseg
        )
        f_sftt_2, t_sftt_2, Zxx_2 = stft(
            do[2], window="hanning", fs=fs, nperseg=nperseg
        )

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

        if file_path != None:
            plt.savefig(file_path)

    def plot_single_stream(self, do, label, fs=100, nperseg=150, file_path=None):
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

        if file_path != None:
            plt.savefig(file_path)

    def plot_stft(self, stream, fs=100, nperseg=155):
        f, t, Zxx = stft(stream, window="hanning", fs=fs, nperseg=nperseg)
        # plt.specgram(x_1[0][0], cmap='plasma', Fs=100)
        plt.pcolormesh(t, f, np.abs(Zxx), shading="auto")


class Model004:
    MODEL_NAME = "GAN-EVENTS"
    BUFFER_SIZE = 6
    BATCH_SIZE = 3
    EPOCHS = 100
    NOISE_DIM = 100
    NUM_EXAMPLES_TO_GENERATE = 1
    FOLDER_NAME = f"{MODEL_NAME} at {strftime('%H:%M')}"
    LOG_DIR = os.path.join("log/", FOLDER_NAME)

    def __init__(self, cfg):
        self._cfg = cfg
        self.GENERATOR = self.make_generator_model()
        self.DISCRIMINATOR = self.make_discriminator_model()
        # Function to calculate cross entropy loss
        self.CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.SEED = tf.random.normal([self.NUM_EXAMPLES_TO_GENERATE, self.NOISE_DIM])
        self.GENERATOR_OPTIMIZER = tf.keras.optimizers.Adam(1e-4)
        self.DISCRIMINATOR_OPTIMIZER = tf.keras.optimizers.Adam(1e-4)

        self.CHECKPOINT_DIR = "./training_checkpoints"
        self.CHECKPOINT_PREFIX = os.path.join(self.CHECKPOINT_DIR, "ckpt")
        self.CHECKPOINT = tf.train.Checkpoint(
            generator_optimizer=self.GENERATOR_OPTIMIZER,
            discriminator_optimizer=self.DISCRIMINATOR_OPTIMIZER,
            generator=self.GENERATOR,
            discriminator=self.DISCRIMINATOR,
        )

        self.CALLBACK = tf.keras.callbacks.TensorBoard(self.LOG_DIR)
        self.CALLBACK.set_model(self.GENERATOR)

    def get_data(self, file_path, idx_start, idx_end, idx_slice):
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

    def build_stfts_single_components(self, x):
        x_train = []

        for triplet in x:
            for stream in triplet:
                _, _, zxx = stft(stream, window="hanning", nperseg=155)
                x_train.append(np.abs(zxx))

        return np.array(x_train)

    def build_stfts_three_components(self, x):
        x_train = []

        for triplet in x:
            inner = []
            for stream in triplet:
                _, _, zxx = stft(stream, window="hanning", nperseg=155)
                inner.append(np.abs(zxx))
            x_train.append(inner)

        return np.array(x_train)

    def make_generator_model(self):
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
                3,
                (10, 10),
                strides=(1, 1),
                padding="same",
                use_bias=False,
                activation="tanh",
            )
        )

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Conv2D(
                64, (5, 5), strides=(2, 2), padding="same", input_shape=[78, 78, 3]
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

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.CROSS_ENTROPY(tf.ones_like(real_output), real_output)
        fake_loss = self.CROSS_ENTROPY(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.CROSS_ENTROPY(tf.ones_like(fake_output), fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    # @tf.function
    def train_step(self, epoch, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.GENERATOR(noise, training=True)

            real_output = self.DISCRIMINATOR(images, training=True)
            fake_output = self.DISCRIMINATOR(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.GENERATOR.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.DISCRIMINATOR.trainable_variables
        )

        self.GENERATOR_OPTIMIZER.apply_gradients(
            zip(gradients_of_generator, self.GENERATOR.trainable_variables)
        )
        self.DISCRIMINATOR_OPTIMIZER.apply_gradients(
            zip(gradients_of_discriminator, self.DISCRIMINATOR.trainable_variables)
        )

    def train(self, dataset):
        for epoch in range(self.EPOCHS):
            start = time.time()

            for image_batch in dataset:
                self.train_step(epoch, image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.GENERATOR, epoch + 1, self.SEED)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.CHECKPOINT.save(file_prefix=self.checkpoint_prefix)

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.GENERATOR, self.EPOCHS, self.SEED)

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        for i in range(predictions.shape[0]):
            inversed_z = istft(
                predictions[i, :, :, 0][:6000], window="hanning", fs=100, nperseg=155
            )
            inversed_n = istft(
                predictions[i, :, :, 1][:6000], window="hanning", fs=100, nperseg=155
            )
            inversed_e = istft(
                predictions[i, :, :, 2][:6000], window="hanning", fs=100, nperseg=155
            )
            self.plot_all(
                [inversed_z[1][:6000], inversed_n[1][:6000], inversed_e[1][:6000]],
                f"GAN Event (epoch {epoch})",
                file_path=f"image_at_epoch_{epoch}.png",
            )

    def run(self):
        try:
            os.makedirs(self.LOG_DIR)
        except OSError as exception:
            print(exception.strerror)
        else:
            print("Successfully created dirs!")

        # (x_1, y_1, evi_1, x_2, y_2, evi_2) = get_data("../data/STEAD-processed-gan.hdf5", 10000, 20000, 18000)
        (x_1, evi_1, x_2, evi_2) = self.get_data(
            self._cfg["stead_path_db_processed_gan"], 10, 10, 18
        )

        x_train = self.build_stfts_three_components(x_1)
        # x_test = self.build_stfts(x_2)

        x_train = x_train.reshape(x_train.shape[0], 78, 78, 3).astype("float32")
        # x_test = x_test.reshape(x_test.shape[0], 78, 78, 1).astype("float32")

        train_dataset = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(self.BUFFER_SIZE)
            .batch(self.BATCH_SIZE)
        )

        # self.train(train_dataset)

        gan = GAN(
            discriminator=self.DISCRIMINATOR,
            generator=self.GENERATOR,
            latent_dim=self.NOISE_DIM,
        )
        
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss_fn=tf.keras.losses.BinaryCrossentropy(),
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.LOG_DIR, histogram_freq=1
        )

        gan.fit(
            train_dataset,
            epochs=self.EPOCHS,
            callbacks=[
                tensorboard_callback,
                GANMonitor(num_img=1, latent_dim=self.NOISE_DIM),
            ],
        )
