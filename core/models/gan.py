import os
import logging
import h5py
import tensorflow as tf

from tensorflow.keras import layers
from time import strftime
from scipy.signal import istft

from core.gan_plotter import GANPlotter

# SCALING_FACTOR = 1


class GAN(tf.keras.Model):
    MODEL_NAME = "GAN-EVENTS"
    BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    EPOCHS = 100
    LATENT_DIM = 100
    FOLDER_NAME = f"{MODEL_NAME} at {strftime('%H:%M')}"
    LOG_DIR = os.path.join("log/", FOLDER_NAME)

    def __init__(self, cfg):
        super(GAN, self).__init__()
        self._cfg = cfg
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        # Function to calculate cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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
        noise = tf.random.normal([self.BATCH_SIZE, self.LATENT_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        # Update metrics
        self.d_loss_metric.update_state(disc_loss)
        self.g_loss_metric.update_state(gen_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def get_stft_data(self, file_path, arr_length):
        with h5py.File(file_path, "r") as f:
            keys = f["keys"][:arr_length]
            components = f["components"][:arr_length]
            data = f["data"][:arr_length]
            return (keys, components, data)

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Dense(3 * 3 * 128, use_bias=False, input_shape=(self.LATENT_DIM,))
        )
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.2))

        model.add(layers.Reshape((3, 3, 128)))

        model.add(
            layers.Conv2DTranspose(
                64, (10, 10), strides=(2, 2), padding="same", use_bias=False
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.2))

        model.add(
            layers.Conv2DTranspose(
                64, (10, 10), strides=(13, 13), padding="same", use_bias=False
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.2))

        model.add(
            layers.Conv2DTranspose(
                1,
                (10, 10),
                strides=(1, 1),
                padding="same",
                use_bias=False,
                activation="linear",
            )
        )

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()

        model.add(
            layers.Conv2D(
                156, (5, 5), strides=(2, 2), padding="same", input_shape=[78, 78, 1]
            )
        )
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(312, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation="sigmoid"))

        return model

    def run(self):
        try:
            os.makedirs(self.LOG_DIR)
        except OSError as exception:
            print(exception.strerror)
        else:
            print("Successfully created dirs!")

        (keys, components, x_train) = self.get_stft_data(
            self._cfg["stead_path_db_processed_stft"], 100000
        )

        x_train = x_train.reshape(x_train.shape[0], 78, 78, 1)

        # Determine the scaling factor based on dataset
        # global SCALING_FACTOR
        # if SCALING_FACTOR == 0:
        #     SCALING_FACTOR = int(
        #         max(
        #             [
        #                 abs(min([x.min() for x in x_train])),
        #                 abs(max([x.max() for x in x_train])),
        #             ]
        #         )
        #     )

        # x_train /= SCALING_FACTOR

        train_dataset = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(self.BUFFER_SIZE)
            .batch(self.BATCH_SIZE)
        )

        self.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss_fn=tf.keras.losses.BinaryCrossentropy(),
        )

        # self.callback.set_model(self)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.LOG_DIR, histogram_freq=1
        )

        self.fit(
            train_dataset,
            epochs=self.EPOCHS,
            callbacks=[
                tensorboard_callback,
                GANMonitor(self.LATENT_DIM),
            ],
        )

        # Save a final models
        self.generator.save("out/gen")
        self.discriminator.save("out/disc")


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.gp = GANPlotter()

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Finished epoch number {epoch}!")

        # Save a snapshot of both models each XX epochs
        if epoch % 25 == 0:
            self.model.generator.save(f"out/gen-epoch-{epoch}")
            self.model.discriminator.save(f"out/disc-epoch-{epoch}")

        random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
        generated = self.model.generator(random_latent_vectors)

        for i in range(generated.shape[0]):
            inversed = istft(
                generated[i, :, :, 0][:6000],
                window="hanning",
                fs=100,
                nperseg=155,
            )
            self.gp.plot_single_stream(
                inversed[1][:6000],
                f"GAN Event (epoch {epoch})",
                file_path=f"out/image_at_epoch_{epoch}.png",
            )
