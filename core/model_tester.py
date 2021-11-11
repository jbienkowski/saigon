import tensorflow as tf

from scipy.signal import istft

from core.gan_plotter import GANPlotter

NOISE_DIM = [1, 100]
FS = 100
NPERSEG = 155
SAMPLE_LENGTH = 6000

class ModelTester:
    def __init__(self, cfg):
        self._cfg = cfg
        self.gp = GANPlotter()
        self.discriminator = tf.keras.models.load_model("out/disc")
        self.generator = tf.keras.models.load_model("out/gen")

    def generate_synth(self):
        noise = tf.random.normal(NOISE_DIM)
        synth = self.generator(noise, training=False)
        return synth

    def plot_synth(self, synth):
        inversed = istft(
            synth[0, :, :, 0],
            window="hanning",
            fs=FS,
            nperseg=NPERSEG,
        )
        self.gp.plot_single_stream(inversed[1][:SAMPLE_LENGTH], f"GAN Event")

    def synth_accuracy(self, sample):
        acc = self.discriminator(sample)
        return acc