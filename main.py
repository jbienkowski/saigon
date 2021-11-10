import json

import logging
from logging.handlers import RotatingFileHandler

from core.stead_reader import SteadReader
from core.stead_plotter import SteadPlotter
from core.models.gan import GAN

logging.basicConfig(
    handlers=[RotatingFileHandler("log/converter.log", maxBytes=10000000, backupCount=10)],
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

def plot_examples(quantity):
    reader = SteadReader(cfg)
    plotter = SteadPlotter()
    stead_data = reader.get_event_data(0, quantity)

    for item in stead_data:
        plotter.plot_all(item)

def prepare_stft_data():
    reader = SteadReader(cfg)
    reader.prepare_stft_data()


if __name__ == "__main__":
    """Script entry point"""
    logging.info("Bonjour!")

    cfg = None
    with open("./config.json", "r") as f:
        cfg = json.load(f)

    m = GAN(cfg)
    m.run()

    logging.info("Au revoir!")
