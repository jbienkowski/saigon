import json

import logging
from logging.handlers import RotatingFileHandler

from core.stead_reader import SteadReader
from core.stead_plotter import SteadPlotter
from core.models.gan64 import GAN

from core.model_tester import ModelTester

logging.basicConfig(
    handlers=[RotatingFileHandler("out/saigon.log", maxBytes=10000000, backupCount=10)],
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

from core.models import cigan64

def plot_examples(cfg, quantity):
    reader = SteadReader(cfg)
    plotter = SteadPlotter()
    stead_data = reader.get_event_data(0, quantity)

    for item in stead_data:
        plotter.plot_all(item)


if __name__ == "__main__":
    """Script entry point"""
    logging.info("Bonjour!")

    cfg = None
    with open("./config.json", "r") as f:
        cfg = json.load(f)

    # mt = ModelTester(cfg)
    # sr = SteadReader(cfg)

    # m = GAN(cfg)
    # m.run()

    logging.info("Au revoir!")
