import json

import logging
from logging.handlers import RotatingFileHandler

from core.stead_reader import SteadReader
from core.stead_plotter import SteadPlotter

from core.gan_plotter import GANPlotter

from core.model_tester import ModelTester

from scipy.signal import resample

logging.basicConfig(
    handlers=[RotatingFileHandler("out/saigon.log", maxBytes=10000000, backupCount=10), logging.StreamHandler()],
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# from core.models import experiment_1
# from core.models import classificator


def plot_examples(cfg, quantity):
    reader = SteadReader(cfg)
    plotter = SteadPlotter()
    stead_data = reader.get_event_data(0, quantity)
    gp = GANPlotter()

    gp.plot_single_stream(
        stead_data[0].get_component("e"),
        f"{stead_data[0].trace_name} (E)",
        fs=100,
        nperseg=155,
        file_path=f"out/{stead_data[0].trace_name}-100Hz.pdf",
    )
    gp.plot_single_stream(
        resample(stead_data[0].get_component("e"), 4000),
        f"{stead_data[0].trace_name} (E)",
        fs=66,
        nperseg=127,
        file_path=f"out/{stead_data[0].trace_name}-66Hz.pdf",
    )

    # for item in stead_data:
    #     plotter.plot_all(item)


if __name__ == "__main__":
    """Script entry point"""
    logging.info("Bonjour!")

    cfg = None
    with open("./config.json", "r") as f:
        cfg = json.load(f)

    # mt = ModelTester(cfg)
    sr = SteadReader(cfg)
    sr.prepare_datasets_case_one()
    # sr.prepare_datasets_case_two()

    # m = GAN(cfg)
    # m.run()

    logging.info("Au revoir!")
