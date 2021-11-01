import logging
import json
import numpy as np

from core.len_reader import LenReader
from core.stead_reader import SteadReader

from core.len_plotter import LenPlotter
from core.stead_plotter import SteadPlotter

import core.models.model_004

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Script entry point"""
    logger.info("Bonjour!")

    cfg = None
    with open("./config.json", "r") as f:
        cfg = json.load(f)

    # m3 = Model003(cfg)

    stead = SteadReader(cfg)
    # stead_plotter = SteadPlotter()

    stead.prepare_gan_data()

    # stead_data = stead.get_event_data(10,11)
    # stead_plotter.plot_all(stead_data[0])

    logger.info("Au revoir!")
