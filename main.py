import logging
import json
import numpy as np

from core.len_reader import LenReader
from core.stead_reader import SteadReader

from core.len_plotter import LenPlotter
from core.stead_plotter import SteadPlotter

from core.models.model_001 import Model001
from core.models.model_002 import Model002

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

    stead = SteadReader(cfg)
    stead.prepare_data()
    # stead_data = stead.get_event_data(1,10000)

    # stead_plotter = SteadPlotter()
    # stead_plotter.plot_all(stead_data[0])

    # len = LenReader(cfg["len_path"])
    # hdf.prepare_data()
    # (x_train, y_train, x_test, y_test) = hdf.get_data(0, 4000, 3000)
    # x_train *= 1000000
    # x_test *= 1000000
    # m2 = Model001(
    #     cfg=cfg, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    # )
    # m2.fit_model()

    # hdf.get_subset_of_processed_data(10000)
    # eq_obc = hdf.get_random_object("EQ")
    # p = LenPlotter()
    # p.plot_all(eq_obc)
    # p.plot_time_series(noise_obj)

    logger.info("Au revoir!")
