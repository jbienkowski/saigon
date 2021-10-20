import logging
import json

from core.hd5_reader import Hd5Reader
from core.plotter import Plotter
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

    hdf = Hd5Reader("data/LEN-DB-processed.hdf5")
    (x_train, y_train, x_test, y_test) = hdf.get_data(0, 10000, 8000)
    # x_train *= 1000000
    # x_test *= 1000000
    # m2 = ModelTwo(
    #     cfg=cfg, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    # )
    # m2.run()

    # hdf.get_subset_of_processed_data(10000)
    # eq_obc = hdf.get_random_object("EQ")
    # p = Plotter()
    # p.plot_all(eq_obc)
    # p.plot_time_series(noise_obj)

    logger.info("Au revoir!")
