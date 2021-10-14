import logging
import json
from core.hd5_reader import Hd5Reader
from core.plotter import Plotter

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

    hdf = Hd5Reader("data/LEN-DB.hdf5")
    eq_obc = hdf.get_random_object("EQ")
    p = Plotter()
    p.plot_all(eq_obc)
    # p.plot_time_series(noise_obj)

    logger.info("Au revoir!")
