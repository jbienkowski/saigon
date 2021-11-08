import logging
import json

from core.stead_reader import SteadReader
from core.stead_plotter import SteadPlotter
from core.models.gan import GAN

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

    m = GAN(cfg)
    m.run()

    reader = SteadReader(cfg)
    plotter = SteadPlotter()

    # stead_data = reader.get_event_data(10,11)
    # plotter.plot_all(stead_data[0])

    logger.info("Au revoir!")
