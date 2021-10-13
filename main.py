import logging
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

    hdf = Hd5Reader("data/LEN-DB.hdf5")
    noise_obj = hdf.find_dataobject("AN/XV_FTGH_1515527757.9699998")
    eq_obc = hdf.find_dataobject("EQ/AF_WDLM_1431223231.095")
    p = Plotter()
    p.plot_spectrogram(noise_obj)
    p.plot_time_series(noise_obj)

    logger.info("Au revoir!")
