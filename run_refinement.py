import sys

if sys.version_info.major < 3:
    print("Please use Python 3.x, your current version is: " + str(sys.version))
    sys.exit(1)

import argparse
import logging
import os

from config import logging_config
from config import GlobalConfig as GlobalConfig
from refiner.models.image_data import ImageData
from refiner.refinement_handler import start_refinement_procedure

logging.basicConfig(**logging_config)
logger = logging.getLogger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--config', type=str)
    return parser.parse_args()


def main(path_data, path_config):
    if not os.path.isfile(os.path.join(path_data, "0_segmentation_0_final.png")):
        raise FileNotFoundError("It seems that output from PlaneRCNN is missing!")

    logger.info(f"Loading config from {path_config}")
    GlobalConfig.load_config(path_config)
    logger.info(f"Loading data from {path_data}")
    image_data = ImageData(0, path_data)
    image_data.load_data()

    # Run refinement
    logger.info("Starting refinement procedure")
    result_dict = start_refinement_procedure(image_data)
    return result_dict


if __name__ == '__main__':
    args = arg_parse()
    main(args.dir_data, args.config)
