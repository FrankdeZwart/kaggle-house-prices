"""
Handles command line input.
"""
import argparse
import logging.config
import os
import sys

from prices.data.make_dataset import MakeDataset

filepath = os.path.dirname(sys.prefix)
log_file_path = os.path.join(filepath, './logger.ini')
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    """
    This function handles command line input and calls on the modules based on the input given.
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datafile', '-d',
        help='Source datafile.',
        dest='datafile',
        type=str
    )

    args, _ = parser.parse_known_args()

    # Process the raw data for model input
    MakeDataset(filepath=filepath, filename=args.datafile).execute()

    logger.info('Script succeeded.')
