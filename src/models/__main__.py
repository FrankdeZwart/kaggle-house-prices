"""
Handles command line input.
"""
import logging, logging.config
from ..data.make_dataset import MakeDataset

logging.config.fileConfig('./logger.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    """
    This function handles command line input and calls on the modules based on the input given.
    :return:
    """
    MakeDataset().execute()

    logger.info('Script succeeded.')


if __name__ == '__main__':
    main()