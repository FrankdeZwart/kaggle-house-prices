"""
Handles command line input.
"""
import argparse
import logging.config
import os
import sys

from prices.config import config
from prices.data.make_dataset import MakeDataset
from prices.features.build_features import BuildFeatures
from prices.models.linear_regression import LinearRegression

log_file_path = os.path.join(sys.prefix, 'prices_data', 'logger.ini')
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


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
    MakeDataset(filename=args.datafile).execute()

    # Feature engineering
    features = config.get('dict_features', 'list_continuous')
    y = BuildFeatures(filename=args.datafile).get_target()
    X = BuildFeatures(filename=args.datafile).select_features(features)

    # Fit a linear regression
    LinearRegression().ols_basic(y, X)

    logger.info('Script succeeded.')
