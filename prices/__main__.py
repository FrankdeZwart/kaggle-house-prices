"""
Handles command line input.
"""
import argparse
import logging.config
import os
import sys

from prices.data.make_dataset import MakeDataset
from prices.features.build_features import FeatureBuilder
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
    # feature_object = FeatureBuilder(filename=args.datafile)
    X, y = FeatureBuilder(filename=args.datafile).build_features()

    # Stepwise feature building
    print(LinearRegression().select_features_forward(X=X, y=yg))

    logger.info('Script succeeded.')
