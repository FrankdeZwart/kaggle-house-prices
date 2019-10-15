"""
Handles command line input.
"""
import argparse
import logging.config
import os
import sys

from prices.features.build_features import FeatureBuilder
from prices.data.dataset_builder import DatasetBuilder
from prices.models.linear_regression import LinearRegression
from prices.models.random_forest import RandomForest

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
        '--train',
        help='Source train data.',
        dest='train',
        type=str
    )

    parser.add_argument(
        '--test',
        help='Source test data.',
        dest='test',
        type=str
    )

    parser.add_argument(
        '--prep',
        help='Boolean whether or not to pre-process raw data.',
        dest='prep',
        type=bool
    )

    args, _ = parser.parse_known_args()

    if args.prep == True:
        # Process the raw data for model input
        DatasetBuilder(filename=args.train).execute()

    # Feature engineering
    X, y = FeatureBuilder(filename=args.train).build_features()

    # Stepwise feature building
    feature_selection = LinearRegression().select_features_forward(X=X, y=y)
    logger.info('Selected features are: ' + str(feature_selection['selected_features']))

    # Basic OLS regression
    results_ols = LinearRegression().ols_basic(y=y, X=X[feature_selection['selected_features']])

    # Random forest regression
    results_rf = RandomForest().rf_basic(y_train=y, X_train=X[feature_selection['selected_features']])

    # Print rf tree
    RandomForest().rf_tree_output(mdl=results_rf['mdl'],
                                  feature_list=['Constant'] + feature_selection['selected_features'])

    # Evaluate performance
    evaluation_ols = LinearRegression().ols_performance(
        fit=results_ols['mdl'],
        true_values=y,
        pred_values=results_ols['is_fit'],
        n_features=results_ols['num_features'])
    logger.info('OLS in-sample RMSLE: %s', evaluation_ols['rmsle'])
    evaluation_rf = RandomForest().rf_performance(true_values=y, pred_values=results_rf['is_fit'])
    logger.info('Random Forest in-sample RMSLE: %s', evaluation_rf['rmsle'])

    logger.info('Script succeeded.')