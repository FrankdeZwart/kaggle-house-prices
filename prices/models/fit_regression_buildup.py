"""
This script fits a linear regression model.
"""
import logging

import pandas as pd
import statsmodels.api as sm
from confload import Config

from ..models.correlation_analysis import CorrelationAnalysis

logger = logging.getLogger(__name__)

if not Config.ready():
    Config.load('./src/model_config.yml')


class RegressionFitter:
    """
    Fit a regression. Select features using backward selection.
    """

    def __init__(self, df, channel):
        self.df = df
        self.target = channel + '_sales_corrected'
        self.weekday = Config.get('aggregated_model')['weekday']
        self.holiday = Config.get('aggregated_model')['holiday']
        self.feature_scale = pd.DataFrame()

    def split_x_y(self):
        """
        Select the columns of the df to use as features and targets.
        :return:
        """
        features = self.df[self.df.drop([self.target, self.holiday, self.weekday, 'datum'], axis=1).columns]
        targets = self.df[self.target].astype(int)
        return features, targets

    @staticmethod
    def run_model(features, target):
        """

        :param features:
        :param target:
        :return:
        """
        # Split data in train and test set.
        train_features, test_features, train_target, test_target = train_test_split(features,
                                                                                    target,
                                                                                    train_size=0.9,
                                                                                    random_state=1)

        # Adding constant for baseline
        train_features = sm.add_constant(train_features, prepend=True, has_constant='skip')

        # OLS estimation of the model
        clf = sm.OLS(train_target, train_features)

        # Determine fit of model
        regression_fit = clf.fit()
        y_hat = regression_fit.predict(train_features)

        # Show statistics of model
        return train_features, test_features, train_target, test_target, regression_fit

    def select_features_forward(self, features, target):
        # Give overview of the inputted features
        logger.info("The setup contains %d input_features, covering %d observations.",
                    features.shape[1],
                    features.shape[0])

        # Apply the variable selection algorithm to remove multicollinearity
        variables_for_analysis = CorrelationAnalysis.multicollinearity_filter(target=target,
                                                                              all_features=features)

        X = features[variables_for_analysis]
        logger.info("The number of input_features is reduced to {k_opt} "
                    "as a result of the selection algorithm.".format(k_opt=X.shape[1])
                    )
        y = target

        self.run_model(features, target)

        residual_correlations = CorrelationAnalysis().correlations(
            primary_variable=current_residuals,
            variables_to_evaluate=exog_features)


def fit_regression(self):
    """
    First fit
    :return:
    """
    # Split features and targets.
    features, target = self.split_x_y()

    self.select_features_forward(features, target)

    # Select the best features
    # sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=40)
    #
    # # Run feature selection
    # results, est = self.select_features(features, target)

    # Unscale the coefficients
    # results['estimated_coefficients'] = self.unscale_coefficients(results['estimated_coefficients'])

    # print(results, est)
