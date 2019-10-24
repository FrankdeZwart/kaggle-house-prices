"""
This script fits a linear regression model.
"""
import logging

import pandas as pd
import statsmodels.api as sm
from confload import Config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    def scale_features(self, features: pd.DataFrame):
        """
        Scale features.
        :param features: the features to scale
        :return:
        """
        features = sm.add_constant(features, prepend=True, has_constant='raise')
        scale_x = StandardScaler().fit(features.values)
        self.feature_scale = pd.DataFrame(
            {
                'scale': scale_x.scale_,
                'mean': scale_x.mean_
            }, columns=['scale'], index=features.columns
        )
        scaled_features = scale_x.transform(features)
        return pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

    def unscale_coefficients(self, coefficients):
        """

        :param coefficients:
        :return:
        """
        # Only unscale the constant
        scale_df = pd.DataFrame(coefficients, index=coefficients.index).join(self.feature_scale, how='inner')
        return scale_df['estimated_coefficients']/scale_df['scale']

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
                                                                                    test_size=0.1,
                                                                                    random_state=1)

        # Adding constant for baseline
        train_features = sm.add_constant(train_features, prepend=True, has_constant='skip')

        # OLS estimation of the model
        clf = sm.OLS(train_target, train_features)

        # Determine fit of model
        regression_fit = clf.fit()

        # Show statistics of model
        return train_features, test_features, train_target, test_target, regression_fit

    def select_features(self, features, target):
        """

        :param features:
        :param target:
        :return:
        """
        train_features, test_features, train_target, test_target, regression_fit = self.run_model(features, target)

        feature_names = pd.DataFrame(
            {
                'features': list(train_features.columns.values)[1:],
                't_values': regression_fit.tvalues[1:]
            },
            columns=['features', 't_values'])

        # Remove the feature with the lowest t value.
        feature_names = feature_names.loc[
            ~(feature_names['t_values'] == min(feature_names['t_values'], key=abs))]

        # Rename features to only include the ones that are not removed.
        features = features[feature_names['features'].values]

        # Continue until all features are significant.
        if min(feature_names['t_values'], key=abs) > 2:
            print('Laatste run.')

            # Run the model
            train_features, test_features, train_target, test_target, regression_fit = self.run_model(features, target)

            # Save the final parameters
            parameters = pd.DataFrame(
                {
                    'features': list(train_features.columns.values),
                    'estimated_coefficients': regression_fit.params,
                    't_values': regression_fit.tvalues
                },
                columns=['features', 'estimated_coefficients', 't_values']).set_index('features', drop=True)

            # Return the parameters and the final regression summary.
            return parameters, regression_fit.summary()

        else:
            print(len(feature_names))
            return self.select_features(features, target)

    def fit_regression(self):
        """
        First fit
        :return:
        """
        # Split features and targets.
        features, target = self.split_x_y()

        # Scale the features
        features = self.scale_features(features)

        # Run feature selection
        results, est = self.select_features(features, target)

        # Unscale the coefficients
        results['estimated_coefficients'] = self.unscale_coefficients(results['estimated_coefficients'])

        print(results, est)
