"""
Build the feature matrix and target vector from the processed data.
"""
import logging

import pandas as pd

from prices.config import config
from ..support_functions.toolbox import calculate_correlations

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Generate and select features.
    """

    def __init__(self, filename: str):
        """
        """
        self.filename = filename
        self.input_filepath = './data/processed/' + self.filename
        self.df = pd.read_csv(self.input_filepath)
        self.target = config.get('dict_features', 'target')
        self.features = config.get('dict_features', 'list_continuous')

    @staticmethod
    def filter_multicollinearity(target: pd.Series, all_features: pd.DataFrame, threshold: float = 0.7) -> list:
        """
        :param target: Variable to predict
        :param all_features: All possible features to be evaluated for possible multicollinearity
        :param threshold: The maximum correlation allowed between two features (degree of multicoll)
        :return: A list with all variables that correlate best with the target, but filtered for multicoll
        """

        # The optimized auto-regressive terms are by default in the model, so need to be excluded from this filter
        exog_cols = ~all_features.columns.str.contains(target.name)
        ar_cols = all_features.columns.str.contains(target.name)
        ar_features = list(all_features.iloc[:, ar_cols].columns)
        exog_features = all_features.iloc[:, exog_cols]

        # Create ordered list with correlations to endogenous variable
        exogenous_corr = calculate_correlations(primary_variable=target, variables_to_evaluate=exog_features)

        # Current feature list (to initialize while-loop)
        current_features_list = ar_features + list(exogenous_corr.index)
        ordered_features = all_features[current_features_list]
        filtered_features = []

        # While loop that goes through all all_features
        while len(current_features_list) > 1:
            # Choose first feature in remaining list (highest ranked)
            current_feature = current_features_list[0]

            # Has not been deleted in previous iteration, so add to feature set
            filtered_features.append(current_feature)

            # Remove added feature from the to be evaluated all_features
            remaining_features = ordered_features.drop(current_feature, axis=1)

            # Determine correlation between current feature and remaining potential all_features
            current_correlations = pd.DataFrame(
                [abs(ordered_features[current_feature].corr(remaining_features.iloc[:, i]))
                 for i in range(0, len(remaining_features.columns))]).set_index(remaining_features.columns)

            # Keep only those all_features which correlation < threshold
            temp_keep_variables = current_correlations[current_correlations.iloc[:, 0] < threshold].index

            # Update remaining feature list for next iteration
            current_features_list = temp_keep_variables
            ordered_features = all_features[current_features_list]

            # For last iteration of the while loop, if one is remaining it does not surpass threshold
            if len(current_features_list) == 1:
                filtered_features.append(current_features_list[0])

        return filtered_features

    def get_features(self) -> pd.DataFrame:
        """
        :return: The features in a df.
        """
        return self.df[self.features]

    def get_target(self) -> pd.Series:
        """

        :return:
        """

        return self.df[self.target]

    def build_features(self, filter_multicollinearity=True):
        """

        :param filter_multicollinearity:
        :return:
        """
        if filter_multicollinearity:
            import pdb
            pdb.set_trace()
            self.features = self.filter_multicollinearity(self.get_target(), self.get_features())

        return self.get_features(), self.get_target()
