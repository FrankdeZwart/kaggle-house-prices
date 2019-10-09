"""
Build the feature matrix and target vector from the processed data.
"""
import logging

import numpy as np
import pandas as pd

from prices.config import config

logger = logging.getLogger(__name__)


class FeatureSelection():
    """
    Generate and select features.
    """

    def __init__(self, filename: str):
        """
        Define the input and output filepath and feature categories.
        """
        self.filename = filename
        self.input_filepath = './data/processed/' + self.filename
        self.df_houses = pd.read_csv(self.input_filepath)
        self.target = config.get_attribute('dict_features', 'target')

    def get_features(self) -> pd.DataFrame:
        """
        Returns the selected features.
        :return: Pandas dataframe with values of selected features.
        """
        X = self.df_houses.drop(self.target, axis=1)

        return X

    def get_target(self) -> pd.Series:
        """
        Returns the log transformation of the target variable.
        :return: Pandas series with log transformed target variable.
        """

        # Log transform the target for official scoring
        y = np.log1p(self.df_houses.loc[:, self.target])

        return y
