"""
Build the feature matrix and target vector from the processed data.
"""
import logging

import pandas as pd

from prices.config import config

logger = logging.getLogger(__name__)


class BuildFeatures():
    """

    """

    def __init__(self, filename: str):
        """

        """
        self.filename = filename
        self.input_filepath = './data/processed/' + self.filename
        self.df_houses = pd.read_csv(self.input_filepath)
        self.target = config.get('dict_features', 'target')

    def select_features(self, features) -> pd.DataFrame:
        """
        
        :return:
        """

        return self.df_houses[features]

    def get_target(self) -> pd.Series:
        """

        :return:
        """

        return self.df_houses[self.target]
