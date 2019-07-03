"""
Process the raw data in such a way that it becomes input for a model.
"""
import logging
from datetime import date

import pandas as pd
from sklearn.preprocessing import StandardScaler

from prices.config import config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MakeDataset():
    """
    Create the input dataset for the model.
    """

    def __init__(self, filename: str):
        """
        Define the input and output filepath and feature categories.
        :param filename: Name of the file that is processed.
        """
        self.filename = filename
        self.input_filepath = './data/raw/' + self.filename
        self.output_filepath = './data/processed/' + self.filename
        self.df_houses = pd.read_csv(self.input_filepath)
        self.list_continuous = config.get('dict_features', 'list_continuous')
        self.list_categorical = config.get('dict_features', 'list_categorical')
        self.list_ordinal = config.get('dict_features', 'list_ordinal')
        self.list_date = config.get('dict_features', 'list_date')

    def replace_nan(self, column_list, replacement):
        """
        Replaces the NaN values in a Pandas dataframe.
        :param column_list: List with selection of features.
        :param replacement: Value that is used to replace NaN's.
        """
        self.df_houses.loc[:, column_list] = self.df_houses.loc[:, column_list].fillna(replacement)

    def map_dictionary(self, dictionary, column_list):
        """
        Changes values in a dataframe by mapping a dictionary.
        :param dictionary: Dictionary with key value pairs.
        :param column_list: List of columns to apply mapping to.
        """
        for column in column_list:
            self.df_houses.loc[:, column] = self.df_houses.loc[:, column].map(dictionary) \
                .fillna(self.df_houses.loc[:, column])

    def standard_scaler(self, column_list):
        """
        Standardizes the values for a selection columns from a dataframe.
        :param column_list: List of columns to apply mapping to.
        """
        input_scaled = StandardScaler().fit_transform(self.df_houses.loc[:, column_list])
        self.df_houses.loc[:, column_list] = pd.DataFrame(data=input_scaled,
                                                          index=self.df_houses.index,
                                                          columns=column_list)

    def execute(self):
        """
        Executes the data processing steps.
        """

        logger.info('Processing data from %s.', self.input_filepath)

        # Create a dictionary with
        # key: value to replace NaNs with
        # value: list of column names
        current_year = date.today().year  # NaNs will become 0 when converting years to age
        dict_nan = {
            0: self.list_continuous + self.list_ordinal,
            'None': self.list_categorical,
            current_year: self.list_date
        }

        # Replace all NaN values for each feature type
        for key, value in dict_nan.items():
            self.replace_nan(value, key)


        # Poor | Fair | Average | Good | Excellent
        dict_ord = {
            'Po': 1,
            'Fa': 2,
            'TA': 3,
            'Gd': 4,
            'Ex': 5
        }

        # Convert ordinal strings to numeric values
        self.map_dictionary(dict_ord, self.list_ordinal)

        # Use standard scaler to standardize the continuous and ordinal features
        self.standard_scaler(self.list_continuous + self.list_ordinal)

        # Convert the categorical features to dummies
        self.df_houses = pd.get_dummies(self.df_houses, columns=self.list_categorical, drop_first=True)

        # Store the processed file
        self.df_houses.to_csv(self.output_filepath)

        logger.info('Written processed file to %s.', self.output_filepath)
