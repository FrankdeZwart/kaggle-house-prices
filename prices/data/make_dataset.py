"""
Process the raw data in such a way that it becomes input for a model.
"""
import logging
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MakeDataset():
    """
    Create the input dataset for the model.
    """

    def __init__(self, filepath: str, filename: str):
        """
        Define the input and output filepath.
        :param filename: Name of the file that is processed.
        """
        self.filepath = filepath
        self.filename = filename
        self.input_filepath = self.filepath + '/data/raw/' + self.filename
        self.output_filepath = self.filepath + '/data/processed/' + self.filename
        self.df_houses = pd.read_csv(self.input_filepath)

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

        :return:
        """

        logger.info('Processing data from %s.', self.input_filepath)

        # Group the features based on their type
        list_continuous = [
            'LotFrontage', 'LotArea', 'MasVnrArea',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
            'LowQualFinSF', 'GrLivArea', 'GarageArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
            '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'MiscVal'
        ]
        list_categorical = [
            'MSSubClass', 'MSZoning', 'Street',
            'Alley', 'LotShape', 'LandContour',
            'Utilities', 'LotConfig', 'LandSlope',
            'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle',
            'RoofMatl', 'Exterior1st', 'Exterior2nd',
            'MasVnrType', 'BedroomAbvGr', 'KitchenAbvGr',
            'Foundation', 'BsmtExposure', 'HalfBath',
            'BsmtFinType1', 'BsmtFinType2', 'SaleCondition',
            'Heating', 'CentralAir', 'Electrical',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
            'TotRmsAbvGrd', 'Functional', 'Fireplaces',
            'GarageType', 'GarageFinish', 'GarageCars',
            'PavedDrive', 'Fence', 'MiscFeature',
            'MoSold', 'YrSold', 'SaleType'
        ]
        list_ordinal = [
            'OverallQual', 'OverallCond', 'ExterQual',
            'ExterCond', 'BsmtQual', 'BsmtCond',
            'HeatingQC', 'KitchenQual', 'PoolQC',
            'FireplaceQu', 'GarageQual', 'GarageCond'
        ]
        list_date = [
            'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'
        ]

        # Create a dictionary with
        # key: value to replace NaNs with
        # value: list of column names
        current_year = datetime.date.today().year  # NaNs will become 0 when converting years to age
        dict_nan = {0: list_continuous + list_ordinal,
                    'None': list_categorical,
                    current_year: list_date}

        # Replace all NaN values for each feature type
        for key, value in dict_nan.items():
            self.replace_nan(value, key)

        # Convert ordinal strings to numeric values
        # Poor | Fair | Average | Good | Excellent
        dict_ord = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        self.map_dictionary(dict_ord, list_ordinal)

        # Use standard scaler to standardize the continuous and ordinal features
        self.standard_scaler(list_continuous + list_ordinal)

        # Convert the categorical features to dummies
        self.df_houses = pd.get_dummies(self.df_houses, columns=list_categorical, drop_first=True)

        # Store the processed file
        self.df_houses.to_csv(self.output_filepath)

        logger.info('Written processed file to %s.', self.output_filepath)
