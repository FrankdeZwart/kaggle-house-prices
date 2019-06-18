"""
Process the raw data in such a way that it becomes input for a model.
"""
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)


class MakeDataset():
    """
    Create the input dataset for the model.
    """

    def __init__(self, input_filepath, output_filepath):
        """
        Define the input and output filepath.
        :param input_filepath: Filepath where raw data is stored.
        :param output_filepath: Filepath where the processed data will be stored.
        """
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.df_input = pd.read_csv(input_filepath)

    def replace_nan(self, column_list, replacement):
        """
        Replaces the NaN values in a Pandas dataframe
        :param df_input: Input Pandas dataframe with NaN's
        :param column_list: List with selection of features
        :param replacement: Value that is used to replace NaN's
        :return: Selection of dataframe with replaced NaN's
        """
        return self.df_input.loc[:, column_list].fillna(replacement)

    def execute(self):
        """

        :return:
        """

        LOGGER.info('making final data set from raw data')

        # Load the raw data in a pandas dataframe
        df_raw = pd.read_excel(self.input_filepath)

        list_continuous = ['LotFrontage', 'LotArea', 'MasVnrArea',
                           'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                           'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                           'LowQualFinSF', 'GrLivArea', 'GarageArea',
                           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                           '3SsnPorch', 'ScreenPorch', 'PoolArea',
                           'MiscVal']

        list_categorical = ['MSSubClass', 'MSZoning', 'Street',
                            'Alley', 'LotShape', 'LandContour',
                            'Utilities', 'LotConfig', 'LandSlope',
                            'Neighborhood', 'Condition1', 'Condition2',
                            'BldgType', 'HouseStyle', 'RoofStyle',
                            'RoofMatl', 'Exterior1st', 'Exterior2nd',
                            'MasVnrType', 'BedroomAbvGr', 'KitchenAbvGr',
                            'Foundation', 'Id',
                            'BsmtFinType1', 'BsmtFinType2',
                            'Heating', 'CentralAir', 'Electrical',
                            'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                            'HalfBath',
                            'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                            'GarageType', 'GarageFinish', 'GarageCars',
                            'PavedDrive', 'Fence', 'MiscFeature',
                            'MoSold', 'YrSold', 'SaleType',
                            'SaleCondition']

        list_ordinal = ['OverallQual', 'OverallCond', 'ExterQual',
                        'ExterCond', 'BsmtQual', 'BsmtCond',
                        'BsmtExposure', 'HeatingQC', 'KitchenQual',
                        'FireplaceQu', 'GarageQual', 'GarageCond',
                        'PoolQC']

        list_date = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

        list_features = list_continuous + list_categorical + list_ordinal + list_date

        df_target = df_raw.loc[:, ~df_raw.columns.isin(list_features)]

        return df_target
