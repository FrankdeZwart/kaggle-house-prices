"""
Process the raw data in such a way that it becomes input for a model.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MakeDataset():
    """
    Create the input dataset for the model.
    """

    def __init__(self):
        """
        Define the input and output filepath.
        :param input_filepath: Filepath where raw data is stored.
        :param output_filepath: Filepath where the processed data will be stored.
        """
        self.input_filepath = './data/raw/train.csv'
        self.output_filepath = './data/processed/train.csv'
        self.df_input = pd.read_csv(self.input_filepath)

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

        logger.info('making final data set from raw data')

        # Group the features based on their type
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

        # Replace the NaN values for each feature type
        df_continuous = self.replace_nan(list_continuous, 0)
        df_categorical = self.replace_nan(list_categorical, 'None')
        # df_ordinal = self.replace_nan(list_ordinal, '')
        # df_date = self.replace_nan(list_date, )

        # Create one list with all features
        list_features = list_continuous + list_categorical + list_ordinal + list_date

        # Store the target value in a separate
        df_target = self.df_input.loc[:, ~self.df_input.columns.isin(list_features)]

        import pdb
        pdb.set_trace()

        return df_target, df_continuous, df_categorical


MakeDataset().execute()
