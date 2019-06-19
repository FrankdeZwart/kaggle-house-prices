"""
Process the raw data in such a way that it becomes input for a model.
"""
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        self.df_raw = pd.read_csv(self.input_filepath)

    def replace_nan(self, column_list, replacement):
        """
        Replaces the NaN values in a Pandas dataframe.
        :param column_list: List with selection of features.
        :param replacement: Value that is used to replace NaN's.
        :return: Selection of dataframe with replaced NaN's.
        """
        return self.df_raw.loc[:, column_list].fillna(replacement)

    @staticmethod
    def map_dictionary(df_input, dictionary):
        """
        Changes values in a dataframe by mapping a dictionary.
        :param df_input: Input Pandas dataframe.
        :param dict: Dictionary with key value pairs.
        :return: Pandas dataframe with transformed values.
        """
        df_output = pd.DataFrame(columns=df_input.columns)
        for column in df_input.columns:
            df_output[column] = df_input[column].map(dictionary).fillna(df_input[column])
        return df_output

    @staticmethod
    def standard_scaler(df_input):
        """
        Standardizes values in Pandas a dataframe.
        :param df_input: Input Pandas dataframe.
        :return: Pandas dataframe with standardized values per column.
        """
        input_scaled = StandardScaler().fit_transform(df_input)
        df_output = pd.DataFrame(data=input_scaled,
                                 index=df_input.index,
                                 columns=df_input.columns)
        return df_output

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
                            'Foundation', 'BsmtExposure', 'HalfBath',
                            'BsmtFinType1', 'BsmtFinType2', 'SaleCondition'
                                                            'Heating', 'CentralAir', 'Electrical',
                            'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                            'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                            'GarageType', 'GarageFinish', 'GarageCars',
                            'PavedDrive', 'Fence', 'MiscFeature',
                            'MoSold', 'YrSold', 'SaleType']
        list_ordinal = ['OverallQual', 'OverallCond', 'ExterQual',
                        'ExterCond', 'BsmtQual', 'BsmtCond',
                        'HeatingQC', 'KitchenQual', 'PoolQC',
                        'FireplaceQu', 'GarageQual', 'GarageCond']
        # list_date = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
        target = 'SalePrice'

        # Replace the NaN values for each feature type
        df_cont_numeric = self.replace_nan(list_continuous, 0)
        df_cat_raw = self.replace_nan(list_categorical, 'None')
        df_ord_raw = self.replace_nan(list_ordinal, 0)

        # Convert ordinal object values to numeric values
        # Poor | Fair | Average | Good | Excellent
        dictionary = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        df_ord_numeric = self.map_dictionary(df_ord_raw, dictionary)

        # Use standard scaler to standardize the continuous and ordinal features
        df_cont_scaled = self.standard_scaler(df_cont_numeric)
        df_ord_scaled = self.standard_scaler(df_ord_numeric)

        # Convert the categorical features to dummies
        df_cat_dummy = pd.get_dummies(df_cat_raw, columns=df_cat_raw.columns, drop_first=True)

        # Concatenate all columns in one processed df_train
        df_train_processed = pd.concat([df_cont_scaled, df_ord_scaled,
                                        df_cat_dummy, self.df_raw.loc[:, target]],
                                       axis=1)

        # Store the processed file
        df_train_processed.to_csv('./data/processed/train.csv')

MakeDataset().execute()
