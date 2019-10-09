"""
Process the raw data in such a way that it becomes input for a model.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import skew

from prices.config import config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DatasetBuilder:
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

        # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
        self.skewness_threshold = 0.5

        self.dict_nan_replacement = config.get_dictionary('dict_nan_replacement')
        self.dict_features_nan = config.get_dictionary('dict_features_nan')
        self.dict_replace = config.get_dictionary('dict_replace')
        self.target = config.get_attribute('dict_features', 'target')
        self.dict_features_simplify = config.get_dictionary('dict_features_simplify')
        self.dict_simplify_replacement = config.get_dictionary('dict_simplify_replacement')

    def replace_nan(self, column_list, replacement):
        """
        Replaces the NaN values in a Pandas dataframe.
        :param column_list: List with selection of features.
        :param replacement: Value that is used to replace NaN's.
        """
        self.df_houses.loc[:, column_list] = self.df_houses.loc[:, column_list].fillna(replacement)

    def combine_features(self):
        """
        Create combinations of the existing features.
        """
        # Overall quality of the house
        self.df_houses["OverallGrade"] = self.df_houses["OverallQual"] * self.df_houses["OverallCond"]
        # Overall quality of the garage
        self.df_houses["GarageGrade"] = self.df_houses["GarageQual"] * self.df_houses["GarageCond"]
        # Overall quality of the exterior
        self.df_houses["ExterGrade"] = self.df_houses["ExterQual"] * self.df_houses["ExterCond"]
        # Overall kitchen score
        self.df_houses["KitchenScore"] = self.df_houses["KitchenAbvGr"] * self.df_houses["KitchenQual"]
        # Overall fireplace score
        self.df_houses["FireplaceScore"] = self.df_houses["Fireplaces"] * self.df_houses["FireplaceQu"]
        # Overall garage score
        self.df_houses["GarageScore"] = self.df_houses["GarageArea"] * self.df_houses["GarageQual"]
        # Overall pool score
        self.df_houses["PoolScore"] = self.df_houses["PoolArea"] * self.df_houses["PoolQC"]
        # Simplified overall quality of the house
        self.df_houses["SimplOverallGrade"] = self.df_houses["SimplOverallQual"] * self.df_houses["SimplOverallCond"]
        # Simplified overall quality of the exterior
        self.df_houses["SimplExterGrade"] = self.df_houses["SimplExterQual"] * self.df_houses["SimplExterCond"]
        # Simplified overall pool score
        self.df_houses["SimplPoolScore"] = self.df_houses["PoolArea"] * self.df_houses["SimplPoolQC"]
        # Simplified overall garage score
        self.df_houses["SimplGarageScore"] = self.df_houses["GarageArea"] * self.df_houses["SimplGarageQual"]
        # Simplified overall fireplace score
        self.df_houses["SimplFireplaceScore"] = self.df_houses["Fireplaces"] * self.df_houses["SimplFireplaceQu"]
        # Simplified overall kitchen score
        self.df_houses["SimplKitchenScore"] = self.df_houses["KitchenAbvGr"] * self.df_houses["SimplKitchenQual"]
        # Total number of bathrooms
        self.df_houses["TotalBath"] = self.df_houses["BsmtFullBath"] + (0.5 * self.df_houses["BsmtHalfBath"]) + \
                                      self.df_houses["FullBath"] + (0.5 * self.df_houses["HalfBath"])
        # Total SF for house (incl. basement)
        self.df_houses["AllSF"] = self.df_houses["GrLivArea"] + self.df_houses["TotalBsmtSF"]
        # Total SF for 1st + 2nd floors
        self.df_houses["AllFlrsSF"] = self.df_houses["1stFlrSF"] + self.df_houses["2ndFlrSF"]
        # Total SF for porch
        self.df_houses["AllPorchSF"] = self.df_houses["OpenPorchSF"] + self.df_houses["EnclosedPorch"] + \
                                       self.df_houses["3SsnPorch"] + self.df_houses["ScreenPorch"]
        # Has masonry veneer or not
        self.df_houses["HasMasVnr"] = self.df_houses.MasVnrType.replace({"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                                                         "Stone": 1, "None": 0})
        # House completed before sale or not
        self.df_houses["BoughtOffPlan"] = self.df_houses.SaleCondition.replace({"Abnorml": 0, "Alloca": 0,
                                                                                "AdjLand": 0, "Family": 0,
                                                                                "Normal": 0, "Partial": 1})

    def differentiate_column_types(self):
        """
        Differentiates numerical features (minus the target) and categorical features.
        :return: List with column names of the categorical features, numerical features, and the target respectively.
        """
        categorical_features = self.df_houses.select_dtypes(include=["object"]).columns
        numerical_features = self.df_houses.select_dtypes(exclude=["object"]).columns.drop(self.target)

        logger.info('Number of categorical features: %s.', len(categorical_features))
        logger.info('Number of numerical features: %s.', len(numerical_features))

        return categorical_features, numerical_features

    def log_transform(self, train_num, threshold):
        """
        Log transform of the skewed numerical features to lessen impact of outliers.
        :return:
        """
        skewness = self.df_houses.loc[:, train_num].apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > threshold]
        skewed_features = skewness.index
        self.df_houses.loc[:,skewed_features] = np.log1p(self.df_houses.loc[:,skewed_features])

        logger.info('%s skewed numerical features are log transformed.', skewness.shape[0])

    def execute(self):
        """
        Executes the data processing steps.
        """

        logger.info('Processing data from %s.', self.input_filepath)

        # Check for duplicates
        idsUnique = len(set(self.df_houses.Id))
        idsTotal = self.df_houses.shape[0]
        idsDupli = idsTotal - idsUnique
        logger.info("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries.")

        # Drop Id column
        self.df_houses.drop('Id', axis=1, inplace=True)

        # Remove outliers (see exploratory data analysis notebook for plot)
        self.df_houses = self.df_houses[self.df_houses.GrLivArea < 4000]

        # Handle missing values for features where median/mean or most common value doesn't make sense
        for key, value in self.dict_features_nan.items():
            self.replace_nan(column_list=value, replacement=self.dict_nan_replacement[key])

        # Convert some numerical features to categorical features
        # Encode some categorical features as ordered numbers when there is information in the order
        self.df_houses = self.df_houses.replace(self.dict_replace)

        # Simplifications of existing features
        for key, value in self.dict_features_simplify.items():
            new_column_list = ['Simpl' + s for s in value]
            self.df_houses[new_column_list] = self.df_houses.loc[:, value].replace(self.dict_simplify_replacement[key])

        # Combinations of existing features based on domain knowledge
        self.combine_features()

        # Differentiate numerical features (minus the target) and categorical features
        categorical_features, numerical_features = self.differentiate_column_types()

        # Log transform of the skewed numerical features to lessen impact of outliers
        self.log_transform(numerical_features, self.skewness_threshold)

        # Handle remaining missing values for numerical features by using median as replacement
        self.df_houses.loc[:, numerical_features] = self.df_houses.loc[:, numerical_features] \
            .fillna(self.df_houses.loc[:, numerical_features].median())

        # Create dummy features for categorical values via one-hot encoding
        self.df_houses = pd.get_dummies(self.df_houses, columns=categorical_features, drop_first=True)

        # Store the processed file
        self.df_houses.to_csv(self.output_filepath, index=False)

        logger.info('Written processed file to %s.', self.output_filepath)
