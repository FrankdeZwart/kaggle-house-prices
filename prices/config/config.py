"""
File with the projects config settings.
"""

import logging
import sys

from datetime import date

logger = logging.getLogger(__name__)

current_config = sys.modules[__name__]


def get(key: str, attribute: str) -> str:
    """
    Get a configuration attribute from a given key.
    """
    return getattr(current_config, key)[attribute]


# Group the features based on their type
dict_features = {
    'list_continuous': ['LotFrontage', 'LotArea',
                        'MasVnrArea', 'BsmtFinSF1',
                        'BsmtFinSF2', 'BsmtUnfSF',
                        'TotalBsmtSF', '1stFlrSF',
                        '2ndFlrSF', 'LowQualFinSF',
                        'GrLivArea', 'GarageArea',
                        'WoodDeckSF', 'OpenPorchSF',
                        'EnclosedPorch', '3SsnPorch',
                        'ScreenPorch', 'PoolArea',
                        'MiscVal'],
    'list_categorical': ['MSSubClass', 'MSZoning',
                         'Street', 'Alley',
                         'LotShape', 'LandContour',
                         'Utilities', 'LotConfig',
                         'LandSlope', 'Neighborhood',
                         'Condition1', 'Condition2',
                         'BldgType', 'HouseStyle',
                         'RoofStyle', 'RoofMatl',
                         'Exterior1st', 'Exterior2nd',
                         'MasVnrType', 'BedroomAbvGr',
                         'KitchenAbvGr', 'Foundation',
                         'BsmtExposure', 'HalfBath',
                         'BsmtFinType1', 'BsmtFinType2',
                         'SaleCondition', 'Heating',
                         'CentralAir', 'Electrical',
                         'BsmtFullBath', 'BsmtHalfBath',
                         'FullBath', 'TotRmsAbvGrd',
                         'Functional', 'Fireplaces',
                         'GarageType', 'GarageFinish',
                         'GarageCars', 'PavedDrive',
                         'Fence', 'MiscFeature',
                         'MoSold', 'YrSold',
                         'SaleType'],
    'list_ordinal': ['OverallQual', 'OverallCond',
                     'ExterQual', 'ExterCond',
                     'BsmtQual', 'BsmtCond',
                     'HeatingQC', 'KitchenQual',
                     'PoolQC', 'FireplaceQu',
                     'GarageQual', 'GarageCond'],
    'list_date': ['YearBuilt', 'YearRemodAdd',
                  'GarageYrBlt'],
    'target': 'SalePrice'
}

# Create a dictionary with
# key: value to replace NaNs with
# value: list of column names
current_year = date.today().year  # NaNs will become 0 when converting years to age
dict_nan = {
    'list_continuous': 0,
    'list_categorical': 'None',
    'list_ordinal': 0,
    'list_date': current_year
}
