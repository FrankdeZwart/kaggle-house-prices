"""
File with the projects config settings.
"""

import logging
import sys

from datetime import date

logger = logging.getLogger(__name__)

current_config = sys.modules[__name__]


def get_attribute(key: str, attribute: str) -> str:
    """
    Get a configuration attribute from a given key.
    """
    return getattr(current_config, key)[attribute]


def get_dictionary(name_dict: str) -> dict:
    """
    Get a dictionary from the config file.
    """
    return getattr(current_config, name_dict)


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

# Handle missing values for features where median/mean or most common value doesn't make sense (45 features)
dict_features_nan = {
    'none': ['Alley', 'MasVnrType'],
    'zero': ['BedroomAbvGr', 'BsmtFullBath',
             'BsmtHalfBath', 'BsmtUnfSF',
             'EnclosedPorch', 'Fireplaces',
             'GarageArea', 'GarageCars',
             'HalfBath', 'KitchenAbvGr',
             'LotFrontage', 'MasVnrArea',
             'MiscVal', 'OpenPorchSF',
             'PoolArea', 'ScreenPorch',
             'TotRmsAbvGrd', 'WoodDeckSF'],
    'no': ['BsmtQual', 'BsmtCond',
           'BsmtExposure', 'BsmtFinType1',
           'BsmtFinType2', 'Fence',
           'FireplaceQu', 'GarageType',
           'GarageFinish', 'GarageQual',
           'GarageCond', 'MiscFeature',
           'PoolQC'],
    'n': ['CentralAir', 'PavedDrive'],
    'norm': ['Condition1', 'Condition2'],
    'ta': ['ExterCond', 'ExterQual',
           'HeatingQC', 'KitchenQual'],
    'typ': ['Functional'],
    'reg': ['LotShape'],
    'normal': ['SaleCondition'],
    'allpub': ['Utilities']
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

dict_nan_replacement = {
    'none': 'None',
    'zero': 0,
    'no': 'No',
    'n': 'N',
    'norm': 'Norm',
    'ta': 'TA',
    'typ': 'Typ',
    'reg': 'Reg',
    'normal': 'Normal',
    'allpub': 'AllPub'
}

# Some numerical features are actually really categories
# Encode some categorical features as ordered numbers when there is information in the order
dict_replace = {
    "MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                   50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                   80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                   150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
    "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"},
    "Alley": {"Grvl": 1, "Pave": 2},
    "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                   "Min2": 6, "Min1": 7, "Typ": 8},
    "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Street": {"Grvl": 1, "Pave": 2},
    "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}
}

# Simplify ordinal features based on the range of the used scale
# Dictionary with features to simplify
dict_features_simplify = {
    'dict_ord_4': ['PoolQC'],
    'dict_ord_5': ['GarageCond', 'GarageQual',
                   'FireplaceQu', 'KitchenQual',
                   'HeatingQC', 'BsmtCond',
                   'BsmtQual', 'ExterCond',
                   'ExterQual'],
    'dict_ord_6': ['BsmtFinType1', 'BsmtFinType2'],
    'dict_ord_8': ['Functional'],
    'dict_ord_10': ['OverallQual', 'OverallCond']
}

# Dictionary with simplified values for each scale
dict_simplify_replacement = {
    'dict_ord_4': {
        1: 1, 2: 1,  # average
        3: 2, 4: 2  # good
    },
    'dict_ord_5': {
        1: 1,  # bad
        2: 1, 3: 1,  # average
        4: 2, 5: 2  # good
    },
    'dict_ord_6': {
        1: 1,  # unfinished
        2: 1, 3: 1,  # rec room
        4: 2, 5: 2, 6: 2  # living quarters
    },
    'dict_ord_8': {
        1: 1, 2: 1,  # bad
        3: 2, 4: 2,  # major
        5: 3, 6: 3, 7: 3,  # minor
        8: 4  # typical
    },
    'dict_ord_10': {
        1: 1, 2: 1, 3: 1,  # bad
        4: 2, 5: 2, 6: 2,  # average
        7: 3, 8: 3, 9: 3, 10: 3  # good
    }
}
