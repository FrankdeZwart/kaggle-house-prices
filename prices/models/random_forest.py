"""
Use the scikit learn package to perform a random forest regression.
"""
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import pydot
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import export_graphviz

logger = logging.getLogger(__name__)


class RandomForest():
    """
        Estimate the house prices with a random forest regression.
    """
    MDL = 'mdl'
    IS_FIT = 'is_fit'
    FEATURE_IMPORTANCE = 'feature_importance'
    NUM_OBS = 'num_obs'
    NUM_FEATURES = 'num_features'
    RMSLE = 'rmsle'

    def rf_basic(self, y_train: pd.Series, X_train: pd.DataFrame) -> Dict[str, Any]:
        """
        "Wrap around sci-kit learn' random forest, primary function used in later expansions"
        :param y: Target column.
        :param X: Feature matrix.
        :return: Dictionary with regression output.
        """

        logger.info('Fitting a random forest regression')

        # Add constant if not present
        X_train = sm.add_constant(data=X_train.values)

        # Save shape; n = obs, k = features
        obs, features = X_train.shape

        # Instantiate model with 1000 decision trees
        mdl = RandomForestRegressor(n_estimators=1000, random_state=42)

        # Train the model on training data
        mdl_fit = mdl.fit(X_train, y_train)

        # In-sample prediction (or fitting)
        y_hat = mdl_fit.predict(X_train)

        feature_importance = mdl_fit.feature_importances_

        return {self.MDL: mdl_fit,
                self.IS_FIT: y_hat,
                self.FEATURE_IMPORTANCE: feature_importance,
                self.NUM_OBS: obs,
                self.NUM_FEATURES: features}

    def rf_tree_output(self, mdl, feature_list):
        # Pull out one tree from the forest
        tree = mdl.estimators_[5]

        # Export the image to a dot file
        export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)

        # Use dot file to create a graph
        (graph,) = pydot.graph_from_dot_file('tree.dot')

        # Write graph to a png file
        graph.write_png('tree.png')

    def rf_performance(self, true_values, pred_values) -> Dict[str, float]:
        """
        Collects essential metrics to evaluate the performance of rf fit.
        """

        # 1. Root mean squared logarithmic error
        rmsle = np.round(np.sqrt(mean_squared_log_error(true_values, pred_values)), 5)

        return {self.RMSLE: rmsle}
