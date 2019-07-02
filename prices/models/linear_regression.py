"""
Use the statsmodels package to perform a linear regression.
"""
import logging
from typing import Dict, Any

import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class LinearRegression():
    """
    Estimate the house prices with a linear regression.
    """
    MDL = 'mdl'
    IS_FIT = 'is_fit'
    COEFFICIENTS = 'coefficients'
    NUM_OBS = 'num_obs'
    NUM_FEATURES = 'num_features'

    LOGL = 'logl'
    AIC = 'aic'
    MSPE = 'mspe'
    R2 = 'r2'
    AR_MODEL = 'ar_model'
    SIGNIFICANCE_95P = 'significance_95p'

    def __init__(self):
        """

        """

    def ols_basic(self, y: pd.Series, X: pd.DataFrame) -> Dict[str, Any]:
        """
        "Wrap around statsmodels' OLS, primary function used in later expansions"
        :param y: Target column.
        :param X: Feature matrix.
        :return: Dictionary with OLS output.
        """

        X = sm.add_constant(X)
        sm.OLS(y, X)

        # Add constant if not present
        X = sm.add_constant(data=X)

        # Save shape; n = obs, k = features
        obs, features = X.shape

        # Main model, simple least squares
        mdl = sm.OLS(y, X, missing='drop')

        # Fit y using X, using B to minimize (y - X*B)
        mdl_fit = mdl.fit()

        # In-sample prediction (or fitting)
        y_hat = mdl_fit.predict(X)

        coeffs = mdl_fit.params

        import pdb
        pdb.set_trace()

        return {self.MDL: mdl_fit,
                self.IS_FIT: y_hat,
                self.COEFFICIENTS: coeffs,
                self.NUM_OBS: obs,
                self.NUM_FEATURES: features}

    def ols_performance(self, fit, true_values, pred_values, n_features) -> Dict[str, float]:
        """
        Collects essential metrics to evaluate the performance of OLS fit.
        """
        # 1. Log-likelihood value to determine AIC
        llf = np.round(fit.llf, 3)

        # 2. Simple R-squared of in-sample fit
        r_squared = np.round(fit.rsquared_adj, 3)

        # 3 Akaike Information Criterion, quick go-to measure for model quality
        aic = np.round(sm.tools.eval_measures.aic(llf, len(true_values), n_features), 3)

        # 4. Mean Squared Prediction Error (in-sample), comparable to R-squared
        mspe = np.round(mean_squared_error(true_values, pred_values), 3)

        return {self.LOGL: llf,
                self.AIC: aic,
                self.MSPE: mspe,
                self.R2: r_squared}
