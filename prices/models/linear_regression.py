"""
Use the statsmodels package to perform a linear regression.
"""
import statsmodels.api as sm


class LinearRegression():
    """
    Estimate the house prices with a linear regression.
    """

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
        X = sm.add_constant(df_features=X)

        # Save shape; n = obs, k = features
        obs, features = X.shape

        # Main model, simple least squares
        mdl = sm.OLS(y, X, missing='drop')

        # Fit y using X, using B to minimize (y - X*B)
        mdl_fit = mdl.fit()

        # In-sample prediction (or fitting)
        y_hat = mdl_fit.predict(X)

        coeffs = mdl_fit.params

        return {self.MDL: mdl_fit,
                self.IS_FIT: y_hat,
                self.COEFFICIENTS: coeffs,
                self.NUM_OBS: obs,
                self.NUM_FEATURES: features}
