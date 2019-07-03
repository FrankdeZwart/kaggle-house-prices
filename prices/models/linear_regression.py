"""
Use the statsmodels package to perform a linear regression.
"""
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

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

    CORRELATION = 'correlation'
    ABS_CORRELATION = 'abs_correlation'

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

        logger.info('Fitting a basic OLS regression')

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

    def correlations(self, primary_variable: pd.Series, variables_to_evaluate: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the correlation between the primary variable and a set of possible variables.
        :param primary_variable: The variable of primary interest (e.g. to predict).
        :param variables_to_evaluate: The entire set of possible variables that could be used in the prediction.
        :return: A DataFrame containing all the correlations to the primary_variable, ordered by their absolute value.
        """

        # Drop a constant variable if present, not relevant for determining correlations
        if 'const' in variables_to_evaluate.columns:
            variables_to_evaluate.drop('const', axis=1, inplace=True)

        # Placeholder DataFrame that has all variables_to_evaluate as index
        all_correlations_raw = pd.DataFrame(index=variables_to_evaluate.columns)

        # Determine correlation of each exogenous feature with the endogenous feature
        all_correlations_raw[self.CORRELATION] = \
            [primary_variable.corr(variables_to_evaluate.iloc[:, i]) for i in
             range(0, len(variables_to_evaluate.columns))]

        # Reindex exogenous features for their magnitude (absolute correlation) with endogenous variable
        all_correlations_raw[self.ABS_CORRELATION] = abs(all_correlations_raw[self.CORRELATION])
        all_correlations_raw.sort_values(self.ABS_CORRELATION, ascending=False, inplace=True)

        return all_correlations_raw[self.CORRELATION]

    # def stepwise_feature_builder(self, input_target: pd.Series, input_features: pd.DataFrame,
    #                              pre_var_selection: bool = True,
    #                              min_corr: float = 0.05, perf_opt: str = 'r2',
    #                              perf_threshold: float = 0.05) -> dict:
    #     """
    #     input_target: Use the target variable after being pre-processed with 'TimeSeriesPrep'
    #     input_target: Use the target variable after being pre-processed with 'TimeSeriesPrep'
    #     input_features: Use the features after being pre-processed with 'TimeSeriesPrep'
    #     pre_var_selection: If true, then the 'multicollinearity_filter' function from 'CorrelationAnalysis.py' is applied
    #     min_corr: The model building stops if the next variable correlates less than 'min_corr' with remaining residuals
    #     perf_threshold: If the improvement in R-squared is less than 'perf_threshold' the model building stops
    #     multicoll_treshold: The model building stops if all variables are correlated more than x to another variable
    #     """
    #
    #     # Give overview of the inputted features
    #     logger.info("The setup contains %d input_features, covering %d observations.",
    #                 input_features.shape[1],
    #                 input_features.shape[0])
    #
    #     # Apply the variable selection algorithm to remove multicollinearity
    #     if pre_var_selection:
    #         variables_for_analysis = CorrelationAnalysis.multicollinearity_filter(target=input_target,
    #                                                                               all_features=input_features)
    #         import pdb
    #         pdb.set_trace()
    #
    #         X = input_features[variables_for_analysis]
    #         logger.info("The number of input_features is reduced to {k_opt} "
    #                     "as a result of the selection algorithm.".format(k_opt=X.shape[1])
    #                     )
    #     else:
    #         X = input_features
    #
    #     Y = input_target
    #
    #     # At this point the final 'Y' and 'X' have been determined for further processing
    #     # The AR terms are the baseline and need to be excluded from the model building stage
    #
    #     # Check for features that have similar name as dependent variable
    #     ar_check = X.columns.str.contains(Y.name)
    #     ar_cols = list(X.iloc[:, ar_check].columns)
    #
    #     # Keep all exogenous features
    #     non_ar_check = ~X.columns.str.contains(Y.name)
    #     exog_cols = list(X.iloc[:, non_ar_check].columns)
    #     exog_features = X[exog_cols]
    #
    #     # Create baseline AR model
    #     optimized_features = ar_cols  # Initialize 'optimized_features' here for while-loop
    #     baseline_output, baseline_perf, __, __ = self.ols_robust(Y, X[optimized_features])
    #     logger.info('The baseline performance is (R2): {}'.format(baseline_perf[self.R2]))
    #
    #     # Determine residuals of basline AR model, see explanation of procedure below
    #     baseline_yhat = baseline_output[self.IS_FIT]
    #     current_residuals = Y - baseline_yhat  # Initialize 'current_residuals' here for while-loop
    #
    #     # Create the full residual matrix
    #     residual_mat = pd.DataFrame(current_residuals)
    #     residual_mat.columns = [self.AR_MODEL]
    #
    #     # Check the model residuals for normality using the Jarque-Bera test
    #     jb_p_value = stats.jarque_bera(current_residuals)[1]
    #     jb_p_values_mat = pd.DataFrame([jb_p_value])
    #     jb_p_values_mat.columns = [self.AR_MODEL]
    #
    #     residual_correlations = CorrelationAnalysis().correlations(
    #         primary_variable=current_residuals,
    #         variables_to_evaluate=exog_features)
    #
    #     # Initialize performance improvement criterium
    #     previous_perf = 1e-9
    #     performance_improvement = perf_threshold
    #
    #     # Run the specific-to-general algorithm while there are still relevant features and it still improves enough
    #     while (abs(residual_correlations[0]) > min_corr) and (performance_improvement >= perf_threshold):
    #         """
    #         Initialize correlations with the residuals of the AR model. With every added feature the residuals should
    #         decrease a little bit (i.e. better fit). To determine which is the best variable to add next,
    #         we select the feature that correlates strongest with the remaining residuals (of the latest model).
    #         """
    #
    #         # Add the feature that correlates most with last estimated model
    #         optimized_features.append(residual_correlations.index[0])
    #         current_feature_name = residual_correlations.index[0]
    #
    #         # Drop features from full 'exog_features' list so it won't be reselected
    #         exog_features.drop(residual_correlations.index[0], axis=1, inplace=True)
    #
    #         # Fit 'optimized_features' set to the dependent variable (Y)
    #
    #         current_mdl, current_perf, __, __ = self.ols_robust(Y, X[optimized_features])
    #         current_yhat = current_mdl[self.IS_FIT]  # Current fit
    #
    #         # As MSPE is better when lower, take the inverse to make a higher value an improvement
    #         if perf_opt == self.MSPE:
    #             perf_metric = 1 / current_perf[perf_opt]
    #         elif perf_opt == self.R2:
    #             perf_metric = current_perf[perf_opt]
    #         else:
    #             logger.error("Unavailable performance metric specified.")
    #             raise ValueError("Unavailable performance metric specified.")
    #
    #         current_perf = np.round(perf_metric, 4)
    #
    #         # Show the made improvement
    #         logger.info("Current {}:\t{}".format(perf_opt, current_perf))
    #
    #         # Update the remaining residuals with current fit
    #         current_residuals = Y - current_yhat
    #
    #         # Examine the normality of the residuals and update residual- and JB matrix accordingly
    #         current_residuals_JB_pval = stats.jarque_bera(current_residuals)[1]
    #
    #         residual_mat[current_feature_name] = current_residuals
    #         jb_p_values_mat[current_feature_name] = current_residuals_JB_pval
    #
    #         # Update the correlations of the new residuals with the remaining features
    #         residual_correlations = CorrelationAnalysis().correlations(
    #             primary_variable=current_residuals,
    #             variables_to_evaluate=exog_features)
    #
    #         # Update the performance improvement in terms of R-squared (in-sample fit)
    #         performance_improvement = (current_perf / previous_perf) - 1
    #
    #         # If model performance decreases quit the while-loop and use previous model
    #         if performance_improvement < 0:
    #             del optimized_features[-1]
    #             logger.info("Performance decreased, so reverting back to previous model with {} {}."
    #                         .format(perf_opt, previous_perf))
    #             break
    #
    #         # Update the performance
    #         previous_perf = current_perf
    #
    #     return {self.SELECTED_FEATURES: optimized_features,
    #             self.RESIDUALS: residual_mat,
    #             self.JARQUE_BERA_PVALUES: jb_p_values_mat.T}