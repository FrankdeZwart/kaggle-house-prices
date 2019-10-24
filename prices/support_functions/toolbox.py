import pandas as pd

def calculate_correlations(primary_variable: pd.Series, variables_to_evaluate: pd.DataFrame) -> pd.DataFrame:
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
    all_correlations_raw['correlation'] = \
        [primary_variable.corr(variables_to_evaluate.iloc[:, i]) for i in
         range(0, len(variables_to_evaluate.columns))]

    # Reindex exogenous features for their magnitude (absolute correlation) with endogenous variable
    all_correlations_raw['abs_correlation'] = abs(all_correlations_raw['correlation'])
    all_correlations_raw.sort_values('abs_correlation', ascending=False, inplace=True)

    return all_correlations_raw['correlation']