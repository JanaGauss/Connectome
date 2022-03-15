"""
contains functions to compute / obtain the basic feature importances and permutation
feature importances from gradient boosting and random forest models
"""


import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Union
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from connectome.models.lgb import GB


def get_fi(model: Union[lgb.LGBMClassifier,
                        lgb.LGBMRegressor,
                        RandomForestClassifier,
                        RandomForestRegressor,
                        GB],
           data: pd.DataFrame,
           feature_names: list = None,
           n: int = 10
           ) -> pd.DataFrame:
    """
    obtains the feature importances from an lgb or RF model

    Examples:
    >>>from connectome.visualization.fi_rf_gb import get_fi
    >>>from sklearn.datasets import make_classification
    >>>import pandas as pd
    >>>from lightgbm import LGBMClassifier
    >>>
    >>># initialize the model
    >>>lgb_class = lgb.LGBMClassifier()
    >>>
    >>># create synthetic data
    >>>X, y = make_classification(n_informative=10)
    >>>X = pd.DataFrame(X, columns=["feature_" + str(i) for i in range(X.shape[1])])
    >>># fit the model
    >>>lgb_class.fit(X, y)
    >>>
    >>># obtain FIs
    >>>fis = get_fi(mod, X)
    >>>fis.plot.bar(x='features', y='importances')
    Args:
        model: lgb or RF model of which the feature importances should be obtained
        data: DataFrame to get the column / feature names
        feature_names: in case data is not in form of a pd.DataFrame
        n: the first n FIs to be obtained - in a descending order

    Returns:
        a pandas DataFrame containing the feature importances and names

    """
    if isinstance(model, GB):
        model = model.lgbm

    return pd.DataFrame({
        "importances": model.feature_importances_,
        "features": data.columns
        if feature_names is None
        and isinstance(data, pd.DataFrame)
        else feature_names}
        ).sort_values(
        by="importances", ascending=False
        ).iloc[:n, :]


def get_pfi(model: Union[lgb.LGBMClassifier,
                         lgb.LGBMRegressor,
                         RandomForestClassifier,
                         RandomForestRegressor],
            x_val: Union[np.ndarray, pd.DataFrame],
            y_val: np.ndarray,
            feature_names: list = None,
            n: int = 10,
            repeats: int = 30
            ) -> pd.DataFrame:
    """
    obtains the permutation feature importances from an lgb or RF model

    Examples:
    >>>from connectome.visualization.fi_rf_gb import get_pfi
    >>>from sklearn.datasets import make_classification
    >>>import pandas as pd
    >>>from lightgbm import LGBMClassifier
    >>>
    >>># initialize the model
    >>>lgb_class = lgb.LGBMClassifier()
    >>>
    >>># create synthetic data
    >>>X, y = make_classification(n_informative=10)
    >>>X = pd.DataFrame(X, columns=["feature_" + str(i) for i in range(X.shape[1])])
    >>># fit the model
    >>>lgb_class.fit(X, y)
    >>>
    >>># obtain FIs
    >>>pfis = get_pfi(mod, X, y, repeats=2)
    >>>pfis.plot.bar(x='features', y='importances_mean')
    Args:
        model: lgb or RF model of which the PFIs should be obtained
        x_val: DataFrame or array containing features of held out data
        y_val target variable for held out data
        feature_names: in case data is not in form of a pd.DataFrame
        n: the first n PFIs to be obtained - in a descending order
        repeats: how often the PFI calculation should be repeated

    Returns:
        a pandas DataFrame containing the PFIs and feature names

    """
    if isinstance(model, GB):
        model = model.lgbm

    r = permutation_importance(
        model, x_val, y_val,
        n_repeats=repeats,
        random_state=0)

    return pd.DataFrame({
        "importances_mean": r["importances_mean"],
        "importances_std": r["importances_std"],
        "features": x_val.columns
        if feature_names is None
        and isinstance(x_val, pd.DataFrame)
        else feature_names}
        ).sort_values(
        by="importances_mean", ascending=False
        ).iloc[:n, :]


if __name__ == "__main__":
    # initialize some models
    lgb_class = lgb.LGBMClassifier()
    lgb_regr = lgb.LGBMRegressor()
    rf = RandomForestClassifier()
    rf_regr = RandomForestRegressor()

    # create synthetic data
    X, y = make_classification(n_informative=10)
    X = pd.DataFrame(
        X,
        columns=["feature_" + str(i)
                 for i in range(X.shape[1])])
    X_regr, y_regr = make_regression(n_informative=15)
    X_regr = pd.DataFrame(
        X_regr,
        columns=["feature_" + str(i)
                 for i in range(X_regr.shape[1])])

    gb_regr = GB(X_regr, y_regr, classification=False, fit_directly=True)
    gb_class = GB(X, y, classification=True, fit_directly=True)


    # fit the models
    lgb_class.fit(X, y)
    lgb_regr.fit(X_regr, y_regr)
    rf.fit(X, y)
    rf_regr.fit(X_regr, y_regr)

    # obtain FIs
    plot_fis = False
    for i, mod in enumerate([lgb_class, lgb_regr, rf, rf_regr, gb_class, gb_regr]):
        if plot_fis:
            get_fi(mod, X_regr if i % 2 != 0 else X).plot.bar(
                x='features', y='importances')
        else:
            print(get_fi(mod, X_regr if i % 2 != 0 else X))

    # obtain PFIs
    plot_pfis = False
    for i, model in enumerate([lgb_class, lgb_regr, rf, rf_regr, gb_class, gb_regr]):
        if plot_fis:
            get_pfi(model, X_regr if i % 2 != 0 else X, y_regr
                    if i % 2 != 0 else y, repeats=2).plot.bar(
                    x='features', y='importances_mean')
        else:
            print(get_pfi(model, X_regr if i % 2 != 0 else X, y_regr
                  if i % 2 != 0 else y, repeats=2))
