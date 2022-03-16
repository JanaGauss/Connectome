"""
wrapper for gradient boosting classification and regression models
"""

import pandas as pd
import numpy as np
import pickle
from typing import Union
from sklearn.datasets import make_classification, make_regression
from lightgbm import LGBMClassifier, LGBMRegressor


class GB:
    """
    wrapper/class for gradient boosting models

    Examples:
    >>> from connectome.models.lgb import GB
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # create synthetic data
    >>> X, y = make_classification(n_informative=15)
    >>> X = pd.DataFrame(
    >>>     X,
    >>>     columns=["feature_" + str(i)
    >>>              for i in range(X.shape[1])])
    >>> X_regr, y_regr = make_regression(n_features=20, n_informative=15)
    >>> X_regr = pd.DataFrame(
    >>>     X_regr,
    >>>     columns=["feature_" + str(i)
    >>>              for i in range(X_regr.shape[1])])
    >>>
    >>> # initialize some models
    >>> gb_class = GB(X, y, classification=True)
    >>> gb_regr = GB(X_regr, y_regr, classification=False)
    >>>
    >>> # get_fis
    >>> print(gb_class.get_feature_importances())
    >>> print(gb_regr.get_feature_importances())
    """
    def __init__(self,
                 features: Union[np.ndarray, pd.DataFrame],
                 target: np.ndarray,
                 feature_names: list = None,
                 classification: bool = True,
                 fit_directly: bool = True,
                 dart: bool = False,
                 **kwargs
                 ):

        if isinstance(features, pd.DataFrame):
            self.pddf = True
            self.feature_names = features.columns
        else:
            self.pddf = False
            self.feature_names = feature_names

        self.features = features
        self.target = target
        self.classification = classification
        self.lgbm = LGBMClassifier(boosting_type='dart' if dart else 'gbdt') \
            if self.classification \
            else LGBMRegressor(boosting_type='dart' if dart else 'gbdt')
        self.fitted = False
        if fit_directly:
            self.fit(**kwargs)
            self.fitted = True

    def fit(self,
            **kwargs):

        self.lgbm = self.lgbm.fit(self.features,
                                  self.target,
                                  **kwargs)
        self.fitted = True

    def predict(self,
                inputs: Union[np.ndarray,
                              pd.DataFrame]
                ) -> np.ndarray:
        return self.lgbm.predict(inputs)

    def predict_proba(self,
                      inputs) -> np.ndarray:
        if self.classification:
            return self.lgbm.predict_proba(inputs)[:, 1]
        else:
            raise ValueError("predict_proba not available for regression")

    def get_feature_importances(self):
        """
        method to get the ordered feature importances in the form of a DataFrame
        Returns:

        """
        if not self.fitted:
            raise Exception("Model has not been fitted yet")
        else:
            return pd.DataFrame({
                    "importances": self.lgbm.feature_importances_,
                    "features": self.feature_names
                    }).sort_values(
                    by="importances", ascending=False
                    )

    def save_model(self, name: str = "lgb"):
        with open(name, "wb") as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    # create synthetic data
    X, y = make_classification(n_informative=15)
    X = pd.DataFrame(
        X,
        columns=["feature_" + str(i)
                 for i in range(X.shape[1])])
    X_regr, y_regr = make_regression(n_features=20, n_informative=15)
    X_regr = pd.DataFrame(
        X_regr,
        columns=["feature_" + str(i)
                 for i in range(X_regr.shape[1])])

    # initialize some models
    gb_class = GB(X, y, classification=True)
    gb_regr = GB(X_regr, y_regr, classification=False)

    # get_fis
    print(gb_class.get_feature_importances())
    print(gb_regr.get_feature_importances())
