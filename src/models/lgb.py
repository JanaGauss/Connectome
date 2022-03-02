import pandas as pd
import numpy as np
from typing import Union
from sklearn.datasets import make_classification, make_regression
from lightgbm import LGBMClassifier, LGBMRegressor


class GB:
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
        elif feature_names is None and isinstance(features, np.ndarray):
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

    def save_model(self,
                   t_dir: str,
                   name: str = "lgb"
                   ):
        pass

    def refit(self):
        pass

    def load_model(self,
                   path: str):
        pass

    def get_feature_importances(self):
        if not self.fitted:
            raise Exception("Model has not been fitted yet")
        else:
            return pd.DataFrame({
                    "importances": self.lgbm.feature_importances_,
                    "features": self.feature_names
                    }).sort_values(
                    by="importances", ascending=False
                    )


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
    gb_class = GB(X, y, classification=True, fit_directly=True)
    gb_regr = GB(X_regr, y_regr, classification=False, fit_directly=True)

    # fit the models
    #ebm_class.fit()
    #ebm_regr.fit()

    # get_fis
    print(gb_class.get_feature_importances())
    print(gb_regr.get_feature_importances())
