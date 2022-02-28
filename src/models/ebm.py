import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression,\
    mutual_info_classif
from typing import Union
from sklearn.datasets import make_classification, make_regression
from interpret.glassbox import ExplainableBoostingRegressor, \
    ExplainableBoostingClassifier
from interpret import show


class EBMmi:

    def __init__(self,
                 features: Union[np.ndarray, pd.DataFrame],
                 target: np.ndarray,
                 feature_names: list = None,
                 classification: bool = True
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
        self.mi = mutual_info_classif(features,
                                      target) \
            if classification \
            else mutual_info_regression(
            features, target)
        self.mi_features = pd.DataFrame({
            "features": self.feature_names,
            "mutual_information": self.mi
        }).sort_values(
            by="mutual_information",
            ascending=False)
        self.ebm = ExplainableBoostingClassifier() \
            if self.classification \
            else ExplainableBoostingRegressor()

    def fit(self,
            n_features: int = 650,
            return_model: bool = False):

        self.get_selected_features(n_features)
        self.ebm = self.ebm.fit(self.x_mi, self.target)
        if return_model:
            return self.ebm.fit(self.x_mi, self.target)

    def get_selected_features(self,
                              n: int = 650
                              ) -> Union[pd.DataFrame, np.ndarray]:

        self.sel_features = list(self.mi_features["features"].iloc[:n])
        cols = [col for col in self.feature_names if col in self.sel_features]

        if not self.pddf:
            indices = np.array([i for i, f in enumerate(self.feature_names)
                                if f in cols])

        self.x_mi = self.features.loc[:, cols].copy() \
            if self.pddf \
            else self.features[:, indices].copy()
        return self.x_mi

    def get_sel_features_names(self) -> list:
        return self.sel_features

    def get_mutual_info(self) -> pd.DataFrame:
        return self.mi_features

    def predict(self,
                inputs: Union[np.ndarray,
                              pd.DataFrame]
                ) -> np.ndarray:
        return self.ebm.predict(inputs)

    def predict_proba(self,
                      inputs) -> np.ndarray:
        if self.classification:
            return self.ebm.predict_proba(inputs)[:, 1]
        else:
            raise ValueError("predict_proba not available for regression")

    def plot_mi(self,
                n: int = 30):
        self.mi_features.iloc[:n, :].plot.bar(
            x="features",
            y="mutual_information")

    def explain_global(self):
        show(self.ebm.explain_global())

    def explain_local(self,
                      inputs: Union[np.ndarray,
                                    pd.DataFrame],
                      target: np.ndarray = None
                      ) -> None:

        if inputs.shape[1] > len(self.sel_features):
            # has to be adjusted to also handle np.ndarrays
            cols = [col for col in inputs.columns if col in self.sel_features]
            inputs = inputs.loc[:, cols]

        show(self.ebm.explain_local(inputs, target))

    def get_contributions(self,
                          inputs: Union[np.ndarray,
                                        pd.DataFrame]
                          ) -> np.ndarray:

        if self.classification:
            res = self.ebm.predict_and_contrib(inputs)
            return res[0][:, 1], res[1]
        else:
            return self.ebm.predict_and_contrib(inputs)

    def get_n_imp_contr(self):
        pass

    def plot_b_imp_contr(self):
        pass

    def save_model(self, name: str = "ebm"):
        pass

    def refit(self):
        pass

    def load_model(self):
        pass


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
    ebm_class = EBMmi(X, y, classification=True)
    ebm_regr = EBMmi(X_regr, y_regr, classification=False)

    # fit the models
    ebm_class.fit(n_features=10)
    ebm_regr.fit(n_features=10)

    # plot functions
    ebm_class.plot_mi(n=5)
    ebm_regr.plot_mi(n=5)

    # explain functions
    #ebm_class.explain_global()
    #ebm_regr.explain_global()

    # local explanations
    #ebm_class.explain_local(X, y)
    #ebm_regr.explain_local(X_regr, y_regr)
