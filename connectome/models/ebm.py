"""
wrapper for the explainable boosting machine with integrated feature selection based
on the mutual information
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression,\
    mutual_info_classif
from typing import Union
from sklearn.datasets import make_classification, make_regression
from interpret.glassbox import ExplainableBoostingRegressor, \
    ExplainableBoostingClassifier
from interpret import show
import sys
import pickle
import os


class EBMmi:
    """
    Class that wraps the explainable boosting machine and performs feature selection
    before fitting the model based on the mutual information scores of the features
    with the target variable. The integrated feature selection step is necessary
    as the explainable boosting machine is computationally very costly especially
    in the case of high number of features.

    Examples:
    >>> import numpy as np
    >>> import pandas as pd
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
    >>> ebm_class = EBMmi(X, y, classification=True)
    >>> ebm_regr = EBMmi(X_regr, y_regr, classification=False)
    >>>
    >>> # check the size
    >>> print(sys.getsizeof(ebm_class)*1e-6)
    >>>
    >>> # plot functions
    >>> ebm_class.plot_mi(n=5)
    >>> ebm_regr.plot_mi(n=5)
    """
    def __init__(self,
                 features: Union[np.ndarray, pd.DataFrame],
                 target: np.ndarray,
                 feature_names: list = None,
                 classification: bool = True,
                 fit_directly: bool = True,
                 **kwargs
                 ):

        if isinstance(features, pd.DataFrame):
            self.pddf = True
            self.feature_names = features.columns
        elif isinstance(features, np.ndarray):
            self.pddf = False
            if feature_names is None:
                raise ValueError("if using numpy arrays, feature names must "
                                 "be provided in a list")
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
        if fit_directly:
            self.fit(**kwargs)

    def fit(self,
            n_features: int = 650,
            return_model: bool = False):
        if n_features > len(self.feature_names):
            n_features = len(self.feature_names)

        self.get_selected_features(n_features)
        if return_model:
            return self.ebm.fit(self.x_mi, self.target)
        else:
            self.ebm = self.ebm.fit(self.x_mi, self.target)

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
                inputs: pd.DataFrame
                ) -> np.ndarray:

        if not isinstance(inputs, pd.DataFrame):
            raise ValueError("EBM needs a DataFrame as input")

        sel_cols = [col for col in self.feature_names if col in self.sel_features]
        input_data = inputs.loc[:, sel_cols].copy()

        return self.ebm.predict(input_data)

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
                          ) -> Union[np.ndarray, tuple]:

        if self.classification:
            res = self.ebm.predict_and_contrib(inputs)
            return res[0][:, 1], res[1]
        else:
            return self.ebm.predict_and_contrib(inputs)

    def save_model(self, name: str = "ebm_trained"):
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
    ebm_class = EBMmi(X, y, classification=True, fit_directly=True)
    ebm_regr = EBMmi(X_regr, y_regr, classification=False, fit_directly=True)

    # fit the models
    #ebm_class.fit()
    #ebm_regr.fit()

    # check the size
    print(sys.getsizeof(ebm_class)*1e-6)

    # plot functions
    ebm_class.plot_mi(n=5)
    ebm_regr.plot_mi(n=5)

    # predict functions
    pred_class = ebm_class.predict(X)
    print(pred_class)
    pred_regr = ebm_regr.predict(X_regr)
    print(pred_regr)

    # explain functions
    #ebm_class.explain_global()
    #ebm_regr.explain_global()

    ebm_class.save_model("ebm")
    # local explanations
    #ebm_class.explain_local(X, y)
    #ebm_regr.explain_local(X_regr, y_regr)
    with open("ebm", "rb") as input_file:
        ebm_class_2 = pickle.load(input_file)

    pred_class = ebm_class_2.predict(X)
    print(pred_class)

    os.remove("ebm")

