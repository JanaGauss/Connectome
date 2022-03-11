from typing import Union
import shap
from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.lgb import GB


class ShapleyLGB:
    """
    class to efficiently use/compute and visualise shapley values
    for a given lgb model
    """
    def __init__(self,
                 model: Union[LGBMModel, GB],
                 data: pd.DataFrame):
        self.model = model.lgbm if isinstance(model, GB) else model
        self.data = data
        self.feature_names = data.columns
        if isinstance(model, LGBMClassifier):
            self.explainer = shap.TreeExplainer(
                model.lgbm if isinstance(model, GB) else model,
                data=data,
                model_output='probability')
        else:
            self.explainer = shap.TreeExplainer(
                model.lgbm if isinstance(model, GB) else model,
                data=data)

        self.shap_values = self.explainer.shap_values(self.data)
        self.abs_shap_values = np.sum(np.abs(self.shap_values), axis=0)

    def plot_single_prediction(self,
                               ind: int
                               ) -> shap.plots._force.AdditiveForceVisualizer:
        """
        visualises the computed shapley values for a single observation
        Args:
            ind: index of the observation / row

        Returns:
            plot of the shapley values

        """
        shap.initjs()
        return shap.force_plot(self.explainer.expected_value,
                               self.shap_values[ind, :],
                               self.data.iloc[ind, :])

    def get_shapley_values(self) -> np.ndarray:
        return self.shap_values

    def get_shapley_values_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.shap_values,
                            columns=self.feature_names)

    def explain_prediction(self,
                           ind: int) -> pd.DataFrame:
        """
        returns the computed shapley values for a given observation
        Args:
            ind: index of the observation / row

        Returns:
            dataframe containing the feature names and associated shapley values
            for the chosen observation

        """
        return pd.DataFrame({
            "features": self.feature_names,
            "shapley_values": self.shap_values[ind, :]
        })

    def depend_plot(self,
                    feature_ind: int) -> None:
        """
        visualises the dependence plot for the selected feature
        Args:
            feature_ind: index of the feature/column

        Returns:
            None
        """
        shap.dependence_plot(feature_ind,
                             self.shap_values,
                             self.data,
                             self.feature_names)

    def summ_plot(self,
                  max_features: int = 25) -> None:
        """

        Args:
            max_features: number of fatures to display

        Returns:
            None

        """
        shap.summary_plot(self.shap_values,
                          self.data,
                          max_display=max_features)

    def shapley_importance(self,
                           n: int = 10):
        """
        returns the n most important features based on the absolute
        value of the shap values

        Args:
            n: the n most important features to return

        Returns:
            DataFrame of the n most important features - including shapley values
            and the index of the respective column

        """
        sh_imp = pd.DataFrame({"features": self.feature_names,
                               "sum_abs_shapley": self.abs_shap_values,
                               "feature_index":
                                   [i for i in range(len(self.feature_names))]})

        self.n_import = sh_imp.sort_values(
            by="sum_abs_shapley",
            ascending=False).iloc[:n, :].copy().reset_index(drop=True)

        return self.n_import

    def plot_n_imp(self,
                   n: int = 10) -> None:
        """
        plots the dependence plots for the n most important features
        (based on the sum of the absolute shapley values)

        Args:
            n: the n most important features to return

        Returns:

        """
        self.shapley_importance(n)

        for i in self.n_import["feature_index"]:
            self.depend_plot(i)

    def update_data(self):
        raise NotImplementedError

    def update_model(self):
        raise NotImplementedError

    def interaction_values(self):
        raise NotImplementedError


if __name__ == "__main__":
    # initialize some models
    lgb_class = LGBMClassifier()
    lgb_regr = LGBMRegressor()

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

    o_lgb_class = GB(X, y, classification=True, fit_directly=True)

    # fit the models
    lgb_class.fit(X, y)
    lgb_regr.fit(X_regr, y_regr)

    # shapley values and analysis for the classification case
    sh_class = ShapleyLGB(lgb_class, X)
    sh_class.summ_plot(5)
    class_imp = sh_class.shapley_importance()
    print(class_imp)
    sh_class.depend_plot(class_imp.iloc[0, 2])

    sh_regr = ShapleyLGB(lgb_regr, X_regr)
    sh_regr.summ_plot(5)
    regr_imp = sh_regr.shapley_importance()
    print(regr_imp)
    sh_regr.depend_plot(regr_imp.iloc[0, 2])

    # shapley values for the GB class
    sh_class = ShapleyLGB(o_lgb_class, X)
    sh_class.summ_plot(5)
    class_imp = sh_class.shapley_importance()
    print(class_imp)
    sh_class.depend_plot(class_imp.iloc[0, 2])
