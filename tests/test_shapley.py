import unittest
from lightgbm import LGBMClassifier, LGBMRegressor
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from connectome.models.lgb import GB
from connectome.visualization.lgb_shapley import ShapleyLGB
from shap.plots._force import AdditiveForceVisualizer

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

gb_class = GB(X, y, classification=True, fit_directly=True)
gb_regr = GB(X_regr, y_regr, classification=False, fit_directly=True)

# fit the models
lgb_class.fit(X, y)
lgb_regr.fit(X_regr, y_regr)

sh_class = ShapleyLGB(lgb_class, X)
sh_regr = ShapleyLGB(lgb_regr, X_regr)
sh_class_GB = ShapleyLGB(gb_class, X)
sh_regr_GB = ShapleyLGB(gb_regr, X_regr)


class TestShapleyGB(unittest.TestCase):
    def test_plot_single_prediction(self):
        plot = sh_class.plot_single_prediction(10)
        self.assertIsInstance(plot, AdditiveForceVisualizer)

        plot = sh_regr.plot_single_prediction(10)
        self.assertIsInstance(plot, AdditiveForceVisualizer)

        plot = sh_class_GB.plot_single_prediction(10)
        self.assertIsInstance(plot, AdditiveForceVisualizer)

        plot = sh_regr_GB.plot_single_prediction(10)
        self.assertIsInstance(plot, AdditiveForceVisualizer)

    def test_get_shapley_values(self):
        expl = sh_class.get_shapley_values()
        self.assertIsInstance(expl, np.ndarray)

        expl = sh_regr.get_shapley_values()
        self.assertIsInstance(expl, np.ndarray)

        expl = sh_class_GB.get_shapley_values()
        self.assertIsInstance(expl, np.ndarray)

        expl = sh_regr_GB.get_shapley_values()
        self.assertIsInstance(expl, np.ndarray)

    def test_get_shapley_values_df(self):
        expl = sh_class.get_shapley_values_df()
        self.assertIsInstance(expl, pd.DataFrame)

        expl = sh_regr.get_shapley_values_df()
        self.assertIsInstance(expl, pd.DataFrame)

        expl = sh_class_GB.get_shapley_values_df()
        self.assertIsInstance(expl, pd.DataFrame)

        expl = sh_regr_GB.get_shapley_values_df()
        self.assertIsInstance(expl, pd.DataFrame)

    def test_explain_prediction(self):
        expl = sh_class.explain_prediction(10)
        self.assertIsInstance(expl, pd.DataFrame)

        expl = sh_regr.explain_prediction(10)
        self.assertIsInstance(expl, pd.DataFrame)

        expl = sh_class_GB.explain_prediction(10)
        self.assertIsInstance(expl, pd.DataFrame)

        expl = sh_regr_GB.explain_prediction(10)
        self.assertIsInstance(expl, pd.DataFrame)

    def test_shapley_importance(self):
        imp = sh_class.shapley_importance()
        self.assertIsInstance(imp, pd.DataFrame)

        imp = sh_regr.shapley_importance()
        self.assertIsInstance(imp, pd.DataFrame)

        imp = sh_class_GB.shapley_importance()
        self.assertIsInstance(imp, pd.DataFrame)

        imp = sh_regr_GB.shapley_importance()
        self.assertIsInstance(imp, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
