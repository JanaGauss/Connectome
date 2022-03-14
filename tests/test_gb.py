import unittest
from connectome.models.lgb import GB
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


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

gb_class_not_dir = GB(X, y, classification=True, fit_directly=False)
gb_regr_not_dir = GB(X, y, classification=False, fit_directly=False)
gb_class_dir = GB(X, y, classification=True, fit_directly=True)
gb_regr_dir = GB(X_regr, y_regr, classification=False, fit_directly=True)


class TestGB(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(gb_class_not_dir, GB)
        self.assertIsInstance(gb_regr_not_dir, GB)

    def test_fit_directly(self):
        self.assertIsInstance(gb_class_dir, GB)
        self.assertIsInstance(gb_regr_dir, GB)

    def test_fit(self):
        gb_class_not_dir.fit()
        self.assertEqual(gb_class_not_dir.fitted, True)

        gb_regr_not_dir.fit()
        self.assertEqual(gb_regr_not_dir.fitted, True)

    def test_predict(self):
        y_pred = gb_class_dir.predict(X)
        self.assertIsInstance(y_pred, np.ndarray)

        y_pred = gb_regr_dir.predict(X_regr)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_predict_proba(self):
        y_pred = gb_class_dir.predict_proba(X)
        self.assertIsInstance(y_pred, np.ndarray)

        with self.assertRaises(ValueError):
            gb_regr_dir.predict_proba(X_regr)

    def test_get_feature_importances(self):
        fi = gb_class_dir.get_feature_importances()
        self.assertIsInstance(fi, pd.DataFrame)

        fi = gb_regr_dir.get_feature_importances()
        self.assertIsInstance(fi, pd.DataFrame)

    def test_save_model(self):
        with self.assertRaises(NotImplementedError):
            gb_class_not_dir.save_model('...')

    def test_refit(self):
        with self.assertRaises(NotImplementedError):
            gb_class_not_dir.refit()

    def test_load_model(self):
        with self.assertRaises(NotImplementedError):
            gb_class_not_dir.load_model('...')


if __name__ == '__main__':
    unittest.main()
