import unittest
from src.models.lgb import GB
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


class TestGB(unittest.TestCase):
    def test_init(self):
        gb = GB(X, y, classification=True, fit_directly=False)
        self.assertIsInstance(gb, GB)

        gb = GB(X_regr, y_regr, classification=False, fit_directly=False)
        self.assertIsInstance(gb, GB)

    def test_fit_directly(self):
        gb = GB(X, y, classification=True, fit_directly=True)
        self.assertIsInstance(gb, GB)

        gb = GB(X_regr, y_regr, classification=False, fit_directly=True)
        self.assertIsInstance(gb, GB)

    def test_fit(self):
        gb = GB(X, y, classification=True, fit_directly=False)
        gb.fit()
        self.assertEqual(gb.fitted, True)

        gb = GB(X_regr, y_regr, classification=False, fit_directly=False)
        gb.fit()
        self.assertEqual(gb.fitted, True)

    def test_predict(self):
        gb = GB(X, y, classification=True, fit_directly=True)
        y_pred = gb.predict(X)
        self.assertIsInstance(y_pred, np.ndarray)

        gb = GB(X_regr, y_regr, classification=False, fit_directly=True)
        y_pred = gb.predict(X_regr)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_predict_proba(self):
        gb = GB(X, y, classification=True, fit_directly=True)
        y_pred = gb.predict_proba(X)
        self.assertIsInstance(y_pred, np.ndarray)

        gb = GB(X_regr, y_regr, classification=False, fit_directly=True)
        with self.assertRaises(ValueError):
            gb.predict_proba(X_regr)

    def test_get_feature_importances(self):
        gb = GB(X, y, classification=True, fit_directly=True)
        fi = gb.get_feature_importances()
        self.assertIsInstance(fi, pd.DataFrame)

    def test_save_model(self):
        gb = GB(X, y, classification=True, fit_directly=False)
        with self.assertRaises(NotImplementedError):
            gb.save_model('...')

    def test_refit(self):
        gb = GB(X, y, classification=True, fit_directly=False)
        with self.assertRaises(NotImplementedError):
            gb.refit()

    def test_load_model(self):
        gb = GB(X, y, classification=True, fit_directly=False)
        with self.assertRaises(NotImplementedError):
            gb.load_model('...')


if __name__ == '__main__':
    unittest.main()
