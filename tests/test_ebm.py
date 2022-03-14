import unittest
from connectome.models.ebm import EBMmi
import numpy as np
import pandas as pd
from typing import Union
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

ebm_class_not_dir = EBMmi(X, y, classification=True, fit_directly=False)
ebm_regr_not_dir = EBMmi(X, y, classification=False, fit_directly=False)
ebm_class_dir = EBMmi(X, y, classification=True, fit_directly=True)
ebm_regr_dir = EBMmi(X_regr, y_regr, classification=False, fit_directly=True)


class TestEBMwi(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(ebm_class_not_dir, EBMmi)
        self.assertIsInstance(ebm_regr_not_dir, EBMmi)

    def test_fit_directly(self):
        self.assertIsInstance(ebm_class_dir, EBMmi)
        self.assertIsInstance(ebm_regr_dir, EBMmi)

    def test_get_selected_features(self):
        fts = ebm_class_dir.get_selected_features()
        self.assertIsInstance(fts, pd.DataFrame)

        fts = ebm_regr_dir.get_selected_features()
        self.assertIsInstance(fts, pd.DataFrame)

    def test_get_sel_feature_names(self):
        self.assertIsInstance(ebm_class_dir.get_sel_features_names(),
                              list)
        self.assertIsInstance(ebm_regr_dir.get_sel_features_names(),
                              list)

    def test_predict(self):
        y_pred = ebm_class_dir.predict(X)
        self.assertIsInstance(y_pred, np.ndarray)

        y_pred = ebm_regr_dir.predict(X_regr)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_predict_proba(self):
        y_pred = ebm_class_dir.predict_proba(X)
        self.assertIsInstance(y_pred, np.ndarray)

        with self.assertRaises(ValueError):
            ebm_regr_dir.predict_proba(X_regr)

    def test_save_model(self):
        with self.assertRaises(NotImplementedError):
            ebm_class_not_dir.save_model('...')

    def test_refit(self):
        with self.assertRaises(NotImplementedError):
            ebm_class_not_dir.refit()

    def test_load_model(self):
        with self.assertRaises(NotImplementedError):
            ebm_class_not_dir.load_model('...')


if __name__ == '__main__':
    unittest.main()
