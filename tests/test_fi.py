import unittest
from src.visualization.fi_rf_gb import get_pfi, get_fi
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Union
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.models.lgb import GB


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

models = [lgb_class, lgb_regr, rf, rf_regr, gb_class, gb_regr]


class TestFiPfi(unittest.TestCase):
    def test_get_fi(self):
        fis = []
        for i, mod in enumerate(models):
            fis.append(get_fi(mod, X_regr if i % 2 != 0 else X))

        res = all(map(lambda x: isinstance(x, pd.DataFrame), fis))
        self.assertEqual(res, True)

    def test_get_pfi(self):
        pfis = []
        for i, mod in enumerate(models):
            pfis.append(get_pfi(mod, X_regr if i % 2 != 0 else X,
                                y_regr if i % 2 != 0 else y))

        res = all(map(lambda x: isinstance(x, pd.DataFrame), pfis))
        self.assertEqual(res, True)


if __name__ == '__main__':
    unittest.main()
