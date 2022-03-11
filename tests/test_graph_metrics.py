import unittest
from src.preprocessing.graph_metrics import get_graph_metrics, \
    get_gms_from_pd, pd_to_arrays, is_conn_col
from src.preprocessing.data_loader import flat_to_mat
import numpy as np
import pandas as pd
from typing import List

# checking the is_conn_col() function
test_names = ["1_2", "asf", "as_asd"]

# checking the get_gms_from_pd function
k = 8  # dim of the conn matrix
obs = 10  # observations
conn_data = np.random.normal(
        loc=0.1,
        scale=1.2,
        size=int((k*(k-1)/2)*obs)).reshape(obs, int((k*(k-1)/2)))

conn_data_list = [flat_to_mat(conn_data[i, :]) for i in range(obs)]

conn = pd.DataFrame(
    conn_data,
    columns=[str(i) + "_" + str(j)
             for i in range(k)
             for j in range(i+1, k)]
)


class TestGraphMetris(unittest.TestCase):
    def test_get_graph_metris(self):
        res = get_graph_metrics(conn_data_list)[0]
        self.assertIsInstance(res, pd.DataFrame)

        res = get_graph_metrics(conn_data_list)[0]
        self.assertIsInstance(res, pd.DataFrame)

    def test_is_conn_col(self):
        res = [is_conn_col(x) for x in test_names]
        self.assertEqual(res, [True, False, False])

    def test_pd_to_arrays(self):
        res = pd_to_arrays(conn, conn.columns)
        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], np.ndarray)
        self.assertEqual(len(res), 10)

    def test_get_gms_from_pd(self):
        res = get_gms_from_pd(conn, cols=conn.columns)
        self.assertIsInstance(res, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
