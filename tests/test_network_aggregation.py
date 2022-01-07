# use python -m tests.test_network_aggregation to run this test file
# note: you have to run this command in your terminal from the Connectome folder
import unittest
import numpy as np
import src.preprocessing.network_aggregation as aggregation


class TestGroupedConnMat(unittest.TestCase):
    """
    tests for different functionalities of the network_aggregation.py file
    """

    matrices = [np.zeros((246, 246)), np.random.uniform(low=-1, high=1, size=(246, 246))]

    matrices[0][5, 6] = 1080
    matrices[0][155, 156] = 20
    matrices[0][212, 213] = -1521


    def test_yeo7_mean(self):
        mat_yeo7_mean = aggregation.grouped_conn_mat(self.__class__.matrices, network = "yeo7", statistic = 'mean')
        self.assertAlmostEqual(mat_yeo7_mean[0][0,0], -1)
        self.assertAlmostEqual(mat_yeo7_mean[0][7,3], 1)

    def test_yeo7_min(self):
        mat_yeo7_min = aggregation.grouped_conn_mat(self.__class__.matrices, network="yeo7", statistic='min')
        self.assertEqual(mat_yeo7_min[0].min(), -1521)
        self.assertGreater(mat_yeo7_min[1].min(), -1)
        self.assertLess(mat_yeo7_min[1].min(), 1)

    def test_yeo17_mean(self):
        mat_yeo17_mean = aggregation.grouped_conn_mat(self.__class__.matrices, network="yeo17", statistic='mean')
        self.assertAlmostEqual(mat_yeo17_mean[0][0, 0], -1)
        self.assertAlmostEqual(mat_yeo17_mean[0][17, 6], 4)
        self.assertGreater(mat_yeo17_mean[1].min(), -1)
        self.assertLess(mat_yeo17_mean[1].max(), 1)

    def test_yeo17_max(self):
        mat_yeo17_max = aggregation.grouped_conn_mat(self.__class__.matrices, network="yeo17", statistic='max')
        self.assertEqual(mat_yeo17_max[0].max(), 1080)
        self.assertGreater(mat_yeo17_max[1].max(), -1)
        self.assertLess(mat_yeo17_max[1].max(), 1)

if __name__ == '__main__':
    unittest.main()





