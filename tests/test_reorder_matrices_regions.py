# use python -m tests.test_reorder_matrices_regions to run this test file
# note: you have to run this command in your terminal from the Connectome folder
import unittest
import numpy as np
import connectome.preprocessing.reorder_matrices_regions as rmr


class TestGroupedConnMat(unittest.TestCase):
    """
    tests for different functionalities of the network_aggregation.py file
    """

    matrices = [np.zeros((246, 246)), np.random.uniform(low=-1, high=1, size=(246, 246))]

    matrices[0][164, 176] = 1080
    matrices[0][186, 187] = 20
    matrices[0][187, 187] = -1521

    def test_reorder_yeo7(self):
        mat_yeo7 = rmr.reorder_matrices_regions(self.__class__.matrices, network="yeo7")
        self.assertEqual(mat_yeo7[0][0, 1], 1080)
        self.assertEqual(mat_yeo7[0][244, 245], 20)
        self.assertEqual(mat_yeo7[0][245, 245], -1521)

        self.assertAlmostEqual(mat_yeo7[1].mean(), self.__class__.matrices[1].mean())
        self.assertAlmostEqual(mat_yeo7[1].min(), self.__class__.matrices[1].min())


    def test_reorder_yeo17(self):
        mat_yeo17 = rmr.reorder_matrices_regions(self.__class__.matrices, network="yeo17")
        self.assertEqual(mat_yeo17[0][0, 1], 1080)

        self.assertAlmostEqual(mat_yeo17[1].mean(), self.__class__.matrices[1].mean())
        self.assertAlmostEqual(mat_yeo17[1].min(), self.__class__.matrices[1].min())

if __name__ == '__main__':
    unittest.main()