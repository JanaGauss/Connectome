#use python -m tests.test_train_test_split to run this test file
import unittest
import logging
import numpy as np
import pandas as pd
from src.preprocessing.preprocessing_matlab_files import create_train_test_split

class TestCreateTrainTestSplit(unittest.TestCase):
    """
    Simple tests for train test split
    """

    def setUp(self):
        # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

    def test_toy_data(self):
        data = pd.DataFrame({
            'target': np.random.choice([0, 1], 100),
            'f1': np.random.normal(0, 10, 100),
            'f2': np.random.poisson(5, 100),
            'f3': np.random.gamma(3, 8, 100),
            'f4': np.random.uniform(0, 10, 100),
            'f5': np.random.beta(2, 5, 100),
            'f6': np.random.normal(0, 10, 100),
            'f7': np.random.poisson(2, 100),
            'f8': np.random.gamma(3, 8, 100),
            'f9': np.random.uniform(0, 10, 100),
            'f10': np.random.beta(2, 5, 100)
        })

        split_size = [.8, 1.5]

        # Test first split size
        df1, df2 = create_train_test_split(data, split_size=split_size[0])
        # Test whether dimensions fit
        self.assertTrue(df1.shape[1] == df2.shape[1])
        self.assertTrue(df1.shape[0] == (df2.shape[0] * 4))

        # Test out of bound splitsize
        # self.assertTrue("split_size out of bounds" in create_train_test_split(data, split_size = split_size[1]))
        with self.assertRaises(AssertionError):
            create_train_test_split(data, split_size=split_size[1])

        # Print Success if no errors
        print("Success! No errors in create_train_test_split!")

if __name__ == '__main__':
    unittest.main()
