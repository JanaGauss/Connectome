# use python -m tests.test_preprocessing_matlab_files to run this test file
# note: you have to run this command in your terminal from the Connectome folder
import unittest
import numpy as np
import logging
import connectome.preprocessing.preprocessing_matlab_files as mth
import os
import pandas as pd

d = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}
excel_data = pd.DataFrame(data=d)

file_list = ["resultsROI_Subject012_Condition001.mat",
             "resultsROI_Subject029_Condition001.mat",
             "resultsROI_Subject090_Condition001"]

stacked_matrices = np.array([[1, 2, 1],
                             [2, 2, 0],
                             [2, 1, 3]])

target_d = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9],
            "IDs": [12, 29, 90], "1_2": [1, 2, 2], "1_3": [2, 2, 1], "2_3": [1, 0, 3]}
target_df = pd.DataFrame(data=target_d)

target_colnames = ['col1', 'col2', 'col3', 'IDs', '1_2', '1_3', '2_3']

conn_matr_colnames = ['1_2', '1_3', '2_3']

conn_matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

stack_matrix = np.array([[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]])

stack_res = np.array([[2, 3, 6],
                      [11, 12, 15]])


class TestMatlabToHdf(unittest.TestCase):
    """
    tests for different functionalities of the preprocessing_matlab_files.py file
    """
    def setUp(self):
        # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)

    def test_write_to_dir(self):

        with self.assertRaises(FileNotFoundError):
            mth.write_to_dir(excel_data, t_direct="no valid directory", file_format="csv")

        with self.assertRaises(AssertionError):
            mth.write_to_dir(excel_data, t_direct=os.getcwd(), file_format="excel")

    def test_get_subject_ids(self):

        target_result = np.array([12, 29, 90])
        result = mth.get_subject_ids(file_list)
        array_eq = np.all(target_result == result)

        self.assertEqual(array_eq, True)

        with self.assertRaises(ValueError):
            test = ["abc", "123", "avt"]
            mth.get_subject_ids(test)

    def test_col_names_conn_matrix(self):
        target_colnames = np.array(conn_matr_colnames)
        test_colnames = np.array(mth.col_names_conn_matrix(3))

        conn_coln_eq = np.all(target_colnames == test_colnames)

        self.assertEqual(conn_coln_eq, True)
        # todo: test that bad input is handled correctly

    def test_flatten_conn_matrix(self):
        target_arr = np.array([2, 3, 6])
        result_arr = mth.flatten_conn_matrix(conn_matrix)

        arr_eq = np.all(target_arr == result_arr)

        self.assertEqual(arr_eq, True)

    def test_col_names_final_df(self):
        test_colnames = np.array(mth.col_names_final_df(excel_data, 3))
        target_cn = np.array(target_colnames)
        coln_eq = np.all(test_colnames == target_cn)

        self.assertEqual(coln_eq, True)

    def test_stack_matrices(self):
        test_list = [conn_matrix, stack_matrix]
        res_test = mth.stack_matrices(test_list)
        stack_eq = np.all(res_test == stack_res)
        self.assertEqual(stack_eq, True)


if __name__ == '__main__':
    unittest.main()
