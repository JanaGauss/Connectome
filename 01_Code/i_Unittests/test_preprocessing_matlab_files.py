import unittest
import numpy as np
import 00_Unittests/preprocessing_matlab_files as mth
import os
import pandas as pd


class TestMatlabToHdf(unittest.TestCase):
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

    def test_write_to_dir(self):

        with self.assertRaises(FileNotFoundError):
            mth.write_to_dir([], "no valid directory", "csv")

        self.assertEqual(mth.write_to_dir([], os.getcwd(), "excel"), "invalid file format selected")

    def test_get_subject_ids(self):

        target_result = np.array([12, 29, 90])
        result = mth.get_subject_ids(self.__class__.file_list)
        array_eq = (target_result == result).all()

        self.assertEqual(array_eq, True)

        with self.assertRaises(IndexError):
            test = ["abc", "123", "avt"]
            mth.get_subject_ids(test)

#    def test_create_train_test_split(self):
#        pass

    def test_create_final_df(self):
        result_df = mth.create_final_df(self.__class__.file_list,
                                        self.__class__.target_colnames,
                                        self.__class__.stacked_matrices,
                                        self.__class__.excel_data)
        df_eq = all(result_df == self.__class__.target_df)
        self.assertEqual(df_eq, True)

    def test_col_names_conn_matrix(self):
        target_colnames = np.array(self.__class__.conn_matr_colnames)
        test_colnames = np.array(mth.col_names_conn_matrix(3))

        conn_coln_eq = all(target_colnames == test_colnames)

        self.assertEqual(conn_coln_eq, True)
        # todo: test that bad input is handled correctly

    def test_flatten_conn_matrix(self):
        target_arr = np.array([2, 3, 6])
        result_arr = mth.flatten_conn_matrix(self.__class__.conn_matrix)

        arr_eq = all(target_arr == result_arr)

        self.assertEqual(arr_eq, True)

    def test_col_names_final_df(self):
        test_colnames = np.array(mth.col_names_final_df(self.__class__.excel_data, 3))
        target_cn = np.array(self.__class__.target_colnames)
        coln_eq = all(test_colnames == target_cn)

        self.assertEqual(coln_eq, True)

#    def test_stack_matrices(self):
#        pass

#    def test_load_matlab_files(self):
#        pass


if __name__ == '__main__':
    unittest.main()
