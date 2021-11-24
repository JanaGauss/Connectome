import unittest
import numpy as np
import matlab_to_hdf as mth
import os
import pandas as pd


class TestMatlabToHdf(unittest.TestCase):
    def test_write_to_dir(self):
        with self.assertRaises(FileNotFoundError):
            mth.write_to_dir([], "no valid directory", "csv")

        self.assertEqual(mth.write_to_dir([], os.getcwd(), "excel"), "invalid file format selected")

    def test_get_subject_ids(self):
        file_list = ["resultsROI_Subject012_Condition001.mat",
                     "resultsROI_Subject029_Condition001.mat",
                     "resultsROI_Subject090_Condition001"]
        target_result = np.array([12, 29, 90])
        result = mth.get_subject_ids(file_list)
        array_eq = (target_result == result).all()

        self.assertEqual(array_eq, True)

        with self.assertRaises(IndexError):
            test = ["abc", "123", "avt"]
            mth.get_subject_ids(test)

#    def test_create_train_test_split(self):
#        pass

    def test_col_names_conn_matrix(self):
        d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]}
        excel_data = pd.DataFrame(data=d)
        test_colnames = mth.col_names_final_df(excel_data, 3)
        target_colnames = ['col1', 'col2', 'col3', 'IDs', '1_2', '1_3', '2_3']

        array_eq2 = (np.array(target_colnames) == np.array(test_colnames)).all()

        self.assertEqual(array_eq2, True)

    def test_create_final_df(self):
        pass



#    def test_col_names_final_df(self):
#        pass

#    def test_flatten_conn_matrix(self):
#        pass

#    def test_stack_matrices(self):
#        pass

#    def test_load_matlab_files(self):
#        pass


if __name__ == '__main__':
    unittest.main()
