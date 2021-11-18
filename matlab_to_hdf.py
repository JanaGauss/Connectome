import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import h5py



def example_av_keys():
    """
    demonstrates what the raw matrix looks like
    displays all available keys or objects that can be imported
    """
    with h5py.File('resultsROI_Subject006_Condition001.mat', 'r') as f:
        return np.array(f.get("Z")), list(f.keys())



def col_names_matrix(n):
    """
    creates the column names for the flattened connecitvity matrix
    """
    return [str(i) + "_" + str(j)  for i in range(1, n+1) for j in range(i+1, n+1)]



def flatten_conn_matrix(matrix):
    """
    turns the connectivity matrix into a 1d array
    """
    
    if not isinstance(matrix, (np.ndarray, np.generic)):
        return "not an ndarray"
    else:
        sh = matrix.shape[0]
        return matrix[np.triu_indices(sh, k = 1)]



def load_matlab_files(directory):
    """
    imports all matlab files from specified directory
    """
    
    try:
        os.chdir(directory)
    except FileNotFoundError:
        print("invalid path")
        return None
    
    mat_files_names = os.listdir()
    conn_matrices = []
    worked = []
    
    for i in mat_files_names:
            
        with h5py.File(i, 'r') as f:
            conn_matrices.append(np.array(f.get("Z")))
            worked.append(i)

    
    
    return conn_matrices, worked



def get_subject_ids(file_names):
    """
    gets the subjectIDs if the filenames correspond to the 
    used format: resultsROI_Subject006_Condition001.mat
    would correspond to subject ID 6
    """
    
    return np.array([int(i.split("Subject",1)[1][0:3]) for i in file_names])


def find_missing_subject(subj_ids, max_subj_id):
    """
    gets missing subject IDs from specified range of IDs
    """

    check = np.array([i for i in range(1, max_subj_id + 1)])

    return np.setdiff1d(check, subj_ids) 




def stack_matrices(matrices):
    
    flattened = []
    for i in matrices:
        #error handling in case one matrix should not work?
        flattened.append(flatten_conn_matrix(i))
        #error handling for stacking
    
    return np.stack(flattened, axis=0)



def main():
	mat_dir = r"C:/Users/leosc/OneDrive/Desktop/WiSe 21-22/Innolab/Test/SBC_01"

	ex_dir = r"C:\Users\leosc\OneDrive\Desktop\WiSe 21-22\Innolab\Test"
	os.chdir(mat_dir)



	#load 
	res = load_matlab_files(r"C:\Users\leosc\OneDrive\Desktop\WiSe 21-22\Innolab\Test\SBC_01")

	#add ID column
	ids = get_subject_ids(res[1])
	ids_added = np.c_[ids, stacked_conn_matrices]

	#load excel
	#Note: There were some rows with Apoe score in the end but no other data
	excel_path = r"C:\Users\leosc\OneDrive\Desktop\WiSe 21-22\Innolab\Test\DELCODE_dataset.xlsx"
	delcode_excel = pd.read_excel(excel_path)


	#create column names
	colnames = ["IDs"]
	colnames = colnames + col_names_matrix(246)

	#creating final df
	final_columns = list(delcode_excel.columns) + colnames
	final_df = np.c_[np.array(delcode_excel), ids_added]
	final_df = pd.DataFrame(final_df, columns = final_columns)

	#getting memory usage
	final_df2.info(memory_usage="deep")

	#first look at DF
	final_df2.shape
	final_df2.head()

	#saving to hdf file
	final_df2.to_hdf('data.h5', key='df', mode='w')

	#checking that it loads correctly
	read_df = pd.read_hdf('data.h5', 'df')
	read_df.head()
	read_df.shape