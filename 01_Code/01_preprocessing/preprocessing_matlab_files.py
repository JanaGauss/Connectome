import os
import numpy as np
import pandas as pd
import h5py
import sklearn


def transform_mat_write_to_hdf(matlab_dir: str, excel_path: str,
                               write_dir: str, flatten: bool = True, split_size: float = .8, seed: int = 42,
                               file_format: str = "csv") -> None:
    """
    final function which combines all the other functions to read in 
    and transform the data.
    Returns a train/test split of datasets for further use in modelling.
    
    Keyword arguments:
    matlab_dir -- path to matlab files
    excel_path -- path to excel list
    write_dir -- path where to write the dataset to
    flatten -- Boolean whether connectivity matrix should be flattened or not <- unused at the Moment!!!
    split_size -- the size of the train dataset (default .8)
    seed -- pass an int for reproducibility purposes (default 42)
    file_format -- str. Pass "hdf" for further modelling in python or "csv" for R (default "csv")
    """

    # load matlab files and excel
    res = load_matlab_files(matlab_dir)
    delcode_excel = pd.read_excel(excel_path)

    # stack matrices
    stacked = stack_matrices(res[0])

    # creating colnames and merging into one df
    colnames = col_names_final_df(delcode_excel, res[0][0].shape[0])
    final_df = create_final_df(file_names=res[1], final_columns=colnames,
                               stacked_matrices=stacked, data_from_excel=delcode_excel)

    # create train test splits
    train, test = create_train_test_split(data=final_df, split_size=split_size, seed=seed)

    write_to_dir(datasets=[train, test], t_direct=write_dir, file_format=file_format)


def load_matlab_files(directory: str) -> list:
    """
    imports all matlab files from specified directory
    """

    try:
        os.chdir(directory)
    except FileNotFoundError:
        print("invalid path")
        return []

    mat_files_names = os.listdir()
    conn_matrices = []
    worked = []

    for i in mat_files_names:
        with h5py.File(i, 'r') as f:
            conn_matrices.append(np.array(f.get("Z")))
            worked.append(i)

    return conn_matrices, worked


def stack_matrices(matrices: list) -> np.ndarray:
    """
    this function stacks the connectivity matrices
    for the subjects upon each other 
    so they can be used in a dataframe 
    """
    flattened = []
    for i in matrices:
        # error handling in case one matrix should not work?
        flattened.append(flatten_conn_matrix(i))
        # error handling for stacking

    return np.stack(flattened, axis=0)


def flatten_conn_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    turns the connectivity matrix into a 1d array
    """

    if not isinstance(matrix, (np.ndarray, np.generic)):
        return "not an ndarray"
    else:
        sh = matrix.shape[0]
        return matrix[np.triu_indices(sh, k=1)]


def col_names_final_df(data_from_excel: pd.DataFrame, shape: int = 246) -> list:
    """
    creates the columns names for the final data frame 
    based on shape / number of columns of the used connectivity matrix
    """
    colnames = ["IDs"]
    colnames = colnames + col_names_conn_matrix(shape)
    final_columns = list(data_from_excel.columns) + colnames
    return final_columns


def col_names_conn_matrix(n: int):
    """
    creates the column names for the flattened connectivity matrix
    """
    return [str(i) + "_" + str(j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]


def create_final_df(file_names: list, final_columns: list,
                    stacked_matrices: np.ndarray, data_from_excel: pd.DataFrame) -> pd.DataFrame:
    """
    this function merges the connectivity matrices, the excel and the subject ids
    """

    ids = get_subject_ids(file_names)
    ids_added = np.c_[ids, stacked_matrices]

    # final_columns = col_names_final_df(data_from_excel = data_from_excel)
    final_df = np.c_[np.array(data_from_excel), ids_added]
    final_df = pd.DataFrame(final_df, columns=final_columns)

    return final_df


def get_subject_ids(file_names: list) -> np.ndarray:
    """
    gets the subjectIDs if the filenames correspond to the 
    used format: resultsROI_Subject006_Condition001.mat
    would correspond to subject ID 6
    """

    return np.array([int(i.split("Subject", 1)[1][0:3]) for i in file_names])


def create_train_test_split(data: pd.DataFrame, split_size: float = .8, seed: int = 42) -> list:
    """
    takes the final data set and splits it into random train and test subsets. 
    Returns a list containing train-test split of inputs
    
    Keyword arguments:
    data -- dataset to be split into train/test
    split_size -- the size of the train dataset (default .8)
    seed -- pass an int for reproducibility purposes
    """
    # assert split size between 0 and 1
    assert 0 <= split_size <= 1, "split_size out of bounds"

    # split into features and target
    #     features = data.drop('target', axis=1)
    #     target = data['target']

    # stratify by the target to ensure equal distribution
    return sklearn.model_selection.train_test_split(data, train_size=split_size, random_state=seed, shuffle=True)


def write_to_dir(datasets: list, t_direct: str, file_format: str = "csv") -> str:
    """
    writes the list of train/test splits to hdf files for future use in python or csv for future use in R 
    in the specified directory

    """

    try:
        os.chdir(t_direct)
    except:
        raise FileNotFoundError("invalid path")

    # Gibts ne elegantere Lösung?
    names = ["train", "test"]
    if file_format == "hdf":
        for i in range(len(datasets)):
            datasets[i].to_hdf(names[i] + '.h5', key='df', mode='w')
    elif file_format == "csv":
        for i in range(len(datasets)):
            datasets[i].to_csv(names[i] + '.csv', index=False)
    else:
        return "invalid file format selected"


def main():
    matlab_dir = r"..."  # enter directories here
    excel_path = r"..."  # enter directories here
    write_dir = r"..."  # enter directories here

    transform_mat_write_to_hdf(matlab_dir=matlab_dir, excel_path=excel_path,
                               write_dir=write_dir)


if __name__ == "__main__":
    main()
