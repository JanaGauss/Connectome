import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split


def transform_mat_write_to_hdf(matlab_dir: str, excel_path: str,
                               write_dir: str, flatten: bool = True, split_size: float = .8, seed: int = 42,
                               file_format: str = "csv") -> None:
    """
    Final function which combines all the other functions to read in
    and transform the data.

    
    Args:
        matlab_dir: path to matlab files
        excel_path: path to excel list
        write_dir: path where to write the dataset to
        flatten: Boolean whether connectivity matrix should be flattened or not <- unused at the Moment!!!
        split_size: the size of the train dataset (default .8)
        seed: pass an int for reproducibility purposes (default 42)
        file_format: str. Pass "hdf" for further modelling in python or "csv" for R (default "csv")

    Returns:
        Two files (a train/test split of datasets) for further use in modelling.
    """
    assert isinstance(matlab_dir, str), "invalid path (matlab files) provided"
    assert isinstance(excel_path, str), "invalid path (excel file) provided"
    assert isinstance(write_dir, str),  "invalid path (write_dir) provided"
    assert isinstance(flatten, bool),   "invalid datatype for argument flatten"
    assert isinstance(split_size, float) & (split_size >= 0.0) & (split_size <= 1.0), "invalid path provided"
    assert isinstance(seed, int),       "provided seed is no integer"
    assert isinstance(file_format) & (file_format == "csv" or file_format == "hdf"), "invalid file format selected"

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


def load_matlab_files(directory: str) -> tuple:
    """
    imports all matlab files from specified directory

    Args:
        directory: Path to Matlab Files

    Returns:
        A list where the first argument is the collection of connectivity matrix
        and the 2nd argument is the names of the connectivity matrix

    Raises:
        KeyError: FileNotFoundError
    """

    try:
        os.chdir(directory)
    except FileNotFoundError:
        print("invalid directory (matlab files)")
        raise

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

    Args:
        matrices: List of connectivity matrix

    Returns:
        A flattenened np.ndarray of connectivtity matrices
    """
    flattened = []
    for i in matrices:
        # error handling in case one matrix should not work?
        flattened.append(flatten_conn_matrix(i))
        # error handling for stacking

    return np.stack(flattened, axis=0)


def flatten_conn_matrix(matrix: np.ndarray, upper: bool = True) -> np.ndarray:
    """
    turns the connectivity matrix into a 1d array

    Args:
        matrix: A connectivity matrix
        upper: whether only the entries above the diagonal should be considered

    Returns:
        A flattenened connectivtity matrices as a 1d array
    """
    assert isinstance(matrix, np.ndarray), "provided matrix is not an ndarray"
    assert isinstance(upper, bool), "invalid option selected - privided input to upper is no bool"


    if upper:
        if not isinstance(matrix, (np.ndarray, np.generic)):
            raise ValueError("not an ndarray")
        else:
            sh = matrix.shape[0]
            return matrix[np.triu_indices(sh, k=1)]
    else:
        if not isinstance(matrix, (np.ndarray, np.generic)):
            raise ValueError("not an ndarray")
        else:
            return matrix.flatten()


def col_names_final_df(data_from_excel: pd.DataFrame, shape: int = 246) -> list:
    """
    creates the columns names for the final data frame 
    based on shape / number of columns of the used connectivity matrix

    Args:
        data_from_excel: A pd.Dataframe
        shape: number of columns in connectivity matrix

    Returns:
        A list of column names for the final dataset
    """
    assert isinstance(data_from_excel, pd.DataFrame), "provided data_from_excel is no pd.DataFrame"

    colnames = ["IDs"]
    colnames = colnames + col_names_conn_matrix(shape)
    final_columns = list(data_from_excel.columns) + colnames
    return final_columns


def col_names_conn_matrix(n: int):
    """
    creates the column names for the flattened connectivity matrix

    Args:
        n: number of columns in connectivity matrix

    Returns:
        Column names for connectivity matrix
    """
    return [str(i) + "_" + str(j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]


def create_final_df(file_names: list, final_columns: list,
                    stacked_matrices: np.ndarray, data_from_excel: pd.DataFrame) -> pd.DataFrame:
    """
    this function merges the connectivity matrices, the excel and the subject ids

    Args:
        file_names:  list of matlab file names
        final_columns:  list of final column names
        stacked_matrices:  a stacked connectivity matrix
        data_from_excel:  a pd Dataframe with extra information on patients


    Returns:
        A Merged dataframe of connectivity matrix + patient information
    """
    assert isinstance(file_names, list), "no list of file names provided"
    assert isinstance(final_columns, list), "no list of final column names provided"
    assert isinstance(stacked_matrices, np.ndarray), "provided connectivity matrices are no array"
    assert isinstance(data_from_excel, pd.DataFrame), "provided data_from_excel is no pd.DataFrame"

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

    Args:
        file_names:  list of matlab file names

    Returns:
        A np.ndarray in a readable format
    """
    assert isinstance(file_names, list), "no list of file names provided"

    return np.array([int(i.split("Subject", 1)[1][0:3]) for i in file_names])


def create_train_test_split(data: pd.DataFrame, split_size: float = .8, seed: int = 42) -> list:
    """
    takes the final data set and splits it into random train and test subsets. 
    Returns a list containing train-test split of inputs
    
    Args:
        data: dataset to be split into train/test
        split_size: the size of the train dataset (default .8)
        seed: pass an int for reproducibility purposes

    Returns:
        A list containing train-test split of inputs
    """
    # assert split size between 0 and 1
    assert 0 <= split_size <= 1, "split_size out of bounds"
    assert isinstance(data, pd.DataFrame), "no DataFrame provided"
    assert isinstance(seed, int), "provided seed is no integer"

    # split into features and target
    #     features = data.drop('target', axis=1)
    #     target = data['target']

    # stratify by the target to ensure equal distribution
    return train_test_split(data, train_size=split_size, random_state=seed, shuffle=True)


def write_to_dir(datasets: list, t_direct: str, file_format: str = "csv") -> str:
    """
    writes the list of train/test splits to hdf files for future use in python or csv for future use in R 
    in the specified directory

    Args:
        datasets: a list of datasets
        t_direct: path where to save the dataframes to
        file_format: The fileformat the data should be saved as (csv of hdf) -> input must be csv or h5

    Returns:
        A train and test dataset as csv or hdf file

    Raises:
        FileNotFoundError
    """
    assert isinstance(t_direct, str), "invalid path (write_dir) provided"
    assert isinstance(datasets, list), "no list of datasets provided"
    assert isinstance(file_format) & (file_format == "csv" or file_format == "h5"), "invalid file format selected"

    try:
        os.chdir(t_direct)
    except FileNotFoundError:
        print("invalid path (write to dir)")
        raise

    # Gibts ne elegantere Lösung?
    names = ["train", "test"]
    names = list(map(lambda k: k + "." + file_format, names))
    if file_format == "h5":
        for i, dataset in enumerate(datasets):
            dataset.to_hdf(names[i], key='df', mode='w')
    elif file_format == "csv":
        for i, dataset in enumerate(datasets):
            dataset.to_csv(names[i], index=False)


def main():
    matlab_dir = input(r'Input your path where the matlab files are stored:')
    excel_path = input(r'Input your path where the excel file is stored (with name + ".xlsx"):')
    write_dir = input(r'Input your path where to write the final file:')

    transform_mat_write_to_hdf(matlab_dir=matlab_dir, excel_path=excel_path,
                               write_dir=write_dir)


if __name__ == "__main__":
    main()
