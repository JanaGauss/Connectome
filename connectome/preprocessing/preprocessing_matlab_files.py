"""
this module is used as a framework to prepare raw connectivity matrices for analysis
**Important**: In your excel sheet with subject information, name the id column: "ConnID".
This column will be used to merge the matlab files.
"""
import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from connectome.preprocessing.network_aggregation import grouped_conn_mat
from connectome.preprocessing.graph_metrics import is_conn_col, pd_to_arrays
from typing import Union


def preprocess_mat_files(matlab_dir: str = None,
                         excel_path: str = None,
                         export_file: bool = False,
                         write_dir: str = None,
                         preprocessing_type: str = 'conn',
                         network: str = 'yeo7',
                         upper: bool = True,
                         statistic: str = 'mean',
                         mat_key: str = 'Z',
                         file_format: str = "csv") -> pd.DataFrame:

    """
    Final function which combines all the other functions to read in
    and transform the data.

    Examples:
    >>> # Preprocess Connectivity Matrices to aggregated matrices
    >>> matlab_dir = r"./Data/MatLab" # Enter the directory for the matlab files
    >>> excel_path = r"./Data/DELCODE_dataset_910.xlsx" # Enter the directory for the corresponding excel sheet
    >>> preprocessing_type = 'aggregation'
    >>> write_dir = r"./path_to_save" # ...
    >>> export_file = True # rename to export file
    >>> statistic = 'greater_zero'
    >>> preprocess_mat_files(matlab_dir=matlab_dir, excel_path=excel_path, preprocessing_type=preprocessing_type,
    >>>                      write_dir=write_dir, export_file=export_file, statistic=statistic)


    Args:
        matlab_dir: path to matlab files
        excel_path: path to excel list
        export_file: If false return as pd dataframe
        write_dir: path where to write the dataset to if save_file = True
        preprocessing_type: conn for connectivity matrix,
            "aggregation" for aggregated conn matrix
        network: yeo7 or yeo17 network (only applicable if preprocessing_type = aggregation)
        statistic: Summary statistic to be applied
            - only applicable if preprocessing_type = aggregation
            - one of (mean, max, min and greater_zero)
        upper: boolean whether only upper diagonal elements of connecivity matrices should be used
        mat_key: the key under which the connectivity data is saved in the matlab files
        file_format: str. Pass "h5" for further modelling in python or "csv" for R (default "csv")

    Returns:
        - DataFrame containing the processes matlab files + excel file
        - optionally saves a file (a train/test split of datasets) for further use in modelling.
    """
    if matlab_dir is None:
        matlab_dir = input(r'Input your path where the matlab files are stored: ')
    if excel_path is None:
        excel_path = input(r'Input your path where the excel '
                           r'file is stored (with name + ".xlsx"): ')
    if write_dir is None and export_file:
        write_dir = input(r'Input your path where '
                          r'to write the final file: ')
        assert isinstance(write_dir, str), "invalid path (write_dir) provided"

    assert isinstance(matlab_dir, str), "invalid path (matlab files) provided"
    assert isinstance(excel_path, str), "invalid path (excel file) provided"
    assert isinstance(export_file, bool), "invalid datatype for argument export_file"
    assert isinstance(preprocessing_type, str) & \
           (preprocessing_type == "conn" or
            preprocessing_type == "aggregation"), "invalid preprocessing type"
    assert isinstance(upper, bool), "invalid datatype for argument flatten"
    assert isinstance(file_format, str) & \
           (file_format == "csv" or file_format == "h5"), "invalid file format selected"

    print('loading files')
    # load matlab files and excel
    res = load_matlab_files(matlab_dir, mat_key)
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError("invalid directory (excel file)")
    delcode_excel = pd.read_excel(excel_path)

    print("Starting Preprocessing")
    if preprocessing_type == 'conn':
        # stack matrices
        stacked = stack_matrices(res[0], upper, preprocessing_type)
        # creating colnames and merging into one df
        colnames = col_names_final_df(delcode_excel,
                                      res[0][0].shape[0],
                                      preprocessing_type)
    elif preprocessing_type == 'aggregation':
        grpd_conn_mat = grouped_conn_mat(res[0], network=network, statistic=statistic)

        # stack matrices
        stacked = stack_matrices(grpd_conn_mat, upper, preprocessing_type)
        # creating colnames and merging into one df
        colnames = col_names_final_df(delcode_excel,
                                      grpd_conn_mat[0].shape[0],
                                      preprocessing_type)

    print("Creating Final Dataset")
    final_df = create_final_df(file_names=res[1],
                               final_columns=colnames,
                               stacked_matrices=stacked,
                               data_from_excel=delcode_excel)
    if export_file:
        write_to_dir(dataset=final_df,
                     t_direct=write_dir,
                     file_format=file_format)
    print("Done!")
    return final_df


def load_matlab_files(directory: str,
                      mat_key: str = "Z") -> tuple:
    """
    imports all matlab files from specified directory

    Args:
        directory: Path to Matlab Files
        mat_key: the key under which the connectivity data is saved in the matlab files

    Returns:
        A list where the first argument is the collection of connectivity matrix
        and the 2nd argument is the names of the connectivity matrix

    Raises:
        KeyError: FileNotFoundError
    """

    if not os.path.exists(directory):
        raise FileNotFoundError("invalid directory (matlab files)")

    mat_files_names = os.listdir(directory)
    mat_files_names = [os.path.join(directory, file) for file in mat_files_names]
    conn_matrices = []
    worked = []

    for i in mat_files_names:
        with h5py.File(i, 'r') as f:
            conn_matrices.append(np.array(f.get(mat_key)))
            worked.append(i)

    return conn_matrices, worked


def stack_matrices(matrices: list,
                   upper: bool = True,
                   preprocessing_type: str = 'conn') -> np.ndarray:
    """
    this function stacks the connectivity matrices
    for the subjects upon each other 
    so they can be used in a dataframe

    Args:
        matrices: List of connectivity matrix
        upper: whether only upper diagonal values should be considered
        preprocessing_type: conn for connectivity matrix,
            "aggregation" for aggregated conn matrix, "graph" for graph matrices

    Returns:
        A flattenened np.ndarray of connectivtity matrices
    """
    flattened = []
    for i in matrices:
        # error handling in case one matrix should not work?
        flattened.append(flatten_conn_matrix(i, upper, preprocessing_type))
        # error handling for stacking

    return np.stack(flattened, axis=0)


def flatten_conn_matrix(matrix: np.ndarray,
                        upper: bool = True,
                        preprocessing_type: str = 'conn') -> np.ndarray:
    """
    turns the connectivity matrix into a 1d array

    Args:
        matrix: A connectivity matrix
        upper: whether only the entries above the diagonal should be considered
        preprocessing_type: conn for connectivity matrix,
            "aggregation" for aggregated conn matrix, "graph" for graph matrices

    Returns:
        flattened connectivity matrix as a 1d array
    """
    assert isinstance(matrix, (np.ndarray, np.generic)), "provided matrix is not an ndarray"
    assert isinstance(upper, bool), "invalid option selected - privided input to upper is no bool"

    if upper:
        if preprocessing_type == 'conn':
            sh = matrix.shape[0]
            return matrix[np.triu_indices(sh, k=1)]
        elif preprocessing_type == 'aggregation':
            sh = matrix.shape[0]
            return matrix[np.triu_indices(sh, k=0)]

    else:
        return matrix.flatten()


def col_names_final_df(data_from_excel: pd.DataFrame,
                       shape: int = 246,
                       preprocessing_type: str = 'conn') -> list:
    """
    creates the columns names for the final data frame 
    based on shape / number of columns of the used connectivity matrix


    Args:
        data_from_excel: A pd.Dataframe
        shape: number of columns in connectivity matrix
        preprocessing_type: conn for connectivity matrix,
            "aggregation" for aggregated conn matrix, "graph" for graph matrices

    Returns:
        A list of column names for the final dataset
    """
    assert isinstance(data_from_excel, pd.DataFrame), "provided data_from_excel is no pd.DataFrame"

    colnames = ["IDs"]
    colnames = colnames + col_names_conn_matrix(shape, preprocessing_type)
    final_columns = list(data_from_excel.columns) + colnames
    return final_columns


def col_names_conn_matrix(n: int,
                          preprocessing_type: str = 'conn'):
    """
    creates the column names for the flattened connectivity matrix


    Args:
        n: number of columns in connectivity matrix
        preprocessing_type: conn for connectivity matrix,
            "aggregation" for aggregated conn matrix, "graph" for graph matrices

    Returns:
        Column names for connectivity matrix
    """
    if preprocessing_type == 'conn':
        return [str(i) + "_" + str(j) for i in range(1, n + 1)
                for j in range(i + 1, n + 1)]
    elif preprocessing_type == 'aggregation':
        return [str(i) + "_" + str(j) for i in range(0, n) for j in range(i, n)]


def create_final_df(file_names: list,
                    final_columns: list,
                    stacked_matrices: np.ndarray,
                    data_from_excel: pd.DataFrame) -> pd.DataFrame:
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
    ids_added_df = pd.DataFrame(ids_added) # create df, first column contains IDs

    # final_columns = col_names_final_df(data_from_excel = data_from_excel)
    final_df_0 = data_from_excel.merge(
        ids_added_df, left_on='ConnID', right_on=0)  # merge two data frames by connID column/first column of ids_added_df
    final_df = pd.DataFrame(np.array(final_df_0), columns=final_columns)  # rename columns

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
    if not all(["Subject" in i for i in file_names]):
        raise ValueError("some filenames do not correspond to the Conn format\n"
                         "example for the Conn format: "
                         "resultsROI_Subject006_Condition001.mat")

    return np.array([int(i.split("Subject", 1)[1].split("_")[0]) for i in file_names])


def create_train_test_split(data: pd.DataFrame,
                            split_size: float = .8,
                            seed: int = 42) -> list:
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


def write_to_dir(dataset: pd.DataFrame,
                 t_direct: str = None,
                 file_format: str = "csv") -> None:
    """
    writes the list of train/test splits to hdf files for future use in python or
    csv for future use in R into the specified directory

    Args:
        dataset: the final dataset to save
        t_direct: path where to save the dataframes to
        file_format: The fileformat the data should be saved as (csv of hdf)
            -> input must be csv or h5

    Returns:
        None - saves a csv or hdf file

    Raises:
        FileNotFoundError
    """
    assert isinstance(t_direct, str), "invalid path (write_dir) provided"
    assert isinstance(dataset, pd.DataFrame), "no DataFrame provided"
    assert isinstance(file_format, str) & \
           ((file_format == "csv") | (file_format == "h5")), \
           "invalid file format selected"

    if not os.path.exists(t_direct):
        raise FileNotFoundError("invalid path (write to dir)")

    filename = os.path.join(t_direct, ("preprocessed_df.h5"
                                       if file_format == "h5"
                                       else "preprocessed_df.csv"))

    if file_format == "h5":
        dataset.to_hdf(filename, key='df', mode='w')
    elif file_format == "csv":
        dataset.to_csv(filename, index=False)


def grouped_conn_df(data: pd.DataFrame,
                    regions: list = None,
                    cols: list = None,
                    return_arrays: bool = True,
                    stack_options: dict = None,
                    **kwargs) -> Union[list, pd.DataFrame]:
    """
    function to compute the grouped / aggregated conn matrices from a pd.DataFrame
    Args:
        data: dataFrame containing the conn data
        regions: list of names of the regions of the conn matrix in case reordered
                 IMPORTANT: region names AFTER aggregation needed
        cols: list of columns of the DataFrame data which contain conn data
        return_arrays: whether the aggregated data should be returned in the form
            of arrays or a dataframe
        stack_options: options passed to "stack_matrices"
        **kwargs: anything that´s passed to "grouped_conn_mat"
    Returns:
        list containing the grouped connectivity matrices or dataFrame
    """
    if stack_options is None:
        stack_options = {"upper": True,
                         "preprocessing_type": 'conn'}

    arrays = pd_to_arrays(data, cols)
    grouped_conn = grouped_conn_mat(conn_matrices=arrays,
                                    **kwargs)
    if return_arrays:
        return grouped_conn

    stacked = stack_matrices(grouped_conn, **stack_options)
    p = grouped_conn[0].shape[0]

    if regions is None:
        regions = [str(i+1) + "_" + str(j+1)
                   for i in range(p)
                   for j in range(i+1, p)]

    return pd.DataFrame(stacked, columns=regions)


def main():
    preprocessing_type = input(r'Input preprocessing type: '
                               r'conn, aggregation or grouped: ')
    if preprocessing_type == "aggregation":
        statistic = input(r'Input summary statistic: ')
        preprocess_mat_files(preprocessing_type=preprocessing_type, statistic=statistic)
    else:
        preprocess_mat_files(preprocessing_type=preprocessing_type)


def test_grouped_conn_df():
    # NOTE: use the following to build unittests later
    # checking the get_gms_from_pd function
    k = 246  # dim of the conn matrix
    obs = 10  # observations
    conn = pd.DataFrame(
        np.random.normal(
            loc=0.1,
            scale=1.2,
            size=int((k*(k-1)/2)*obs)).reshape(obs, int((k*(k-1)/2))),
            columns=[str(i) + "_" + str(j)
                     for i in range(k)
                     for j in range(i+1, k)])
    print("data")
    print(conn)
    # check option returning arrays
    res = grouped_conn_df(conn)
    print("result")
    print(res)

    print(len(res))
    print("checking for correctness of shape")
    print([r.shape for r in res])

    # check option returning DataFrame
    res = grouped_conn_df(conn, return_arrays=False)
    print("result DF")
    print(res)

    print(len(res))
    print("checking for correctness of shape")
    print(res.shape)


if __name__ == "__main__":
    test_grouped_conn_df()
