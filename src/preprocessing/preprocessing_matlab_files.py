import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from src.preprocessing.network_aggregation import grouped_conn_mat


def preprocess_mat_files(matlab_dir: str = None,
                         excel_path: str = None,
                         export_file: bool = False,
                         write_dir: str = None,
                         preprocessing_type: str = 'conn',
                         network: str = 'yeo7',
                         upper: bool = True,
                         statistic: str = 'mean',
                         split_size: float = .8,
                         seed: int = 42,
                         file_format: str = "csv") -> None:
    """
    Final function which combines all the other functions to read in
    and transform the data.

    Args:
        matlab_dir: path to matlab files
        excel_path: path to excel list
        export_file: If false return as pd dataframe
        write_dir: path where to write the dataset to if save_file = True
        preprocessing_type: conn for connectivity matrix,
            "aggregation" for aggregated conn matrix, "graph" for graph matrices
        network: Yeo7 or Yeo17 network (only applicable if preprocessing_type = aggregation)
        statistic: Summary statistic to be applied
            - only applicable if preprocessing_type = aggregation
            - one of (mean, max, min and greater_zero)
        upper: boolean whether only upper diagonal elements of connecivity matrices should be used
        split_size: the size of the train dataset (default .8)
        seed: pass an int for reproducibility purposes (default 42)
        file_format: str. Pass "h5" for further modelling in python or "csv" for R (default "csv")
    Returns:
        Two files (a train/test split of datasets) for further use in modelling.
    """

    if matlab_dir is None:
        matlab_dir = input(r'Input your path where the matlab files are stored: ')
    if excel_path is None:
        excel_path = input(r'Input your path where the excel '
                           r'file is stored (with name + ".xlsx"): ')
    if write_dir is None:
        write_dir = input(r'Input your path where '
                          r'to write the final file: ')

    if preprocessing_type == "aggregation":
        statistic = input(r'Input summary statistic: ')

    assert isinstance(matlab_dir, str), "invalid path (matlab files) provided"
    assert isinstance(excel_path, str), "invalid path (excel file) provided"
    assert isinstance(export_file, bool), "invalid datatype for argument export_file"
    assert isinstance(write_dir, str), "invalid path (write_dir) provided"
    assert isinstance(preprocessing_type, str) & \
           (preprocessing_type == "conn" or
            preprocessing_type == "aggregation" or
            preprocessing_type == "graph"), "invalid preprocessing type"
    assert isinstance(upper, bool), "invalid datatype for argument flatten"
    assert isinstance(split_size, float) & (split_size >= 0.0) & \
           (split_size <= 1.0), "invalid path provided"
    assert isinstance(seed, int), "provided seed is no integer"
    assert isinstance(file_format, str) & \
           (file_format == "csv" or file_format == "h5"), "invalid file format selected"

    print('loading files')
    # load matlab files and excel
    res = load_matlab_files(matlab_dir)
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

    elif preprocessing_type == 'graph':
        pass

    print("Creating Final Dataset")
    final_df = create_final_df(file_names=res[1],
                               final_columns=colnames,
                               stacked_matrices=stacked,
                               data_from_excel=delcode_excel)
    if export_file:
        write_to_dir(dataset=final_df,
                     t_direct=write_dir,
                     file_format=file_format)
    else:
        print("Done!")
        return final_df


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

    if not os.path.exists(directory):
        raise FileNotFoundError("invalid directory (matlab files)")

    mat_files_names = os.listdir(directory)
    mat_files_names = [os.path.join(directory, file) for file in mat_files_names]
    conn_matrices = []
    worked = []

    for i in mat_files_names:
        with h5py.File(i, 'r') as f:
            conn_matrices.append(np.array(f.get("Z")))
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
        A flattenened connectivtity matrices as a 1d array
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
        elif preprocessing_type == 'graph':
            pass
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
                 save_file: bool = False,
                 t_direct: str = None,
                 file_format: str = "csv") -> str:
    """
    writes the list of train/test splits to hdf files for future use in python or
    csv for future use in R into the specified directory

    Args:
        dataset: the final dataset to save
        save_file: If false return pd.DataFrame
        t_direct: path where to save the dataframes to
        file_format: The fileformat the data should be saved as (csv of hdf)
            -> input must be csv or h5

    Returns:
        A train and test dataset as csv or hdf file

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

    if save_file:
        filename = os.path.join(t_direct, ("preprocessed_df.h5"
                                           if file_format == "h5"
                                           else "preprocessed_df.csv"))
        if file_format == "h5":
            dataset.to_hdf(filename, key='df', mode='w')
        elif file_format == "csv":
            dataset.to_csv(filename, index=False)
    else:
        return dataset


def main():
    preprocessing_type = input(r'Input preprocessing type: conn, aggregation or grouped: ')
    if preprocessing_type == "aggregation":
        statistic = input(r'Input summary statistic: ')
        preprocess_mat_files(preprocessing_type=preprocessing_type, statistic=statistic)
    else:
        preprocess_mat_files(preprocessing_type=preprocessing_type)


if __name__ == "__main__":
    main()
