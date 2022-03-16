import unittest
import numpy as np
import shutil
from random import choices
import hdf5storage
from connectome.preprocessing.graph_metrics import pd_to_arrays
import os
import pandas as pd
import tempfile
from connectome.preprocessing.preprocessing_matlab_files import preprocess_mat_files
from connectome.preprocessing.data_preparation import prepare_data
from connectome.models.framework import model_framework
from connectome.models.evaluation import model_evaluation
from connectome.visualization.viz_framework import visualization_framework


def get_fake_data(n: int = 50,
                  k: int = 50):
    """
    little helper function to create connectivity like artificial data

    Examples:
    >>> arrs, excel = get_fake_data(k=5)
    Args:
        n: number of observations
        k: number of brain regions

    Returns:
        a list of numpy arrays (connectivity data) as well as
        a DataFrame (structural data)

    """
    m = int((k*k)/2 - k/2)
    fake_conn_data = np.random.normal(loc=0, scale=2, size=m*n).reshape(n, m)
    colnames = [str(i) + "_" + str(j) for i in range(k) for j in range(i+1, k)]
    df = pd.DataFrame(fake_conn_data, columns=colnames)
    arrays = pd_to_arrays(df, None)

    excel_data = pd.DataFrame()
    excel_data["age"] = choices(range(60, 88), k=n)
    excel_data["sex"] = choices([0, 1], k=n)
    excel_data["edyears"] = choices(range(9, 20), k=n)
    excel_data["Apoe"] = choices([0, 1], k=n)
    excel_data["target"] = choices([0, 1], k=n)
    excel_data["subject_id"] = range(n)
    excel_data["ConnID"] = range(n)

    return arrays, excel_data


def gen_files(dims: tuple = (10, 25, 50, 90, 246),
              obs: int = 50) -> tuple:

    base_path = os.getcwd()
    res = {}
    generated_files = []
    for dim in dims:
        curr_res = {}
        # creating the parent directory for the current dim
        path = os.path.join(base_path, str(dim))
        if not os.path.exists(path):
            os.mkdir(path, 0o777)
        curr_res["main_dir"] = path
        generated_files.append(path)
        os.chdir(path)

        # creating the artificial data
        arrs, excel = get_fake_data(k=dim, n=obs)
        file_name = "example_data_" + str(dim) + ".xlsx"
        excel.to_excel(file_name)

        curr_res["excel_file"] = os.path.join(path, file_name)

        generated_files.append(os.path.join(path, file_name))

        # creating the directory for the conn_data
        path = os.path.join(path, "conn_data")
        curr_res["conn_directory"] = path

        if not os.path.exists(path):
            os.mkdir(path, 0o777)
        os.chdir(path)
        generated_files.append(path)

        curr_res["conn_files"] = []
        for i, arr in enumerate(arrs):
            dict_data = {'Z': arr}
            n = 3 - len(str(i))
            x = ["0" for i in range(n)] + [str(i)]
            id = "".join(x)
            file_name = "Subject" + id + "_Condition001.mat"
            hdf5storage.write(dict_data, '.', file_name,
                              matlab_compatible=True)
            path = os.path.join(path, file_name)
            generated_files.append(path)
            curr_res["conn_files"].append(path)
        res[dim] = curr_res
    return generated_files, res


def clear_gen_files(files: list):
    directories = []
    for f in files:
        if os.path.isdir(f):
            directories.append(f)
        else:
            os.remove(f)

    del files
    directories = sorted(directories, key=lambda x: len(x), reverse=True)

    failed = []
    for dirct in directories:
        try:
            shutil.rmtree(dirct)
        except PermissionError:
            failed.append(dirct)
            print("PermissionError for directory ", dirct)


class TestPipeline(unittest.TestCase):
    def test_pipeline_elnet(self):
        dims = (10, 25, 50)
        obs = 15
        files, res = gen_files(dims=dims, obs=obs)

        for dim in res.values():
            # begin pipeline test
            df = preprocess_mat_files(
                matlab_dir=dim["conn_directory"],
                excel_path=dim["excel_file"])

            classification = True
            columns_drop = ["ConnID", "Apoe", "subject_id"]
            target = "target"
            y_0 = [0]
            y_1 = [1]
            train_size = 0.8
            seed = 1855
            split = True

            X_train, y_train, X_test, y_test = prepare_data(data=df,
                                                            classification=classification,
                                                            columns_drop=columns_drop,
                                                            target=target, y_0=y_0,
                                                            y_1=y_1,
                                                            train_size=train_size,
                                                            seed=seed, split=split)
            model = model_framework(X_train=X_train,
                                    y_train=y_train,
                                    model="elnet",
                                    pretrained=False,
                                    model_path=None,
                                    verbose=0)
            print(model_evaluation(model, X_test, y_test))

            viz = visualization_framework(model=model, X=X_test,
                                          y=y_test, viz_method="elastic_net")


if __name__ == '__main__':
    unittest.main()
