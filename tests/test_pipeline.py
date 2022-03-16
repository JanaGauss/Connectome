import unittest
import os
from connectome.preprocessing.preprocessing_matlab_files import preprocess_mat_files
from connectome.preprocessing.data_preparation import prepare_data
from connectome.models.framework import model_framework
from connectome.models.evaluation import model_evaluation
from connectome.visualization.viz_framework import visualization_framework
from connectome.models.brainnet_cnn import preprocess_test_data_for_cnn


DIMS = (10, 25, 50)
OBS = 50
path_dict = {}

for dim in DIMS:
    cr = {}
    cr["matlab_dir"] = os.path.join(os.path.dirname(__file__),
                                    'test_data', str(dim), 'conn_data')
    cr["excel_path"] = os.path.join(os.path.dirname(__file__),
                                    'test_data', str(dim), "example_data_" + str(dim) + ".xlsx")
    path_dict[dim] = cr


class TestPipeline(unittest.TestCase):
    def test_pipeline_elnet(self):

        for value in path_dict.values():
            df = preprocess_mat_files(
                matlab_dir=value["matlab_dir"],
                excel_path=value["excel_path"])

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

    def test_pipeline_cnn(self):

        for value in path_dict.values():
            df = preprocess_mat_files(
                matlab_dir=value["matlab_dir"],
                excel_path=value["excel_path"])

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
                                    model="cnn",
                                    pretrained=False,
                                    model_path=None,
                                    epochs=1,
                                    patience=1)

            X_test, y_test = preprocess_test_data_for_cnn(X_test, y_test, aggregation=False, reorder=False)

            print(model_evaluation(model, X_test, y_test))

            viz = visualization_framework(model=model, X=X_test, y=y_test,
                                          viz_method="feature_attribution", method='saliency', average=True,
                                          ordered=False)


if __name__ == '__main__':
    unittest.main()
