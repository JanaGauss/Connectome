"""
Framework function for modelling
"""
import pandas as pd
import keras
import tensorflow as tf
from keras.models import load_model
import pickle
from src.models.brainnet_cnn import model_brainnet_cnn
from src.models.ebm import EBMmi
from src.models.lgb import GB
from src.models.pipeline_elastic_net import model_elastic_net
from src.models.pipeline_RF import run_random_forest


def model_framework(X_train, y_train,  # Brauchen wir die 2 für pretrained models?
                    model: str,
                    pretrained: bool = True,
                    model_path: str = None,
                    **kwargs):
    """
    Function that lets the user decide what type of model he wants to use for his data and which parameters to use.
    Args:
        X_train: The training dataset
        y_train: The true labels
        pretrained: If a new model should be trained or a pretrained model should be used
                    (True/False, default True uses pretrained model
        model: A string to enter which model to use ("elnet" = elastic net, "gboost" = gradient boosting,
                                                     "rf" = random forest, "cnn" = convolutional neural network,
                                                     "ebm" = explainable boosting machine)
        model_path: Full path to the folder where the selected pretrained model is stored
        **kwargs: Additional parameters and options depending on the selected model
    Returns:
        A fitted model

    Raises:
        FileNotFoundError
        IOError
    """
    # to do: gboost, rf, pretrained models aufrufen? assert pd.dataframe?
    assert isinstance(model, str), "invalid option, must be string"
    assert model in ["elnet", "gboost", "rf", "cnn"], "please provide a valid model (elnet, gboost, rf or cnn)"
    assert isinstance(pretrained, bool), "invalid pretrained, must be True/False"
    if pretrained:
        pass
    else:
        assert len(X_train) == len(y_train), "X_train and y_train must be the same length"

    if len(set(y_train)) == 2:  # Checking whether classification or regression is needed
        classification = True  # machen wir das so, Problem mit fällen wo 2 Label ls negativ gewerted werden oder so

    # loading pretrained models
    if pretrained:
        if not os.path.exists("model_path"):
            raise FileNotFoundError("File not found, please specify exact name and make sure the location is correct")
        if model == "elnet":
            rmodel = pickle.load(open(model_path, 'rb'))
        elif model == "gboost":
            rmodel = pickle.load(open(model_path, 'rb'))
        elif model == "ebm":
            rmodel = pickle.load(open(model_path, 'rb'))
        elif model == "rf":
            rmodel = pickle.load(open(model_path, 'rb'))
        else:
            try:
                rmodel = load_model(model_path)
            except IOError:
                print("File not found, please specify exact name and make sure the location is correct")
                raise

    else:  # training new models
        if model == "elnet":
            rmodel = model_elastic_net(X_train, y_train, classification, **kwargs)

        elif model == "ebm":
            rmodel = EBMmi(X_train, y_train, classification=classification, **kwargs)
        elif model == "gboost":
            rmodel = GB(X_train, y_train, classification=True)

        elif model == "rf":
            rmodel = run_random_forest(X_train, y_train, classification, **kwargs)

        else:
            rmodel = model_brainnet_cnn(X_train, y_train, **kwargs)

        # Missing: save model to disk

    return rmodel
