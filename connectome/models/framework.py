"""
Framework to open up saved models as well as train and save new models
"""

import pandas as pd
import keras
import tensorflow as tf
import os
from keras.models import load_model
from keras.models import save
import pickle
from connectome.models.brainnet_cnn import model_brainnet_cnn
from connectome.models.ebm import EBMmi
from connectome.models.lgb import GB
from connectome.models.pipeline_elastic_net import model_elastic_net
from connectome.models.pipeline_RF import run_random_forest


def model_framework(X_train, y_train,
                    model: str,
                    pretrained: bool = True,
                    model_path: str = None,
                    save: bool = False,
                    t_direct: str = None,
                    **kwargs):
    """
    Function that lets the user decide what type of model he wants to use for his data and which parameters to use. 
    See the documentation of the modelling functions for more information on further parameters that can be specified.

    Examples:
    >>> # CNN:
    >>> # Different optimizer and loss function
    >>> import tensorflow as tf
    >>> from tf.keras.optimizers import Nadam
    >>> from tf.keras.losses import Hinge
    >>> model = model_framework(X_train = X_train,
                                y_train = y_train,
                                model = "cnn",
                                pretrained = False,
                                model_path = None,
                                epochs = 500,
                                patience = 10,
                                optimizer= Nadam(),
                                loss= Hinge())
    >>> 
    >>> # Elastic Net:
    >>> # classification task:
    >>> model = model_framework(X_train = X_train,
                                y_train = y_train,
                                model = "elnet",
                                pretrained = False,
                                n_alphas_logreg = 5, 
                                cv_logreg = 3, 
                                l1_ratios_logreg = [0.0, 0.1])
    >>> # you can transform the values before modelling, e.g. take the absolute values (remember to also transform X_test for evaluation):
    >>> X_train = prepare_data_elastic_net(data = X_train,
							 option = "abs")
    >>> X_test = prepare_data_elastic_net(data = X_test,
							 option = "abs")
    >>> # regression task:
    >>> model = model_framework(X_train = X_train,
                                y_train = y_train,
                                model = "elnet",
                                pretrained = False,
                                n_alphas_linreg = 20,
                                cv_linreg = 5, 
                                l1_ratios_linreg = [0.01, 0.1])
    >>> # you don't have to specifiy n_alphas_logreg/linreg, cv_logreg/linreg or l1_ratios_logreg/linreg and take the default values instead (see documentation of elastic net function)
    >>> 
    >>> # Random Forest:
    >>> model = model_framework(X_train = X_train,
                                y_train = y_train,
                                model = "rf",
                                pretrained = False,
                                n_estimators = 100)  


    Args:
        X_train: The training dataset
        y_train: The true labels
        pretrained: If a new model should be trained or a pretrained model should be used
                    (True/False, default True uses pretrained model
        model: A string to enter which model to use ("elnet" = elastic net, "gboost" = gradient boosting,
                                                     "rf" = random forest, "cnn" = convolutional neural network,
                                                     "ebm" = explainable boosting machine)
        model_path: Full path to the folder where the selected pretrained model is stored
        save: If the newly trained model should be saved
        t_direct: Target directory to save the model in
        **kwargs: Additional parameters and options depending on the selected model
    Returns:
        A fitted model
    Raises:
        FileNotFoundError
        IOError
    """
    assert isinstance(model, str), "invalid option, must be string"
    assert model in ["elnet", "gboost", "rf", "cnn",
                     "ebm"], "please provide a valid model (elnet, gboost, ebm, rf or cnn)"
    assert isinstance(pretrained, bool), "invalid pretrained, must be True/False"
    if pretrained:
        pass
    else:
        assert len(X_train) == len(y_train), "X_train and y_train must be the same length"

    classification = False
    if len(set(y_train)) == 2:  # Checking whether classification or regression is needed
        classification = True

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

    # saving model

    if save:
        if not os.path.exists(t_direct):
            raise FileNotFoundError("Invalid path (write to dir)")
        if model == "elnet":
            filename = os.path.join(t_direct, "trained_elnet")
            pickle.dump(model, open(filename, 'wb'))
        elif model == "ebm":
            filename = os.path.join(t_direct, "trained_ebm")
            pickle.dump(model, open(filename, 'wb'))
        elif model == "gboost":
            filename = os.path.join(t_direct, "trained_gboost")
            pickle.dump(model, open(filename, 'wb'))
        elif model == "rf":
            filename = os.path.join(t_direct, "trained_randomforest")
            pickle.dump(model, open(filename, 'wb'))

        else:
            filename = os.path.join(t_direct, "trained_cnn")
            rmodel.save(filename)

    return rmodel
