import numpy as np
import pandas as pd
from viz_utils import plot_feature_map
from viz_nn import nn_feature_visualization


def visualization_framework(model,
                            X,
                            y,
                            viz_method: str = 'GFI',
                            **kwargs):
    """
    Returns feature importance and other visualization techniques

    Args:
        model:
        X:
        y:
        viz_method

    Returns:
        List of reordered connectvity Matrices
    """

    if type(X) == list:
            assert X[0].shape[0] == len(y), 'X_test and y_test are not of equal length'
    else:
        assert X.shape[0] == len(y), 'X_test and y_test are not of equal length'

    #
    if viz_method == "GFI":
        pass

    #Elastic Net
    elif viz_method == "elastic_net":
       pass

    # Shapley Values
    elif viz_method == "shapley":
        pass

    ## Neural Network Feature Attribution
    elif viz_method == "feature_attribution":
        return nn_feature_visualization(model, X, y, **kwargs)
