import numpy as np
import pandas as pd
from viz_utils import plot_feature_map, plot_coef_elastic_net, plot_grouped_FI
from viz_nn import nn_feature_visualization
from group_imp import grouped_permutation_FI, group_only_permutation_FI
from src.preprocessing.colnames_to_yeo7 import colnames_to_yeo7


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
	# groups_df = pd.read_csv("references/colnames_to_yeo7.csv")
	groups_df = get_colnames_df()
	groups_df = groups_df.loc[np.in1d(groups_df["conn_name"], X.columns)] # remove entries of groups_df that are not in colnames of X
	df_importance = grouped_permutation_FI(model, X, y, groups_df)
	return plot_grouped_FI(df_importance)
        
    elif viz_method == "GFI_only":
	# groups_df = pd.read_csv("references/colnames_to_yeo7.csv")
	groups_df = get_colnames_df()
	groups_df = groups_df.loc[np.in1d(groups_df["conn_name"], X.columns)] # remove entries of groups_df that are not in colnames of X
	df_importance = group_only_permutation_FI(model, X, y, groups_df)
	return plot_grouped_FI(df_importance)

    #Elastic Net
    elif viz_method == "elastic_net":
       return plot_coef_elastic_net(model)

    # Shapley Values
    elif viz_method == "shapley":
        pass

    ## Neural Network Feature Attribution
    elif viz_method == "feature_attribution":
        return nn_feature_visualization(model, X, y, **kwargs)
