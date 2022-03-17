"""
Framework function for vizualisation
"""
import numpy as np
import pandas as pd
from connectome.visualization.viz_utils import plot_feature_map, plot_coef_elastic_net, plot_grouped_FI
from connectome.visualization.viz_nn import nn_feature_visualization
from connectome.visualization.group_imp import grouped_permutation_FI, group_only_permutation_FI
from connectome.preprocessing.colnames_to_yeo7 import get_colnames_df
from connectome.visualization.lgb_shapley import ShapleyLGB


def visualization_framework(model,
                            X,
                            y=None,
                            viz_method: str = 'GFI',
                            **kwargs):
    """
    Returns feature importance and other visualization techniques. Methods: "GFI" and "GFI_only" work for elastic net and random forest, if an aggregation by yeo7 is possible. 
    "FI" and "FI_only" work for elastic net and random forest.
    "elastic_net" works for elastic net models.
    "shapley" works for random forest and gradient boosting.
    "feature_attribution" works for CNN.
    For more details on the methods, see the documentations of the respective functions.

    Examples:
    >>> # Visualize Saliency Maps for neural networks
    >>> visualization_framework(model = model,
                                X = X_test,
                                y= y_test,
                                viz_method = 'feature_attribution',
                                method='saliency',
                                average=True,
                                ordered = True)
    >>> # Calculate and visualize the Grouped Permutation Feature Importance, e.g. for an elastic net model. Works similar for 'GFI_only', 'FI' and 'FI_only'.
    >>> visualization_framework(model = model,
                                X = X_test,
                                y= y_test,
                                viz_method = 'GFI',
                                m = 20) 
    >>> # the higher m (number of permutations) the more accurate the result, but the longer the runtime
    >>>
    >>> # Plot coefficients of an elastic net model
    >>> visualization_framework(model = model,
                                X = X_test,
                                y= y_test,
                                viz_method = 'elastic_net')

    Args:
        model: a trained ML Model
        X: A  dataframe
        y: the labels
        viz_method: Choice  of "GFI", "GFI_only", "FI", "FI_only", "elastic_net", "shapley", or "feature_attribution"

    Returns:
        List of reordered connectvity Matrices
    """
    assert isinstance(viz_method, str), "invalid viz_method, must be string"
    assert viz_method in ["GFI", "GFI_only", "FI", "FI_only", "elastic_net", "shapley",
                          "feature_attribution"], "please provide a valid viz_method (GFI, GFI_only, FI, FI_only, elastic_net, shapley, feature_attribution)"

    if type(X) == list:
        assert X[0].shape[0] == len(y), 'X_test and y_test are not of equal length'
    else:
        assert X.shape[0] == len(y), 'X_test and y_test are not of equal length'

    #
    if viz_method == "GFI":
        assert model.__class__.__name__ in ['LogisticRegressionCV', 'ElasticNetCV', 'RandomForestClassifier', 'RandomForestRegressor'], "invalid viz_method for supplied model"
        groups_df = get_colnames_df()
        groups_df = groups_df.loc[
            np.in1d(groups_df["conn_name"], X.columns)]  # remove entries of groups_df that are not in colnames of X
        df_importance = grouped_permutation_FI(model, X, y, groups_df, **kwargs)
        return plot_grouped_FI(df_importance)

    elif viz_method == "GFI_only":
        assert model.__class__.__name__ in ['LogisticRegressionCV', 'ElasticNetCV', 'RandomForestClassifier', 'RandomForestRegressor'], "invalid viz_method for supplied model"
        groups_df = get_colnames_df()
        groups_df = groups_df.loc[
            np.in1d(groups_df["conn_name"], X.columns)]  # remove entries of groups_df that are not in colnames of X
        df_importance = group_only_permutation_FI(model, X, y, groups_df, **kwargs)
        return plot_grouped_FI(df_importance, title="Group only Permutation Feature Importance")

    elif viz_method == "FI":  # normal permutation feature importance -> use function for grouped importance
        assert model.__class__.__name__ in ['LogisticRegressionCV', 'ElasticNetCV', 'RandomForestClassifier', 'RandomForestRegressor'], "invalid viz_method for supplied model"
        ind_conn_cols = []  # extract all connectivity variables
        for x in range(len(model.feature_names_in_)):
            if len(model.feature_names_in_[x].split("_")) > 1 and model.feature_names_in_[x].split("_")[0].isdigit() and \
                    model.feature_names_in_[x].split("_")[1].isdigit():
                ind_conn_cols.append(x)

        groups_df = pd.DataFrame({'conn_name': model.feature_names_in_[ind_conn_cols],
                                  'region': model.feature_names_in_[
                                      ind_conn_cols]})  # create groups_df, but region contains the same entries as conn_name
        df_importance = grouped_permutation_FI(model, X, y, groups_df, **kwargs)
        return plot_grouped_FI(df_importance, title="Permutation Feature Importance")

    elif viz_method == "FI_only":  # normal permutation only feature importance -> use function for grouped importance
        assert model.__class__.__name__ in ['LogisticRegressionCV', 'ElasticNetCV', 'RandomForestClassifier', 'RandomForestRegressor'], "invalid viz_method for supplied model"
        ind_conn_cols = []  # extract all connectivity variables
        for x in range(len(model.feature_names_in_)):
            if len(model.feature_names_in_[x].split("_")) > 1 and model.feature_names_in_[x].split("_")[0].isdigit() and \
                    model.feature_names_in_[x].split("_")[1].isdigit():
                ind_conn_cols.append(x)

        groups_df = pd.DataFrame({'conn_name': model.feature_names_in_[ind_conn_cols],
                                  'region': model.feature_names_in_[
                                      ind_conn_cols]})  # create groups_df, but region contains the same entries as conn_name
        df_importance = group_only_permutation_FI(model, X, y, groups_df, **kwargs)
        return plot_grouped_FI(df_importance, title="Permutation Feature Importance")

    # Elastic Net
    elif viz_method == "elastic_net":
        assert model.__class__.__name__ in ['LogisticRegressionCV',
                                            'ElasticNetCV'], "if viz_method = elastic net, an elastic net model has to be provided"
        return plot_coef_elastic_net(model)

    # Shapley Values
    elif viz_method == "shapley":
        shapley = ShapleyLGB(model, X)
        return shapley.summ_plot(**kwargs)

    ## Neural Network Feature Attribution
    elif viz_method == "feature_attribution":
        return nn_feature_visualization(model, X, y, **kwargs)
