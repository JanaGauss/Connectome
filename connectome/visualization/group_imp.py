"""
functions to calculate the group feature importance
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def grouped_permutation_FI(model, Xtest, ytest, groups_df, m = 10):
    """
    grouped permutation feature importance, see paper: Au, Quay, et al. "Grouped feature importance and combined features effect plot." arXiv preprint arXiv:2104.11688 (2021), section 2.2.1. 
    Shuffle observations of group g and calculate metric after shuffle.
    For more information on how to use this function within the framework, see the documentation of the viz_framework function.
  
  
    Args:
        model: fitted model
        Xtest: pd.Dataframe containing test data
        ytest: pd.Series containing labels of test data
        groups_df: pd.Dataframe with columns region (e.g. yeo7) and conn_name (names of connectivity matrix)
        m: number of random shuffles per group, default 10   
        
    Returns:
        a pd.DataFrame containing the regions and the associated increase in MSE/decrease in accuracy if features from that group are shuffled
  
    """
    # check input types
    assert isinstance(Xtest, pd.DataFrame), "provided Xtest is no pd.DataFrame"
    assert isinstance(ytest, pd.Series), "provided ytest is no pd.Series"
    assert isinstance(groups_df, pd.DataFrame), "provided groups_df is no pd.DataFrame"
    assert isinstance(m, int), "provided m is no integer" 
 
    # check if groups_df is valid
    assert "region" in groups_df.columns, "column 'region' not found in columns of groups_df"
    assert "conn_name" in groups_df.columns, "column 'conn_name' not found in columns of groups_df"
    assert all(x in Xtest.columns for x in groups_df["conn_name"]), "entries of groupf_df['conn_name'] not found in columns of Xtest"

  
    if len(ytest.unique()) == 2:
        classification = True
    else: 
        classification = False

    predictions = model.predict(Xtest)
    if classification:
        metric_test = accuracy_score(ytest, np.round(predictions))
    else:
        metric_test = mean_squared_error(ytest, predictions)

  
    result = []

    for g in groups_df["region"].unique():

        metric_group = [] # store result

        conn_names_region = groups_df.loc[groups_df["region"] == g]["conn_name"]
        subset_group = Xtest[conn_names_region] # extract variables belonging to region g

        for i in range(m):
            subset_group_shuffled = subset_group.sample(frac = 1) # return rows in random order
            
            Xtest_shuffle = Xtest.copy().reset_index(drop = True)
            Xtest_shuffle[conn_names_region] = subset_group_shuffled.reset_index(drop = True) # replace variables belonging to g by randomly shuffled observations
            
            predictions_shuffled = model.predict(Xtest_shuffle)
            
            if classification:
                metric_shuffled = accuracy_score(ytest, np.round(predictions_shuffled))
            else:
                metric_shuffled = mean_squared_error(ytest, predictions_shuffled)
                
            metric_group.append(metric_shuffled)
            
        # calculate mean change in metric for group g over m repetitions
        if classification:
            mean_decr_metric = metric_test - sum(metric_group)/len(metric_group) # calculate mean decrease accuracy
            result.append([g, mean_decr_metric]) 
    
        else: 
            mean_incr_metric = sum(metric_group)/len(metric_group) - metric_test # calculate mean increase MSE
            result.append([g, mean_incr_metric])
    

    if classification:
        result_df = pd.DataFrame(data = result, columns = ["region", "mean decrease accuracy"])
    else:
        result_df = pd.DataFrame(data = result, columns = ["region", "mean increase MSE"])

    return(result_df)   



def group_only_permutation_FI(model, Xtest, ytest, groups_df, m = 10):
    """
    group only permutation feature importance, see paper: Au, Quay, et al. "Grouped feature importance and combined features effect plot." arXiv preprint arXiv:2104.11688 (2021), section 2.2.2.
    Compare the metric after permuting all features jointly with the metric after permuting all features except the considered group.
    For more information on how to use this function within the framework, see the documentation of the viz_framework function.
    
    Args:
      model: fitted model
      Xtest: pd.Dataframe containing test data
      ytest: pd.Series containing labels of test data
      groups_df: pd.Dataframe with columns region (e.g. yeo7) and conn_name (names of connectivity matrix)
      m: number of random shuffles per group, default 10   
      
    Returns:
      a pd.DataFrame containing the regions and the associated increase in MSE/decrease in accuracy 

    """
    # check input types
    assert isinstance(Xtest, pd.DataFrame), "provided Xtest is no pd.DataFrame"
    assert isinstance(ytest, pd.Series), "provided ytest is no pd.Series"
    assert isinstance(groups_df, pd.DataFrame), "provided groups_df is no pd.DataFrame"
    assert isinstance(m, int), "provided m is no integer" 
 
    # check if groups_df is valid
    assert "region" in groups_df.columns, "column 'region' not found in columns of groups_df"
    assert "conn_name" in groups_df.columns, "column 'conn_name' not found in columns of groups_df"
    assert all(x in Xtest.columns for x in groups_df["conn_name"]), "entries of groupf_df['conn_name'] not found in columns of Xtest"

    if len(ytest.unique()) == 2:
      classification = True
    else: 
      classification = False


    result = []

    for g in groups_df["region"].unique():

      metric_group = [] # store result

      conn_names_region = groups_df.loc[groups_df["region"] == g]["conn_name"]
      subset_group = Xtest[conn_names_region] # extract variables belonging to region g

      for i in range(m):

        all_shuffled = Xtest.sample(frac = 1).reset_index(drop = True) # return rows of whole dataset in random order
        all_shuffled_except_g = all_shuffled.copy()
        all_shuffled_except_g[conn_names_region] = subset_group.reset_index(drop = True) # all variables shuffled except group g

        predictions_shuffled_all = model.predict(all_shuffled)
        predictions_shuffled_all_except_g = model.predict(all_shuffled_except_g)

        if classification:        
          metric_shuffled_all = accuracy_score(ytest, np.round(predictions_shuffled_all))
          metric_shuffled_all_except_g = accuracy_score(ytest, np.round(predictions_shuffled_all_except_g))
          metric_group.append(metric_shuffled_all_except_g - metric_shuffled_all) # calculate decrease accuracy
        else:
          metric_shuffled_all = mean_squared_error(ytest, predictions_shuffled_all)
          metric_shuffled_all_except_g = mean_squared_error(ytest, predictions_shuffled_all_except_g)
          metric_group.append(metric_shuffled_all - metric_shuffled_all_except_g) # calculate increase MSE

      # calculate mean change in metric for group g over m repetitions      
      if classification:
        mean_decr_metric = sum(metric_group)/len(metric_group) # calculate mean decrease accuracy
        result.append([g, mean_decr_metric]) 
      else: 
        mean_incr_metric = sum(metric_group)/len(metric_group) # calculate mean increase MSE
        result.append([g, mean_incr_metric])

    
    if classification:
      result_df = pd.DataFrame(data = result, columns = ["region", "mean decrease accuracy"])
    else:
      result_df = pd.DataFrame(data = result, columns = ["region", "mean increase MSE"])

    return(result_df) 
