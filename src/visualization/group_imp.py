import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def grouped_permutation_FI(model, Xtest, ytest, groups_df, m):
  """
  grouped permutation feature importance, see paper, section 2.2.1. Shuffle observations of group g and calculate metric after shuffle
  
  Args:
    model: fitted model
    Xtest: pd.Dataframe
    ytest: array containing labels
    groups_df: pd.Dataframe with columns region (yeo7) and conn_name (names of connectivity matrix)
    m: number of random shuffles per group   
        
  Returns:
    a pd.DataFrame containing the regions and the associated increase in MSE/decrease in accuracy if features from that group are shuffled
  
  """
  # ToDo: check input

  if len(ytest.unique()) == 2:
    classification = True
  else: 
    classification = False

  predictions = model.predict(Xtest)
  if classification:
    metric_test = accuracy_score(ytest, predictions)
  else:
    metric_test = mean_squared_error(ytest, predictions)

  
  result = []

  for g in groups["region"].unique():

    metric_group = [] # store result

    conn_names_region = groups.loc[groups["region"] == g]["conn_name"]
    subset_group = Xtest[conn_names_region] # extract variables belonging to region g

    for i in range(m):
      subset_group_shuffled = subset_group.sample(frac = 1) # return rows in random order

      Xtest_shuffle = Xtest.copy().reset_index(drop = True)
      Xtest_shuffle[conn_names_region] = subset_group_shuffled.reset_index(drop = True) # replace variables belonging to g by randomly shuffled observations

      predictions_shuffled = model.predict(Xtest_shuffle)

      if classification:        
        metric_shuffled = accuracy_score(ytest, predictions_shuffled)
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



def group_only_permutation_FI(model, Xtest, ytest, groups_df, m):
  """
  grouped permutation feature importance, see paper, section 2.2.2. Compare the metric after permuting all features jointly with
  the metric after permuting all features except the considered group
  
  Args:
    model: fitted model
    Xtest: pd.Dataframe
    ytest: array containing labels
    groups_df: pd.Dataframe with columns region (yeo7) and conn_name (names of connectivity matrix)
    m: number of random shuffles per group   
        
  Returns:
    a pd.DataFrame containing the regions and the associated increase in MSE/decrease in accuracy 
  
  """
  # ToDo: check input

  if len(ytest.unique()) == 2:
    classification = True
  else: 
    classification = False


  result = []

  for g in groups["region"].unique():

    metric_group = [] # store result

    conn_names_region = groups.loc[groups["region"] == g]["conn_name"]
    subset_group = Xtest[conn_names_region] # extract variables belonging to region g

    for i in range(m):

      all_shuffled = Xtest.sample(frac = 1).reset_index(drop = True) # return rows of whole dataset in random order
      all_shuffled_except_g = all_shuffled.copy()
      all_shuffled_except_g[conn_names_region] = subset_group.reset_index(drop = True) # all variables shuffled except group g

      predictions_shuffled_all = model.predict(all_shuffled)
      predictions_shuffled_all_except_g = model.predict(all_shuffled_except_g)

      if classification:        
        metric_shuffled_all = accuracy_score(ytest, predictions_shuffled_all)
        metric_shuffled_all_except_g = accuracy_score(ytest, predictions_shuffled_all_except_g)
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