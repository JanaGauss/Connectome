"""
Prepare data for elastic net (e.g. calculate squared or absolute values or interactions) and run elastic net model

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ElasticNetCV



def prepare_data_elastic_net(data: pd.DataFrame, 
                             option: str = "abs") -> pd.DataFrame:
    """
    Function that transforms the data for elastic net modelling (absolute values, squared values, quadratic functions). Remember to run this function both on X_train and X_test.

    Examples:
    >>> # calculate absolute values of all features:
    >>> X_train = prepare_data_elastic_net(data = X_train,
							 option = "abs")
    >>> X_test = prepare_data_elastic_net(data = X_test,
							 option = "abs")
    
    Args:
        data: A pd.Dataframe
        option: option for transformation (abs = absolute values, squ = squared values, quadr = quadratic functions)

    Returns:
        A pd.Dataframe with transformed values
    """

    assert isinstance(data, pd.DataFrame), "provided data is no pd.DataFrame"
    assert isinstance(option, str), "invalid option, must be string"
    assert option in ["abs", "squ", "quadr"], "please provide a valid option (abs, squ or quadr)"
    


    if option == "abs":
      transf = data.apply(lambda x: abs(x) if np.issubdtype(x.dtype, np.number) else x)


    if option == "squ":
      transf = data.apply(lambda x: x**2 if np.issubdtype(x.dtype, np.number) else x)

    if option == "quadr":
      transf_qu = data.apply(lambda x: x**2 if np.issubdtype(x.dtype, np.number) else x)
      transf_qu.columns = transf_qu.columns + "_squ"
      transf = pd.concat([data, transf_qu], axis=1)   

    return transf

def model_elastic_net(X_train, y_train, 
			    classification: bool = True, 
                      n_alphas_logreg = 10, 
                      n_alphas_linreg = 10,
                      cv_logreg = 5, 
                      cv_linreg = 5, 
		          l1_ratios_logreg = np.linspace(0,  1, 11).tolist(),
                      l1_ratios_linreg = np.linspace(0.01,  1, 11).tolist(), 
		          verbose = 0,
                      **kwargs):
    """
    Function that fits an elastic net model and searches for best parameters via CV. 
    See also https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html and https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html for further arguments that can be specified.
    See the documentation of the framework function for examples how to use this function within the framework.
    Information on arguments: The higher the values for n_alphas and cv and the higher the amount of l1_ratios, the more accurate the search for good parameters, but the longer the runtime. 
    Running this function with the default values on the Connectome dataset (not aggregated) may take a while, especially for classification.
  
    Examples:
    >>> model = model_elastic_net(X_train = X_train,
                                  y_train = y_train,
                                  classification = True) 
 
    Args:
        X_train: The training dataset
        y_train: The true labels
        classification: classification task -> logistic regression or regression task -> linear model
        n_alphas_logreg: number of alphas along the regularization path (logreg), default 10
        n_alphas_linreg: number of alphas along the regularization path (linreg), default 10
        cv_logreg: number of folds (logreg), default 5
        cv_linreg: number of folds (linreg), default 5
        l1_ratios_logreg: (list of) float for l1_ratio, 0 is L2, 1 is L1 (logreg)
        l1_ratios_linreg: (list of) float for l1_ratio, 0 is L2, 1 is L1 (linreg). Default with 0.01 instead of 0, because for l1_ratio = 0, automatic alpha grid generation is not supported 
        verbose: amount of verbosity, default = 1 
        
    Returns:
        Returns fitted model
    """
  
    assert isinstance(X_train, pd.DataFrame), "provided X_train is no pd.DataFrame"
    assert isinstance(y_train, pd.Series), "provided y_train is no pd.Series"
    assert isinstance(n_alphas_logreg, int), "invalid n_alphas_logreg"
    assert isinstance(n_alphas_linreg, int), "invalid n_alphas_linreg"
    assert isinstance(cv_logreg, int), "invalid cv_logreg"
    assert isinstance(cv_linreg, int), "invalid cv_linreg"
    assert isinstance(l1_ratios_logreg, list), "invalid l1_ratios_logreg"
    assert isinstance(l1_ratios_linreg, list), "invalid l1_ratios_linreg"
    assert all(isinstance(x, float) for x in l1_ratios_logreg), "invalid l1_ratios_logreg"
    assert all(isinstance(x, float) for x in l1_ratios_linreg), "invalid l1_ratios_linreg"


    if classification:
    
      model = LogisticRegressionCV(Cs = n_alphas_logreg, penalty = "elasticnet", 
                                 cv = cv_logreg, solver = "saga", 
                                 l1_ratios = l1_ratios_logreg, 
				 verbose = verbose,
                                 **kwargs)
    
    else:

      model = ElasticNetCV(l1_ratio = l1_ratios_linreg, 
                         n_alphas = n_alphas_linreg,
			 cv = cv_linreg, 
			 verbose = verbose,
			 **kwargs)
    

    fit_model = model.fit(X_train, y_train)

    return fit_model  
