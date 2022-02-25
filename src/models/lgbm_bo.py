import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from typing import Union
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os


def bayes_parameter_opt_lgb(
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        random_seed: int = 6,
        init_points: int = 15,
        n_iter: int = 3,
        sklearn_cv: bool = False,
        ranges: dict = None,
        default_params: dict = None
        ) -> tuple:

    """
    Bayesian Optimization for lightgbm model
    Args:
        X: features
        y: target
        n_folds: number of folds for cv
        random_seed: for reproducibility
        init_points: BO parameter - How many steps of random exploration you want to perform.
            Random exploration can help by diversifying the exploration space
        n_iter: BO parameter - How many steps of bayesian optimization you want to perform.
            The more steps the more likely to find a good maximum you are
        sklearn_cv: whether the sklearn RepeatedKFold should be used or basic lightgbm cv
        ranges: ranges for the parameters to be tuned
        default_params: fixed parameters

    Returns:

    """

    # parameters
    if ranges is None:
        ranges = {
            'learning_rate': (0.001, 0.25),
            'num_leaves': (10, 100),
            'feature_fraction': (0.1, 1.0),
            'max_depth': (5, 40),
            'min_split_gain': (0.001, 0.1),
            'min_child_weight': (1e-4, 0.3),
            'subsample': (0.01, 1.0),
            'num_iterations': (30, 2500)
        }

    if default_params is None:
        default_params = {
            'application': 'binary',
            'histogram_pool_size': 1000,  # reduce to lower RAM usage
            'metric': 'auc',
            'verbose': -1,
            'early_stopping_round': 100,
            'max_bin': 60,  # default 255, reduce to lower RAM usage
            'n_jobs': -1
            # 'n_threads': 2
        }
    params = default_params.copy()

    def lgb_eval(
            learning_rate,
            num_leaves,
            feature_fraction,
            max_depth,
            min_split_gain,
            min_child_weight,
            subsample,
            num_iterations
    ) -> float:
        """
        fits an lgb model using the chosen cv method given the parameters

        Args:
            learning_rate: Boosting learning rate
            num_leaves: Maximum tree leaves for base learners.
            feature_fraction: LightGBM will randomly select a subset of features
                on each iteration (tree) if feature_fraction is smaller than 1.0
            max_depth: Maximum tree depth for base learners, <=0 means no limit
            min_split_gain: Minimum loss reduction required to make a
                further partition on a leaf node of the tree.
            min_child_weight: Minimum sum of instance weight (hessian)
                needed in a child (leaf).
            subsample: Subsample ratio of the training instance
            num_iterations: Number of boosted trees to fit

        Returns:

        """

        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['subsample'] = max(min(subsample, 1), 0)
        params['num_iterations'] = int(num_iterations)

        if sklearn_cv:
            kf = RepeatedKFold(n_splits=n_folds, n_repeats=3)
            splits = kf.split(X, y)

        train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
        cv_results = lgb.cv(params,
                            train_data,
                            seed=1234,
                            stratified=False,
                            verbose_eval=50,
                            nfold=None if sklearn_cv else n_folds,
                            shuffle=(False if sklearn_cv else True),
                            folds=(splits if sklearn_cv else None),
                            metrics=['auc'])

        return max(cv_results['auc-mean'])

    lgbBO = BayesianOptimization(lgb_eval, ranges, random_state=random_seed)

    lgbBO.maximize(init_points=init_points, n_iter=n_iter)

    model_auc = []
    for model in range(len(lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])

    # return best parameters
    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'], lgbBO.res[pd.Series(model_auc).idxmax()][
        'params'], default_params


def hpo_lgbm(
        X: np.ndarray,
        y: np.ndarray,
        save_dir: str = None,
        return_model: bool = False,
        save_model: bool = True,
        name: str = None,
        **kwargs
) -> Union[lgb.sklearn.LGBMClassifier, dict]:
    """
    performs Bayesian Optimisation for LGBM model

    Args:
        X:
        y:
        save_dir:
        return_model:
        save_model:
        name:

    Returns:
        either a trained LGBM model or a dictionary with the best parameters
    """

    if save_model and save_dir is None:
        save_dir = input(r'Input the directory where to save the resulting model')

        if not os.path.exists(save_dir):
            raise FileNotFoundError("invalid directory specified")

        if name is None:
            name = input(r'Input the name under which the model '
                         r'should be saved (without filename extension)')

    # perform BO based on given parameters
    res = bayes_parameter_opt_lgb(X, y, **kwargs)
    default_params = res[2]
    best_params = res[1]

    # transform some parameters to integer
    int_params = ['num_iterations', 'max_depth', 'num_leaves']

    for key in best_params.keys():
        if key in int_params:
            best_params[key] = int(best_params[key])

    best_params = {**best_params, **default_params}

    if return_model:
        lgb_tuned = lgb.LGBMClassifier(
            **best_params,
            tree_method='gpu_hist'
        )
        X_tr, X_v, y_tr, y_v = train_test_split(X, y,
                                                test_size=0.15)

        lgb_tuned.fit(X_tr, y_tr,
                      eval_metric='auc',
                      eval_set=(X_v, y_v),
                      verbose=25)

        # save model before returning
        if save_model:
            lgb_tuned.booster_.save_model(name + ".txt")

        return lgb_tuned

    else:
        return best_params


if __name__ == "__main__":
    X, y = make_classification(n_informative=10)
    print(hpo_lgbm(
        X, y,
        n_folds=2,
        init_points=2,
        n_iter=1,
        sklearn_cv=True,
        save_model=False)
    )
