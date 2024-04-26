""" Search the hyperparameter space for XGBoost.

Notes:
    (1) weld_num is proxy for wear.
"""
from typing import Dict, List

import numpy as np
import sklearn
import hyperopt
import pandas as pd
import pickle

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.util.functions import log_msg


def preprocess_data(df: pd.DataFrame) -> List[pd.DataFrame]:
    """ Preprocesses the data in one routine.

    Args:
        df (pd.DataFrame): full input dataframe.

    Returns:
        List[ pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ]: x_train, x_test, y_train, y_test.
    """
    y_train = df[["diameter"]]
    x_train = df.drop("diameter", axis="columns")

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, train_size=0.8, test_size=0.2,
    )
    return x_train, x_test, y_train, y_test


def train_model_AL5(hyper_parameters: Dict[str, str]):
    df = pd.read_parquet("src/al_rsw_baseline/data/data_AL5.parquet")
    x_train, x_test, y_train, y_test = preprocess_data(df)

    xgb = XGBRegressor(
        n_estimators=int(hyper_parameters['n_estimators']),
        max_depth=int(hyper_parameters['max_depth']),
        gamma=hyper_parameters['gamma'],
        reg_alpha=int(hyper_parameters['reg_alpha']),
        #min_child_weight=int(hyper_parameters['min_child_weight']),
        #colsample_bytree=int(hyper_parameters['colsample_bytree']),
        #max_leaves=int(hyper_parameters['max_leaves']),
        #max_bin=int(hyper_parameters['max_bin']),
    )
    xgb.fit(x_train, y_train)

    y_pred_xgb = xgb.predict(x_test)
    loss = float(sklearn.metrics.mean_squared_error(y_pred_xgb, y_test))

    log_msg(f"R2: {round(sklearn.metrics.r2_score(y_pred_xgb, y_test), 3)}\n",)

    return {'loss': loss, 'status': hyperopt.STATUS_OK,}


def train_model_AL6(hyper_parameters: Dict[str, str],):
    df = pd.read_parquet("src/al_rsw_baseline/data/data_AL6.parquet")
    x_train, x_test, y_train, y_test = preprocess_data(df)

    xgb = XGBRegressor(
        n_estimators=int(hyper_parameters['n_estimators']),
        max_depth=int(hyper_parameters['max_depth']),
        gamma=hyper_parameters['gamma'],
        reg_alpha=int(hyper_parameters['reg_alpha']),
        #min_child_weight=int(hyper_parameters['min_child_weight']),
        #colsample_bytree=int(hyper_parameters['colsample_bytree']),
        #max_leaves=int(hyper_parameters['max_leaves']),
        #max_bin=int(hyper_parameters['max_bin']),
    )
    xgb.fit(x_train, y_train)

    y_pred_xgb = xgb.predict(x_test)
    loss = float(sklearn.metrics.mean_squared_error(y_pred_xgb, y_test))

    log_msg(f"R2: {round(sklearn.metrics.r2_score(y_pred_xgb, y_test), 3)}\n",)

    return {'loss': loss, 'status': hyperopt.STATUS_OK,}


def main():
    search_space = {
        'max_depth': hyperopt.hp.loguniform("max_depth", np.log(3), np.log(1000)),
        'gamma': hyperopt.hp.loguniform('gamma', np.log(1), np.log(9)),
        'reg_alpha' : hyperopt.hp.quniform('reg_alpha', 40, 180, 1),
        'reg_lambda' : hyperopt.hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree' : hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight' : hyperopt.hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hyperopt.hp.loguniform("n_estimators", np.log(3), np.log(1000)),
        'max_bin': hyperopt.hp.loguniform("max_bin", np.log(3), np.log(200)),
        'max_leaves': hyperopt.hp.loguniform("max_leaves", np.log(3), np.log(200)),
        'seed': 6020,
    }

    best_params = hyperopt.fmin(
        fn=train_model_AL5,
        space=search_space,
        algo=hyperopt.tpe.suggest,
        max_evals=300,
        trials=hyperopt.Trials(),
        verbose=True,
    )

    log_msg(f"The best params for AL5 are: {best_params}.")
    
    pickle.dump(best_params, open("src/al_rsw_baseline/models/best_params_AL5.pkl", "wb"))
    log_msg("Saved best AL5 parameters.")

    best_params = hyperopt.fmin(
        fn=train_model_AL6,
        space=search_space,
        algo=hyperopt.tpe.suggest,
        max_evals=300,
        trials=hyperopt.Trials(),
        verbose=True,
    )

    log_msg(f"The best params for AL6 are: {best_params}.")

    pickle.dump(best_params, open("src/al_rsw_baseline/models/best_params_AL6.pkl", "wb"))
    log_msg("Saved best AL6 parameters.")


if __name__ == "__main__":
    main()
