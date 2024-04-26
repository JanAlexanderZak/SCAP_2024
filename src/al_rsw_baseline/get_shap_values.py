from typing import Dict

import shap
import pandas as pd

import sklearn
import matplotlib.pyplot as plt
import pickle

from xgboost import XGBRegressor

from src.al_rsw_baseline.hyperparameter_search import preprocess_data
from src.util.functions import log_msg


def get_shap_values(df: pd.DataFrame, best_params: Dict, alloy: str = "AL5",) -> None:
    x_train, x_test, y_train, y_test = preprocess_data(df)

    feature_names = ["current_max", "force_max", "weld_num", "sheets_thk",]
    x_train = pd.DataFrame(x_train, columns=feature_names)
    x_test = pd.DataFrame(x_test, columns=feature_names)
    y_train = pd.DataFrame(y_train, columns=["diameter"])
    y_test = pd.DataFrame(y_test, columns=["diameter"])

    xgb = XGBRegressor(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        gamma=best_params['gamma'],
        reg_alpha=int(best_params['reg_alpha']),
        #min_child_weight=int(best_params['min_child_weight']),
        #colsample_bytree=int(best_params['colsample_bytree']),
        #max_leaves=int(best_params['max_leaves']),
        #max_bin=int(best_params['max_bin']),
    )
    xgb.fit(x_train, y_train)

    y_pred_xgb = xgb.predict(x_test)
    log_msg(f"R2: {round(sklearn.metrics.r2_score(y_pred_xgb, y_test), 3)}\n",)

    # * Shap values
    x_test_shap = x_test
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(x_test_shap)
    
    shap.dependence_plot(
        "current_max", shap_values, x_test_shap, interaction_index="force_max", show=False, alpha=0.4, x_jitter=1,
    )
    ax = plt.gca()
    ax.set_ylim(-4, 4)
    plt.savefig(f'src/al_rsw_baseline/plots/{alloy}/dependence_plot_current_max.png', dpi=300)

    shap.dependence_plot(
        "force_max", shap_values, x_test_shap, interaction_index="current_max", show=False, alpha=0.4, x_jitter=0.2,
    )
    ax = plt.gca()
    ax.set_ylim(-4, 4)
    plt.savefig(f'src/al_rsw_baseline/plots/{alloy}/dependence_plot_force_max.png', dpi=300)

    shap.dependence_plot(
        "sheets_thk", shap_values, x_test_shap, interaction_index="force_max", show=False, alpha=0.4, x_jitter=0.5,
    )
    ax = plt.gca()
    ax.set_ylim(-4, 4)
    ax.set_xlim(1.75, 6.25)
    plt.savefig(f'src/al_rsw_baseline/plots/{alloy}/dependence_plot_sheets_thk.png', dpi=300)


def main():
    df_AL5 = pd.read_parquet("src/al_rsw_baseline/data/data_AL5.parquet")
    best_params_AL5 = pickle.load(open("src/al_rsw_baseline/models/best_params_AL5.pkl", "rb"))

    df_AL6 = pd.read_parquet("src/al_rsw_baseline/data/data_AL6.parquet")
    best_params_AL6 = pickle.load(open("src/al_rsw_baseline/models/best_params_AL6.pkl", "rb"))

    get_shap_values(df_AL5, best_params_AL5, "AL5")
    get_shap_values(df_AL6, best_params_AL6, "AL6")


if __name__ == "__main__":
    main()
