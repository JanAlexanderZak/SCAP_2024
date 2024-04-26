""" Executes training of selected models and generates the visualization in on go.

Selected Models are:
    (1) XGBoost
    (1) Random Forest
    (1) SVR
    (1) Neural Net
    (1) Linear Regression
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn

from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.util.constants import DEFAULT_LAYOUT, DEFAULT_AXIS, PLOTLY_DEFAULT_COLORS


def get_sorted_residuals(
    y_test,
    y_pred,
) -> np.ndarray:
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()

    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    print(mse, sklearn.metrics.r2_score(y_test, y_pred))

    df = pd.DataFrame(data=[y_test, y_pred], index=["y_test", "y_pred"]).T
    df["residual"] = (df["y_test"] - df["y_pred"]) ** 2 / len(df)

    df = df.sort_values(by=["residual"]).reset_index(drop=True)
    df["residual_cumsum"] = df["residual"].cumsum()

    return mse, df["residual"], df["residual_cumsum"]


def main():
    df = pd.read_parquet("src/residual_distribution/data/data.parquet")

    x_train = df[["current_max", "force_max", "cap_thk", "weld_num", "room_temp"]]
    y_train = df[["diameter"]]

    # * Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # * Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, train_size=0.8, test_size=0.2,
    )
    print(len(y_test))

    # * Linear Regression
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    y_pred_lm = lm.predict(x_test)
    mse_lm, residuals_lm, residuals_cumsum_lm = get_sorted_residuals(y_test, y_pred_lm)
    
    # * Support Vector Regression
    svr = SVR()
    svr.fit(x_train, y_train)

    y_pred_svr = svr.predict(x_test)
    mse_svr, residuals_svr, residuals_cumsum_svr = get_sorted_residuals(y_test, y_pred_svr)

    # * Random Forest 
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)

    y_pred_rf = rf.predict(x_test)
    mse_rf, residuals_rf, residuals_cumsum_rf = get_sorted_residuals(y_test, y_pred_rf)

    # * XGBoost
    xgb = XGBRegressor()
    xgb.fit(x_train, y_train)

    y_pred_xgb = xgb.predict(x_test)
    mse_xgb, residuals_xgb, residuals_cumsum_xgb = get_sorted_residuals(y_test, y_pred_xgb)

    # * Neural Network
    nn = MLPRegressor()
    nn.fit(x_train, y_train)

    y_pred_nn = nn.predict(x_test)
    mse_nn, residuals_nn, residuals_cumsum_nn = get_sorted_residuals(y_test, y_pred_nn)


    #
    # ! Visualizations are intentionally not out-sourced.
    #
    max_x = len(y_test)
    MODELS = ["Linear Regression", "Support Vector Regression", "Random Forest", "XGBoost", "Neural Network"]

    # * Fist plot
    fig = make_subplots(rows=1, cols=2)
    
    for idx, residuals_cumsum in enumerate([
        residuals_cumsum_lm,
        residuals_cumsum_svr,
        residuals_cumsum_rf,
        residuals_cumsum_xgb,
        residuals_cumsum_nn,
    ]):
        fig = fig.add_trace(go.Scatter(
            y=np.array(residuals_cumsum).flatten(),
            line=dict(color=PLOTLY_DEFAULT_COLORS[idx]),
            name=MODELS[idx],
        ), row=1, col=2,)

    fig = fig.add_vline(
        x=1026,
        line_dash="dot",
        line_color="grey",
        annotation=dict(
            text="SVR outperforms the NN<br>on 97% of the data",
            font_size=14,
            align="right",
            ),
        annotation_position="top left",
        )
    
    fig = fig.update_layout(
        DEFAULT_LAYOUT,
        xaxis2=DEFAULT_AXIS,
        yaxis2=DEFAULT_AXIS,
    )
    fig = fig.update_layout(
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font_size=12,
        ),
        xaxis=dict(
            title=r"$\text{Actual values}, \ y \ [\text{mm}]$",
            range=[0, 12],
            tickmode="array",
            tickvals=[0, 2, 4, 6, 8, 10, 12],
        ),
        yaxis=dict(
            title=r"$\text{Predicted values}, \ \hat{y} \ [\text{mm}]$",
            range=[0, 12],
            tickmode="array",
            tickvals=[0, 2, 4, 6, 8, 10, 12],
        ),
        xaxis2=dict(
            title=r"$\text{Percentage of data points}$",
            range=[0, max_x],
            tickmode="array",
            tickvals=[perc/100 * max_x for perc in range(0, 120, 20)],
            ticktext=["0", "20", "40", "60", "80", "100"],
        ),
        yaxis2=dict(
            title=r"$\text{MSE}, \ \sum_{i=1}^N (y_i - \hat{y}_i)^2 \times N^{-1}$",
            range=[0, max(mse_lm, mse_svr, mse_rf, mse_xgb, mse_nn) * 1.05],
        ),
    )
    fig.write_html("src/residual_distribution/plots/residuals_cumsum.html")
    #fig.write_image("src/residual_distribution/plots/residuals_cumsum.png", scale=5)
    
    # * Second plot
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.2,
    )
    
    # Left
    for idx, residuals_cumsum in enumerate([
        residuals_cumsum_lm,
        residuals_cumsum_svr,
        residuals_cumsum_rf,
        residuals_cumsum_xgb,
        residuals_cumsum_nn,
    ]):
        fig = fig.add_trace(go.Scatter(
            y=np.array(residuals_cumsum).flatten(),
            line=dict(color=PLOTLY_DEFAULT_COLORS[idx]),
            name=MODELS[idx],
        ), row=1, col=1,)

    # Right
    for idx, residuals in enumerate([
        residuals_lm,
        residuals_svr,
        residuals_rf,
        residuals_xgb,
        residuals_nn,
    ]):
        fig = fig.add_trace(go.Scatter(
            y=np.array(residuals).flatten(),
            line=dict(color=PLOTLY_DEFAULT_COLORS[idx]),
            showlegend=False,
        ), row=1, col=2)

    fig = fig.add_vline(
        x=1026,
        line_dash="dot",
        line_color="grey",
        annotation=dict(
            text="SVR outperforms the NN<br>on 97% of the data",
            font_size=14,
            align="right",
            ),
        annotation_position="top left",
        row=1, col=2
        )
    fig = fig.add_vline(
        x=1026,
        line_dash="dot",
        line_color="grey",
        annotation=dict(
            text="",
            font_size=14,
            align="right",
            ),
        annotation_position="top left",
        row=1, col=1
        )
    
    fig = fig.update_layout(
        DEFAULT_LAYOUT,
        xaxis2=DEFAULT_AXIS,
        yaxis2=DEFAULT_AXIS,
    )
    fig = fig.update_layout(
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font_size=12,
        ),
        # Left
        xaxis=dict(
            title=r"$\text{Percentage of data points}$",
            range=[0.9 * max_x, max_x],
            tickmode="array",
            tickvals=[perc/100 * max_x for perc in np.arange(0, 102.5, 2.5)],
            ticktext=[f"{perc}" for perc in np.arange(0, 102.5, 2.5)],
        ),
        yaxis=dict(
            title=r"$\text{MSE}, \ \sum_{i=1}^N (y_i - \hat{y}_i)^2 \times N^{-1}$",
            range=[0, max(mse_lm, mse_svr, mse_rf, mse_xgb, mse_nn) * 1.05],
        ),
        # Right
        xaxis2=dict(
            title=r"$\text{Percentage of data points}$",
            range=[0.9 * max_x, max_x],
            tickmode="array",
            tickvals=[perc/100 * max_x for perc in np.arange(0, 102.5, 2.5)],
            ticktext=[f"{perc}" for perc in np.arange(0, 102.5, 2.5)],
        ),
        yaxis2=dict(
            title=r"$\text{Data point MSE}, \ (y_i - \hat{y}_i)^2 \times N^{-1}$",
            range=[0, 0.05],
            #type="log",
            #showticklabels=False,
        ),
    )
    fig.write_html("src/residual_distribution/plots/residuals_alt.html")
    #fig.write_image("src/residual_distribution/plots/residuals_alt.png", scale=5)
    

if __name__ == "__main__":
    main()
