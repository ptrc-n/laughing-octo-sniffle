import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .find_gaps import find_gaps
from .predictor import Avocato
import tensorflow as tf
import numpy as np

YEAR_OF_DATA = 2019

_first_row = [
    "USFLUX",
    "TOTPOT",
]

_second_row = [
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "TOTUSJZ",
    "TOTUSJH",
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "MEANSHR",
    "SHRGT45",
    "SIZE",
    "SIZE_ACR",
    "NACR",
    "NPIX",
]

_third_row = [
    "MEANJZD",
    "MEANALP",
    "MEANJZH",
]

load_sharp_columns = [
    "USFLUX",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "TOTPOT",
    "TOTUSJZ",
    "TOTUSJH",
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "MEANSHR",
    "SHRGT45",
    "SIZE",
    "SIZE_ACR",
    "NACR",
    "NPIX",
    "MEANJZD",
    "MEANALP",
    "MEANJZH",
]

_sharp_columns = [*_first_row, *_second_row, *_third_row]


@st.cache
def _load_sharp_data(year_of_data):
    df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/sharp/{year_of_data}.csv'
    )
    df = df[["timestamp", "harp", *_sharp_columns]]
    for col in _sharp_columns:
        df[col] = df[col] / df[col].mean()
    return df


@st.cache
def _load_sharp_data_prediction(year_of_data):
    df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/sharp/{year_of_data}.csv'
    )
    df = df[["timestamp", "harp", *load_sharp_columns]]
    return df


_sharp_data = _load_sharp_data(YEAR_OF_DATA)
# _sharp_data_prediction = _load_sharp_data_prediction(YEAR_OF_DATA)
_model = Avocato("8h-2")


def plot_sharp_data(placeholder, data_horizon, placeholder_prediction,
                    gs_long_future, gp_long_future):
    start, end = data_horizon

    display_data = _sharp_data[(_sharp_data["timestamp"] >= start)
                               & (_sharp_data["timestamp"] <= end)]
    _sharp_data_prediction = _load_sharp_data_prediction(YEAR_OF_DATA)[
        (_load_sharp_data_prediction(YEAR_OF_DATA)["timestamp"] >= start)
        & (_load_sharp_data_prediction(YEAR_OF_DATA)["timestamp"] <= end)]
    _sharp_data_prediction = _sharp_data_prediction[
        (_sharp_data_prediction["timestamp"] >
         (datetime.fromtimestamp(end) - timedelta(hours=1)).timestamp())
        & (_sharp_data_prediction["timestamp"] <= end)]

    x_ray_pred = _model(_sharp_data_prediction)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
    )

    for subplot_row, columns in enumerate(
        [_first_row, _second_row, _third_row]):
        for column in columns:
            x, y = display_data['timestamp'], display_data[column]

            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers', name=column),
                row=subplot_row + 1,
                col=1,
            )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=700,
        showlegend=False,
        hovermode="x",
    )

    fig.update_traces(marker=dict(size=3))

    fig.update_xaxes(range=(start, end))
    fig.update_xaxes(title_text="<b>Timestamp</b> [s]", row=3, col=1)
    placeholder.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    gap_threshold = 3 * 12 * 60
    gaps = find_gaps(display_data, "timestamp", gap_threshold, end)

    x_ray_pred_plot = tf.math.reduce_max(x_ray_pred, axis=-1).numpy()[0]

    x_ray_pred = tf.math.reduce_max(tf.math.reduce_max(x_ray_pred, axis=-2),
                                    axis=0).numpy()[0][-1]

    x_ray_class_pred = "A - ????"

    if x_ray_pred > 1e-7:
        x_ray_class_pred = "B - ????"
    elif x_ray_pred > 1e-6:
        x_ray_class_pred = "C - ????"
    elif x_ray_pred > 1e-5:
        x_ray_class_pred = "M - ????"
    elif x_ray_pred > 1e-4:
        x_ray_class_pred = "X - ????"

    x = np.array([1, 2, 3, 4, 5])

    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(x=x,
                   y=gs_long_future["Long"],
                   mode='markers',
                   name="gs_long_future"))
    fig_pred.add_trace(
        go.Scatter(x=x,
                   y=gp_long_future["Long"],
                   mode='markers',
                   name="gp_long_future"))
    fig_pred.add_trace(
        go.Scatter(x=x, y=x_ray_pred_plot, mode='markers', name="x_ray_pred"))

    # placeholder_prediction.plotly_chart(
    #     fig_pred,
    #     use_container_width=True,
    #     config={"displayModeBar": False},
    # )

    return gaps, x_ray_class_pred