import streamlit as st
import pandas as pd
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .find_gaps import find_gaps

_first_row = [
    "USFLUX",
    "TOTPOT",
]

_second_row = [
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
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

_sharp_columns = [*_first_row, *_second_row, *_third_row]


@st.cache
def _load_sharp_data():
    df = pd.read_csv(
        '/mnt/hackathon2021/Weltraumwetterlage/own_data/sharp/2019.csv')
    df = df[["timestamp", *_sharp_columns]]
    for col in _sharp_columns:
        df[col] = df[col] / df[col].mean()
    return df


_sharp_data = _load_sharp_data()


def plot_sharp_data(placeholder, data_horizon) -> go.Figure:
    start, end = data_horizon

    display_data = _sharp_data[(_sharp_data['timestamp'] >= start)
                               & (_sharp_data['timestamp'] <= end)]

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
    fig.update_xaxes(range=(start, end))
    fig.update_xaxes(title_text="<b>Timestamp</b> [s]", row=3, col=1)
    placeholder.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    gap_threshold = 3 * 12 * 60
    gaps = find_gaps(display_data, "timestamp", gap_threshold, end)
    return gaps
