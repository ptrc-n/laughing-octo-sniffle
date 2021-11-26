import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .find_gaps import find_gaps

YEAR_OF_DATA = 2019


@st.cache
def _load_noaa_data(year_of_data):
    """
    1. Loading NOAA data
    2. Filtering physically unrealistic NOAA data
    """
    gs_df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/noaa/{year_of_data}_Gs_xr_1m.csv'
    )
    gp_df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/noaa/{year_of_data}_Gp_xr_1m.csv'
    )
    ace_epam_df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/noaa/{year_of_data}_ace_epam_5m.csv'
    )
    ace_mag_df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/noaa/{year_of_data}_ace_mag_1m.csv'
    )
    ace_sis_df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/noaa/{year_of_data}_ace_sis_5m.csv'
    )
    ace_swepam_df = pd.read_csv(
        f'/mnt/hackathon2021/Weltraumwetterlage/own_data/noaa/{year_of_data}_ace_swepam_1m.csv'
    )

    # X-Ray
    gs_short_df = gs_df[gs_df["Short"] > 0]
    gs_long_df = gs_df[gs_df["Long"] > 0]
    gp_short_df = gp_df[gp_df["Short"] > 0]
    gp_long_df = gp_df[gp_df["Long"] > 0]
    # ACE EPAM
    ace_epam_E_38_53_df = ace_epam_df[ace_epam_df["Electron 38-53"] > 0][[
        "Electron 38-53", "timestamp"
    ]]
    ace_epam_E_175_315_df = ace_epam_df[ace_epam_df["Electron 175-315"] > 0][[
        "Electron 175-315", "timestamp"
    ]]
    ace_epam_P_47_68_df = ace_epam_df[ace_epam_df["Protons keV 47-68"] > 0][[
        "Protons keV 47-68", "timestamp"
    ]]
    ace_epam_P_115_195_df = ace_epam_df[
        ace_epam_df["Protons keV 115-195"] > 0][[
            "Protons keV 115-195", "timestamp"
        ]]
    ace_epam_P_310_580_df = ace_epam_df[
        ace_epam_df["Protons keV 310-580"] > 0][[
            "Protons keV 310-580", "timestamp"
        ]]
    ace_epam_P_795_1193_df = ace_epam_df[
        ace_epam_df["Protons keV 795-1193"] > 0][[
            "Protons keV 795-1193", "timestamp"
        ]]
    ace_epam_P_1060_1900_df = ace_epam_df[
        ace_epam_df["Protons keV 1060-1900"] > 0][[
            "Protons keV 1060-1900", "timestamp"
        ]]
    # ACE MAG
    ace_mag_Bx_df = ace_mag_df[ace_mag_df["GSM_Bx"].abs() < 50][[
        "GSM_Bx", "timestamp"
    ]]
    ace_mag_By_df = ace_mag_df[ace_mag_df["GSM_By"].abs() < 50][[
        "GSM_By", "timestamp"
    ]]
    ace_mag_Bz_df = ace_mag_df[ace_mag_df["GSM_Bz"].abs() < 50][[
        "GSM_Bz", "timestamp"
    ]]
    ace_mag_Bt_df = ace_mag_df[ace_mag_df["GSM_Bt"].abs() < 50][[
        "GSM_Bt", "timestamp"
    ]]
    # ACE SIS
    ace_sis_10_df = ace_sis_df[
        ace_sis_df["Integral Proton Flux > 10 MeV"] > 0][[
            "Integral Proton Flux > 10 MeV", "timestamp"
        ]]
    ace_sis_30_df = ace_sis_df[
        ace_sis_df["Integral Proton Flux > 30 MeV"] > 0][[
            "Integral Proton Flux > 30 MeV", "timestamp"
        ]]
    # ACE SWEPAM
    ace_swepam_density_df = ace_swepam_df[
        ace_swepam_df["Solwar Wind Density"] > 0][[
            "Solwar Wind Density", "timestamp"
        ]]
    ace_swepam_speed_df = ace_swepam_df[
        ace_swepam_df["Solwar Wind Speed"] > 0][[
            "Solwar Wind Speed", "timestamp"
        ]]
    ace_swepam_temp_df = ace_swepam_df[
        ace_swepam_df["Solwar Wind Temperature"] > 0][[
            "Solwar Wind Temperature", "timestamp"
        ]]
    return gs_short_df, gs_long_df, gp_short_df, gp_long_df, ace_epam_E_38_53_df, ace_epam_E_175_315_df, ace_epam_P_47_68_df, ace_epam_P_115_195_df, ace_epam_P_310_580_df, ace_epam_P_795_1193_df, ace_epam_P_1060_1900_df, ace_mag_Bx_df, ace_mag_By_df, ace_mag_Bz_df, ace_mag_Bt_df, ace_sis_10_df, ace_sis_30_df, ace_swepam_density_df, ace_swepam_speed_df, ace_swepam_temp_df


_noaa_data = _load_noaa_data(YEAR_OF_DATA)


def plot_noaa_data(placeholder, start, end):
    filter_dataframe = lambda df: df[
        (df["timestamp"] >= start) & (df["timestamp"] <= end)]

    gs_short_df, gs_long_df, gp_short_df, gp_long_df, ace_epam_E_38_53_df, ace_epam_E_175_315_df, ace_epam_P_47_68_df, ace_epam_P_115_195_df, ace_epam_P_310_580_df, ace_epam_P_795_1193_df, ace_epam_P_1060_1900_df, ace_mag_Bx_df, ace_mag_By_df, ace_mag_Bz_df, ace_mag_Bt_df, ace_sis_10_df, ace_sis_30_df, ace_swepam_density_df, ace_swepam_speed_df, ace_swepam_temp_df = _noaa_data
    health = {}
    # Reducing data to inspected timerange
    # X-Ray
    display_gs_short_df = filter_dataframe(gs_short_df)
    display_gs_long_df = filter_dataframe(gs_long_df)
    gs_long_future = gs_long_df[
        (gs_long_df["timestamp"] >
         (datetime.fromtimestamp(end) + timedelta(hours=7)).timestamp())
        & (gs_long_df["timestamp"] <=
           (datetime.fromtimestamp(end) + timedelta(hours=8)).timestamp())]
    gs_long_future = gs_long_future.iloc[::12, :]

    gs_gaps = find_gaps(display_gs_short_df, "timestamp", 5 * 60, end)
    gs_up = len(gs_gaps) == 0 or gs_gaps[-1][-1] != end
    health["GS"] = (gs_gaps, gs_up)

    display_gp_short_df = filter_dataframe(gp_short_df)
    display_gp_long_df = filter_dataframe(gp_long_df)
    gp_long_future = gp_long_df[
        (gp_long_df["timestamp"] >
         (datetime.fromtimestamp(end) + timedelta(hours=7)).timestamp())
        & (gp_long_df["timestamp"] <=
           (datetime.fromtimestamp(end) + timedelta(hours=8)).timestamp())]
    gp_long_future = gp_long_future.iloc[::12, :]

    gp_gaps = find_gaps(display_gp_short_df, "timestamp", 5 * 60, end)
    gp_up = len(gp_gaps) == 0 or gp_gaps[-1][-1] != end
    health["GP"] = (gp_gaps, gp_up)

    # ACE EPAM
    display_ace_epam_E_38_53_df = filter_dataframe(ace_epam_E_38_53_df)
    display_ace_epam_E_175_315_df = filter_dataframe(ace_epam_E_175_315_df)
    display_ace_epam_P_47_68_df = filter_dataframe(ace_epam_P_47_68_df)
    display_ace_epam_P_115_195_df = filter_dataframe(ace_epam_P_115_195_df)
    display_ace_epam_P_310_580_df = filter_dataframe(ace_epam_P_310_580_df)
    display_ace_epam_P_795_1193_df = filter_dataframe(ace_epam_P_795_1193_df)
    display_ace_epam_P_1060_1900_df = filter_dataframe(ace_epam_P_1060_1900_df)
    # ACE MAG
    display_ace_mag_Bx_df = filter_dataframe(ace_mag_Bx_df)
    display_ace_mag_By_df = filter_dataframe(ace_mag_By_df)
    display_ace_mag_Bz_df = filter_dataframe(ace_mag_Bz_df)
    display_ace_mag_Bt_df = filter_dataframe(ace_mag_Bt_df)
    # ACE SIS
    display_ace_sis_10_df = filter_dataframe(ace_sis_10_df)
    display_ace_sis_30_df = filter_dataframe(ace_sis_30_df)
    # ACE SWEPAM
    display_ace_swepam_density_df = filter_dataframe(ace_swepam_density_df)
    display_ace_swepam_speed_df = filter_dataframe(ace_swepam_speed_df)
    display_ace_swepam_temp_df = filter_dataframe(ace_swepam_temp_df)

    ace_gaps = find_gaps(display_ace_mag_Bt_df, "timestamp", 16 * 60, end)
    ace_up = len(ace_gaps) == 0 or ace_gaps[-1][-1] != end
    health["ACE"] = (ace_gaps, ace_up)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
    )
    for satellite, key in [("GS", "Short"), ("GP", "Short"), ("GS", "Long"),
                           ("GP", "Long")]:
        if key == "Short":
            fig.add_trace(
                go.Scatter(
                    x=display_gs_short_df["timestamp"]
                    if satellite == "GS" else display_gp_short_df["timestamp"],
                    y=display_gs_short_df["Short"]
                    if satellite == "GS" else display_gp_short_df["Short"],
                    mode='markers',
                    name=f'{satellite}-Short',
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=display_gs_long_df["timestamp"]
                    if satellite == "GS" else display_gp_long_df["timestamp"],
                    y=display_gs_long_df["Long"]
                    if satellite == "GS" else display_gp_long_df["Long"],
                    mode='markers',
                    name=f'{satellite}-Long',
                ),
                row=1,
                col=1,
            )

    fig.update_yaxes(title_text="<b>XRay</b> <br> [W/m<sup>2</sup>]",
                     row=1,
                     col=1)

    for df_ace_epam, key in [
        (display_ace_epam_E_38_53_df, "Electron 38-53"),
        (display_ace_epam_E_175_315_df, "Electron 175-315"),
        (display_ace_epam_P_47_68_df, "Protons keV 47-68"),
        (display_ace_epam_P_115_195_df, "Protons keV 115-195"),
        (display_ace_epam_P_310_580_df, "Protons keV 310-580"),
        (display_ace_epam_P_795_1193_df, "Protons keV 795-1193"),
        (display_ace_epam_P_1060_1900_df, "Protons keV 1060-1900"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df_ace_epam["timestamp"],
                y=df_ace_epam[key],
                mode='markers',
                name=key,
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(
        title_text="<b>EPAM</b> <br> [part./(cm<sup>2</sup>-s-ster-MeV)]",
        type="log",
        row=2,
        col=1)

    for df_ace_mag, key in [
        (display_ace_mag_Bx_df, "GSM_Bx"),
        (display_ace_mag_By_df, "GSM_By"),
        (display_ace_mag_Bz_df, "GSM_Bz"),
        (display_ace_mag_Bt_df, "GSM_Bt"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df_ace_mag["timestamp"],
                y=df_ace_mag[key],
                mode='markers',
                name=key,
            ),
            row=3,
            col=1,
        )
    fig.update_yaxes(title_text="<b>MAG</b> <br> [nT]", row=3, col=1)

    for df_ace_sis, key in [
        (display_ace_sis_10_df, "Integral Proton Flux > 10 MeV"),
        (display_ace_sis_30_df, "Integral Proton Flux > 30 MeV"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df_ace_sis["timestamp"],
                y=df_ace_sis[key],
                mode='markers',
                name=key,
            ),
            row=4,
            col=1,
        )
    fig.update_yaxes(
        title_text="<b>SIS</b> <br> [p/(cs<sup>2</sup>-sec-ster)]",
        row=4,
        col=1)

    for i, (df_ace_swepam, key) in enumerate([
        (display_ace_swepam_density_df, "Solwar Wind Density"),
        (display_ace_swepam_speed_df, "Solwar Wind Speed"),
        (display_ace_swepam_temp_df, "Solwar Wind Temperature"),
    ]):
        fig.add_trace(
            go.Scatter(
                x=df_ace_swepam["timestamp"],
                y=df_ace_swepam[key],
                mode='markers',
                name=key,
            ),
            secondary_y=i == len(df_ace_swepam) - 1,
            row=5,
            col=1,
        )
    fig.update_yaxes(title_text="<b>SWEPAM</b> <br> [p/cc, km/s, K]",
                     type="log",
                     row=5,
                     col=1)

    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x",
    )

    fig.update_traces(marker=dict(size=1))

    fig.update_xaxes(range=(start, end))
    fig.update_xaxes(title_text="<b>Timestamp</b> [s]", row=5, col=1)

    placeholder.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    xray_now = max(display_gs_long_df["Long"].iloc[-1],
                   display_gp_long_df["Long"].iloc[-1])

    xray_future = max(gs_long_future["Long"].max(),
                      gp_long_future["Long"].max())

    x_ray_class_future = "A - 😊"
    x_ray_class_now = "A - 😊"

    if xray_future > 1e-7:
        x_ray_class_future = "B - 🤔"
    elif xray_future > 1e-6:
        x_ray_class_future = "C - 🤨"
    elif xray_future > 1e-5:
        x_ray_class_future = "M - 😦"
    elif xray_future > 1e-4:
        x_ray_class_future = "X - 😱"

    if xray_now > 1e-7:
        x_ray_class_now = "B - 🤔"
    elif xray_now > 1e-6:
        x_ray_class_now = "C - 🤨"
    elif xray_now > 1e-5:
        x_ray_class_now = "M - 😦"
    elif xray_now > 1e-4:
        x_ray_class_now = "X - 😱"

    return health, x_ray_class_now, x_ray_class_future, gs_long_future, gp_long_future
