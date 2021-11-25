import time
from datetime import datetime, timedelta
import streamlit as st

# NOTE: must be called before any other streamlit function
st.set_page_config(layout="wide")

# NOTE: data imports include streamlit functions
from data.sharp import plot_sharp_data
from data.noaa import plot_noaa_data
from data.images import plot_sharp_image

delta = timedelta(hours=1)

ranges = {
    "1 Day": timedelta(days=1),
    "3 Days": timedelta(days=3),
    "1 Week": timedelta(weeks=1),
}

st.header("Final Space Weather")
range_ = st.sidebar.selectbox("Range to inspect", list(ranges.keys()), 0)

play_option = st.sidebar.selectbox(
    "Select run mode",
    ["Auto-Run", "Slider"],
    index=1,
)

if play_option == "Slider":
    if range_ == "1 Day":
        offset = st.sidebar.slider("Offset in Hours", 0, 364 * 24, 0)
        delta = timedelta(hours=offset)
    elif range_ == "3 Days":
        offset = st.sidebar.slider("Offset in Days", 0, 364, 0)
        delta = timedelta(days=offset)
    elif range_ == "1 Week":
        offset = st.sidebar.slider("Offset in Weeks", 0, 46, 0)
        delta = timedelta(weeks=offset)
else:
    offset = timedelta(weeks=0)

start_time = (datetime(2019, 1, 1) + delta).timestamp()
end_time = (datetime.fromtimestamp(start_time) + ranges[range_]).timestamp()

health_cols = st.columns(4)
with health_cols[0]:
    xray_status = st.empty()
    xray_metric = st.empty()
with health_cols[1]:
    ace_status = st.empty()
    ace_metric = st.empty()
with health_cols[2]:
    sharp_status = st.empty()
    sharp_metric = st.empty()
with health_cols[3]:
    placeholder_sharp_image = st.empty()

plot_cols = st.columns(2)

with plot_cols[0]:
    # st.write("### NOAA Activity")
    with st.expander("NOAA Data"):
        st.markdown("X-Ray, Sources: GOES-14, GOES-15", unsafe_allow_html=True)
        st.markdown("""ACE Satellite <br> 
            EPAM: Electron, Proton and Alpha Monitor <br> 
            MAG: Magnetometer <br> 
            SIS: Solar Isotope Spectrometer <br> 
            SWEPAM: Solar Wind Electron Proton Alpha Monitor""",
                    unsafe_allow_html=True)
    placeholder_noaa = st.empty()

with plot_cols[1]:
    # st.write("### SHARP Activity")
    with st.expander("SHARP Data"):
        st.markdown("""USFLUX: Total unsigned flux [Mx] <br> 
            MEANGAM: Mean inclination angle [deg] <br>
            MEANGBT: Mean value of the total field gradient [G/Mm] <br> 
            MEANGBZ: Mean value of the vertical field gradient [G/Mm] <br> 
            MEANGBH: Mean value of the horizontal field gradient [G/Mm] <br> 
            MEANJZD: Mean vertical current density [mA/m<sup>2</sup>] <br> 
            TOTUSJZ: Total unsigned vertical current [A] <br> 
            MEANALP: Total twist parameter [1/Mm] <br> 
            MEANJZH: Mean current helicity [G^2/m] <br> 
            TOTUSJH: Total unsigned current helicity [G^2/m] <br> 
            ABSNJZH: Absolute value of the net current helicity [G^2/m] <br> 
            SAVNCPP: Sum of the Absolute Value of the Net Currents Per Polarity [A] <br> 
            MEANPOT: Mean photospheric excess magnetic energy density [ergs/cm^3] <br> 
            TOTPOT: Total photospheric magnetic energy density [ergs/cm^3] <br> 
            MEANSHR: Mean shear angle [deg] <br>
            SHRGT45: Percentage of pixels with a mean shear angle greater than 45 deg [%] <br>
            SIZE: Projected area of patch in image on microhemishere <br> 
            SIZE_ACR: Projected area of active pixels on image in microhemisphere <br> 
            NACR: Number of active pixles in patch <br> 
            NPIX: Number of pixels within patch
            """,
                    unsafe_allow_html=True)
    placeholder_sharp = st.empty()


def draw_charts(start, end):
    noaa_health = plot_noaa_data(placeholder_noaa, start, end)

    xray_gaps = noaa_health["GS"][0] + noaa_health["GP"][0]
    xray_up = noaa_health["GS"][1] + noaa_health["GP"][1]
    xray_status.metric(
        "X-Ray current Status",
        {
            0: "DOWN",
            1: "50%",
            2: "OK",
        }[xray_up],
    )
    xray_metric.metric("X-Ray Gaps", f"{len(xray_gaps)}")

    ace_gaps = noaa_health["ACE"][0]
    ace_up = noaa_health["ACE"][1]
    ace_status.metric("ACE current Status", "OK" if ace_up else "DOWN")
    ace_metric.metric("ACE Gaps", f"{len(ace_gaps)}")

    starp_ok = plot_sharp_image(placeholder_sharp_image, end_time)
    sharp_status.metric(
        "SHARP current Status",
        "OK" if starp_ok else "DOWN",
    )
    gaps = plot_sharp_data(placeholder_sharp, (start, end))
    sharp_metric.metric(
        f"SHARP Gaps",
        len(gaps),
    )


draw_charts(start_time, end_time)

if play_option == "Auto-Run":
    while end_time < datetime(2019, 12, 31).timestamp():
        time.sleep(1)
        start_time = (datetime.fromtimestamp(start_time) + delta).timestamp()
        end_time = (datetime.fromtimestamp(end_time) + delta).timestamp()
        draw_charts(start_time, end_time)
