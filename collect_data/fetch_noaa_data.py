"""
This script is used to fetch the data from the NOAA FTP server.
"""

import requests
import pandas as pd
import io
from tqdm import tqdm
from datetime import datetime, timedelta

ACE_BASE_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"
EPAM = f"{ACE_BASE_URL}%Y%m%d_ace_epam_5m.txt"
EPAM_KEYS = [
    "YR", "MO", "DA", "HHMM", "Modified Julian Day", "Seconds of the Day",
    "Electron S", "Electron 38-53", "Electron 175-315", "Protons keV S",
    "Protons keV 47-68", "Protons keV 115-195", "Protons keV 310-580",
    "Protons keV 795-1193", "Protons keV 1060-1900", "Anis. Index"
]

MAG = f"{ACE_BASE_URL}%Y%m%d_ace_mag_1m.txt"
MAG_KEYS = [
    "YR",
    "MO",
    "DA",
    "HHMM",
    "Modified Julian Day",
    "Seconds of the Day",
    "GSM_S",
    "GSM_Bx",
    "GSM_By",
    "GSM_Bz",
    "GSM_Bt",
    "GSM_Lat.",
    "GSM_Long.",
]

SIS = f"{ACE_BASE_URL}%Y%m%d_ace_sis_5m.txt"
SIS_KEYS = [
    "YR",
    "MO",
    "DA",
    "HHMM",
    "Modified Julian Day",
    "Seconds of the Day",
    "Integral Proton Flux S",
    "Integral Proton Flux > 10 MeV",
    "Integral Proton Flux S2",
    "Integral Proton Flux > 30 MeV",
]

SWEPAM = f"{ACE_BASE_URL}%Y%m%d_ace_swepam_1m.txt"
SWEPAM_KEYS = [
    "YR",
    "MO",
    "DA",
    "HHMM",
    "Modified Julian Day",
    "Seconds of the Day",
    "Solwar Wind S",
    "Solwar Wind Density",
    "Solwar Wind Speed",
    "Solwar Wind Temperature",
]

XRAY_BASE_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/xray/"
GS = f"{XRAY_BASE_URL}%Y%m%d_Gs_xr_1m.txt"
GP = f"{XRAY_BASE_URL}%Y%m%d_Gp_xr_1m.txt"

GS_GP_KEYS = [
    "YR", "MO", "DA", "HHMM", "Modified Julian Day", "Seconds of the Day",
    "Short", "Long"
]


def query_data_for_date(filename, column_names, date):
    """
    Query the NOAA FTP server for the data for a given date and return a pandas dataframe
    """
    url = datetime.strftime(date, filename)
    response = requests.get(url)
    if response.status_code != 200:
        print("Could not Query data for {} Error: {}".format(
            url, response.status_code))
    data = response.text.split("\n")
    csv_string = ",".join(column_names) + "\n"
    for line in data:
        if line == "" or line[0] == "#" or line[0] == ":":
            continue
        line = line.split(" ")
        line = [l for l in line if l != ""]
        line = ",".join(line)
        csv_string += line + "\n"

    return pd.read_csv(io.StringIO(csv_string))


def query_year(filename, column_names, year):
    """
    Query the NOAA FTP server for the data for a given year and return a pandas dataframe
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    n_days = (end_date - start_date).days + 1

    df = query_data_for_date(filename, column_names, start_date)
    for _ in tqdm(range(n_days)):
        start_date += timedelta(days=1)
        df = df.append(query_data_for_date(filename, column_names, start_date))

    df["timestamp"] = df.apply(
        lambda row: datetime(
            int(row["YR"]),
            int(row["MO"]),
            int(row["DA"]),
            int(row['HHMM']) // 100,
            int(row['HHMM']) % 100,
        ).timestamp(),
        axis=1,
    )
    return df


def main(year):
    print("Fetching ACE Data")
    for filename, column_names in [
        (EPAM, EPAM_KEYS),
        (MAG, MAG_KEYS),
        (SIS, SIS_KEYS),
        (SWEPAM, SWEPAM_KEYS),
        (GS, GS_GP_KEYS),
        (GP, GS_GP_KEYS),
    ]:
        print(filename)
        df = query_year(filename, column_names, year)
        target_path = filename.split("/")[-1]
        df.to_csv(
            f"data/noaa/{year}" + target_path[6:].replace(".txt", ".csv"),
            index=None,
        )


if __name__ == "__main__":
    main(2019)
