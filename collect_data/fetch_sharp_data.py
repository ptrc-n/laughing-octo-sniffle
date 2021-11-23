"""
This script uses the DRMS API to query the HARP for a given year and series.
"""

import drms
from datetime import datetime
from tqdm import tqdm
import pandas as pd

client = drms.Client()


def query_harp_for_year(harp_number, year, series):
    """Performs a DRMS query for a single day.
    Args:
        year int: year in the format YYYY
        harp_numbers (int): HARP to query
        series (str): series to query
    Returns:
        pandas.DataFrame: dataframe containing the query results
    """
    series_info = client.info(series)
    keys = [keyword for keyword in series_info.keywords.index]
    date_string = f"{year}.01.01_00:00:00_TAI"
    duration = f"{365 * 24}h"
    dataframe = client.query(
        f"{series}[{harp_number}][{date_string}/{duration}]",
        key=", ".join(keys),
    )
    if len(dataframe) > 0:
        dataframe["timestamp"] = dataframe["T_REC"].apply(
            lambda value: datetime.strptime(
                value,
                "%Y.%m.%d_%H:%M:%S_TAI",
            ).timestamp())
        dataframe["harp"] = harp_number
    return dataframe


def main(series: str, year: int, harp_numbers: list):
    dataframes = []
    for harp_number in tqdm(harp_numbers):
        print(f"{datetime.now()} Querying HARP {harp_number}")
        dataframe = query_harp_for_year(harp_number, year, series)
        if len(dataframe) > 0:
            dataframes.append(dataframe)

    dataframe = pd.concat(dataframes)
    dataframe.sort_values(by="timestamp", inplace=True)
    dataframe.to_csv(f"data/sharp/{year}.csv")


if __name__ == '__main__':
    harp_numbers = []
    with open("files/all_harps_with_noaa_ars.txt", "r") as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            harp_numbers.append(int(line.split(" ")[0]))

    # SERIES_NAME = 'hmi.sharp_720s'
    # SERIES_NAME = 'hmi.sharp_720s_nrt'
    SERIES_NAME = 'hmi.sharp_cea_720s'
    # SERIES_NAME = 'hmi.sharp_cea_720s_nrt'

    main(SERIES_NAME, 2019, harp_numbers)
