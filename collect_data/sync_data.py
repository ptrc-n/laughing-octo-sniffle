"""
This script synchronises datapoints from NOAA and SHARP.

The problem:

SHARP has different time intervals than NOAA (5 or 1 minutes vs 12 minutes).
The goal is to have one valid NOAA data point for each SHARP data point.
Datapoints in NOAA, where no SHARP data exists, are dropped.
"""
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm


def main(year):
    sharp_df = pd.read_csv(f"data/sharp/{year}.csv")
    timestamps = sharp_df["timestamp"].unique()

    noaa_files = [
        filename for filename in os.listdir("data/noaa")
        if f"{year}" in filename and "m.csv" in filename
    ]

    for noaa_file in noaa_files:
        noaa_df = pd.read_csv(f"data/noaa/{noaa_file}")

        new_df = noaa_df[noaa_df["HHMM"] == -1].copy(
        )  # create empty dataframe with same columns as NOAA

        print(f"{noaa_file}")
        for timestamp in tqdm(timestamps):

            delta_t = noaa_df["timestamp"] - timestamp
            date = datetime.fromtimestamp(timestamp)
            closest_index = delta_t.abs().idxmin()
            closest_df = noaa_df.iloc[closest_index].copy()

            closest_df["timestamp"] = timestamp
            closest_df["YR"] = date.year
            closest_df["MO"] = date.month
            closest_df["DA"] = date.day
            closest_df["HHMM"] = date.hour * 100 + date.minute
            closest_df["Modified Julian Day"] = date.strftime("%j")
            closest_df["Seconds of the Day"] = date.strftime("%S")

            new_df = new_df.append(closest_df)

        new_df.to_csv(
            f"data/noaa/{noaa_file.split('.')[0]}_harmonized.csv",
            index=None,
        )


if __name__ == "__main__":
    main(2019)
