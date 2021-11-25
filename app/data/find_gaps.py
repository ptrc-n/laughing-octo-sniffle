from typing import List
from datetime import timedelta
import pandas as pd


def find_gaps(
    df: pd.DataFrame,
    timecol: str,
    gap_threshold: timedelta,
    end: float = None,
):
    """
    Drop all rows with missing values.
    Check if there are any gaps in the data.
    Returns a list of tuples with the start and end of the gaps.
    """
    df = df.dropna()
    if len(df) == 0:
        return [(-1, end)]
    timecol = df[timecol]
    time_diff = timecol.diff()
    gaps = [i for i, x in enumerate(time_diff) if x > gap_threshold]
    # print(timecol)
    # print(gaps)
    gaps = [(timecol.iloc[i - 1], timecol.iloc[i]) for i in gaps]
    last_timestamp = df['timestamp'].iloc[-1]
    if end is not None and end - last_timestamp > gap_threshold:
        gaps.append((last_timestamp, end))
    return gaps
