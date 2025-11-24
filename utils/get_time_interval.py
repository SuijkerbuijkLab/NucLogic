from datetime import datetime
import tables
import h5py
import numpy as np


def get_time_interval(input_file):
    # Try original method first
    try:
        with tables.open_file(input_file, "r") as hf:
            time_values = hf.root.DataSetTimes.Time.read()
        timestamps = np.array([row[2] for row in time_values])
        timestamps = timestamps // 3.6e9  # gets data in nanoseconds, calculate to hours
        time_interval = timestamps[1] - timestamps[0]
    except Exception:
        # Try new method from TimeInfo attributes
        try:
            with tables.open_file(input_file, "r") as hf:
                time_info = hf.root.DataSetInfo.TimeInfo

                # Get TimePoint attributes
                timepoint_attrs = [
                    attr
                    for attr in time_info._v_attrs._f_list()
                    if attr.startswith("TimePoint")
                ]
                timepoint_attrs_sorted = sorted(
                    timepoint_attrs, key=lambda x: int(x.replace("TimePoint", ""))
                )

                # Extract first two timestamps
                time_str_1 = "".join(
                    [
                        b.decode("utf-8")
                        for b in time_info._v_attrs[timepoint_attrs_sorted[0]]
                    ]
                )
                time_str_2 = "".join(
                    [
                        b.decode("utf-8")
                        for b in time_info._v_attrs[timepoint_attrs_sorted[1]]
                    ]
                )

                # Parse to datetime
                dt1 = datetime.strptime(time_str_1, "%Y-%m-%d %H:%M:%S")
                dt2 = datetime.strptime(time_str_2, "%Y-%m-%d %H:%M:%S")

                # Calculate interval in hours
                time_interval = (dt2 - dt1).total_seconds() / 3600
        except Exception:
            # If both methods fail, default to 1
            time_interval = 1

    return time_interval
