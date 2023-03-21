import os, wfdb, pandas as pd
import numpy as np, neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def DriveDB(path, missing, sample_rate, gt_type, streams):

    assert gt_type == "EDA", "Invalid ground truth"
    if "BR" in streams:
        streams.remove("BR")
        streams.append("RESP_rate")
    assert all(s in ["HR", "RESP_rate"] for s in streams), "Invalid streams"

    def lowpass_filter(ts=None, freq=15.5, cut=0.25):
        b, a = butter(3, cut, fs=freq, btype="low")
        return filtfilt(b, a, ts)

    def mask_intervals(signal=None, missing=missing):
        intervals, durations = [], []
        min_win, max_win = 0 * len(signal), 0.01 * len(signal)

        def cap(a, b):
            return [i for i in a if i in b]

        while sum(durations) < missing * len(signal):
            random_start = np.random.randint(0, len(signal) - max_win)
            random_end = random_start + np.random.randint(min_win, max_win)
            random_win = np.arange(random_start, random_end)

            intersections = [len(cap(p, random_win)) for p in intervals]
            if sum(intersections) >= random_end - random_start:
                continue

            intervals.append(random_win)
            durations.append(random_end - random_start - sum(intersections))

        # interpolate
        for interval in intervals:
            signal[interval] = signal[interval[0] - 1] if interval[0] else signal[0]

        return signal

    data, gt_data, names = [], [], []
    for file in sorted(os.listdir(path)):
        if not file.endswith(".dat"):
            continue

        ### data loading
        signals, fields = wfdb.rdsamp(path + os.path.splitext(file)[0])
        this_df = pd.DataFrame(signals, columns=fields["sig_name"])

        ### respiration processing
        try:
            signal = this_df["RESP"].to_numpy()
            _ = this_df["HR"].to_numpy()
        except:
            continue
        out, _ = nk.rsp_process(signal, sampling_rate=fields["fs"])
        this_df["RESP_amp"] = out[["RSP_Amplitude"]]
        this_df["RESP_rate"] = out[["RSP_Rate"]]

        ### masking for missing data
        if missing > 0:
            this_df[streams] = this_df[streams].apply(mask_intervals)

        ### lowpass filter (0.05Hz) + downsample
        this_df.index = pd.date_range(
            start="1/1/2022", periods=len(this_df), freq="0.065S"
        )
        down = int(1 / sample_rate)
        this_df = this_df.apply(lowpass_filter).resample(f"{down}S").mean()

        ### specify ground truth
        try:
            gt_signal = this_df["hand GSR"].to_numpy()
        except:
            gt_signal = this_df["foot GSR"].to_numpy()
        gt_signal = lowpass_filter(gt_signal, freq=sample_rate, cut=0.05)

        data.append(this_df[streams])
        gt_data.append(gt_signal)
        names.append(file.split(".")[0])

    return data, gt_data, names
