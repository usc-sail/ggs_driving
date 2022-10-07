from turtle import clear
import pandas as pd, neurokit2 as nk
from scipy.signal import butter, filtfilt

def HCIDriving(path, missing, sample_rate, gt_type, streams):

    assert streams == ["HR"], "Invalid streams"

    def lowpass_filter(ts=None, freq=1024, cut=0.25):
        b, a = butter(3, cut, fs=freq, btype='low')
        return filtfilt(b, a, ts)

    def fuse_gt(s1, s2):
        combined = ...
        return combined

    data, gt_data, names = [], [], []
    for i in range(1, 11):

        ### data loading
        this_path = path + f"participant_{i}.csv"
        this_df = pd.read_csv(this_path, delimiter=";")[
            ["Time_Light", "ECG", "HR", "HRV_LF", "SCR", "Speed_GPS", "Rating_Videorating"]
        ]

        ### process ECG independently
        ...
        ### masking of "missing" values
        ...

        ### lowpass filter (0.1Hz) + downsample to 0.5Hz
        this_df["Time_Light"] = pd.to_datetime(
            this_df["Time_Light"],
            format='%H:%M:%S:%f'
        )
        this_df = this_df.drop_duplicates().set_index("Time_Light")
        down = int(1/sample_rate)
        clean_data = this_df.apply(lowpass_filter).resample(f"{down}S").mean()

        ### smooth + same for ground truth
        stream1 = clean_data["SCR"].to_numpy().squeeze()
        stream2 = clean_data["Rating_Videorating"].to_numpy().squeeze()
        if gt_type == "EDA":
            gt_signal = stream1
        elif gt_type == "Rating":
            gt_signal = stream2
        else:
            gt_signal = fuse_gt(stream1, stream2)
        gt_signal = lowpass_filter(gt_signal, freq=sample_rate, cut=0.1)

        data.append(clean_data[streams])
        gt_data.append(gt_signal)
        names.append(f"participant_{i}")

    return data, gt_data, names