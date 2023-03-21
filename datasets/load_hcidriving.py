import pandas as pd, numpy as np, neurokit2 as nk
from scipy.signal import butter, filtfilt
from raaw import compute_EWE
import matplotlib.pyplot as plt


def HCIDriving(path, missing, sample_rate, gt_type, streams):

    assert all(s in ["HR", "BR"] for s in streams), "Invalid stream set"

    def lowpass_filter(ts=None, freq=1024, cut=0.25):
        b, a = butter(3, cut, fs=freq, btype="low")
        return filtfilt(b, a, ts)

    def mask_intervals(signal=None, missing=missing):
        # undo signal upsampling
        down_signal = np.array(signal)[::8]

        intervals, durations = [], []
        min_win, max_win = 0 * len(down_signal), 0.01 * len(down_signal)

        def cap(a, b):
            return [i for i in a if i in b]

        while sum(durations) < missing * len(down_signal):
            random_start = np.random.randint(0, len(down_signal) - max_win)
            random_end = random_start + np.random.randint(min_win, max_win)
            random_win = np.arange(random_start, random_end)

            intersections = [len(cap(p, random_win)) for p in intervals]
            if sum(intersections) >= random_end - random_start:
                continue

            intervals.append(random_win)
            durations.append(random_end - random_start - sum(intersections))

        # interpolate
        for interval in intervals:
            down_signal[interval] = (
                down_signal[interval[0] - 1] if interval[0] else down_signal[0]
            )

        # re-apply to the original
        upsampled = np.repeat(down_signal, 8)
        for i in range(len(signal)):
            signal[i] = upsampled[i]

        return signal

    def fuse_gt(s1, s2):
        s1 = (s1 - np.mean(s1)) / np.std(s1)
        s2 = (s2 - np.mean(s2)) / np.std(s2)
        input = np.expand_dims(np.array([s1, s2]), axis=0)
        output = compute_EWE(np.expand_dims(input, axis=-1))
        return output[0].squeeze()

    data, gt_data, names = [], [], []
    for i in range(1, 11):

        ### data loading
        this_path = path + f"participant_{i}.csv"
        this_df = pd.read_csv(this_path, delimiter=";")[
            [
                "Time_Light",
                "ECG",
                "HR",
                "HRV_LF",
                "SCR",
                "Speed_GPS",
                "Rating_Videorating",
            ]
        ]

        ### extract ECG-derived Respiration
        if "BR" in streams:
            rpeaks, _ = nk.ecg_peaks(this_df["ECG"], sampling_rate=1024)
            ecg_rate = nk.ecg_rate(
                rpeaks, sampling_rate=1024, desired_length=len(this_df["ECG"])
            )
            # this_df["HR"] = ecg_rate
            this_df["BR"] = nk.ecg_rsp(ecg_rate, sampling_rate=1024)

        ### masking of "missing" values
        if missing > 0:
            this_df[streams] = this_df[streams].apply(mask_intervals)

        ### lowpass filter (0.1Hz) + downsample to 0.5Hz
        this_df["Time_Light"] = pd.to_datetime(
            this_df["Time_Light"], format="%H:%M:%S:%f"
        )
        this_df = this_df.drop_duplicates().set_index("Time_Light")
        down = int(1 / sample_rate)
        clean_data = this_df.apply(lowpass_filter).resample(f"{down}S").mean()

        ### smooth + same for ground truth
        stream1 = clean_data["SCR"].to_numpy().squeeze()
        stream1 = lowpass_filter(stream1, freq=sample_rate, cut=0.05)
        stream2 = clean_data["Rating_Videorating"].to_numpy().squeeze()
        stream2 = lowpass_filter(stream2, freq=sample_rate, cut=0.05)

        if gt_type == "EDA":
            gt_signal = stream1
        elif gt_type == "Rating":
            gt_signal = stream2
        else:
            gt_signal = fuse_gt(stream1, stream2)

        data.append(clean_data[streams])
        gt_data.append(gt_signal)
        names.append(f"participant_{i}")

    return data, gt_data, names
