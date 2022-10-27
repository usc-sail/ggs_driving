import os, pandas as pd, numpy as np
from scipy.signal import butter, filtfilt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def lowpass_filter_bio(ts=None, freq=1, cut=0.05):
    b, a = butter(3, cut, fs=freq, btype="low")
    return filtfilt(b, a, ts)


def lowpass_filter(ts=None, freq=4, cut=0.05):
    b, a = butter(3, cut, fs=freq, btype="low")
    return filtfilt(b, a, ts)


def preprocess_bioharness_data(bioharness_path, start_index, stop_index):

    # load the data
    df = pd.read_csv(bioharness_path, delimiter=";")
    # start index to end index
    df = df[start_index:stop_index]
    df["Time"] = pd.to_datetime(df[df.columns[0]])
    return df.drop_duplicates().set_index("Time")


def process_e4_df(e4_path, sample_rate, start_index, end_index):

    # load the EDA data
    e4_df = pd.read_csv(e4_path, header=None)

    # extract the unix timestamp and sample rate for left and right EDA
    unix_timestamp_e4 = e4_df[0].iloc[0]
    sample_rate_e4 = e4_df[0].iloc[1]

    # drop the first two rows
    e4_df = e4_df.drop([0, 1])

    timestep = 1 / sample_rate_e4
    base_timestamp = datetime.fromtimestamp(int(unix_timestamp_e4))
    index_timestamp = [
        base_timestamp + i * timedelta(seconds=timestep)
        for i in np.arange(e4_df.shape[0])
    ]
    e4_df["Time"] = index_timestamp
    e4_df = e4_df.rename(columns={0: "EDA"})
    e4_df["Time"] = pd.to_datetime(e4_df["Time"])

    # start index to end index
    e4_df = e4_df.drop_duplicates().set_index("Time")
    e4_df = e4_df[start_index:end_index]
    # e4_df = e4_df.iloc[selected]
    t = e4_df.index
    e4_df = e4_df.apply(lowpass_filter).resample(f"{1 / sample_rate}S").mean()

    return e4_df, t


def process_e4_df_1_2(
    e4_path_1,
    e4_path_2,
    sample_rate,
    start_index,
    end_index,
    column_name="EDA",
):

    # load the eda data
    e4_df_1 = pd.read_csv(e4_path_1, header=None)
    e4_df_2 = pd.read_csv(e4_path_2, header=None)

    # extract the unix timestamp and sample rate for left and right EDA
    unix_timestamp_e4_1 = e4_df_1[0].iloc[0]
    sample_rate_e4_1 = e4_df_1[0].iloc[1]

    # drop the first two rows
    e4_df_1 = e4_df_1.drop([0, 1])
    e4_df_2 = e4_df_2.drop([0, 1])

    e4_df = e4_df_1.append(e4_df_2)

    timestep = 1 / sample_rate_e4_1
    base_timestamp = datetime.fromtimestamp(int(unix_timestamp_e4_1))
    index_timestamp = [
        base_timestamp + i * timedelta(seconds=timestep)
        for i in np.arange(e4_df.shape[0])
    ]
    e4_df["Time"] = index_timestamp
    e4_df = e4_df.rename(columns={0: column_name})
    e4_df["Time"] = pd.to_datetime(e4_df["Time"])

    # start index to end index
    down = 1 / sample_rate
    e4_df = e4_df.drop_duplicates().set_index("Time")
    e4_df = e4_df[start_index:end_index]
    e4_df = e4_df.apply(lowpass_filter).resample(f"{down}S").mean()

    return e4_df


def generate_numeric_id(filename):
    res = [int(i) for i in filename if i.isdigit()]
    return str(res[0]) if len(res) == 1 else str(res[0] * 10 + res[1])


def AffectiveROAD(path, missing, sample_rate, gt_type, streams):
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

    bio_path = path + "Bioharness/"
    e4_path = path + "E4/"
    metrics_path = path + "Subj_metric/"

    # load the E4 and bioharness metadata
    bio_annot_df = pd.read_csv(bio_path + "Annot_Bioharness.csv")
    e4_annot_l_df = pd.read_csv(e4_path + "Annot_E4_Left.csv")
    e4_annot_r_df = pd.read_csv(e4_path + "Annot_E4_Right.csv")
    subj_metric_df = pd.read_csv(metrics_path + "Annot_Subjective_metric.csv")

    # start and end indices
    start = "Z_Start"
    end = "Z_End.1"

    data, gt_data = [], []
    names = [d.split("_")[-1].split(".")[0] for d in os.listdir(bio_path) if "Bio" in d]
    names.remove("Bioharness")
    names.remove("Drv2")

    for drive in names:

        # bio annot for specific driver with start and end indices
        bio_annot_driv_metadata = bio_annot_df[bio_annot_df["Drive_id"] == drive]
        start_index_bio = bio_annot_driv_metadata[start].iloc[0]
        stop_index_bio = bio_annot_driv_metadata[end].iloc[0]

        # extract the start and stop indices for left and right EDA
        e4_annot_l_metadata = e4_annot_l_df[e4_annot_l_df["Drive-id"] == drive]
        start_index_l = e4_annot_l_metadata[start].iloc[0]
        stop_index_l = e4_annot_l_metadata[end].iloc[0]

        e4_annot_r_metadata = e4_annot_r_df[e4_annot_r_df["Drive-id"] == drive]
        start_index_r = e4_annot_r_metadata[start].iloc[0]
        stop_index_r = e4_annot_r_metadata[end].iloc[0]

        # metrics annot for specific driver with start and end indices
        subj_annot_driv_metadata = subj_metric_df[subj_metric_df["Drive_id"] == drive]
        start_index_subj = subj_annot_driv_metadata[start].iloc[0]
        stop_index_subj = (
            start_index_subj + stop_index_r - start_index_r
        )  # align with E4

        # bioharness csv file processing for HR and BR
        bio_harness_csv_file = os.path.join(bio_path, "Bio_" + drive + ".csv")
        this_df = preprocess_bioharness_data(
            bio_harness_csv_file, start_index_bio, stop_index_bio
        )
        # apply masking and downsampling to bioharness
        this_df[streams] = this_df[streams].apply(mask_intervals)
        down = int(1 / sample_rate)
        if "Unnamed: 0" in this_df.columns:
            this_df = this_df.drop(columns=["Unnamed: 0"])
        this_df = this_df.apply(lowpass_filter_bio).resample(f"{down}S").mean()

        # process left and right E4 data
        numeric_id = generate_numeric_id(drive)
        select = "Left2" if drive == "Drv2" else "Left"
        e4_left_file = os.path.join(
            e4_path, numeric_id + "-E4-" + drive, select, "EDA.csv"
        )
        e4_left_df, t = process_e4_df(
            e4_left_file, sample_rate, start_index_l, stop_index_l
        )

        e4_right_file = os.path.join(
            e4_path, numeric_id + "-E4-" + drive, "Right", "EDA.csv"
        )
        e4_right_df, _ = process_e4_df(
            e4_right_file, sample_rate, start_index_r, stop_index_r
        )

        # process subjective metrics data
        subj_metrics_csv_file = os.path.join(metrics_path, f"SM_{drive}.csv")
        metric = pd.read_csv(subj_metrics_csv_file)
        metric = metric[start_index_subj:stop_index_subj]

        # align time with E4
        temp = metric.to_numpy().squeeze()
        temp = np.pad(temp, (0, len(t) - len(metric)), "mean")
        metric = metric.reindex(t, fill_value=0)
        metric["Rating"] = temp

        metric = metric.iloc[:, 1:]
        metric = metric.apply(lowpass_filter).resample(f"{1 / sample_rate}S").mean()

        data.append(this_df[streams])
        # mean of left and right EDA
        if gt_type == "EDA":
            gt_signal = (
                e4_left_df["EDA"].to_numpy() + e4_right_df["EDA"].to_numpy()
            ) / 2
        elif gt_type == "Rating":
            gt_signal = metric.to_numpy().squeeze()
        else:
            gt_eda = (e4_left_df["EDA"].to_numpy() + e4_right_df["EDA"].to_numpy()) / 2
            gt_eda /= np.max(gt_eda)
            gt_signal = (gt_eda + metric.to_numpy().squeeze()) / 2

        gt_signal = lowpass_filter(gt_signal, freq=0.5, cut=0.01)
        gt_data.append(gt_signal)

    return data, gt_data, names


if __name__ == "__main__":

    data, gt_data, _ = AffectiveROAD(
        path="/home/kavra/Datasets/AffectiveROAD/Database/",
        missing=0,
        sample_rate=0.5,
        gt_type="Fused",
        streams=["HR"],
    )
    print(f"Successfully loaded AffectiveROAD with length {len(data)}.")
