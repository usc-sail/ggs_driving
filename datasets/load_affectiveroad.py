import os, pandas as pd
from scipy.signal import butter, filtfilt


def Bioharness():
    return


def Empatica4():
    return


def SubjMetric():
    return


def AffectiveROAD(path, missing, sample_rate, gt_type, streams):

    bio_path = path + "Bioharness/"
    e4_path = path + "E4/"
    sm_path = path + "Subj_metric/"

    bio_annot = bio_path + "Annot_Bioharness.csv"
    e4_annot_l = e4_path + "Annot_E4_Left.csv"
    e4_annot_r = e4_path + "Annot_E4_Left.csv"
    sm_annot = sm_path + "Annot_Subjective_metric.csv"

    def lowpass_filter(ts=None, freq=1, cut=0.05):
        b, a = butter(3, cut, fs=freq, btype="low")
        return filtfilt(b, a, ts)

    data, gt_data, names = [], [], []
    for drive in os.listdir(bio_path):
        if not drive.startswith("Bio"):
            continue

        ### data loading
        this_df = pd.read_csv(bio_path + drive, delimiter=";")
        this_df["Time"] = pd.to_datetime(this_df[this_df.columns[0]])
        if this_df.columns[0] != "Time":
            this_df = this_df.drop(columns=[this_df.columns[0]], axis=1)
        this_df = this_df.drop_duplicates().set_index("Time")

        ### lowpass filter (0.05Hz) + downsample to 0.5Hz
        down = int(1 / sample_rate)
        this_df = this_df.apply(lowpass_filter).resample(f"{down}S").mean()

        ### specify ground truth and smooth
        gt_signal = this_df["Activity"].to_numpy()
        gt_signal = lowpass_filter(gt_signal, freq=0.5, cut=0.01)

        data.append(this_df[["HR", "BR"]])
        gt_data.append(gt_signal)
        names.append(drive.split(".")[0])

    return data, gt_data, names
