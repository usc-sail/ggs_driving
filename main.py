import numpy as np
from utils import *
from datasets import load_dataset
from tslearn.clustering import TimeSeriesKMeans


def cluster_eval(gt_data, gt_bps, bps, n_clusters):
    X = segment_ts(gt_data, gt_bps)
    model = TimeSeriesKMeans(n_clusters, metric="dtw", random_state=42)
    clusters = model.fit_predict(X)
    # plot_cluster(gt_signal, gt_bps, clusters)

    old = gt_bps.copy()
    for i in range(1, len(old) - 1):
        if clusters[i] == clusters[i - 1]:
            gt_bps.remove(old[i])

    return covering_metric(bps, gt_bps, len(gt_data))


if __name__ == "__main__":

    dataset = "DriveDB"  # choose from ["DriveDB", "HCIDriving", "AffectiveROAD"]
    missing = 0  # percentage of data points to be removed
    sample_rate = 0.5  # final sample rate of signals (in Hz)
    gt_type = "EDA"  # choose from ["EDA", "Rating", "Fuse"]
    lmbda = 1  # hyperparameter of GGS algorithm
    n_clusters = 3  # number of clusters for the ground truth
    streams = [  # physio signals to experiment with
        "HR",
        # "BR",
        "RESP_rate",
        "RESP_amp",
    ]

    print("Dataset:", dataset)
    data, gt_data, names = load_dataset(dataset, missing, sample_rate, gt_type, streams)

    scores = []
    for i in range(len(names)):
        print(f"\n{names[i]}")
        score = cluster_eval(
            gt_data[i],
            gt_bps=apply_ggs(gt_data[i], lmbda=lmbda),
            bps=apply_ggs(data[i].to_numpy(), lmbda=lmbda),
            n_clusters=n_clusters,
        )
        print("Cover Metric:", np.around(score, 3))
        scores.append(score)

    print("\nAverage Score:", np.around(np.mean(scores), 3))
