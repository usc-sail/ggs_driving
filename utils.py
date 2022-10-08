import matplotlib.pyplot as plt, numpy as np
from itertools import combinations
from ggs import GGS


def apply_ggs(data, kmax=20, lmbda=15):
    data = data.T if len(data.shape) != 1 else data[None, ...]
    bps, _ = GGS(data, kmax, lmbda)
    return bps[-1] if isinstance(bps[0], list) else bps


def plot_ggs(signal, bps):
    plt.figure(figsize=(20, 4))
    plt.plot(signal)
    for x in bps:
        plt.axvline(x=x, linestyle="--", color="black")
    # plt.yticks([])
    plt.show()


def segment_ts(ts, bps):
    X = [ts[bps[i] : bps[i + 1]] for i in range(len(bps) - 1)]
    lens = [len(li) for li in X]
    X = [
        np.pad(X[i], (0, max(lens) - len(X[i])), constant_values=(0, np.nan))
        for i in range(len(X))
    ]
    return np.stack(X)


def plot_cluster(signal, bps, clusters):
    plt.figure(figsize=(20, 4))
    plt.plot(signal)
    colors = ["red", "green", "blue", "orange"]
    for i in range(len(bps) - 1):
        plt.axvspan(bps[i], bps[i + 1], facecolor=colors[clusters[i]], alpha=0.25)
    # plt.yticks([])
    plt.show()


def jaccard(set1, set2):
    intersection = len(np.intersect1d(set1, set2))
    union = len(set1) + len(set2) - intersection
    return float(intersection) / union


def covering_metric(bps, gt_bps, length):
    cover = 0
    for i in range(len(gt_bps) - 1):
        set1 = np.arange(gt_bps[i], gt_bps[i + 1])
        jaccards = [
            jaccard(set1, np.arange(bps1, bps2))
            for (bps1, bps2) in combinations(bps, 2)
        ]
        cover += len(set1) * max(jaccards)
    return cover / length
