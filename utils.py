import matplotlib.pyplot as plt, numpy as np
from itertools import combinations
from ggs import GGS


def apply_ggs(data, fs=0.5, lmbda=1):
    data = data.T if len(data.shape) != 1 else data[None, ...]
    kmax = data.shape[-1] // (60 * 3 * fs)
    bps, _ = GGS(data, int(kmax), lmbda)
    return bps[-1] if isinstance(bps[0], list) else bps


def plot_ggs(signal, bps):
    plt.figure(figsize=(20, 4))
    plt.plot(signal)
    for x in bps:
        plt.axvline(x=x, linestyle="--", color="black")
    plt.show()


def segment_ts(ts, bps):
    X = [ts[bps[i] : bps[i + 1]] for i in range(len(bps) - 1)]
    lens = [len(li) for li in X]
    X = [
        np.pad(X[i], (0, max(lens) - len(X[i])), constant_values=(0, np.nan))
        for i in range(len(X))
    ]
    return np.stack(X)


def plot_cluster(signal, gt_bps, clusters, bps):
    plt.figure(figsize=(20, 4))
    norm = 60 * 0.5

    time = np.linspace(0, len(signal) / norm, num=len(signal))
    plt.plot(time, signal, color="black")

    plt.xlim(0, len(signal) / norm)
    plt.xticks(np.arange(0, len(signal) / norm, 5))
    plt.xlabel("Time (min)")
    plt.ylabel("Clustered Ground Truth")
    plt.title("Breakpoint Proposals")

    num = len(set(clusters))
    colors = (
        ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]
        if num == 5
        else ["#003f5c", "#7a5195", "#ef5675", "#ffa600"]
        if num == 4
        else ["#003f5c", "#bc5090", "#ffa600"]
        if num == 3
        else ["#003f5c", "#ffa600"]
    )
    for i in range(len(gt_bps) - 1):
        plt.axvspan(
            gt_bps[i] / norm,
            gt_bps[i + 1] / norm,
            facecolor=colors[clusters[i]],
            alpha=0.3,
        )
    for x in bps:
        plt.axvline(x=x / norm, linestyle="--", color="black")
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
