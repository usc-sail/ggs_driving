import numpy as np, json, os, argparse
from utils import *
from datasets import load_dataset
from tqdm import tqdm
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
    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default="DriveDB", help="dataset used for experiments: choose from DriveDB, HCIDriving, AffectiveROAD")
    parser.add_argument('--data_dir', default="/media/data/public-data/drive/drivedb/1.0.0/")
    parser.add_argument('--missing', default=0.0, type=float, help="data missing rate")
    parser.add_argument('--gt_type', default="EDA", type=str, help="choose from EDA, Rating, Fuse")
    parser.add_argument('--lmbda', default=1.0, type=float, help="hyperparameter of GGS algorithm")
    parser.add_argument('--ncluster', default=3, type=int, help="number of clusters")
    parser.add_argument('--sample_rate', default=0.5, type=float, help="sampling rate")
    parser.add_argument("--streams", nargs="*", type=str, default=["HR"], help="physio signals to experiment with: HR, BR, RESP_rate, RESP_amp")
    args = parser.parse_args()

    # load data set
    data, gt_data, names = load_dataset(args.data_dir, args.dataset, args.missing, args.sample_rate, args.gt_type, args.streams)
    scores, compute_avg = dict(), list()
    for i in tqdm(range(len(names))):
        score = cluster_eval(
            gt_data[i],
            gt_bps=apply_ggs(gt_data[i], lmbda=args.lmbda),
            bps=apply_ggs(data[i].to_numpy(), lmbda=args.lmbda),
            n_clusters=args.ncluster,
        )
        compute_avg.append(score)
        scores[names[i]] = np.around(score, 3)
    scores["mean"] = np.around(np.mean(compute_avg), 3)

    ### logging configuration
    dir_name = f"runs/{args.dataset}/"
    # import pdb
    # pdb.set_trace()
    log_name = f"{'_'.join(args.streams)}_gt_{args.gt_type}_lambda_{args.lmbda}_missing_{args.missing}_clusters_{args.ncluster}"

    os.makedirs(dir_name, exist_ok=True)
    with open(dir_name + log_name + ".json", "w") as f:
        f.write(json.dumps(scores, indent=4))
