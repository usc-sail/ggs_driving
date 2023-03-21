import numpy as np, json, os, argparse
from utils import *
from datasets import load_dataset
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans

def cluster_eval(gt_data, gt_bps, bps, n_clusters, plot=False):
    X = segment_ts(gt_data, gt_bps)
    model = TimeSeriesKMeans(n_clusters, metric="dtw", random_state=42)
    clusters = model.fit_predict(X)
    if plot:
        plot_cluster(gt_data, gt_bps, clusters, bps)

    old = gt_bps.copy()
    for j in range(1, len(old) - 1):
        if clusters[j] == clusters[j - 1]:
            gt_bps.remove(old[j])

    return covering_metric(bps, gt_bps, len(gt_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        default="DriveDB",
        help="choose from DriveDB, HCIDriving, AffectiveROAD",
    )
    parser.add_argument(
        "--data_dir", default="/media/data/public-data/drive/drivedb/1.0.0/"
    )
    parser.add_argument("--missing", default=0.0, type=float, help="data missing rate")
    parser.add_argument(
        "--gt_type", default="EDA", type=str, help="choose from EDA, Rating, Fuse"
    )
    parser.add_argument(
        "--lmbda", default=1.0, type=float, help="hyperparameter of GGS algorithm"
    )
    parser.add_argument("--ncluster", default=3, type=int, help="number of clusters")
    parser.add_argument("--sample_rate", default=0.5, type=float, help="sampling rate")
    parser.add_argument(
        "--streams",
        nargs="*",
        type=str,
        default=["HR"],
        help="physio signals to experiment with: HR, BR, RESP_rate, RESP_amp",
        # help="physio signals to experiment with: HR, BR, HRV",
    )
    parser.add_argument("--plot", default=False, type=bool, help="whether to plot seg")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    data, gt_data, names = load_dataset(
        args.data_dir,
        args.dataset,
        args.missing,
        args.sample_rate,
        args.gt_type,
        args.streams,
    )

    print("Loaded. Now running ...")
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
    scores["std"] = np.around(np.std(compute_avg), 3)

    dir_name = f"runs/{args.dataset}_new/"
    os.makedirs(dir_name, exist_ok=True)
    log_name = f"{'_'.join(args.streams)}_gt_{args.gt_type}_lambda_{args.lmbda}_missing_{args.missing}_clusters_{args.ncluster}"
    with open(dir_name + log_name + ".json", "w") as f:
        f.write(json.dumps(scores, indent=4))
