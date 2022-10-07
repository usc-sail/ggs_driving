from .load_affectiveroad import AffectiveROAD
from .load_drivedb import DriveDB
from .load_hcidriving import HCIDriving

def load_dataset(
    dataset, missing, sample_rate, gt_type, streams
):
    if dataset == "AffectiveROAD":
        path = f"/home/kavra/Datasets/{dataset}/Database/"
        return AffectiveROAD(path, missing, sample_rate, gt_type, streams)
    elif dataset == "DriveDB":
        path = f"/home/kavra/Datasets/{dataset}/"
        return DriveDB(path, missing, sample_rate, gt_type, streams)
    elif dataset == "HCIDriving":
        path = f"/home/kavra/Datasets/{dataset}/"
        return HCIDriving(path, missing, sample_rate, gt_type, streams)
    else:
        return NotImplementedError