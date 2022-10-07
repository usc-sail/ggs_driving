from .load_affectiveroad import AffectiveROAD
from .load_drivedb import DriveDB
from .load_hcidriving import HCIDriving


def load_dataset(dataset, missing, sample_rate, gt_type, streams):
    if dataset == "AffectiveROAD":
        path = f"/home/kavra/Datasets/AffectiveROAD/Database/"
        return AffectiveROAD(path, missing, sample_rate, gt_type, streams)
    elif dataset == "DriveDB":
        path = f"/home/kavra/Datasets/physionet.org/files/drivedb/1.0.0/"
        return DriveDB(path, missing, sample_rate, gt_type, streams)
    elif dataset == "HCIDriving":
        path = f"/home/kavra/Datasets/hcilab_driving_dataset/"
        return HCIDriving(path, missing, sample_rate, gt_type, streams)
    else:
        return NotImplementedError
