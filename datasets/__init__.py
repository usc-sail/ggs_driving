from .load_affectiveroad import AffectiveROAD
from .load_drivedb import DriveDB
from .load_hcidriving import HCIDriving


def load_dataset(path, dataset, missing, sample_rate, gt_type, streams):
    if dataset == "AffectiveROAD":
        return AffectiveROAD(path, missing, sample_rate, gt_type, streams)
    elif dataset == "DriveDB":
        return DriveDB(path, missing, sample_rate, gt_type, streams)
    elif dataset == "HCIDriving":
        return HCIDriving(path, missing, sample_rate, gt_type, streams)
    else:
        return NotImplementedError
