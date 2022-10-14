#!/usr/bin/env bash
for ncluster in 2 3 4 5
  do
  for dataset in HCIDriving DriveDB
    do
    # change to your own data folder location
    if [ "$dataset" = "DriveDB" ]; then
      data_dir="/home/kavra/Datasets/physionet.org/files/drivedb/1.0.0/"
    elif [ "$dataset" = "HCIDriving" ]; then
      data_dir="/home/kavra/Datasets/hcilab_driving_dataset/"
    else
      data_dir="/home/kavra/Datasets/AffectiveROAD/"
    fi

    python3 ./main.py \
      --dataset $dataset \
      --streams HR \
      --sample_rate 0.5 \
      --ncluster $ncluster \
      --lmbda 15 \
      --gt_type EDA \
      --missing 0 \
      --data_dir $data_dir
  done
done
