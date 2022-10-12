#!/usr/bin/env bash
for ncluster in 3 4 5
  do
  for dataset in DriveDB
    do
    # change to your own data folder location
    if [ "$dataset" = "DriveDB" ]; then
      data_dir="/media/data/public-data/drive/drivedb/1.0.0/"
    elif [ "$dataset" = "HCIDriving" ]; then
      data_dir="/media/data/public-data/drive/drivedb/1.0.0/"
    else
      data_dir="/media/data/public-data/drive/drivedb/1.0.0/"
    fi

    python3 ./main.py \
      --dataset $dataset \
      --streams HR \
      --sample_rate 0.5 \
      --ncluster $ncluster \
      --lmbda 10.0 \
      --gt_type EDA \
      --missing 0.25 \
      --data_dir $data_dir
  done
done
