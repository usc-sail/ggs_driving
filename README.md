# Tracking Arousal-level Changes on Drivers

```
git clone https://github.com/usc-sail/ggs_driving
cd ggs_driving && pip install -r requirements.txt
```

## Datasets

To run experiments you need to download either of the following datasets:

* [AffectiveROAD](https://www.media.mit.edu/tools/affectiveroad/)
* [DriveDB](https://physionet.org/content/drivedb/1.0.0/)
* [HCIDriving](https://www.hcilab.org/research/hcilab-driving-dataset/)

After extracting them you should modify the paths in `datasets/__init__.py` accordingly.

## Experiments

You can modify parameters of the experiment at the `main.py` file.
```
./run.sh
```

Results are stored under `runs/{dataset}/` for each utilized dataset.