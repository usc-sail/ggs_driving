# Tracking Stress-level Changes on Drivers

Applying GGS to physio- and bio-signals.

## Datasets

To run experiments you need to download either of the following datasets:

* [AffectiveROAD](https://www.media.mit.edu/tools/affectiveroad/)
* [DriveDB](https://physionet.org/content/drivedb/1.0.0/)
* [HCIDriving](https://www.hcilab.org/research/hcilab-driving-dataset/)

After extracting them you should modify the paths in `datasets/__init__.py` accordingly.

## Experiments

You can modify parameters of the experiment at the `main.py` file.
```
pip install -r requirements.txt
python main.py
```