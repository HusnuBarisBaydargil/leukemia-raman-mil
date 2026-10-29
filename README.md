# Leukemia Raman classification

Classify label-free Raman cell maps as AML or ALL using five selected MIL models.
Parameters are loaded automatically from `config.json`.
The encoder widths and dropout under `model` apply to all five models.
Model-specific attention, pooling, and projection settings are under `model.heads`.

Install dependencies with Python 3.11:

```bash
pip install -r requirements.txt
```

## 1. Download the dataset

Download the nine `.tar` files from [the Vanna leukemia Raman dataset](https://zenodo.org/records/11195904)
into `data/raw/`. You can leave the archives compressed. Use either archives or extracted
Feather files, not both copies of the same maps.

## 2. Preprocess

```bash
python preprocess.py
```

This uses the release-level spectra, averages non-overlapping 8×8 spatial patches,
and writes patch bags to `data/cache/`. For another dataset location:

```bash
python preprocess.py --data-dir /path/to/downloaded/dataset
```

## 3. Train

```bash
python train.py
```

This trains the top five models from the consolidated single-view comparison using
the supplied grouped four-fold split. Normalization is fitted on inner-training data only.

| Model | Historical fold-mean balanced accuracy |
| --- | ---: |
| UG-DEC-MIL | 0.7352 |
| DEC-MIL | 0.7192 |
| Attention-MIL | 0.7117 |
| Gated ABMIL | 0.7031 |
| CLAM-SB | 0.6913 |

To train one model:

```bash
python train.py --model ug-dec-mil
```

## 4. Model weights are saved automatically

Training saves `outputs/<model>/fold_1.pt` through `fold_4.pt`, along with held-out
predictions and metrics. Each checkpoint includes its preprocessing settings and
training-fitted normalization statistics. Existing caches and model folders are not overwritten.
If preprocessing or training stops, the error names the partial folder and explains how
to restart. Rename that folder as a backup or choose a fresh cache/output path in
`config.json`, then rerun the command. Training restarts from the beginning; use
`--model <name>` to restart just the affected model. Ctrl+C exits with code 130.

## 5. Run inference

```bash
python predict.py --checkpoint outputs/ug-dec-mil/fold_1.pt --input /path/to/new_cell.feather
```

Input can also be a folder or tar archive. Predictions are written to `predictions.csv`;
use `--output another_file.csv` to choose a different destination.

## 6. Done

Historical scores are from the preserved study; this simplified implementation has
been tested on synthetic data, and its full real-data rerun is pending dataset restoration.
This is a research classifier. Dataset groups are patient surrogates, and attention
is not localization ground truth.
