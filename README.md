# Leukemia Raman classification

This repository contains research code for classifying label-free Raman
cell maps as acute myeloid leukemia (AML) or acute lymphoblastic leukemia (ALL).
It brings together five multiple instance learning (MIL) models and a simple
workflow: download the data, preprocess it, train the models, and make predictions.

The related research was presented as a poster at **SPEC 2026**. If this project
helps your research, please see [Citing this work](#citing-this-work) below.

To get started, open a terminal in this project folder with **Python 3.11** and install
the dependencies:

```bash
python -m pip install -r requirements.txt
```

All parameters are loaded automatically from `config.json`, so you can use the
commands below without choosing settings first.

## 1. Download the dataset

Download the nine `.tar` files from [the Vanna leukemia Raman dataset](https://zenodo.org/records/11195904)
and place them in a `data/raw/` folder inside this project. You can leave the archives
compressed. Extracted Feather files are also supported; use one copy of each map.

The data were created and shared by Vanna and colleagues. Please credit their
[dataset and original study](#dataset-and-original-study) when using these data.

## 2. Preprocess

```bash
python preprocess.py
```

This prepares the downloaded, release-level spectra for the models by averaging
non-overlapping 8×8 spatial patches. The results are cached in `data/cache/`, so you
only need to preprocess once. If your downloaded data are somewhere else, use:

```bash
python preprocess.py --data-dir /path/to/downloaded/dataset
```

## 3. Train

```bash
python train.py
```

One command trains all five models below using the supplied grouped four-fold split.
Normalization is fitted on inner-training data only. The table shows the 
results used to select these models from the consolidated single-view comparison.

| Model | Fold-mean balanced accuracy |
| --- | ---: |
| UG-DEC-MIL | 0.7352 |
| DEC-MIL | 0.7192 |
| Attention-MIL | 0.7117 |
| Gated ABMIL | 0.7031 |
| CLAM-SB | 0.6913 |

Prefer to start with just one model? Run:

```bash
python train.py --model ug-dec-mil
```

## 4. Model weights are saved automatically

There is no separate saving step. Training writes `outputs/<model>/fold_1.pt` through
`fold_4.pt`, along with held-out predictions and metrics. Each checkpoint includes
the preprocessing settings and training-fitted normalization statistics needed for inference.

Existing caches and model folders are kept safe and are not overwritten.
If preprocessing or training stops, the error names the partial folder and explains how
to restart. Rename that folder as a backup or choose a fresh cache/output path in
`config.json`, then rerun the command. Training restarts from the beginning; use
`--model <name>` to restart just the affected model. Ctrl+C exits with code 130.

## 5. Run inference

Use a saved checkpoint to classify another map:

```bash
python predict.py --checkpoint outputs/ug-dec-mil/fold_1.pt --input /path/to/new_cell.feather
```

Replace `/path/to/new_cell.feather` with your input file. You can also give a folder
or tar archive. The command loads the weights and normalization automatically and
writes labels and AML probabilities to `predictions.csv`. Use
`--output another_file.csv` to choose a different destination.

## 6. View your results

Each model's output folder contains `metrics.csv` for the overall scores,
`fold_metrics.csv` for the individual folds, and `predictions.csv` for the held-out
cell-map predictions.

The historical scores above come from the preserved study. This simplified implementation
has passed synthetic workflow tests; its full real-data rerun is still pending dataset
restoration. This is research code: dataset groups are patient surrogates, and attention
maps are qualitative rather than localization ground truth.

### Changing the defaults

Edit `config.json` if you want to change the settings. Encoder widths and dropout
under `model` apply to all five models. Model-specific attention, pooling, and
projection settings are under `model.heads`.

## Citing this work

### Dataset and original study

If you use the dataset, please cite the original data release and the associated paper:

- **Data release:** Vanna and colleagues (2024), Zenodo, version 1.
  [doi:10.5281/zenodo.11195904](https://doi.org/10.5281/zenodo.11195904).
- **Original paper:** Vanna et al. (2024), *High-Resolution Raman Imaging of >300
  Patient-Derived Cells from Nine Different Leukemia Subtypes: A Global Clustering
  Approach*. *Analytical Chemistry*.
  [doi:10.1021/acs.analchem.4c00787](https://doi.org/10.1021/acs.analchem.4c00787).

### This repository and our poster

If you use this code or build on it in your research, please link to
[this repository](https://github.com/HusnuBarisBaydargil/leukemia-raman-mil), ideally
including the commit or release you used, and consider citing our related poster:

> Baydargil, H. B., & Bocklitz, T. (2026, May 16–20). *Grouped Evaluation of Deep
> Learning on Label-Free Leukemia Raman Cell Maps* [Poster presentation]. SPEC 2026,
> Vanderbilt University, Nashville, TN, United States.

Thank you for acknowledging both the original dataset creators
and the work behind this repository.
