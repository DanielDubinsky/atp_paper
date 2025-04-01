# Non-Invasive Quantification of Viability in Spheroids using Deep Learning
This repository contains the code for the Neural Viability Regression (NViR) model introduced in our paper on cell viability assessment using deep learning. The NViR model is designed to non-invasively assess the viability of spheroids using bright-field microscopy images, offering a novel approach in the field of cellular biology.

## setup
To set up your environment to run these scripts, you'll need Python3.10 installed on your system. Once Python is installed, you can set up the project dependencies with the following commands:
```
pip install -r requirements.txt
pip install -e .
```

## Download data
The necessary datasets for training and evaluating the NViR model can be downloaded using the following instructions:
```
WILL BE UPDATED BEFORE PUBLICATION
```

## Train NViR
To train the models that reproduce the R-squared (R2) and Mean Absolute Error (MAE) tables for the NViR model as presented in our paper, run the following script:
```
./scripts/experiments/0001.sh
```

## Evaluate NViR
For evaluating the model with the checkpoint file, use:
```
python scripts/evaluate.py data/model_11lrj9za.ckpt data/data_labeled.csv data/images_labeled_256/
```

## Predict
For generating predictions with the checkpoint file use:
```
python scripts/predict.py data/model_11lrj9za.ckpt data/data_unlabeled.csv data/images_unlabeled_256 data/unlabeled_w_preds.csv
```

## Train Baselines
To reproduce the R2 and MAE tables for the baseline models used in the paper:
1. Generate classic features
```
python scripts/baselines/cve.py
```
2. Train
```
python scripts/baselines/train.py
```
## Staining Intensities and Correlations
To analyze staining intensities and their correlations:
1. crop spheroids
```
python scripts/crop_spheroid.py data/data_labeled.csv data/images_labeled/ data/images_labeled_cropped_0.2/
python scripts/crop_spheroid.py data/data_chromalive.csv data/images_chromalive/ data/images_chromalive_cropped_0.2/
```
2. Calculate intensities and correlations
```
python scripts/stainings/stainings.py
```


## Drug-Induced Liver Injury (DILI) Prediction
For DILI prediction using the NViR model, given a csv with predictions:
1. calculate ic50
```
python scripts/dili/calc_ic50.py data/unlabeled_w_preds.csv data/ic50.csv
```
2. calculate roc auc of MOS
```
python scripts/dili/auc.py data/ic50.csv
```
