# Metsa Group Project

## Bleach ratio prediction

### Requirements

1. Access to the data
2. Python 3.8.5

### Installation

1. Clone this repository
2. Export bleach ratio data from Azure in csv format, name it for example "some_samples.csv", and put it into directory ```data/raw```
3. pip3 install -r requirements

### How to use

1. Create datasets

```bash
python3 create_datasets.py -s <csv file with samples, e.g. some_samples.csv>
```

2. Train a model

```bash
python3 train_models_to_predict_bleach_ratio.py
```

3. Test the model (TO DO)

```bash
python3 test_models_to_predict_bleach_ratio.py -s <csv file with samples, e.g. more_samples.csv>
```

### Results

1. Prediction errors for different delay classes and different models can be found [here](results/predict_bleach_ratio/predict_bleach_ratio_error.csv)
2. Figures with train and validation errors during the training stage (TO DO)
