# Metsa Group Project

## Bleach ratio prediction

### Requirements

Python 3.8.5

### Installation

1. Clone this repository
2. Export bleach ratio data from Azure in csv format, name it for example "some_samples.csv", and put it into directory ```data/raw```.
3. pip3 install -r requirements

### How to use

1. Create datasets

```bash
python3 create_datasets.py -s <csv file with samples, e.g. samples_16062021.csv>
```

2. Train a model

```bash
python3 train_models_to_predict_bleach_ratio.py
```

3. Test the model

```bash
python3 test_models_to_predict_bleach_ratio.py
```

### Results

1. Prediction errors for different delay classes and different models can be found [here](results/predict_bleach_ratio/predict_bleach_ratio_error.csv)
