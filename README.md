# Metsa Group Project

## Bleach ratio prediction

### Requirements

1. Access to the data
2. Python 3.8.5

### Installation

1. Clone this repository
2. Create directory ```raw``` in directory ```data```
3. Export bleach ratio data from Azure in csv format, name it for example "some_samples.csv", and put it into ```data/raw```
4. pip3 install -r requirements

### How to use

1. Create datasets:

```bash
python3 create_datasets.py -s <csv file with samples, e.g. some_samples.csv>
```

2. Calculate feature correlation with the target variable:

```bash
python3 calculate_target_correlation.py
```

3. Calculate permutation feature importance:

```bash
python3 calculate_permutation_importance.py -m <prediction model, e.g. cnn> -l <model layer sizes, e.g. 2048 2048>
```

4. Calculate feature importance based on prediction error:

```bash
python3 calculate_prediction_error.py -e <evaluation method: selected, not-selected or permuted> -m <prediction model, e.g. cnn> -l <model layer sizes, e.g. 2048 2048>
```

5. Plot feature importance:

```bash
python3 plot_feature_importance.py
```
<img src="figures/predict_bleach_ratio/features_ranked.pdf" width="800"/>

6. One can also plot a particular feature values against the target variable:

```bash
python3 plot_feature_values.py -f <feature name, e.g. 126A0333-QI>
```
