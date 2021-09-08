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

### Train and deploy prediction models

1. Create datasets:

```bash
python3 create_datasets.py
```

2. Train prediction models:

```bash
python3 train_prediction_models.py -e <feature extractor, e.g. cnn1>
```

3. Register model on Azure:

```bash
az ml model register -n metsa_brp -p <path to the model, e.g. models/predict_bleach_ratio/production>
```

4. Deploy the model on Azure using ```scoring.py``` as the entry script and ```environment.yml``` as the dependencies file.

5. Test the deployment, e.g.:

```bash
python3 simple_endpoint_test.py
```

### Calculate feature importance (this part is now a bit broken, need to edit)

1. Calculate feature correlation with the target variable:

```bash
python3 calculate_target_correlation.py
```

2. Calculate permutation feature importance:

```bash
python3 calculate_permutation_importance.py -m <prediction model, e.g. cnn> -l <model layer sizes, e.g. 2048 2048>
```

3. Calculate feature importance based on prediction error:

```bash
python3 calculate_prediction_error.py -e <evaluation method: selected, not-selected or permuted> -m <prediction model, e.g. cnn> -l <model layer sizes, e.g. 2048 2048>
```

4. Plot feature importance:

```bash
python3 plot_feature_importance.py
```
<img src="figures/predict_bleach_ratio/features_ranked.png" width="800"/>

5. One can also plot a particular feature values against the target variable:

```bash
python3 plot_feature_values.py -f <feature name, e.g. 126A0333-QI>
```
<img src="figures/predict_bleach_ratio/126A0333-QI_vs_126A0466-QI.png" width="260"/> <img src="figures/predict_bleach_ratio/126A0535-QIC_vs_126A0466-QI.png" width="260"/> <img src="figures/predict_bleach_ratio/126A0118-QI_vs_126A0466-QI.png" width="260"/> 
