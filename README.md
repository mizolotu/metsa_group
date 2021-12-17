# Metsa Group Project

## Bleach ratio prediction

### Requirements

1. Access to the data
2. Python 3.8
3. Azure inference cluster by default has tensorflow 2.5.0, models trained in tf 2.6.0 cannot be loaded in keras due to some compatibility issue, keep that in mind
4. Looks like requests 2.26 has bug with json (probably, allow_nan set as False), therefore requests 2.25.1 should be used for the endpoint testing

### Installation

1. Clone this repository
2. Create directory ```raw``` in directory ```data```
3. Export bleach ratio data files from Azure in csv format and put them into ```data/raw```
4. pip3 install -r requirements

### Train and deploy prediction models

1. Create datasets:

```bash
python3 create_datasets.py
```

2. Train prediction models:

```bash
python3 train_prediction_models.py -e <feature extractor, e.g. mlp> -m <mode: development or production>
```

3. Deploy the model on Azure using ```scoring.py``` as the entry script and ```environment.yml``` as the dependencies file.

### Calculate feature importance

1. Calculate feature correlation with the target variable:

```bash
python3 calculate_feature_correlations.py
```

2. Calculate permutation feature importance:

```bash
python3 permutation_feature_test.py -e <prediction model, e.g. mlp> -c <correlation type used to eliminate the most correlated features: pearson or spearman> -d <maximum delay class>
```

3. Calculate feature importance based on prediction error:

```bash
python3 bruteforce_feature_test.py -e <evaluation method: not-selected or permuted> -d <maximum delay class>
```

4. Plot feature importance:

```bash
python3 plot_feature_importance.py
```
<img src="figures/predict_bleach_ratio/features_ranked_anonymized_4.png" width="400"/>
<img src="figures/predict_bleach_ratio/features_ranked_anonymized_5.png" width="400"/>

5. One can also plot a particular feature values against the target variable:

```bash
python3 plot_feature_values.py -f <feature name>
```
<img src="figures/predict_bleach_ratio/126A0436-WIC_vs_126A0466-QI_anonymized.png" width="260"/> <img src="figures/predict_bleach_ratio/126A0546-QI1_vs_126A0466-QI_anonymized.png" width="260"/> <img src="figures/predict_bleach_ratio/126A0503-QI_A2_vs_126A0466-QI_anonymized.png" width="260"/> 
