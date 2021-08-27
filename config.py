# directories and files

data_dir = 'data'
raw_data_dir = f'{data_dir}/raw'
processed_data_dir = f'{data_dir}/features'
models_dir = 'models'
results_dir = 'results'
figures_dir = 'figures'
tags_fname = 'tags.xlsx'
correlation_csv = 'correlation.csv'
prediction_error_csv = 'prediction_error.csv'
permutation_error_csv = 'permutation_error.csv'
prediction_results_csv = 'prediction_results.csv'
prediction_errors_csv = 'prediction_errors.csv'
summary_txt = 'summary.txt'

# data

ts_key = 'LDTS'
br_key = '126A0466-QI'
br_thr = 80.0
br_min = 86.0
br_max = 92.0
stages = ['training', 'validation', 'inference']
validation_share = 0.2
train_test_ratio = 0.5
yellow_tags = ['126A0159-HS', '126A0125-UI', '126A0164-UI', '126A7035-DLY', '126A7036-DLY', '126A7037-DLY']
position_column = 'Positio'
delay_class_column = 'Viiveluokka'
nan_value = -1

# ml

seed = 0
batch_size = 512
epochs = 10000
patience = 100

# other

eps = 1e-10
csv = '.csv'
pdf = '.pdf'
