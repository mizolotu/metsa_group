# directories and files

data_dir = 'data'
raw_data_dir = f'{data_dir}/raw'
models_dir = 'models'
results_dir = 'results'
figures_dir = 'figures'
functions_dir = 'functions'

tags_fname = 'tags.xlsx'
features_fname = 'features.csv'
meta_fname = 'metainfo.json'

xx_pearson_correlation_csv = 'feature_vs_feature_pearson_correlation.csv'
xx_spearman_correlation_csv = 'feature_vs_feature_spearman_correlation.csv'
xy_correlation_csv = 'feature_vs_target_correlation.csv'
less_correlated_json = 'less_correlated_{0}.json'

prediction_importance_csv = 'prediction_importance.csv'
permutation_importance_csv = 'permutation_importance.csv'

prediction_results_fname = 'prediction_results.csv'
prediction_mean_errors_fname = 'prediction_mean_errors.csv'
prediction_min_errors_fname = 'prediction_min_errors.csv'
prediction_max_errors_fname = 'prediction_max_errors.csv'

anomaly_detection_results_fname = 'anomaly_detection_results.csv'
anomaly_detection_mean_aucs_fname = 'anomaly_detection_mean_aucs.csv'

summary_txt = 'summary.txt'
example_samples_fname = 'example.json'
test_samples_fname = 'test_samples.csv'
error_cdf_fname = 'error_cdf.pdf'
error_for_real_fname = 'error_for_real.pdf'

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
series_step_prefix = 'step'

# ml

seed = 0
batch_size = 64
epochs = 1 # 0000
patience = 100
series_len = 64
threshold_variable_name = 'threshold'
ae_models = ['aen', 'som']

# azure

endpoint_jyu = 'http://20.93.236.203/api/v1/service/metsa-brp/score'
insert_data_sample_function_url = 'http://localhost:7071/api/insert_sample_data_row'

# other

dc_combs_col_name = 'Delay classes'
modes = ['development', 'production']
eps = 1e-10
csv = '.csv'
pdf = '.pdf'

