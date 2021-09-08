import json, requests
import numpy as np
import argparse as arp
import os.path as osp

from train_prediction_models import set_seeds, load_meta, load_data
from config import *

if __name__ == '__main__':

    # aprse args

    parser = arp.ArgumentParser(description='Test prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=seed)
    parser.add_argument('-n', '--new', help='New example data?', type=bool, default=False)
    args = parser.parse_args()

    # task dir

    task_dir = osp.join(data_dir, args.task)

    # set seed for results reproduction

    set_seeds(seed)

    # try to load example data

    do_generate_new_data = True
    if args.new is None or args.new is False:
        try:
            with open(osp.join(task_dir, example_samples_fname), 'r') as f:
                example_data = json.load(f)
                do_generate_new_data = False
        except Exception as e:
            print(e)

    # otherwise generate new from features.csv

    if do_generate_new_data:

        # laod meta

        task_dir = osp.join(data_dir, args.task)
        meta = load_meta(osp.join(task_dir, meta_fname))
        features = meta['features']
        classes = meta['classes']

        # load data

        example_data = []
        u_classes = np.unique(classes)
        for i in range(len(u_classes)):
            features_selected, feature_classes_selected = [list(item) for item in zip(*[(f, c) for f, c in zip(features, classes) if c in u_classes[:i+1]])]
            values, labels, timestamps = load_data(osp.join(task_dir, features_fname), features_selected)
            idx = np.random.randint(len(labels))
            inf_x = {}
            for i, fs in enumerate(features_selected):
                inf_x[fs] = values[idx, i]
            inf_x[br_key] = labels[idx]
            example_data.append(inf_x)

        # save data

        with open(osp.join(task_dir, example_samples_fname), 'w') as f:
            json.dump(example_data, f)

    # scoring init

    #init()

    # scoring run

    # endpoint

    endpoint_url = 'http://20.93.236.203/api/v1/service/metsa-brp/score'

    # scoring

    for i, sample in enumerate(example_data):
        label = sample.pop(br_key)
        for key in sample:
            if np.isnan(sample[key]):
                print(key, sample[key], type(sample[key]))
        r = requests.post(url=endpoint_url, json=sample)
        print(r)
        jdata = r.json()
        print(f'Example {i}: prediction = {jdata[br_key]}, label = {label}')

    for i, sample in enumerate(example_data):
        label = sample.pop(br_key)
        print('here')
        for key in sample:
            if np.isnan(sample[key]):
                print(key, sample[key], type(sample[key]))
        data = json.dumps(sample)
        result = run(data)
        print(f'Example {i}: prediction = {result}, label = {label}')