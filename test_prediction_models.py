import argparse as arp
import pandas as pd
import os.path as osp
import numpy as np

from train_prediction_models import set_seeds, load_meta, load_data
from config import *
from scoring import *

def generate_test_data(data, features, feature_classes, nsamples, outpit_dir):
    X, Y = {}, {}
    return X, Y

if __name__ == '__main__':

    # aprse args

    parser = arp.ArgumentParser(description='Test prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=seed)
    parser.add_argument('-n', '--nsamples', help='Number of test samples', type=int, default=10)
    args = parser.parse_args()

    # set seed for results reproduction

    set_seeds(seed)

    # laod meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']

    # generate/load test data


    test_classes = [1, 2]
    features_selected, feature_classes_selected = [list(item) for item in zip(*[(f, c) for f, c in zip(features, classes) if c in test_classes])]
    values, labels, timestamps = load_data(osp.join(task_dir, features_fname), features_selected)
    idx = np.random.choice(len(labels), args.nsamples)
    inf_x = {}
    for i, fs in enumerate(features_selected):
        inf_x[fs] = values[idx, i].tolist()
    data = json.dumps(inf_x)

    # scoring init

    init()

    # scoring run

    run(data)