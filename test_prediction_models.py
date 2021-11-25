import json, requests
import numpy as np
import argparse as arp
import os.path as osp

from common.utils import set_seeds, load_meta, load_data
from config import *

if __name__ == '__main__':

    # aprse args

    parser = arp.ArgumentParser(description='Test prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=seed)
    parser.add_argument('-n', '--nsamples', help='Number of samples', type=int, default=1)
    parser.add_argument('-e', '--endpoint', help='Endpoint', default='http://20.103.112.91:80/api/v1/service/metsa-brp/score')
    parser.add_argument('-k', '--key', help='Endpoint key', default='I70BZiKrv2PmXsYDDIJ40vO87hRs23ou')
    args = parser.parse_args()

    # task dir

    task_dir = osp.join(data_dir, args.task)

    # set seed for results reproduction

    set_seeds(seed)

    # laod meta

    task_dir = osp.join(data_dir, args.task)
    meta = load_meta(osp.join(task_dir, meta_fname))
    features = meta['features']
    classes = meta['classes']

    # load data

    data_batches = []
    u_classes = np.unique(classes)
    for i in range(len(u_classes)):
        features_selected, feature_classes_selected = [list(item) for item in zip(*[(f, c) for f, c in zip(features, classes) if c in u_classes[:i+1]])]
        values, labels, timestamps = load_data(osp.join(raw_data_dir, test_samples_fname), features_selected, dtype='object', nan_to_none=True)
        idx = np.random.randint(0, len(labels), args.nsamples)
        vals = values[idx, :]
        cols = np.hstack([meta['timestamp'], features_selected, meta['label']]).tolist()
        rows = np.hstack([timestamps[idx].reshape(-1, 1), vals, labels[idx].reshape(-1, 1)]).tolist()
        data_batches.append({'rows': rows, 'cols': cols})
        for j in u_classes[1:i + 1][::-1]:
            data_batches.append({'rows': rows, 'cols': cols, 'dc': int(j)})
        for j in u_classes[1:i+1][::-1]:
            vals[:, np.where(feature_classes_selected == j)] = None
            rows = np.hstack([timestamps[idx].reshape(-1, 1), vals, labels[idx].reshape(-1, 1)]).tolist()
            data_batches.append({'rows': rows, 'cols': cols})

    # scoring

    for data_batch in data_batches:
        r = requests.post(url=args.endpoint, json=data_batch, headers={'Authorization': (f'Bearer {args.key}')})
        jdata = r.json()
        if 'status' in jdata.keys():
            if jdata['status'] == 'ok' and br_key in jdata.keys() and 'errors' in jdata.keys():
                assert len(jdata[br_key]) == len(jdata['errors']), 'Results shape mismatch!'
                for i, (p, e) in enumerate(zip(jdata[br_key], jdata['errors'])):
                    print(f'Example {i}: prediction = {p}, error = {e}')
            else:
                print(jdata['status'])
        else:
            print(jdata)