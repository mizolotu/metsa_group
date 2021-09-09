import pandas as pd
import requests, json
import os.path as osp

from config import *

if __name__ == '__main__':

    # task dir

    task = 'predict_bleach_ratio'
    task_dir = osp.join(data_dir, task)

    # load example data

    with open(osp.join(task_dir, example_samples_fname), 'r') as f:
        example_data = json.load(f)

    # scoring

    dc_comb = []
    for i, sample in enumerate(example_data):
        dc_comb.append(str(i + 1))
        label = sample.pop(br_key)
        for key in sample.keys():
            if pd.isna(sample[key]):
                sample[key] = None
        r = requests.post(url=endpoint_jyu, json=sample)
        jdata = r.json()
        print(f"Example {i + 1} (features of classes {', '.join(dc_comb)}):\nInput: {sample}")
        if jdata['status'] == 'ok':
            print(f"Real value: {label}, predicted value: {jdata[br_key]}, model used: {jdata['model']}\n")
        else:
            print('Something went wrong :(')