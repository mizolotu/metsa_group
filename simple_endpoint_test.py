import requests, json
import argparse as arp
import os.path as osp

from config import *

if __name__ == '__main__':

    # aprse args

    parser = arp.ArgumentParser(description='Test prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    args = parser.parse_args()

    # task dir

    task_dir = osp.join(data_dir, args.task)

    # load example data

    with open(osp.join(task_dir, example_samples_fname), 'r') as f:
        example_data = json.load(f)

    # scoring

    dc_comb = []
    for i, sample in enumerate(example_data):
        dc_comb.append(str(i + 1))
        label = sample.pop(br_key)
        r = requests.post(url=endpoint_jyu, json=sample)
        jdata = r.json()
        print(f"Example {i + 1} (features of classes {', '.join(dc_comb)}):\nInput: {sample}')")
        if jdata['status'] == 'ok':
            print(f"Real value: {label}, predicted value: {jdata[br_key]}, model used: {jdata['model']}\n")
        else:
            print('Something went wrong :(')