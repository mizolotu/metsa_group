import logging, os
import argparse as arp
import os.path as osp

from flask import Flask, request
from scoring import init, run
from config import *

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/score', methods=['GET', 'POST'])
def score():
    data = request.data.decode('utf-8')
    resp = run(data)
    return resp

if __name__ == '__main__':

    # parse args

    parser = arp.ArgumentParser(description='Train prediction models')
    parser.add_argument('-t', '--task', help='Task', default='predict_bleach_ratio')
    args = parser.parse_args()

    # set environment variable

    os.environ['AZUREML_MODEL_DIR'] = osp.join(osp.join(models_dir, args.task))

    # init

    init()

    # start flask app

    app.run(host='0.0.0.0')