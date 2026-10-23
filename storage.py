import os
import pickle
import warnings
from datetime import datetime
import numpy as np
import config as C
from calib_model import NAMES, build_model

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



def load_calibration(path=C.CALIB_FILE):
    if not os.path.exists(path):
        warnings.warn('No calibration found - running uncorrected.')
        return None
    calib = load_pickle(path)
    if not all(k in calib for k in NAMES):
        warnings.warn('Calibration missing some of the 8 directions - running uncorrected.')
        return None
    model = build_model(calib)
    if model is None:
        warnings.warn('Could not build calibration model - running uncorrected.')
        return None
    calib['model'] = model        # computed at load time, so model changes need no recalibration
    return calib


def save_calibration(calib, path=C.CALIB_FILE):
    save_pickle(calib, path)
    print(f'Saved calibration: {path}')


def save_log(log, meta, exp_name):
    os.makedirs(C.LOG_DIR, exist_ok=True)
    fname = os.path.join(C.LOG_DIR, f"log_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    save_pickle({'log': log, 'meta': meta}, fname)
    print(f'Saved log: {fname}')
    return fname