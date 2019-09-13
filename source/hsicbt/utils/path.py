import os
from .path import *
from .const import *
from .color import *

def attaching_timestamp_filepath(filepath):
    filename = os.path.basename(filepath)
    dirname  = os.path.dirname(filepath)
    filename, ext = os.path.splitext(filename)
    filename_time = "{}_{}.{}".format(TIMESTAMP_CODE, filename, ext)
    timestamp_path = os.path.join(dirname, 'raw', filename_time)
    return timestamp_path

def make_symlink(src_path, sym_path):
    if os.path.exists(sym_path):
        os.remove(sym_path)
    os.symlink(src_path, sym_path)
    print_highlight("Symlink [{}]".format(sym_path), ctype="blue")

def get_batch_log_filepath(task, ttype, dtype, idx=None):
    if idx:
        filepath = "{}/assets/logs/{}-batch-{}-{}-{:04d}.npy".format(os.getcwd(), task, ttype, dtype, idx)
    else:
        filepath = "{}/assets/logs/{}-batch-{}-{}.npy".format(os.getcwd(),task, ttype, dtype)
    return filepath

def get_epoch_log_filepath(task, ttype, dtype, idx=None):
    if idx:
        filepath = "{}/assets/logs/{}-epoch-{}-{}-{:04d}.npy".format(os.getcwd(),task, ttype, dtype, idx)
    else:
        filepath = "{}/assets/logs/{}-epoch-{}-{}.npy".format(os.getcwd(),task, ttype, dtype)
    return filepath

def get_log_filepath(task, ttype, dtype, idx=None):
    if idx:
        filepath = "{}/assets/logs/{}-{}-{}-{:04d}.npy".format(os.getcwd(),task, ttype, dtype, idx)
    else:
        filepath = "{}/assets/logs/{}-{}-{}.npy".format(os.getcwd(),task, ttype, dtype)
    return filepath

def get_log_raw_filepath(task, ttype, dtype, idx=None):
    if idx:
        filepath = "{}/assets/logs/raw/{}-{}-{}-{:04d}.npy".format(os.getcwd(),task, ttype, dtype, idx)
    else:
        filepath = "{}/assets/logs/raw/{}-{}-{}.npy".format(os.getcwd(),task, ttype, dtype)
    return filepath

def get_plot_filename(config_dict):
    return "{}-{}".format(config_dict['task'], config_dict['data_code'])

def get_exp_path(filename):
    return "{}/assets/exp/{}".format(os.getcwd(), filename)

def get_exp_raw_path(filename):
    return "{}/assets/exp/raw/{}".format(os.getcwd(), filename)

def get_model_path(filename, idx=None):
    if idx:
        filepath = "./assets/models/{}-{:04d}.pt".format(
            os.path.splitext(filename)[0], idx)
    else:
        filepath = "./assets/models/{}".format(filename)
    return filepath

def get_model_raw_path(filename, idx=None):
    if idx:
        filepath = "./assets/models/raw/{}-{:04d}.pt".format(
            os.path.splitext(filename)[0], idx)
    else:
        filepath = "./assets/models/raw/{}".format(filename)
    return filepath

def get_tmp_path(filename):
    return "./assets/tmp/{}".format(filename)


