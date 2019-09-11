import os

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

def get_plot_filename(config_dict):
    return "{}-{}".format(config_dict['task'], config_dict['data_code'])

def get_exp_path(filename):
    return "./assets/exp/{}".format(filename)

def get_model_path(filename, idx=None):
    if idx:
        filepath = "./assets/models/{}-{:04d}.pt".format(
            os.path.splitext(filename)[0], idx)
    else:
        filepath = "./assets/models/{}".format(filename)
    return filepath

def get_tmp_path(filename):
    return "./assets/tmp/{}".format(filename)


