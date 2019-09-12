from . import *


def task_assigner(ttype):
    func = None
    if ttype == TTYPE_HSICTRAIN:
        func = training_hsic
    elif ttype == TTYPE_STANDARD:
        func = training_standard
    elif ttype == TTYPE_FORMAT:
        func = training_format
    return func

def get_experiment_fig_filename(config_dict, etype):
    filepath = ''
    if etype == 'needle-1':
        filepath = get_exp_path("fig3b-needle-1d-dist-{}.{}".format(config_dict['training_type'], config_dict['ext']))
    elif etype == 'needle-2':
        filepath = get_exp_path("fig3a-needle-1d-dist-{}.{}".format(config_dict['training_type'], config_dict['ext']))
    return filepath

def save_experiment_fig(filepath):
    filename = os.path.basename(filepath)
    filename_noext, ext = os.path.splitext(filename)
    dirname = os.path.dirname(filepath)
    timestamp_filename = "{}-{}{}".format(filename_noext, TIMESTAMP_CODE, ext)
    timestamp_filepath = os.path.join(dirname, timestamp_filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    plot.save_figure(timestamp_filepath)
    os.symlink(timestamp_filepath, filepath)
    print_highlight("Symlink [{}]".format(filepath), 'blue')

