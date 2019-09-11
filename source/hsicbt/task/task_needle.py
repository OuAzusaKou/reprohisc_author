from . import *


def task_assigner(ttype):
    func = None
    if ttype == TTYPE_HSICTRAIN:
        func = training_hsic
    elif ttype == TTYPE_STANDARD:
        func = training_standard
    return func

def get_fig_number(ttype):
    text = None
    if ttype == TTYPE_HSICTRAIN:
        text = 'fig3b'
    elif ttype == TTYPE_STANDARD:
        text = 'fig3a'
    return text

def plot_result(ttype, ext):
    plot.plot_1d_activation_kde(get_tmp_path('activation-needle-{}.npy').format(ttype))
    plot.save_figure(get_exp_path("{}-needle-1d-dist-{}.{}".format(get_fig_number(ttype), ttype, ext)))

def task_needle_func(config_dict):

    func = task_assigner(config_dict['training_type'])
    if config_dict['do_training']:  
        func(config_dict)
    try:
        plot_result(config_dict['training_type'], config_dict['ext'])
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program exits.".format(e), 'red')
        quit()

