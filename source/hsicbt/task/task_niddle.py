from . import *

def task_niddle_func(config_dict):

    if config_dict['do_training']:
        training_standard(config_dict)
        training_hsic(config_dict)
    try:
        plot.plot_1d_activation_kde(get_tmp_path('activation-niddle-hsic.npy'))
        plot.save_figure(get_exp_path("niddle-1d-dist-hsic.{}".format(
            config_dict['ext'])))
        plot.plot_1d_activation_kde(get_tmp_path('activation-niddle-standard.npy'))
        plot.save_figure(get_exp_path("niddle-1d-dist-standard.{}".format(
            config_dict['ext'])))
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

