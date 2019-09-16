from . import *




def plot_needle_result(config_dict):
    try:
        fig = plot.plot_1d_activation_kde(get_tmp_path('activation-needle-{}.npy').format(config_dict['training_type']))
        plot.adding_footnote(fig, config_dict['footnote'])

        if config_dict['training_type'] == TTYPE_HSICTRAIN:
            filepath = get_experiment_fig_filename(config_dict, 'needle-1')
        elif config_dict['training_type'] == TTYPE_STANDARD:
            filepath = get_experiment_fig_filename(config_dict, 'needle-2')
        save_experiment_fig(filepath)
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program exits.".format(e), 'red')
        quit()    

def task_needle_func(config_dict):

    func = task_assigner(config_dict['training_type'])
    func(config_dict)


