from . import *

def plot_hsicsolve_result(config_dict):

    try:
        out_epoch      = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code']))['epoch_log_dict']
        out_hsic_epoch = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_HSICTRAIN, config_dict['data_code']))['epoch_log_dict']
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    metadata = {
        #'title':'{} test performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epochs',
        'ylabel': 'test accurarcy',
        'label': ['backpropagation', 'unformatted-training']
    }
    filename = ""
    plot.plot_epoch_log([out_epoch, out_hsic_epoch], 'test_acc', metadata)
    plot.save_figure(get_exp_path("{}-test-acc-{}.{}".format(
        get_plot_filename(config_dict), TIMESTAMP_CODE, config_dict['ext'])))

    metadata = {
        #'title':'{} test performance'.format(config_dict['data_code']),
        'title': '',
        'xlabel': 'epochs',
        'ylabel': 'training accurarcy',
        'label': ['backpropagation', 'unformatted-training']
    }
    plot.plot_epoch_log([out_epoch, out_hsic_epoch], 'train_acc', metadata)
    plot.save_figure(get_exp_path("fig5-{}-train-acc-{}.{}".format(
        get_plot_filename(config_dict), TIMESTAMP_CODE, config_dict['ext'])))

    plot.plot_activation_distribution()
    plot.save_figure(get_exp_path("fig4-hsic-solve-actdist-{}-{}.{}".format(
        config_dict['data_code'], TIMESTAMP_CODE, config_dict['ext'])))
    

def task_hsicsolve_func(config_dict):

    if config_dict['do_training']:
        func = task_assigner(config_dict['training_type'])
        if config_dict['training_type'] == TTYPE_HSICTRAIN:
            config_dict['last_hidden_width'] = 10 # since we are using hsic to solve classification            
        func(config_dict)
