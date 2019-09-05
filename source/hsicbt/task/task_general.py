from . import *

def task_general_func(config_dict):

    if config_dict['do_training']:
        training_format(config_dict)
        training_standard(config_dict)

    try:
        out_standard_batch = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code']))
        out_format_batch = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code']))        
        out_standard_epoch = load_logs(get_epoch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code']))
        out_format_epoch = load_logs(get_epoch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code']))
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    metadata = {
        'title':'{} batch training performance'.format(config_dict['data_code']),
        'xlabel': 'epochs',
        'ylabel': 'training batch accurarcy',
        'label': ['stadnard-train', 'format-train']
    }


    plot.plot_batches_log([out_standard_batch, out_format_batch], 'batch_acc', metadata)
    plot.save_figure(get_exp_path("{}-batch.{}".format(
        get_plot_filename(config_dict), config_dict['ext'])))

    metadata = {
        'title':'{} training performance'.format(config_dict['data_code']),
        'xlabel': 'epochs',
        'ylabel': 'training accurarcy (eval at the end of epoch)',
        'label': ['stadnard-train', 'format-train']
    }
    plot.plot_epoch_log([out_standard_epoch, out_format_epoch], 'train_acc', metadata)
    plot.save_figure(get_exp_path("{}-epoch-train-acc.{}".format(
        get_plot_filename(config_dict), config_dict['ext'])))

    metadata = {
        'title':'{} test performance'.format(config_dict['data_code']),
        'xlabel': 'epochs',
        'ylabel': 'test accurarcy (eval at the end of epoch)',
        'label': ['stadnard-train', 'format-train']
    }
    plot.plot_epoch_log([out_standard_epoch, out_format_epoch], 'test_acc', metadata)
    plot.save_figure(get_exp_path("{}-epoch-test-acc.{}".format(
        get_plot_filename(config_dict), config_dict['ext'])))
