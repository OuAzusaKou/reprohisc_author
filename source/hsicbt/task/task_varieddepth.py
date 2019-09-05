from . import *

def task_varieddepth_func(config_dict):

    if config_dict['do_training']:
        
        config_dict['n_layers'] = 5
        config_dict['exp_index'] = 1
        training_standard(config_dict) 

        config_dict['n_layers'] = 10
        config_dict['exp_index'] = 2
        training_standard(config_dict)

        config_dict['n_layers'] = 15
        config_dict['exp_index'] = 3
        training_standard(config_dict)

        config_dict['n_layers'] = 20
        config_dict['exp_index'] = 4
        training_standard(config_dict)

    try:
        out_standard_batch_05 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 1))
        out_standard_batch_10 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 2))
        out_standard_batch_15 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 3))
        out_standard_batch_20 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 4))
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    input_list = [out_standard_batch_05, out_standard_batch_10, out_standard_batch_15, out_standard_batch_20]
    label_list = ['depth-05', 'depth-10', 'depth-15', 'depth-20']

    metadata = {
        'title':'HSIC(X, Z_L) of Varied-depth',
        'xlabel': 'epochs',
        'ylabel': 'HSIC(X, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hx', metadata)
    plot.save_figure(get_exp_path("varied-depth-hsic_xz-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))

    metadata = {
        'title':'HSIC(Y, Z_L) of Varied-depth',
        'xlabel': 'epochs',
        'ylabel': 'HSIC(Y, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hy', metadata)
    plot.save_figure(get_exp_path("varied-depth-hsic_yz-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))

    metadata = {
        'title':'performance of Varied-depth',
        'xlabel': 'epochs',
        'ylabel': 'training accuracy',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_acc', metadata)
    plot.save_figure(get_exp_path("varied-depth-acc-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))