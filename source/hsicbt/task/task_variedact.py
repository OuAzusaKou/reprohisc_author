from . import *

def task_variedact_func(config_dict):

    if config_dict['do_training']:
        
        config_dict['atype'] = 'relu'
        config_dict['exp_index'] = 1
        training_standard(config_dict) 

        config_dict['atype'] = 'tanh'
        config_dict['exp_index'] = 2
        training_standard(config_dict)

        config_dict['atype'] = 'elu'
        config_dict['exp_index'] = 3
        training_standard(config_dict)

        config_dict['atype'] = 'sigmoid'
        config_dict['exp_index'] = 4
        training_standard(config_dict)

    try:
        out_standard_batch_relu = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 1))
        out_standard_batch_tanh = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 2))
        out_standard_batch_elu = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 3))
        out_standard_batch_sigmoid = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 4))
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()


    input_list = [out_standard_batch_relu, out_standard_batch_tanh, out_standard_batch_elu, out_standard_batch_sigmoid]
    label_list = ['relu', 'tanh', 'elu', 'sigmoid']

    metadata = {
        #'title':'nHSIC(X, Z_L) of Varied-activation',
        'title': '',
        'xlabel': 'epochs',
        'ylabel': 'nHSIC(X, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hx', metadata)
    plot.save_figure(get_exp_path("fig2a-varied-activation-hsic_xz-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))

    metadata = {
        #'title':'nHSIC(Y, Z_L) of Varied-activation',
        'title':'',
        'xlabel': 'epochs',
        'ylabel': 'nHSIC(Y, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hy', metadata)
    plot.save_figure(get_exp_path("fig2b-varied-activation-hsic_yz-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))

    metadata = {
        #'title':'performance of Varied-activation',
        'title':'',
        'xlabel': 'epochs',
        'ylabel': 'training accuracy',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_acc', metadata)
    plot.save_figure(get_exp_path("fig2c-varied-activation-acc-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))
