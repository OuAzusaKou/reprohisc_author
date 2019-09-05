from . import *

def task_variedep_func(config_dict):

    model_filename = config_dict['model_file']

    if config_dict['do_training']:

        config_dict['epochs_hsic'] = 1
        config_dict['exp_index'] = 1
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_hsic(config_dict) 

        config_dict['epochs_hsic'] = 5
        config_dict['exp_index'] = 2
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_hsic(config_dict) 

        config_dict['epochs_hsic'] = 10
        config_dict['exp_index'] = 3
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_hsic(config_dict) 

        config_dict['epochs_hsic'] = 20
        config_dict['exp_index'] = 4
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_hsic(config_dict) 

        config_dict['batch_size'] = 256
        config_dict['exp_index'] = 1
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_format(config_dict)

        config_dict['exp_index'] = 2
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_format(config_dict)

        config_dict['exp_index'] = 3
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_format(config_dict)

        config_dict['exp_index'] = 4
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_format(config_dict)



    try:
        out_standard_batch_001 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))
        out_standard_batch_005 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))
        out_standard_batch_010 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))
        out_standard_batch_025 = load_logs(get_batch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 4))
        out_standard_epoch_001 = load_logs(get_epoch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))
        out_standard_epoch_005 = load_logs(get_epoch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))
        out_standard_epoch_010 = load_logs(get_epoch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))
        out_standard_epoch_025 = load_logs(get_epoch_log_filepath(
            config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 4))
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    input_batch_list = [out_standard_batch_001, out_standard_batch_005, out_standard_batch_010, out_standard_batch_025]
    input_epoch_list = [out_standard_epoch_001, out_standard_epoch_005, out_standard_epoch_010, out_standard_epoch_025]
    label_list = ['epoch-001', 'epoch-005', 'epoch-010', 'epoch-025']

    metadata = {
        'title':'HSIC(X, Z_L) of Varied-epoch',
        'xlabel': 'epochs',
        'ylabel': 'HSIC(X, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_batch_list, 'batch_hsic_hx', metadata)
    plot.save_figure(get_exp_path("varied-epoch-hsic_xz-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))

    metadata = {
        'title':'HSIC(Y, Z_L) of Varied-epoch',
        'xlabel': 'epochs',
        'ylabel': 'HSIC(Y, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_batch_list, 'batch_hsic_hy', metadata)
    plot.save_figure(get_exp_path("varied-epoch-hsic_yz-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))

    metadata = {
        'title':'performance of Varied-epoch',
        'xlabel': 'epochs',
        'ylabel': 'training accuracy',
        'label': label_list
    }
    plot.plot_batches_log(input_batch_list, 'batch_acc', metadata)
    plot.save_figure(get_exp_path("varied-epoch-acc-{}.{}".format(
        config_dict['data_code'], config_dict['ext'])))

    metadata = {
        'title':'{} test performance of Varied-epoch'.format(config_dict['data_code']),
        'xlabel': 'epochs',
        'ylabel': 'test accurarcy',
        'label': label_list
    }
    plot.plot_epoch_log(input_epoch_list, 'test_acc', metadata)
    plot.save_figure(get_exp_path("{}-epoch-train-acc.{}".format(
        get_plot_filename(config_dict), config_dict['ext'])))