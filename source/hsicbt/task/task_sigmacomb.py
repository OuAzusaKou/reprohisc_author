from . import *

def task_sigmacomb_func(config_dict):

    model_filename = config_dict['model_file']

    if config_dict['do_training_hsic']:

        config_dict['sigma_hx'] = 5
        config_dict['sigma_hy'] = 5
        config_dict['exp_index'] = 1
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_hsic(config_dict)

        config_dict['sigma_hx'] = 10
        config_dict['sigma_hy'] = 10
        config_dict['exp_index'] = 2
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_hsic(config_dict)

        config_dict['sigma_hx'] = 15
        config_dict['sigma_hy'] = 15
        config_dict['exp_index'] = 3
        config_dict['model_file'] = "{}-{:04d}.pt".format(
            os.path.splitext(model_filename)[0], config_dict['exp_index'])
        training_hsic(config_dict)

    if config_dict['do_training_format']:

        config_dict['batch_size'] = 512
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
        config_dict['model_file'] = [ \
            "{}-{:04d}.pt".format(os.path.splitext(model_filename)[0], 1),\
            "{}-{:04d}.pt".format(os.path.splitext(model_filename)[0], 2),\
            "{}-{:04d}.pt".format(os.path.splitext(model_filename)[0], 3),\
            ]
        training_format_combined(config_dict)

    if config_dict['do_plot']:
        
        try:
            out_standard_batch_1 = load_logs(get_batch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))
            out_standard_batch_2 = load_logs(get_batch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))
            out_standard_batch_3 = load_logs(get_batch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))
            out_standard_batch_4 = load_logs(get_batch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 4))
            out_standard_epoch_1 = load_logs(get_epoch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 1))
            out_standard_epoch_2 = load_logs(get_epoch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 2))
            out_standard_epoch_3 = load_logs(get_epoch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 3))
            out_standard_epoch_4 = load_logs(get_epoch_log_filepath(
                config_dict['task'], TTYPE_FORMAT, config_dict['data_code'], 4))
        except IOError as e:
            print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
            quit()

        input_list = [out_standard_batch_1, out_standard_batch_2, out_standard_batch_3, out_standard_batch_4]
        label_list = ['sigma=5', 'sigma=10', 'sigma=15', 'sigma-combined']
        metadata = {
            #'title':'{} training perf of sigma-combined net'.format(config_dict['data_code']),
            'title': '',
            'xlabel': 'epochs',
            'ylabel': 'training accurarcy',
            'label': label_list
        }
        plot.plot_batches_log(input_list, 'batch_acc', metadata)
        plot.save_figure(get_exp_path("fig7b-{}-sigmacomb-train-acc.{}".format(
            get_plot_filename(config_dict), config_dict['ext'])))

        # input_list = [out_standard_epoch_1, out_standard_epoch_2, out_standard_epoch_3, out_standard_epoch_4]
        # label_list = ['sigma=5', 'sigma=10', 'sigma=15', 'sigma-combined']
        # metadata = {
        #     #'title':'{} test perf of sigma-combined net'.format(config_dict['data_code']),
        #     'title': '',
        #     'xlabel': 'epochs',
        #     'ylabel': 'test accurarcy',
        #     'label': label_list
        # }
        # plot.plot_epoch_log(input_list, 'test_acc', metadata)
        # plot.save_figure(get_exp_path("{}-sigmacomb-test-acc.{}".format(
        #     get_plot_filename(config_dict), config_dict['ext'])))
