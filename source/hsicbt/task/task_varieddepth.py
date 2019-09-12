from . import *


def plot_varieddepth_result(config_dict):
    try:
        out_standard_batch_05 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 1))['batch_log_list']
        out_standard_batch_10 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 2))['batch_log_list']
        out_standard_batch_15 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 3))['batch_log_list']
        out_standard_batch_20 = load_logs(get_log_filepath(
            config_dict['task'], TTYPE_STANDARD , config_dict['data_code'], 4))['batch_log_list']
    except IOError as e:
        print_highlight("{}.\nPlease do training by setting do_training key to True in config. Program quits.".format(e), 'red')
        quit()

    input_list = [out_standard_batch_05, out_standard_batch_10, out_standard_batch_15, out_standard_batch_20]
    label_list = ['depth-05', 'depth-10', 'depth-15', 'depth-20']

    metadata = {
        'title':'',
        'xlabel': 'epochs',
        'ylabel': 'HSIC(X, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hx', metadata)
    plot.save_figure(get_exp_path("fig2d-varied-depth-hsic_xz-{}-{}.{}".format(
        config_dict['data_code'], TIMESTAMP_CODE, config_dict['ext'])))

    metadata = {
        'title':'',
        'xlabel': 'epochs',
        'ylabel': 'HSIC(Y, Z_L)',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_hsic_hy', metadata)
    plot.save_figure(get_exp_path("fig2e-varied-depth-hsic_yz-{}-{}.{}".format(
        config_dict['data_code'], TIMESTAMP_CODE, config_dict['ext'])))

    metadata = {
        'title':'',
        'xlabel': 'epochs',
        'ylabel': 'training accuracy',
        'label': label_list
    }
    plot.plot_batches_log(input_list, 'batch_acc', metadata)
    plot.save_figure(get_exp_path("fig2f-varied-depth-acc-{}-{}.{}".format(
        config_dict['data_code'], TIMESTAMP_CODE, config_dict['ext'])))
    

def task_varieddepth_func(config_dict):

    func = task_assigner(config_dict['training_type'])
    func(config_dict) 

