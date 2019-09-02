from .. import *

from .engine import training_standard
from .engine import training_format
from .engine import training_hsic

from ..utils import plot



def job_execution(config_dict):

    if config_dict['task'] == 'standard-train':
        out_batch, out_epoch = training_standard(config_dict)

    elif config_dict['task'] == 'hsic-train':
        out_batch, out_epoch = training_hsic(config_dict)

    elif config_dict['task'] == 'format-train':
        out_batch, out_epoch = training_format(config_dict)


    elif config_dict['task'] == 'general':

        out_format_batch  , out_format_epoch   = training_format(config_dict)
        out_standard_batch, out_standard_epoch = training_standard(config_dict)
        
        metadata = {
            'title':'comparison of format-train and standard-train',
            'xlabel': 'epochs',
            'ylabel': 'training batch accurarcy',
            'label': ['stadnard-train', 'format-train']
        }
        plot.plot_batches_log([out_standard_batch, out_format_batch], 'batch_acc', metadata)
        plot.save_figure("./assets/standard-hsic-comparison-{}-{}-batch.{}".format(
            config_dict['data_code'], config_dict['model'], config_dict['ext']))

        metadata = {
            'title':'comparison of format-train and standard-train',
            'xlabel': 'epochs',
            'ylabel': 'training accurarcy (eval at the end of epoch)',
            'label': ['stadnard-train', 'format-train']
        }
        plot.plot_epoch_log([out_standard_epoch, out_format_epoch], 'train_acc', metadata)
        plot.save_figure("./assets/standard-hsic-comparison-{}-{}-epoch-train-acc.{}".format(
            config_dict['data_code'], config_dict['model'], config_dict['ext']))

        metadata = {
            'title':'comparison of format-train and standard-train',
            'xlabel': 'epochs',
            'ylabel': 'test accurarcy (eval at the end of epoch)',
            'label': ['stadnard-train', 'format-train']
        }
        plot.plot_epoch_log([out_standard_epoch, out_format_epoch], 'test_acc', metadata)
        plot.save_figure("./assets/standard-hsic-comparison-{}-{}-epoch-test-acc.{}".format(
            config_dict['data_code'], config_dict['model'], config_dict['ext']))

    elif config_dict['task'] == 'hsic-solve':
        
        out_batch, out_epoch = training_standard(config_dict)
        config_dict['last_hidden_width'] = 10 # since we are using hsic to solve classification
        out_hsic_batch, out_hsic_epoch = training_hsic(config_dict)
        metadata = {
            'title':'comparison of hsic-solve and standard-train',
            'xlabel': 'epochs',
            'ylabel': 'accurarcy',
            'label': ['stadnard-training', 'hsic-solve']
        }
        plot.plot_epoch_log([out_epoch, out_hsic_epoch], 'test_acc', metadata)
        plot.save_figure("./assets/hsic-solve-comparison-{}-test-acc.{}".format(
            config_dict['data_code'], config_dict['ext']))
        plot.plot_epoch_log([out_epoch, out_hsic_epoch], 'train_acc', metadata)
        plot.save_figure("./assets/hsic-solve-comparison-{}-train-acc.{}".format(
            config_dict['data_code'], config_dict['ext']))
        plot.plot_activation_distribution()
        plot.save_figure("./assets/hsic-solve-actdist-{}.{}".format(
            config_dict['data_code'], config_dict['ext']))


    elif config_dict['task'] == 'niddle':

        out_standard_batch, out_standard_epoch = training_standard(config_dict)
        out_hsic_batch    , out_hsic_epoch     = training_hsic(config_dict)

        plot.plot_1d_activation_kde('assets/activation-niddle-hsic.npy')
        plot.save_figure("./assets/activation-1d-dist-hsic.{}".format(config_dict['ext']))
        plot.plot_1d_activation_kde('assets/activation-niddle-standard.npy')
        plot.save_figure("./assets/activation-1d-dist-standard.{}".format(config_dict['ext']))
        

    elif config_dict['task'] == 'varied-activation':

        config_dict['atype'] = 'relu'
        out_standard_batch_relu  , out_standard_epoch_relu   = training_standard(config_dict) 

        config_dict['atype'] = 'tanh'
        out_standard_batch_tanh  , out_standard_epoch_tanh   = training_standard(config_dict)

        config_dict['atype'] = 'elu'
        out_standard_batch_elu  , out_standard_epoch_elu   = training_standard(config_dict)

        config_dict['atype'] = 'sigmoid'
        out_standard_batch_sigmoid  , out_standard_epoch_sigmoid   = training_standard(config_dict)

        input_list = [out_standard_batch_relu, out_standard_batch_tanh, out_standard_batch_elu, out_standard_batch_sigmoid]
        label_list = ['relu', 'tanh', 'elu', 'sigmoid']

        metadata = {
            'title':'HSIC value between input and last hidden (Varied-activation)',
            'xlabel': 'epochs',
            'ylabel': 'HSIC(X, Z_L)',
            'label': label_list
        }
        plot.plot_batches_log(input_list, 'batch_hsic_hx', metadata)
        plot.save_figure("./assets/varied-activation-hsic_xz-{}.{}".format(
            config_dict['data_code'], config_dict['ext']))

        metadata = {
            'title':'HSIC value between label and last hidden (Varied-activation)',
            'xlabel': 'epochs',
            'ylabel': 'HSIC(Y, Z_L)',
            'label': label_list
        }
        plot.plot_batches_log(input_list, 'batch_hsic_hy', metadata)
        plot.save_figure("./assets/varied-activation-hsic_yz-{}.{}".format(
            config_dict['data_code'], config_dict['ext']))

        metadata = {
            'title':'Standard training accuracy (Varied-activation)',
            'xlabel': 'epochs',
            'ylabel': 'training accuracy',
            'label': label_list
        }
        plot.plot_batches_log(input_list, 'batch_acc', metadata)
        plot.save_figure("./assets/varied-activation-acc-{}.{}".format(
            config_dict['data_code'], config_dict['ext']))


    elif config_dict['task'] == 'varied-depth':

        config_dict['n_layers'] = 5
        out_standard_batch_05, _ = training_standard(config_dict) 

        config_dict['n_layers'] = 10
        out_standard_batch_10, _ = training_standard(config_dict)

        config_dict['n_layers'] = 15
        out_standard_batch_15, _ = training_standard(config_dict)

        config_dict['n_layers'] = 20
        out_standard_batch_20, _ = training_standard(config_dict)

        input_list = [out_standard_batch_05, out_standard_batch_10, out_standard_batch_15, out_standard_batch_20]
        label_list = ['depth-05', 'depth-10', 'depth-15', 'depth-20']

        metadata = {
            'title':'HSIC value between input and last hidden (Varied-depth model)',
            'xlabel': 'epochs',
            'ylabel': 'HSIC(X, Z_L)',
            'label': label_list
        }
        plot.plot_batches_log(input_list, 'batch_hsic_hx', metadata)
        plot.save_figure("./assets/varied-depth-hsic_xz-{}.{}".format(
            config_dict['data_code'], config_dict['ext']))

        metadata = {
            'title':'HSIC value between label and last hidden (Varied-depth model)',
            'xlabel': 'epochs',
            'ylabel': 'HSIC(Y, Z_L)',
            'label': label_list
        }
        plot.plot_batches_log(input_list, 'batch_hsic_hy', metadata)
        plot.save_figure("./assets/varied-depth-hsic_yz-{}.{}".format(
            config_dict['data_code'], config_dict['ext']))

        metadata = {
            'title':'Standard training accuracy (Varied-depth model)',
            'xlabel': 'epochs',
            'ylabel': 'training accuracy',
            'label': label_list
        }
        plot.plot_batches_log(input_list, 'batch_acc', metadata)
        plot.save_figure("./assets/varied-depth-acc-{}.{}".format(
            config_dict['data_code'], config_dict['ext']))


    else:
        raise ValueError("Unknown given task [{}], please check \
            hsicbt.dispatcher.job_execution".format(config_dict['task']))
    
