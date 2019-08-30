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
        plot.save_figure("./assets/standard-hsic-comparison-{}-{}-batch.jpg".format(config_dict['data_code'], config_dict['model']))

        metadata = {
            'title':'comparison of format-train and standard-train',
            'xlabel': 'epochs',
            'ylabel': 'training accurarcy (eval at the end of epoch)',
            'label': ['stadnard-train', 'format-train']
        }
        plot.plot_epoch_log([out_standard_epoch, out_format_epoch], 'train_acc', metadata)
        plot.save_figure("./assets/standard-hsic-comparison-{}-{}-epoch-train-acc.jpg".format(config_dict['data_code'], config_dict['model']))

        metadata = {
            'title':'comparison of format-train and standard-train',
            'xlabel': 'epochs',
            'ylabel': 'test accurarcy (eval at the end of epoch)',
            'label': ['stadnard-train', 'format-train']
        }
        plot.plot_epoch_log([out_standard_epoch, out_format_epoch], 'test_acc', metadata)
        plot.save_figure("./assets/standard-hsic-comparison-{}-{}-epoch-test-acc.jpg".format(config_dict['data_code'], config_dict['model']))

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
        plot.save_figure("./assets/hsic-solve-comparison-test-acc.jpg")
        plot.plot_epoch_log([out_epoch, out_hsic_epoch], 'train_acc', metadata)
        plot.save_figure("./assets/hsic-solve-comparison-train-acc.jpg")
        plot.plot_activation_distribution()
        plot.save_figure("./assets/hsic-solve-actdist.jpg")


    elif config_dict['task'] == 'niddle':

        out_standard_batch, out_standard_epoch = training_standard(config_dict)
        out_hsic_batch    , out_hsic_epoch     = training_hsic(config_dict)

        plot.plot_1d_activation_kde('assets/activation-niddle-hsic.npy')
        plot.save_figure("./assets/activation-1d-dist-hsic.jpg")
        plot.plot_1d_activation_kde('assets/activation-niddle-standard.npy')
        plot.save_figure("./assets/activation-1d-dist-standard.jpg")
        
    else:
        raise ValueError("Unknown given task [{}], please check \
            hsicbt.dispatcher.job_execution".format(config_dict['task']))
    
