import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .color import *
from .const import *
from .misc import *
from .path import *

def plot_epoch_log(curve_list, ptype, metadata):


    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    ax = fig.add_subplot(111)

    n = len(curve_list[0][ptype])
    xticks_idx = np.arange(n).tolist()
    xticks_val = np.arange(1,n+1).tolist()
    max_y = -1
    min_y = 1E5
    for i, curve_dict in enumerate(curve_list):
        ax.plot(curve_dict[ptype], linewidth=4, label=metadata['label'][i])
        max_y = max(max_y, np.max(curve_dict[ptype]))
        min_y = min(min_y, np.min(curve_dict[ptype]))
    plt.legend(fontsize=FONTSIZE_LEDEND)
    margin = 5
    max_y += margin
    min_y -= margin
    
    yticks_idx = np.linspace(min_y, max_y, 100)[::10]
    yticks_val = [np.round(x, 1) for x in np.linspace(min_y, max_y, 100)[::10]]

    ax.set_title(metadata['title'], fontsize=FONTSIZE_TITLE)
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xticks_val, fontsize=FONTSIZE_LEDEND)
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_val,  fontsize=FONTSIZE_YTICKS)
    ax.set_xlabel(metadata['xlabel'], fontsize=FONTSIZE_XLABEL)
    ax.set_ylabel(metadata['ylabel'], fontsize=FONTSIZE_YLABEL)


def plot_batches_log(curve_list, ptype, metadata):

    assert len(curve_list)>1, "this is for multiple curve plotting"

    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    ax = fig.add_subplot(111)


    n = len(curve_list[0][0][ptype])
    
    xticks_idx = np.arange(0, n*(len(curve_list[0])+1), n).tolist()
    xticks_val = np.arange(len(xticks_idx)).tolist()

    for i, curve_dict in enumerate(curve_list):

        val = [x[ptype] for x in curve_dict]
        val = [y for x in val for y in x]
        ax.plot(val, linewidth=4, label=metadata['label'][i])
        
    plt.legend(fontsize=FONTSIZE_LEDEND)
    ax.set_title(metadata['title'], fontsize=FONTSIZE_TITLE)
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xticks_val, fontsize=FONTSIZE_LEDEND)
    ax.set_xlabel(metadata['xlabel'], fontsize=FONTSIZE_XLABEL)
    ax.set_ylabel(metadata['ylabel'], fontsize=FONTSIZE_YLABEL)
    

def plot_activation_distribution():


    data = np.load('assets/tmp/activation-onehot.npy', allow_pickle=True)[()]
    activation_data = data['activation']
    label_data = data['label']
    label_index = []

    # # # calc average acc
    avg_acc = 0
    for i in range(10):
        indices = np.where(label_data==i)[0]
        select_item = activation_data[indices] # get all associated classed activation
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        num_correct = np.where(out==np.argmax(y))[0] # comparing how many samples match the maximum of mean
        accuracy = float(num_correct.shape[0]/out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.
    ylim_min = np.min(activation_data)
    ylim_max = np.max(activation_data)

    fig, ax = plt.subplots(2, 5, figsize=(30,10))
    ax = [x for a in ax for x in a]
    
    shuffled_list = []
    for i in range(10):

        subplot = ax[i]

        indices = np.where(label_data==i)[0]
        select_item = activation_data[indices] # extract activation associated with label
        out = np.array([np.argmax(vec) for vec in select_item]) # find the maximum arg of activation dist

        y = np.mean(select_item, axis=0)

        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0]/out.shape[0])

        e = np.std(select_item, axis=0)
        idx = np.arange(10).tolist()
        shuffled_list.append(int(np.argmax(y)))
        subplot.plot(y)
        subplot.fill_between(np.arange(10), y-e, y+e, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        #subplot.set_title("img class/argmax: {}/{}; category acc {:.2f}".format(i,int(np.argmax(y)), accuracy))
        subplot.set_xticks(idx, idx)
        subplot.set_ylim(ylim_min-2, ylim_max)
        # subplot.set_xlabel('10 dimension output activation indices')
        # subplot.set_ylabel('activation value')

    fig.suptitle("Output of HSIC network activations", fontsize=FONTSIZE_TITLE)
    fig.text(0.15, 0.05, "10 dimension output activation indices; shuffled argmax list {} Avg acc {:.2f}".format(shuffled_list, avg_acc), fontsize=FONTSIZE_XLABEL)
    fig.text(0.1, 0.6, "activation value", fontsize=FONTSIZE_YLABEL, rotation='vertical')

def plot_1d_activation_kde(datapath):

    data = np.load(datapath, allow_pickle=True)[()]
    activation_data = data['activation']
    label_data = data['label']

    from scipy.stats import gaussian_kde

    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    ax = fig.add_subplot(111)

    sample_idx = np.linspace(-3,3,150)
    for i in range(10):
        indices = np.where(label_data==i)[0]
        select_item = activation_data[indices] # get all associated classed activation

        kernel = gaussian_kde(np.squeeze(select_item))

        sampling = kernel(sample_idx)

        plt.plot(sampling, linewidth=3, label="c:{} m:{:.2f}".format(i, float(np.mean(select_item))))


    xticks_idx = np.arange(len(sample_idx))
    xticks_idx = list(xticks_idx[::25]) + [xticks_idx[-1]]
    xticks_val = list(sample_idx[::25]) + [sample_idx[-1]]
    xticks_val = [int(x) for x in xticks_val]

    # plt.legend(fontsize=22) 
    ax.set_title('class signals of dataset', fontsize=FONTSIZE_TITLE)
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xticks_val,  fontsize=FONTSIZE_XTICKS)
    ax.set_yticklabels(np.arange(10),  fontsize=FONTSIZE_YTICKS)
    ax.set_xlabel('tanh activation', fontsize=FONTSIZE_XLABEL)
    ax.set_ylabel('KDE density', fontsize=FONTSIZE_YLABEL)

def save_figure(filepath):
    timestamp_path = attaching_timestamp_filepath(filepath)
    plt.savefig(timestamp_path, bbox_inches='tight')
    make_symlink(timestamp_path, filepath)
    plt.clf()
    print_highlight("Saved [{}]".format(filepath))
