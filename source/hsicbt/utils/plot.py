import matplotlib.pyplot as plt
import numpy as np
from .color import *

def plot_epoch_log(curve_list, ptype, metadata):

    n = len(curve_list[0][ptype])
    xticks_idx = np.arange(n)
    xticks_val = np.arange(n)
    for i, curve_dict in enumerate(curve_list):
        plt.plot(curve_dict[ptype], label=metadata['label'][i])
        
    plt.legend()
    plt.title(metadata['title'])
    plt.xticks(xticks_idx, xticks_val)
    plt.xlabel(metadata['xlabel'])
    plt.ylabel(metadata['ylabel'])


def plot_batches_log(curve_list, ptype, metadata):

    assert len(curve_list)>1, "this is for multiple curve plotting"

    n = len(curve_list[0][0][ptype])
    
    xticks_idx = np.arange(0, n*len(curve_list[0])+1, n)
    xticks_val = np.arange(len(xticks_idx))

    for i, curve_dict in enumerate(curve_list):

        val = [x[ptype] for x in curve_dict]
        val = [y for x in val for y in x]
        plt.plot(val, label=metadata['label'][i])
        
    plt.legend()
    plt.title(metadata['title'])
    plt.xticks(xticks_idx, xticks_val)
    plt.xlabel(metadata['xlabel'])
    plt.ylabel(metadata['ylabel'])
    

def plot_activation_distribution():


    data = np.load('assets/activation-onehot.npy', allow_pickle=True)[()]
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
        subplot.set_title("img class/argmax: {}/{}; category acc {:.2f}".format(i,int(np.argmax(y)), accuracy, ))
        subplot.set_xticks(idx, idx)
        subplot.set_ylim(ylim_min-2, ylim_max)
        subplot.set_xlabel('10 dimension output activation indices')
        subplot.set_ylabel('activation value')
    fig.suptitle("Output of HSIC network activations \n shuffled argmax list {} Avg acc {:.2f}".format(shuffled_list, avg_acc), fontsize=18)


def plot_1d_activation_kde(datapath):

    data = np.load(datapath, allow_pickle=True)[()]
    activation_data = data['activation']
    label_data = data['label']

    from scipy.stats import gaussian_kde

    sample_idx = np.linspace(-3,3,150)
    for i in range(10):
        indices = np.where(label_data==i)[0]
        select_item = activation_data[indices] # get all associated classed activation

        kernel = gaussian_kde(np.squeeze(select_item))

        sampling = kernel(sample_idx)

        plt.plot(sampling, label="class:{} mean:{:.2f}".format(i, float(np.mean(select_item))))



    xticks_idx = np.arange(len(sample_idx))
    xticks_idx = list(xticks_idx[::25]) + [xticks_idx[-1]]
    xticks_val = list(sample_idx[::25]) + [sample_idx[-1]]
    xticks_val = [int(x) for x in xticks_val]
    plt.xticks(xticks_idx, xticks_val)
    plt.legend()

def save_figure(filepath):
    plt.savefig(filepath)
    plt.clf()
    print_highlight("Saved [{}]".format(filepath))