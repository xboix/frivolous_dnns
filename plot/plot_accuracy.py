import argparse
import logging
import sys

import numpy as np
import pandas as pd
import glob
from plot import plot, mock_cross_validation_if_necessary, WHOLE_NETWORK
import seaborn
import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import rc
rc('text', usetex=True)
_logger = logging.getLogger(__name__)

DATASET = "ImageNet"

def plot_single_data(data):
    fig, ax = pyplot.subplots()
    data = data[data['layer'] == 'all'].copy()
    nuance_settings = data[['training_dropout', 'random_labels', 'training_weight_decay', 'training_data_augmentation']].drop_duplicates()
    idx = []
    for k in nuance_settings.iterrows():
        if k[1]['training_dropout'] == 0.5:
            if k[1]['training_data_augmentation'] == True:
                idx.append(1)
            else:
                idx.append(0)
        elif k[1]['training_weight_decay'] != 0:
            if k[1]['training_data_augmentation'] == True:
                idx.append(1)
            else:
                idx.append(3)
        elif k[1]['training_data_augmentation'] != 0:
            if k[1]['training_weight_decay'] == True:
                idx.append(1)
            else:
                idx.append(2)
        elif k[1]['random_labels'] == True:
            idx.append(5)
        else:
            idx.append(4)

    idx.append(6)

    nuance_settings = (nuance_setting.to_dict() for _, nuance_setting in nuance_settings.iterrows())
    name_settings = ['Dropout', 'All Regularizers','Data Augment.', 'Weight Decay', 'Not Regularized',  'Random Labels', 'Random Guess'  ]
    cc = ['skyblue', 'green', 'purple', 'olive', 'darkblue', 'red']

    linestyles = ["--","-","--", "--","-","-"]
    ll_width = [3, 4, 3, 3, 4, 4]

    for idx_plot, num_data in enumerate(nuance_settings):
        layer_data = data[data['training_dropout'] == num_data['training_dropout']].copy()
        layer_data = layer_data[layer_data['training_weight_decay'] == num_data['training_weight_decay']].copy()
        layer_data = layer_data[layer_data['random_labels'] == num_data['random_labels']].copy()
        layer_data = layer_data[layer_data['training_data_augmentation'] == num_data['training_data_augmentation']].copy()
        mm = layer_data
        if len(layer_data) == 0:
            _logger.warning("No data for num_data {}".format(num_data))
            continue
        # plot means and stds
        layer_data = layer_data.groupby('multiplier_layer')  # group across cross_validations
        means = 100*layer_data['performance'].mean()
        #errs = layer_data['num_components_per_neuron'].std()
        means.plot(xticks=means.index, label=name_settings[idx[idx_plot]],linewidth=ll_width[idx[idx_plot]],  linestyle=linestyles[idx[idx_plot]],
                   color=cc[idx[idx_plot]], logx=True, ax=ax),#yerr=errs)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    mm['performance'] = 0.1
    mm = mm.groupby('multiplier_layer')  # group across cross_validations
    means = 100 * mm['performance'].mean()
    # errs = layer_data['num_components_per_neuron'].std()
    means.plot(xticks=means.index, label="Random Guess", color='black', linestyle=':',
               logx=True, ax=ax),  # yerr=errs)

    handles, labels = pyplot.gca().get_legend_handles_labels()
    order = [5, 4, 1, 0, 2, 3, 6]

    #pyplot.legend([handles[np.where(np.array(idx) == i)[0][0]] for i in order],
    #              [labels[np.where(np.array(idx) == i)[0][0]] for i in order],
    #              bbox_to_anchor=(1., 0.7), frameon= True)
    pyplot.ylim([0, 100])
    if np.unique(data['evaluation_set']) == 'train set':
        ax.set_title(DATASET + ' - Accuracy Train Set')
    else:
        ax.set_title(DATASET + ' - Accuracy Test Set')
    ax.set_xlabel('Model Size Factor')
    ax.set_ylabel('Accuracy (\%)')
    pyplot.xticks(means.index, ['1/4x', '1/2x', '1x', '2x', '4x'])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)

    pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16, right=0.96)
    return fig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    data = pd.concat((pd.read_csv(f) for f in glob.glob(args.files[0])))
    data = mock_cross_validation_if_necessary(data)

    data = data[data['preserved_energy'] == 0.95].copy()

    #data = add_whole_network_redundancy(data)

    plot(data, plot_single_data,
         plotting_columns=['not_selective','cross_validation','NN_ave','NN_std','training_dropout', 'random_labels', 'training_weight_decay', 'training_data_augmentation']+
                        ['layer',  'cross_validation', 'num_components', 'performance'] +
                          ['non_zero', 'num_components',  'selectivity_mean','selectivity_std','selectivity_gen_mean','selectivity_gen_std']
                         + [column for column in data if column.startswith('num_neurons_layer')]
                          + [column for column in data if column.startswith('multiplier_layer')],
         ignored_key_columns=['num_components'], results_subdirectory='accuracy')


if __name__ == '__main__':
    main()
