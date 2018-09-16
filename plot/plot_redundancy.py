import argparse
import logging
import sys

import numpy as np
import pandas as pd
import glob
from robustness.plot import plot, mock_cross_validation_if_necessary, WHOLE_NETWORK
import seaborn
import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import rc
rc('text', usetex=True)
_logger = logging.getLogger(__name__)

DATASET = "ImageNet"

def plot_single_data(data):
    fig, ax = pyplot.subplots()
    #data = data[~(data['layer'] == '4')].copy()
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


    nuance_settings = (nuance_setting.to_dict() for _, nuance_setting in nuance_settings.iterrows())
    name_settings = ['Dropout', 'All Regularizers','Data Augment.', 'Weight Decay', 'Not Regularized',  'Random Labels'  ]
    cc = ['skyblue', 'green', 'purple', 'olive', 'darkblue', 'red']

    linestyles = ["--","-","--", "--","-","-"]
    ll_width = [3, 4, 3, 3, 4, 4]

    for idx_plot, num_data in enumerate(nuance_settings):
        layer_data = data[data['training_dropout'] == num_data['training_dropout']].copy()
        layer_data = layer_data[layer_data['training_weight_decay'] == num_data['training_weight_decay']].copy()
        layer_data = layer_data[layer_data['random_labels'] == num_data['random_labels']].copy()
        layer_data = layer_data[layer_data['training_data_augmentation'] == num_data['training_data_augmentation']].copy()
        layer_data['num_components_per_neuron'] = 100*information_per_neuron(layer_data)
        if len(layer_data) == 0:
            _logger.warning("No data for num_data {}".format(num_data))
            continue
        # plot means and stds
        layer_data = layer_data.groupby('multiplier_layer')  # group across cross_validations
        means = layer_data['num_components_per_neuron'].mean()
        errs = layer_data['num_components_per_neuron'].std()
        means.plot(yerr=5*errs, xticks=means.index, linewidth=ll_width[idx[idx_plot]],
                   label=name_settings[idx[idx_plot]],  linestyle=linestyles[idx[idx_plot]], color=cc[idx[idx_plot]], logx=True, ax=ax),#yerr=errs)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())


    handles, labels = pyplot.gca().get_legend_handles_labels()
    #order = [5, 4, 1, 0, 2, 3]

    #pyplot.legend([handles[np.where(np.array(idx) == i)[0][0]] for i in order],
    #              [labels[np.where(np.array(idx) == i)[0][0]] for i in order],
    #              bbox_to_anchor=(1.05, 0.5), frameon=True)
    pyplot.xticks(means.index, ['1/4x', '1/2x', '1x', '2x', '4x'])

    #pyplot.ylim([10, 45])
    ax.set_title(DATASET + ' - Compressability ')
    ax.set_xlabel('Model Size Factor')
    ax.set_ylabel('Compressability (\%)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)

    pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.16,right=0.96)

    return fig


def information_per_neuron(layer_data):
    # get the number of neurons that is relevant for each row, denoted by that row's considered layer
    layer_num_neurons = [row['num_neurons_layer{}'.format(row['layer'])] for _, row in layer_data.iterrows()]
    return (layer_num_neurons - layer_data['num_components']) / layer_num_neurons


def add_whole_network_redundancy(data):
    grouped_data = data.groupby(list(set(data.columns) - {'layer', 'num_components', 'performance'}))

    def insert_whole_network(group):
        assert len(np.unique(group['layer'])) == len(group)  # no duplicate layers
        group = group.copy()
        whole_network_row = next(group.iterrows())[1]
        whole_network_row['layer'] = WHOLE_NETWORK
        # sum as the upper bound of the number of components
        whole_network_row['num_components'] = group['num_components'].sum()
        group = group.append(whole_network_row)
        # sum num_neurons of first row, all rows are the same
        num_neurons = sum(whole_network_row[[column for column in group.columns
                                             if column.startswith('num_neurons_layer')]])
        group['num_neurons_layer{}'.format(WHOLE_NETWORK)] = [num_neurons] * len(group)
        return group

    data = grouped_data.apply(insert_whole_network).reset_index(drop=True)
    assert WHOLE_NETWORK in data['layer'].values
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    data = pd.concat((pd.read_csv(f) for f in glob.glob(args.files[0])))
    del data['performance']
    data = mock_cross_validation_if_necessary(data)

    data = data[data['preserved_energy'] == 0.95].copy()
    #data = data[data['training_amount_data'] == 0.95].copy()


    #data = add_whole_network_redundancy(data)

    plot(data, plot_single_data,
         plotting_columns=[ 'not_selective','cross_validation','NN_ave','NN_std','training_dropout', 'training_weight_decay', 'training_data_augmentation', 'random_labels', 'non_zero', 'num_components', 'cross_validation', 'selectivity_mean','selectivity_std','selectivity_gen_mean','selectivity_gen_std']
                          + [column for column in data if column.startswith('multiplier_layer')]
                          + [column for column in data if column.startswith('num_neurons_layer')],
         ignored_key_columns=['num_components'], results_subdirectory='redundancy')


if __name__ == '__main__':
    main()
