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

import itertools
import seaborn

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

    cc_normal = itertools.cycle(seaborn.light_palette("navy", reverse=False))
    cc_random = itertools.cycle(seaborn.light_palette("red", reverse=False))

    linestyles = ["--","-","--", "--","-","-"]
    ll_width = [3, 4, 3, 3, 4, 4]
    ll_factor = ['1/4x', '', '1x', '', '4x']
    for idx_plot, num_data in enumerate(nuance_settings):
        if idx[idx_plot] == 0 or  idx[idx_plot] == 2 or idx[idx_plot] == 3 or idx[idx_plot] == 1:
            continue

        if idx[idx_plot] == 4:
            cc = cc_normal
        else:
            cc = cc_random

        for idx_multi, mult in enumerate(sorted(np.unique(data['multiplier_layer']))):
            if mult == 0.5 or mult == 2:
                continue
            col = next(cc)
            layer_data = data[data['training_dropout'] == num_data['training_dropout']].copy()
            layer_data = layer_data[layer_data['multiplier_layer'] == mult]

            layer_data = layer_data[layer_data['training_weight_decay'] == num_data['training_weight_decay']].copy()
            layer_data = layer_data[layer_data['random_labels'] == num_data['random_labels']].copy()
            layer_data = layer_data[layer_data['training_data_augmentation'] == num_data['training_data_augmentation']].copy()
            if len(layer_data) == 0:
                _logger.warning("No data for num_data {}".format(num_data))
                continue
            # plot means and stds
            layer_data['not_selective'] = 100*layer_data['not_selective']
            layer_data = layer_data.groupby('layer')  # group across cross_validations
            means = layer_data['not_selective'].mean()
            errs = layer_data['not_selective'].std()
            means.plot(yerr=errs, xticks=means.index,linewidth=ll_width[idx[idx_plot]],
                       label=name_settings[idx[idx_plot]] + ' ' + ll_factor[idx_multi],
                       linestyle=linestyles[idx[idx_plot]], color=col, ax=ax),#yerr=errs)
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            col = next(cc)


    handles, labels = pyplot.gca().get_legend_handles_labels()
    order = [5, 4, 1, 0, 2, 3]

    pyplot.legend(
                  loc='upper right', frameon= True)
    pyplot.xticks(np.arange(4), np.arange(1, 5))

    pyplot.ylim([0, 100])
    if np.unique(data['evaluation_set']) == 'train set':
        ax.set_title(DATASET + ' - Train Set\n Non Selective Neurons')
    else:
        ax.set_title(DATASET + ' - Test Set\n Non Selective Neurons')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Amount of Non Selective Neurons (\%)')
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
    del data['performance']
    data = mock_cross_validation_if_necessary(data)

    data = data[data['preserved_energy'] == 0.95].copy()
    data = data[data['training_amount_data'] == 0.95].copy()
    data = data[data['layer'] != "all"].copy()

    #data = add_whole_network_redundancy(data)

    plot(data, plot_single_data,
         plotting_columns=[ 'layer','not_selective', 'cross_validation','NN_ave','NN_std','training_dropout', 'training_weight_decay', 'training_data_augmentation', 'random_labels', 'non_zero', 'num_components', 'cross_validation', 'selectivity_mean','selectivity_std','selectivity_gen_mean','selectivity_gen_std']
                          + [column for column in data if column.startswith('multiplier_layer')]
                          + [column for column in data if column.startswith('num_neurons_layer')],
         ignored_key_columns=['num_components'], results_subdirectory='Not_Selectivity_layer')


if __name__ == '__main__':
    main()
