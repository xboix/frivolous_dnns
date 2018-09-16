import argparse
import logging
import sys
import glob
import numpy as np
import pandas as pd

from robustness.plot import plot, mock_cross_validation_if_necessary
from robustness.plot.plot_redundancy import information_per_neuron
#from robustness.plot.plot_redundancy import total_modes_network


import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams.update({'font.size': 14})
mpl.rc('axes', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('xtick', labelsize=14)

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    fig, ax = pyplot.subplots()
    layer_multiplier = 'multiplier_layer'
    #layer = '2'
    for multiplier in sorted(np.unique(data[layer_multiplier])):
        #layer_data = data[data['layer'] == layer].copy()
        layer_data = data[data[layer_multiplier] == multiplier]


        layer_data['num_components_per_neuron'] = information_per_neuron(layer_data)
        # plot means and stds
        layer_data = layer_data.groupby('training_amount_data')  # group across cross_validations
        means = layer_data['num_components_per_neuron'].mean()
        errs = layer_data['num_components_per_neuron'].std()
        means.plot( xticks=means.index, label=multiplier, logx=True, ax=ax)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.legend()
    ax.set_xlabel('training_amount_data')
    ax.set_ylabel('information per neuron\n= necessary components / number of neurons')
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


    #data = data.groupby(data.columns.difference(['layer', 'num_components']).tolist()).apply(total_modes_network).reset_index(drop=True)
    #data['num_neurons_layer5'] = data[
    #    [column for column in data if column.startswith('num_neurons_layer')]].sum(axis=1)

    data = data[data['preserved_energy'] == 0.95].copy()

    plot(data, plot_single_data,
         plotting_columns=['non_zero', 'training_amount_data', 'num_components', 'cross_validation', 'selectivity_mean','selectivity_std','selectivity_gen_mean','selectivity_gen_std']
                          + [column for column in data if column.startswith('multiplier_layer')]
                          + [column for column in data if column.startswith('num_neurons_layer')],
         ignored_key_columns=['num_components'],
         results_subdirectory='numdata_redundancy')


if __name__ == '__main__':
    main()
