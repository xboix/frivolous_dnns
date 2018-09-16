import argparse
import logging
import sys
import glob
import numpy as np
import pandas as pd

from robustness.plot import plot, mock_cross_validation_if_necessary
from robustness.plot.plot_redundancy import information_per_neuron, add_whole_network_redundancy

import seaborn
from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    plot_data = []
    for tt in np.unique(data['num_neurons_layerall']):
        for layer in np.unique(data['layer']):
            layer_data = data[data['layer'] == layer].copy()
            layer_data = layer_data[layer_data['num_neurons_layerall'] == tt].copy()

            train_data = layer_data[layer_data['evaluation_set'] == 'train set'].copy().reset_index(drop=True)
            test_data = layer_data[layer_data['evaluation_set'] == 'test set'].copy().reset_index(drop=True)
            train_data['num_components_per_neuron'] = information_per_neuron(train_data)

            # link train and test
            train_data.drop(['evaluation_set', 'performance'], axis=1, inplace=True)
            test_data.drop(['evaluation_set', 'num_components'], axis=1, inplace=True)
            redundancy_generalization_data = pd.merge(train_data, test_data)
            assert len(train_data) == len(test_data) == len(redundancy_generalization_data)
            plot_data.append(redundancy_generalization_data)

    # plot, averaged across cross_validations
    fig, _ = pyplot.subplots()
    plot_data = pd.concat(plot_data)
    ax = seaborn.lmplot(data=plot_data, x='num_components_per_neuron', y='performance',
                        hue='num_neurons_layerall', x_estimator=np.mean, fit_reg=True, ci=False, line_kws={'alpha': 0.3})
    ax.set_axis_labels('information per neuron', 'generalization = test performance')
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

    data = add_whole_network_redundancy(data)

    plot(data, plot_single_data,
         plotting_columns=['training_amount_data', 'layer', 'evaluation_set', 'training_amount_data', 'cross_validation',
                           'num_components', 'performance'] +
                          [column for column in data if column.startswith('num_neurons_layer')]
                          + [column for column in data if column.startswith('multiplier_layer')],
         ignored_key_columns=['num_components', 'performance'],
         results_subdirectory='generalization_redundancy')


if __name__ == '__main__':
    main()
