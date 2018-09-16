import argparse
import logging
import sys

import numpy as np
import pandas as pd
import glob

from robustness.plot import plot, mock_cross_validation_if_necessary
from robustness.plot.plot_redundancy import information_per_neuron

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    fig, ax = pyplot.subplots()
    for layer in np.unique(data['layer']):
        layer_data = data[data['layer'] == layer].copy()
        layer_data['num_components_per_neuron'] = information_per_neuron(layer_data)
        layer_data = layer_data.groupby('training_amount_data')  # group across cross_validations
        layer_data = layer_data['num_components_per_neuron', 'performance']
        means, errs = layer_data.mean(), layer_data.std()
        means.plot(x='num_components_per_neuron', y='performance', yerr=errs, label=layer, ax=ax)
    ax.legend()
    ax.set_xlabel('information per neuron')
    ax.set_ylabel('performance')
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

    plot(data, plot_single_data,
         plotting_columns=['layer', 'training_amount_data', 'cross_validation', 'num_components', 'performance'],
         results_subdirectory='redundancy_performance')


if __name__ == '__main__':
    main()
