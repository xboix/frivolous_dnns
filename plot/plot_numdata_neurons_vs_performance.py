import argparse
import logging
import sys
import glob
import numpy as np
import pandas as pd
from matplotlib.mlab import griddata

from robustness.plot import plot, mock_cross_validation_if_necessary

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    #assert len(np.unique(data['perturbation_layer'])) == 1

    grouped = data.groupby(['training_amount_data', 'multiplier_layer'])  # across cross-validations
    performances = grouped[['training_amount_data', 'multiplier_layer', 'performance']].mean()
    x, y, z = performances['training_amount_data'], performances['multiplier_layer'], performances['performance']


    fig, ax = pyplot.subplots()
    if not len(x) == len(y) == len(z) == 25:
        return fig
    len_x, len_y = len(np.unique(x.round(decimals=8))), len(np.unique(y.round(decimals=8)))
    X = np.reshape(x, [len_x, len_y]).T
    Y = np.reshape(y, [len_x, len_y]).T
    Z = np.reshape(z, [len_x, len_y]).T
    contour = ax.contourf(X, Y, Z)
    fig.colorbar(contour, ax=ax)

    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xlabel('training_amount_data')
    ax.set_ylabel('multiplier')
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
    plot(data, plot_single_data,
         plotting_columns=['layer', 'training_amount_data', 'cross_validation', 'num_components', 'performance'] +
                          ['non_zero', 'num_components',  'selectivity_mean','selectivity_std','selectivity_gen_mean','selectivity_gen_std']
                         + [column for column in data if column.startswith('num_neurons_layer')]
                          + [column for column in data if column.startswith('multiplier_layer')],
         ignored_key_columns=['performance'],
         results_subdirectory='numdata_neurons_performance')


if __name__ == '__main__':
    main()
